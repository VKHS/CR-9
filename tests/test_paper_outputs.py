"""scripts/paper_outputs.py and hpc/produce_paper_outputs_arrhenius.sh: the one-command production of the
paper's measured outputs (torch-free).

Planning runs on a reproduction suite laid out from the bundled example runs (synthetic, so --allow-fixture);
the complete job graph of all seven datasets is planned with a stand-in for the runners' own plan steps;
submission, status and the Slurm side run against mock sbatch/sacct/scontrol on PATH (the simulated-Slurm
approach of tests/test_reproduction_launcher.py). Every planned command line is parsed by its runner's own
argparse parser. No cluster, GPU or benchmark is contacted.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts import paper_outputs as po  # noqa: E402

EXAMPLES = PROJECT / "examples" / "runs"
MOCKS = {
    "sbatch": r'''#!/bin/bash
n_file="$MOCK_ROOT/next_id"; n=$(( $(cat "$n_file" 2>/dev/null || echo 1000) + 1 )); echo "$n" > "$n_file"
python3 -c 'import json,sys,os; open(os.environ["MOCK_ROOT"]+"/sbatch.jsonl","a").write(json.dumps({"id":sys.argv[1],"args":sys.argv[2:]})+"\n")' "$n" "$@"
if [[ " $* " == *" --wait "* ]]; then
  script="" args=() exports=()
  for a in "$@"; do
    if [[ -n "$script" ]]; then args+=("$a"); continue; fi
    case "$a" in --export=*) IFS=, read -r -a exports <<< "${a#--export=}" ;; --*) ;; *) script="$a" ;; esac
  done
  for e in "${exports[@]}"; do [[ "$e" == ALL ]] || export "$e"; done
  SLURM_JOB_ID="$n" bash "$script" "${args[@]}" > "$MOCK_ROOT/wait_$n.out" 2>&1 || { cat "$MOCK_ROOT/wait_$n.out" >&2; exit 1; }
fi
if [[ " $* " == *" --parsable "* ]]; then echo "$n"; else echo "Submitted batch job $n"; fi
''',
    "sacct": '#!/bin/bash\ncat "$MOCK_ROOT/sacct.txt" 2>/dev/null || true\n',
    "scontrol": '#!/bin/bash\necho "$*" >> "$MOCK_ROOT/scontrol.log"\nexit 0\n',
}


def make_suite(base: Path) -> Path:
    """A reproduction suite (000_MANIFEST/suite.json, runs/NN_<dataset>) holding the two example runs."""
    suite = base.resolve() / "CROSS_MORSE_REPRODUCTION_0001"
    runs = []
    for key, dataset, example in (("01_arxiv", "ogbn-arxiv", "arxiv"),
                                  ("07_peptides_func", "Peptides-func", "peptides_func")):
        path = suite / "runs" / key
        shutil.copytree(EXAMPLES / example, path)
        runs.append({"key": key, "dataset": dataset, "stem": key[3:], "kind": "single", "path": str(path),
                     "splits": None, "config": f"configs/{key[3:]}_smoke.json",
                     "launcher": str(path / "source" / "run_cross_morse.sh")})
    (suite / "000_MANIFEST").mkdir(parents=True)
    (suite / "000_MANIFEST" / "suite.json").write_text(json.dumps({
        "schema": "cross-morse-reproduction-suite-1", "tag": "CMR_0001", "number": 1, "suite": str(suite),
        "profile": "smoke", "created_utc": "2026-11-07T00:00:00Z", "datasets": ["ogbn-arxiv", "Peptides-func"],
        "data_root": str(base / "data"), "source": {"path": str(PROJECT), "runtime_sha256": "fixture"},
        "report_source": str(suite / "source"), "paper_comparison": False, "settings": {}, "runs": runs}))
    return suite


def fake_executor(argv):
    """Stands in for a runner's plan step: writes <output>/plan.json shaped like that runner's plan."""
    args = [str(a) for a in argv]
    out = Path(args[args.index("--output") + 1])
    parents = [Path(args[i + 1]) for i, a in enumerate(args) if a == "--parent"]
    members = sum(len(list(p.glob("split_*"))) or 1 for p in parents)
    script = Path(args[2]).name
    out.mkdir(parents=True, exist_ok=True)
    plan = {"schema": "fixture-plan", "argv": args}
    if script == "run_sweep.py":
        sweep = args[args.index("--sweep") + 1]
        if sweep == "buffer_sensitivity":
            plan.update(sweep=sweep, variants=[], settings={"jobs": [{"id": f"buffer{i}"} for i in range(len(parents))]})
        else:
            plan.update(sweep=sweep, settings={}, variants=[{"id": f"{sweep}_{i}", "output": f"runs/{i}"}
                                                           for i in range(members)])
    elif script == "run_experiment_b.py":
        plan["experiment"] = args[3]
    elif "modules" in args:
        plan["jobs"] = [{"kind": "lifecycle"}] * (3 * members) + [{"kind": "measure"}] * members
    elif "factorial-fit" in args:
        plan["jobs"] = [{"kind": "lifecycle"}] * 3
        (out / "factorial").mkdir(exist_ok=True)
        (out / "factorial" / "plan.json").write_text(json.dumps({"jobs": [{"cell": i} for i in range(8)]}))
    else:
        plan["jobs"] = [{"job": i} for i in range(members)]
    (out / "plan.json").write_text(json.dumps(plan))
    return 0, "fixture plan written"


def seven_runs(base: Path) -> dict:
    """Stand-ins for the seven sealed runs (Roman-empire: a campaign of ten split runs), as verify_runs returns them."""
    runs = {}
    for dataset in po.DATASETS:
        root = base / "suite" / "runs" / f"{po.DATASETS.index(dataset) + 1:02d}_{po.STEMS[dataset]}"
        members = [root / f"split_{i:02d}" for i in range(10)] if dataset == "Roman-empire" else [root]
        for index, member in enumerate(members):
            member.mkdir(parents=True)
            split = index if dataset == "Roman-empire" else None
            (member / "config.json").write_text(json.dumps({"benchmark_dataset": dataset, "profile": "full",
                                                            "data": {"split_index": split},
                                                            "training": {"seeds": list(range(1, 13))}}))
        runs[dataset] = {"path": str(root), "kind": "campaign" if dataset == "Roman-empire" else "single",
                         "key": root.name, "synthetic": False,
                         "members": [{"run": str(m), "training_seeds": list(range(1, 13)), "profile": "full",
                                      "scientific_run": True} for m in members]}
    return runs


def baseline_plans(base: Path) -> Path:
    directory = base / "baselines"
    for dataset, method, splits in (("ogbn-arxiv", "AMP", [0]), ("Roman-empire", "ADMP-GNN", list(range(10)))):
        target = directory / f"{po.STEMS[dataset]}_{method}"
        target.mkdir(parents=True)
        (target / "plan.json").write_text(json.dumps({
            "schema": "cm-upstream-study-1", "dataset": dataset, "method": method,
            "trials": [{"lr": 10.0 ** -k, "trial": k} for k in range(48)], "seed_ids": list(range(12)),
            "selection_role": "internal_training_validation", "direction": "maximize", "metric_unit": "native_0_to_1",
            "source_commit": "a" * 40, "adapter_commit": "b" * 40, "adapter_argv": ["{python}", "adapter.py"],
            "source_directory": "upstream", "adapter_directory": "adapter", "official_splits": splits}))
    return directory


def options(**overrides) -> dict:
    value = {"only": [], "allow_fixture": False, "allow_subset": False,
             "negative_controls_dataset": po.DEFAULT_NEGATIVE_CONTROLS_DATASET,
             "band_audit_dataset": po.DEFAULT_BAND_AUDIT_DATASET, "band_audit_baseline": po.DEFAULT_BAND_AUDIT_BASELINE,
             "arm_seeds": list(po.DEFAULT_ARM_SEEDS), "provisional": [], "data_root": "/data", "artifacts": {},
             "budget_campaign": None, "confirmation_campaign": None, "baselines": None, "environment": None}
    value.update(overrides)
    return value


def full_plan(base: Path) -> dict:
    """The complete graph of all seven datasets, every runner's plan stood in by fake_executor."""
    work = base / "work"
    work.mkdir()
    plan = po.build_plan(work, seven_runs(base), options(baselines=str(baseline_plans(base))),
                         required=po.required_artifact_ids(), executor=fake_executor,
                         repository={"git_commit": "c" * 40, "dirty": False, "runtime_sha256": po.runtime_fingerprint()})
    po.write_json(work / "plan.json", plan)
    return plan


class MockSlurm:
    def __init__(self, root: Path):
        self.root = root / "mock"
        (self.root / "bin").mkdir(parents=True)
        for name, text in MOCKS.items():
            path = self.root / "bin" / name
            path.write_text(text)
            path.chmod(0o755)
        (root / "home").mkdir()
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("CM_", "SLURM_", "MOCK_"))}
        self.env.update(PATH=str(self.root / "bin") + os.pathsep + self.env.get("PATH", ""), HOME=str(root / "home"),
                        MOCK_ROOT=str(self.root), CM_BASE=str(root / "base"), CM_ACCOUNT="fixture-account")

    def submissions(self) -> list[dict]:
        path = self.root / "sbatch.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []

    def run(self, *args, env=None, timeout=300):
        return subprocess.run([sys.executable, "-B", str(PROJECT / "scripts" / "paper_outputs.py"), *args],
                              env=dict(self.env, **(env or {})), capture_output=True, text=True, timeout=timeout,
                              cwd=PROJECT)


def option(args, name):
    return next((a.split("=", 1)[1] for a in args if a.startswith(name + "=")), None)


class PaperOutputsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cm paper outputs ")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)

    # ------------------------------------------------------------------ planning
    def plan_examples(self, *extra, executor=fake_executor):
        suite = make_suite(self.base)
        args = po.parser().parse_args(["plan", "--runs", str(suite), "--work", str(self.base / "work"),
                                       "--allow-fixture", "--negative-controls-dataset", "Peptides-func", *extra])
        with contextlib.redirect_stdout(io.StringIO()) as summary:
            self.assertEqual(po.command_plan(args, executor=executor), 0)
        self.assertIn("plan.json", summary.getvalue())
        return suite, po.load_plan(self.base / "work")

    def test_example_suite_is_discovered_verified_and_planned(self):
        suite, plan = self.plan_examples()
        self.assertEqual(plan["schema"], po.SCHEMA)
        self.assertEqual(set(plan["runs"]), {"ogbn-arxiv", "Peptides-func"})
        self.assertEqual(plan["artifacts"]["run:ogbn-arxiv"], {"status": "run", "path": str(suite / "runs/01_arxiv")})
        self.assertEqual(plan["artifacts"]["run:Peptides-func"]["path"], str(suite / "runs/07_peptides_func"))
        self.assertIn("no sealed ogbn-proteins run", plan["artifacts"]["run:ogbn-proteins"]["reason"])
        self.assertEqual(sorted(plan["artifacts"]), sorted(po.required_artifact_ids()))
        nodes = {n["id"]: n for n in plan["nodes"]}
        # one input gate per admitted run (here the two example runs), behind which the controls are planned
        self.assertEqual(nodes["input_gate.run"]["array"], 2)
        self.assertEqual(nodes["negative_controls.run"]["dependencies"], [{"node": "input_gate.run", "type": "afterok"}])
        gate = next(t for t in nodes["input_gate.run"]["tasks"] if "07_peptides_func" in " ".join(t["fresh"]["plan"]))
        control = nodes["negative_controls.run"]["tasks"][0]["fresh"]["plan"]
        self.assertEqual(control[control.index("--gate-from") + 1], gate["fresh"]["directory"])
        # only PascalVOC-SP holds the horizon and beam sweep: no run, nothing planned
        self.assertIn("no sealed run of PascalVOC-SP", plan["artifacts"]["records:horizon_beam_sweep"]["reason"])
        self.assertTrue(plan["options"]["allow_fixture"])

    def test_synthetic_runs_are_refused_without_the_fixture_flag(self):
        suite = make_suite(self.base)
        args = po.parser().parse_args(["plan", "--runs", str(suite / "runs"), "--work", str(self.base / "work")])
        with self.assertRaises(po.PlanError) as caught:
            po.command_plan(args, executor=fake_executor)
        self.assertIn("not a full scientific run", str(caught.exception))
        self.assertIn("sealed by runtime source", str(caught.exception))
        self.assertFalse((self.base / "work" / "plan.json").exists())

    def test_runs_are_given_per_dataset_and_checked(self):
        found, _ = po.discover_runs(None, [f"ogbn-arxiv={EXAMPLES / 'arxiv'}"])
        self.assertEqual(found["ogbn-arxiv"]["kind"], "single")
        with self.assertRaises(po.PlanError):
            po.discover_runs(None, [f"COCO-SP={EXAMPLES / 'arxiv'}"])
        with self.assertRaises(po.PlanError):
            po.discover_runs(None, ["arxiv=/nonexistent"])
        broken = self.base / "broken"
        shutil.copytree(EXAMPLES / "arxiv", broken)
        (broken / "final" / "results_summary.json").write_text("{}")
        found, _ = po.discover_runs(None, [f"ogbn-arxiv={broken}"])
        with self.assertRaises(po.PlanError) as caught:
            po.verify_runs(found, runtime=po.runtime_fingerprint(), allow_fixture=True)
        self.assertIn("not a sealed eight-stage run", str(caught.exception))

    def test_real_runner_plans_size_the_arrays(self):
        names = ["ensemble_grid", "tube_audit", "depth_diagnostics", "runtime_scaling", "timing_anchors",
                 "mechanism_contrasts", "oracle_dispersion"]
        _, plan = self.plan_examples("--only", *names, executor=po._subprocess_executor)
        nodes = {n["id"]: n for n in plan["nodes"]}
        for name in names:
            node = nodes[f"{name}.run"]
            self.assertFalse(node.get("blocked"), node.get("blocked"))
            jobs = po.read_json(Path(node["outputs"][0]) / "plan.json")["jobs"]
            self.assertEqual(len(node["tasks"]), len(jobs), name)
            self.assertEqual([t["commands"][0][t["commands"][0].index("--job") + 1] for t in node["tasks"]],
                             [str(k) for k in range(len(jobs))])
        self.assertEqual(nodes["timing_anchors.run"]["resource"], po.CPU)
        self.assertEqual(nodes["timing_anchors.run"]["dependencies"], [{"node": "runtime_scaling.run", "type": "afterok"}])
        self.assertIn("--scaling-output", po.read_json(Path(nodes["timing_anchors.run"]["outputs"][0]) / "plan.json")
                      .get("scaling", {}) and ["--scaling-output"])
        self.assertEqual(plan["artifacts"]["records:input_gate"]["status"], "not_selected")
        self.assertEqual(plan["artifacts"]["records:tube_audit"]["status"], "planned")

    def test_complete_graph_dependencies_resources_and_outputs(self):
        plan = full_plan(self.base)
        self.assertEqual({a for a, e in plan["artifacts"].items() if e["status"] not in ("planned", "run")}, set())
        nodes = {n["id"]: n for n in plan["nodes"]}
        deps = {k: {(d["node"], d["type"]) for d in n["dependencies"]} for k, n in nodes.items()}
        afterok = lambda *names: {(n, "afterok") for n in names}  # noqa: E731
        self.assertEqual(deps["negative_controls.run"], afterok("input_gate.run"))
        self.assertEqual(deps["module_ablations.plan"], afterok("input_gate.run", "negative_controls.run"))
        self.assertEqual(deps["module_ablations.lifecycles"], afterok("module_ablations.plan"))
        self.assertEqual(deps["module_ablations.measure"], afterok("module_ablations.lifecycles"))
        self.assertEqual(deps["decision_audit.plan"], afterok("input_gate.run"))
        self.assertEqual(deps["decision_audit.run"], afterok("decision_audit.plan"))
        self.assertEqual(deps["timing_anchors.run"], afterok("runtime_scaling.run"))
        self.assertEqual(deps["s14_campaign.assemble"], afterok("s14_campaign.run"))
        self.assertEqual(deps["confirmation.plan"], afterok("s14_campaign.assemble"))
        self.assertEqual(deps["confirmation.run"], afterok("confirmation.plan"))
        self.assertEqual(deps["factorials.cells"], afterok("factorials.lifecycles"))
        self.assertEqual(deps["budget.studies.roman_empire_split_03"], afterok("budget.parent.roman_empire_split_03"))
        self.assertEqual(deps["baselines.arxiv_AMP.select"], {("baselines.arxiv_AMP.hpo", "afterany")})
        self.assertEqual(deps["baselines.roman_empire_ADMP-GNN.final.split9"], afterok("baselines.roman_empire_ADMP-GNN.select"))
        order = [n["id"] for n in plan["nodes"]]
        for node, before in deps.items():
            for dep, _ in before:
                self.assertLess(order.index(dep), order.index(node), f"{dep} is not planned before {node}")
        exports = {e["id"]: e for e in plan["exports"]}
        self.assertEqual(exports["export.ensemble_grid_scores"]["after"], ["ensemble_grid.run"])
        # resources and array sizes
        gpu = {k for k, n in nodes.items() if n["resource"] == po.GPU}
        cpu = set(nodes) - gpu
        self.assertEqual(cpu, {"timing_anchors.run", "buffer_sensitivity.run", "module_ablations.plan",
                               "decision_audit.plan", "confirmation.plan", "s14_campaign.assemble",
                               "baselines.arxiv_AMP.select", "baselines.roman_empire_ADMP-GNN.select"})
        self.assertEqual(nodes["input_gate.run"]["array"], 15)               # 5 single admitted runs + 10 Roman splits
        self.assertEqual(nodes["ensemble_grid.run"]["array"], 15)
        self.assertEqual(nodes["module_ablations.lifecycles"]["array"], 48)
        self.assertEqual(nodes["module_ablations.measure"]["array"], 16)
        self.assertEqual(nodes["module_ablations.measure"]["tasks"][0]["job_in"]["index"], 48)
        cell = nodes["factorials.cells"]["tasks"][0]["commands"][0]
        self.assertEqual(cell[cell.index("--job") + 1], "3")                # the cells follow the 3 lifecycles
        self.assertEqual(nodes["factorials.cells"]["array"], 8)
        self.assertEqual(nodes["decision_audit.run"]["array"], 16)
        self.assertEqual(nodes["confirmation.run"]["array"], 36)              # every trained seed of 3 runs (bound)
        self.assertTrue(nodes["confirmation.run"]["tasks"][0]["job_in"]["upper_bound"])
        self.assertEqual(len([k for k in nodes if k.startswith("budget.parent.")]), 15)
        self.assertEqual(nodes["budget.studies.arxiv"]["array"], 12)
        self.assertEqual(nodes["baselines.arxiv_AMP.hpo"]["array"], 48)
        self.assertEqual(len([k for k in nodes if k.startswith("baselines.roman_empire_ADMP-GNN.final")]), 10)
        self.assertEqual(nodes["sensitivity_grids.variants"]["array"], 16)
        self.assertEqual(nodes["buffer_sensitivity.run"]["array"], 2)
        # the inputs of the deferred plans are the sealed outputs of the earlier nodes
        modules = nodes["module_ablations.plan"]["plan_step"]["argv"]
        self.assertEqual(modules.count("--input-gate"), 15)
        self.assertEqual(modules.count("--negative-controls"), 1)
        audit = nodes["decision_audit.plan"]["plan_step"]["argv"]
        self.assertEqual(audit.count("--gate-from"), 15)
        roman_gate = next(t["fresh"]["directory"] for t in nodes["input_gate.run"]["tasks"]
                          if t["fresh"]["directory"].endswith("roman_empire_split_09"))
        self.assertIn("Roman-empire/split_09=" + roman_gate, audit)
        self.assertIn("Peptides-func=" + nodes["input_gate.run"]["tasks"][-1]["fresh"]["directory"], audit)
        self.assertEqual(audit[audit.index("--stress-dataset") + 1], "PascalVOC-SP")
        confirmation = nodes["confirmation.plan"]["plan_step"]["argv"]
        self.assertEqual(confirmation[confirmation.index("--campaign") + 1], str(self.base / "work/campaigns/confirmation"))
        parent = nodes["budget.parent.roman_empire_split_03"]["tasks"][0]["commands"]
        self.assertEqual([c[c.index("--stage") + 1] for c in parent], list(po.PRE_ADMISSION))
        self.assertEqual(parent[0][parent[0].index("--split-index") + 1], "3")
        study = nodes["budget.studies.arxiv"]["tasks"][5]
        self.assertEqual(study["commands"], [["bash", "hpc/run_budget_arms_arrhenius.sh"]])
        self.assertEqual(study["env"]["CM_ARM_SEED"], "5")
        self.assertEqual(study["env"]["CM_BASE_RUN"], str(self.base / "work/budget/parents/arxiv"))
        # outputs: every planned artifact has its directory and its producing nodes
        for artifact, entry in plan["artifacts"].items():
            if entry["status"] == "planned":
                self.assertTrue(entry["path"], artifact)
                self.assertTrue(all(n in nodes for n in entry["nodes"]), artifact)
        self.assertEqual(plan["artifacts"]["records:decision_traces"]["path"], str(self.base / "work/records/decision_traces"))

    def test_internal_baselines_nodes_feed_the_baselines_export(self):
        plan = full_plan(self.base)
        nodes = {n["id"]: n for n in plan["nodes"]}
        internal = nodes["baselines.internal.run"]
        self.assertEqual((internal["resource"], internal["array"], internal["dependencies"]), (po.GPU, 16, []))
        self.assertFalse(internal["requeue"])
        self.assertFalse(any(t["requeue"] for t in internal["tasks"]))       # one allocation per timing job
        runner_plan = po.read_json(Path(internal["outputs"][0]) / "plan.json")["argv"]
        self.assertEqual(runner_plan.count("--parent"), 7)                   # every run; Roman: its campaign
        self.assertEqual([t["commands"][0][t["commands"][0].index("--job") + 1] for t in internal["tasks"]],
                         [str(k) for k in range(16)])
        self.assertIn("--trust-local-checkpoints", internal["tasks"][0]["commands"][0])
        export = {e["id"]: e for e in plan["exports"]}["export.baselines"]
        self.assertIn("baselines.internal.run", export["after"])
        self.assertIn("baselines.roman_empire_ADMP-GNN.final.split9", export["after"])
        capture, command = export["steps"][0]["argv"], export["steps"][1]["argv"]
        self.assertIn("{jobs:baselines.internal.run}", capture[2])          # the timing allocations' sacct records
        self.assertEqual(command[command.index("--internal") + 1], internal["outputs"][0])
        self.assertEqual(command[command.index("--checkouts") + 1], str((self.base / "baselines").resolve()))
        self.assertEqual(command.count("--study"), 2)
        self.assertEqual(export["provenance"]["internal"], internal["outputs"][0])
        self.assertIn("baselines.internal.run", plan["artifacts"]["records:baselines"]["nodes"])
        # map coverage: the baselines set is published once its export completed, awaiting otherwise
        results = {e["id"]: {"status": "COMPLETE"} for e in plan["exports"]}
        mapping = po.publication_map(plan, results, self.base / "work")
        self.assertEqual(mapping["records:baselines"], str(self.base / "work/records/baselines"))
        failed = dict(results, **{"export.baselines": {"status": "FAILED", "reason": "the baselines set is incomplete"}})
        mapping = po.publication_map(plan, failed, self.base / "work")
        self.assertEqual((mapping["records:baselines"], mapping["_awaiting"]["records:baselines"]),
                         ("", "the baselines set is incomplete"))
        # the runner's own plan step sizes the node; a refused plan leaves the set awaiting with its reason
        work = self.base / "work2"
        work.mkdir()

        def refusing(argv):
            if any(str(a).endswith("run_internal_baselines.py") for a in argv):
                return 2, "ERROR: arxiv: the lifecycle ran on cpu, not on CUDA"
            return fake_executor(argv)
        refused = po.build_plan(work, seven_runs(self.base / "b"), options(baselines=str(baseline_plans(self.base / "b"))),
                                required=po.required_artifact_ids(), executor=refusing,
                                repository={"git_commit": "c" * 40, "dirty": False})
        self.assertEqual(refused["artifacts"]["records:baselines"]["status"], "awaiting")
        self.assertIn("not on CUDA", refused["artifacts"]["records:baselines"]["reason"])

    def test_the_real_internal_baselines_plan_sizes_its_node_on_a_smoke_suite(self):
        directory = baseline_plans(self.base / "plans")
        _, plan = self.plan_examples("--only", "baselines", "--baselines", str(directory),
                                     executor=po._subprocess_executor)
        nodes = {n["id"]: n for n in plan["nodes"]}
        internal = nodes["baselines.internal.run"]
        self.assertFalse(internal.get("blocked"), internal.get("blocked"))
        runner_plan = po.read_json(Path(internal["outputs"][0]) / "plan.json")
        self.assertEqual(runner_plan["runner"], "internal_baselines")
        self.assertTrue(runner_plan["allow_fixture"])                      # the smoke runs ran on CPU
        self.assertEqual(len(internal["tasks"]), len(runner_plan["jobs"]))
        self.assertEqual(plan["artifacts"]["records:baselines"]["status"], "planned")

    def test_baselines_hardware_includes_the_internal_jobs(self):
        study = self.base / "studies" / "arxiv_AMP"
        (study / "hpo" / "trial-000").mkdir(parents=True)
        po.write_json(study / "hpo" / "trial-000" / "invocation.json", {"python": "3.13.5 (main)", "slurm": {}})
        internal = self.base / "internal"
        po.write_json(internal / "jobs" / "ogbn-arxiv" / "arxiv" / "environment.json",
                      {"gpu": "NVIDIA GH200 96GB", "torch": "2.12.1", "python": "3.13.5"})
        hardware, _ = po.derive_provenance({"kind": "baselines", "studies": [str(study)], "internal": str(internal),
                                            "accounting": str(self.base / "absent.psv")})
        self.assertIn("NVIDIA GH200 96GB; torch 2.12.1; python 3.13.5", hardware)

    def test_every_planned_command_parses_with_its_runner(self):
        plan = full_plan(self.base)
        commands = []
        for node in plan["nodes"]:
            for task in node["tasks"]:
                commands += [(c, task.get("env") or {}) for c in task["commands"]]
                if task.get("fresh", {}).get("plan"):
                    commands.append((task["fresh"]["plan"], {}))
            if node.get("plan_step"):
                commands.append((node["plan_step"]["argv"], {}))
        for export in plan["exports"]:
            commands += [(s["argv"], {}) for s in export["steps"] if "argv" in s]
        checked = set()
        for argv, env in commands:
            shape = tuple(a for a in argv[:5] if not a.startswith("/")) + tuple(sorted(a for a in argv if a.startswith("--")))
            if shape in checked:
                continue
            checked.add(shape)
            if argv[0] == "{python}":
                po.check_cli(argv)
            else:
                self.assertEqual(argv[0], "bash")
                text = (PROJECT / argv[1]).read_text()
                for name in env:
                    self.assertIn(name, text, f"{argv[1]} reads no {name}")
        self.assertGreater(len(checked), 40)
        with self.assertRaises(po.PlanError), contextlib.redirect_stderr(io.StringIO()):
            po.check_cli(po.py("scripts/run_experiment.py", "ensemble-grid", "run", "--output", "/x", "--no-such-flag"))

    def test_experiment_datasets_follow_the_experiment_registry(self):
        registry = json.loads((PROJECT / "release/experiment_code.json").read_text())["experiments"]
        stems = {v: k for k, v in po.STEMS.items()}
        compared = 0
        for entry in registry:
            datasets = {stems[Path(c).name.rsplit("_full", 1)[0]] for c in entry["configurations"]}
            for output in entry["outputs"]:
                experiment = po.ARTIFACT_EXPERIMENT.get(output)
                if experiment and datasets:
                    self.assertEqual(set(po.EXPERIMENT_DATASETS[experiment]), datasets, output)
                    compared += 1
        self.assertGreaterEqual(compared, 20)
        required = [a for a in po.required_artifact_ids() if not a.startswith("run:")]
        self.assertEqual([a for a in required if a not in po.ARTIFACT_EXPERIMENT], [])

    def test_provided_artifacts_replace_their_production(self):
        work = self.base / "work"
        work.mkdir()
        campaign = self.base / "s14"
        campaign.mkdir()
        plan = po.build_plan(work, seven_runs(self.base), options(confirmation_campaign=str(campaign),
                                                                  budget_campaign=str(self.base),
                                                                  artifacts={"records:baselines": str(self.base)}),
                             required=po.required_artifact_ids(), executor=fake_executor,
                             repository={"git_commit": "c" * 40, "dirty": False})
        ids = {n["id"] for n in plan["nodes"]}
        self.assertFalse({i for i in ids if i.startswith(("s14_campaign", "budget.", "baselines."))})
        self.assertEqual(plan["artifacts"]["campaign:confirmation"], {"status": "provided", "path": str(campaign)})
        self.assertEqual(plan["artifacts"]["records:baselines"]["status"], "provided")
        nodes = {n["id"]: n for n in plan["nodes"]}
        self.assertNotIn("confirmation.plan", nodes)           # planned now: the campaign exists
        self.assertEqual(nodes["confirmation.run"]["dependencies"], [])
        exports = {e["id"]: e for e in plan["exports"]}
        self.assertIn("export.budget_matched", exports)
        self.assertEqual(plan["artifacts"]["records:budget_matched"]["status"], "planned")

    def test_missing_inputs_leave_their_artifacts_awaiting_with_the_reason(self):
        work = self.base / "work"
        work.mkdir()
        plan = po.build_plan(work, seven_runs(self.base), options(), required=po.required_artifact_ids(),
                             executor=fake_executor, repository={"git_commit": "c" * 40, "dirty": True})
        self.assertIn("--baselines DIR", plan["artifacts"]["records:baselines"]["reason"])
        self.assertIn("clean git checkout", plan["artifacts"]["campaign:budget"]["reason"])

        def refusing(argv):
            if "input-gate" in argv:
                return 2, "ERROR: the parent was produced by another runtime source"
            return fake_executor(argv)
        work = self.base / "work2"
        work.mkdir()
        plan = po.build_plan(work, seven_runs(self.base / "b"), options(), required=po.required_artifact_ids(),
                             executor=refusing, repository={"git_commit": "c" * 40, "dirty": False})
        for artifact in ("records:input_gate", "records:negative_controls", "records:module_ablations",
                         "records:decision_traces"):
            self.assertEqual(plan["artifacts"][artifact]["status"], "awaiting", artifact)
        self.assertIn("another runtime source", plan["artifacts"]["records:input_gate"]["reason"])
        self.assertFalse(any(Path(n["outputs"][0]).exists() for n in plan["nodes"] if n["id"] == "input_gate.run"))

    # ------------------------------------------------------------------ submission and status
    def test_submit_chains_afterok_dependencies_with_the_site_settings(self):
        plan = full_plan(self.base)
        slurm = MockSlurm(self.base)
        settings = {"CM_PARTITION": "gpu-a", "CM_GPU_REQUEST": "--gres=gpu:1", "CM_THREADS": "12",
                    "CM_REPORT_PARTITION": "cpu-b", "CM_REPORT_GPU_REQUEST": "", "CM_MEM_ARXIV": "200G",
                    "CM_SBATCH_EXTRA": "--qos=normal", "CM_PRODUCE_GPU_TIME": "36:00:00"}
        work = str(self.base / "work")
        dry = slurm.run("submit", "--work", work, "--dry-run", env=settings)
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertEqual(slurm.submissions(), [])
        self.assertIn("sbatch --parsable", dry.stdout)
        self.assertIn("publish_paper_artifacts_arrhenius.sh all", dry.stdout)
        result = slurm.run("submit", "--work", work, env=settings)
        self.assertEqual(result.returncode, 0, result.stderr)
        jobs = po.read_json(Path(work) / "jobs.json")["nodes"]
        submitted = slurm.submissions()
        self.assertEqual(len(submitted), len(plan["nodes"]) + 1)
        by_node = {Path(s["args"][-1]).name[:-len(".sbatch")]: s for s in submitted}
        nodes = {n["id"]: n for n in plan["nodes"]}
        for node_id, record in by_node.items():
            args = record["args"]
            self.assertIn("--account=fixture-account", args)
            self.assertIn("--qos=normal", args)
            if node_id == "collect":
                continue
            self.assertEqual(jobs[node_id]["job_id"], record["id"])
            node = nodes[node_id]
            if node["resource"] == po.GPU:
                self.assertEqual(option(args, "--partition"), "gpu-a")
                self.assertIn("--gres=gpu:1", args)
                self.assertEqual(option(args, "--cpus-per-task"), "12")
                self.assertEqual(option(args, "--time"), "36:00:00")
            else:
                self.assertEqual(option(args, "--partition"), "cpu-b")
                self.assertNotIn("--gres=gpu:1", args)
                self.assertEqual(option(args, "--cpus-per-task"), "8")
            self.assertEqual(option(args, "--array"), f"0-{node['array'] - 1}" if node.get("array") else None)
            expected = {}
            for dep in node["dependencies"]:
                expected.setdefault(dep["type"], []).append(by_node[dep["node"]]["id"])
            dependency = option(args, "--dependency")
            if expected:
                self.assertEqual(dependency, ",".join(f"{k}:{':'.join(v)}" for k, v in expected.items()))
                self.assertIn("--kill-on-invalid-dep=yes", args)
                self.assertTrue(all(int(v) < int(record["id"]) for ids in expected.values() for v in ids))
            else:
                self.assertIsNone(dependency)
        self.assertEqual(option(by_node["budget.parent.arxiv"]["args"], "--mem"), "200G")
        self.assertIsNone(option(by_node["budget.parent.proteins"]["args"], "--mem"))
        self.assertIn("--signal=B:USR1@180", by_node["ensemble_grid.run"]["args"])
        self.assertNotIn("--requeue", by_node["input_gate.run"]["args"])      # group-B runs restart fresh
        collect = by_node["collect"]["args"]
        self.assertEqual(sorted(option(collect, "--dependency").split(":")[1:]),
                         sorted(r["id"] for k, r in by_node.items() if k != "collect"))
        script = (Path(work) / "slurm" / "module_ablations.plan.sbatch").read_text()
        self.assertIn('eval "$(bash "$ROOT/run_cross_morse.sh" site-env)"', script)
        self.assertIn("execute --work \"$WORK\" --node module_ablations.plan", script)
        self.assertNotIn("torch.cuda.is_available", script)
        self.assertIn("torch.cuda.is_available", (Path(work) / "slurm" / "ensemble_grid.run.sbatch").read_text())
        self.assertIn('paper_outputs.py" collect --work', (Path(work) / "slurm" / "collect.sbatch").read_text())
        again = slurm.run("submit", "--work", work, env=settings)
        self.assertEqual(again.returncode, 2)
        self.assertIn("--resume", again.stderr)
        # resume: complete nodes are not resubmitted and drop out of the dependencies
        for k in range(15):
            po.write_marker(work, "input_gate.run", k, "COMPLETE", exit_code=0)
        before = len(slurm.submissions())
        resumed = slurm.run("submit", "--work", work, "--resume", env=settings)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        later = {Path(s["args"][-1]).name[:-len(".sbatch")]: s for s in slurm.submissions()[before:]}
        self.assertNotIn("input_gate.run", later)
        self.assertIsNone(option(later["negative_controls.run"]["args"], "--dependency"))
        self.assertIn("input_gate.run", resumed.stdout)

    def test_blocked_nodes_and_their_dependents_are_not_submitted(self):
        _, plan = self.plan_examples()
        nodes = {n["id"]: n for n in plan["nodes"]}
        nodes["input_gate.run"]["blocked"] = "fixture: refused"
        po.write_json(self.base / "work" / "plan.json", dict(plan, nodes=list(nodes.values())))
        order, skipped, _ = po.submission_order(po.load_plan(self.base / "work"), self.base / "work", resume=False)
        self.assertIn("input_gate.run", skipped)
        self.assertIn("negative_controls.run", skipped)
        self.assertIn("module_ablations.plan", skipped)
        self.assertNotIn("input_gate.run", [n["id"] for n in order])

    def test_status_from_sacct_and_from_task_markers(self):
        full_plan(self.base)
        slurm = MockSlurm(self.base)
        work = self.base / "work"
        result = slurm.run("submit", "--work", str(work))
        self.assertEqual(result.returncode, 0, result.stderr)
        jobs = po.read_json(work / "jobs.json")["nodes"]
        grid, gate = jobs["ensemble_grid.run"]["job_id"], jobs["input_gate.run"]["job_id"]
        (slurm.root / "sacct.txt").write_text(f"{grid}_0|COMPLETED\n{grid}_1|RUNNING\n{gate}_0|FAILED\n")
        status = slurm.run("status", "--work", str(work), "--json")
        rows = {r["node"]: r for r in json.loads(status.stdout)["nodes"]}
        self.assertEqual(json.loads(status.stdout)["source"], "sacct")
        self.assertEqual(rows["ensemble_grid.run"]["state"], "RUNNING")
        self.assertEqual(rows["input_gate.run"]["state"], "FAILED")
        (slurm.root / "bin" / "sacct").unlink()
        po.write_marker(work, "environment.capture", 0, "COMPLETE", exit_code=0)
        po.write_marker(work, "tube_audit.run", 0, "FAILED", exit_code=2)
        status = slurm.run("status", "--work", str(work))
        self.assertIn("task markers", status.stdout)
        self.assertRegex(status.stdout, r"environment.capture\s+COMPLETE")
        self.assertRegex(status.stdout, r"tube_audit.run\s+FAILED")

    # ------------------------------------------------------------------ execution inside a job
    def tiny_plan(self, nodes):
        work = self.base / "work"
        work.mkdir(exist_ok=True)
        plan = {"schema": po.SCHEMA, "work": str(work), "options": {"data_root": "/data"},
                "repository": {"runtime_sha256": po.runtime_fingerprint()}, "nodes": nodes, "exports": [],
                "artifacts": {}}
        po.write_json(work / "plan.json", plan)
        return work

    def execute(self, work, node, task=None, **env):
        command = ["execute", "--work", str(work), "--node", node] + ([] if task is None else ["--task", str(task)])
        old = dict(os.environ)
        os.environ.update({k: str(v) for k, v in env.items()})
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return po.main(command)
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_execute_runs_resumes_skips_and_requeues(self):
        out = self.base / "out"
        writer = ["{python}", "-c", f"import pathlib,sys; pathlib.Path(r'{out}').mkdir(exist_ok=True); "
                                    f"(pathlib.Path(r'{out}') / ('done' + sys.argv[1])).write_text('{{threads}}')", "x"]
        plan_file = self.base / "exp" / "plan.json"
        plan_file.parent.mkdir()
        plan_file.write_text(json.dumps({"jobs": [{"kind": "lifecycle"}, {"kind": "measure"}]}))
        work = self.tiny_plan([
            {"id": "a", "resource": "cpu", "tasks": [{"commands": [writer[:-1] + ["0"]]},
                                                      {"commands": [writer[:-1] + ["1"]]}], "dependencies": []},
            {"id": "bound", "resource": "gpu", "tasks": [
                {"commands": [writer[:-1] + [str(k)]], "job_in": {"plan": str(plan_file), "index": k, "kind": None,
                                                                  "upper_bound": True}} for k in range(4)],
             "dependencies": []},
            {"id": "kinds", "resource": "gpu", "tasks": [{"commands": [writer], "job_in": {
                "plan": str(plan_file), "index": 0, "kind": "measure", "upper_bound": False}}], "dependencies": []},
            {"id": "checkpoint", "resource": "gpu", "requeue": True, "tasks": [
                {"commands": [["{python}", "-c", "raise SystemExit(75)"]]}], "dependencies": []},
            {"id": "fails", "resource": "gpu", "tasks": [{"commands": [["{python}", "-c", "raise SystemExit(3)"]]}],
             "dependencies": []}])
        self.assertEqual(self.execute(work, "a", SLURM_ARRAY_TASK_ID=1, SLURM_CPUS_PER_TASK=7), 0)
        self.assertEqual((out / "done1").read_text(), "7")
        self.assertEqual(po.read_json(po.task_marker(work, "a", 1))["status"], "COMPLETE")
        (out / "done1").unlink()
        self.assertEqual(self.execute(work, "a", task=1), 0)                 # complete: not executed again
        self.assertFalse((out / "done1").exists())
        self.assertEqual(self.execute(work, "bound", task=3), 0)
        self.assertEqual(po.read_json(po.task_marker(work, "bound", 3))["status"], "SKIPPED")
        self.assertEqual(self.execute(work, "bound", task=1), 0)
        self.assertTrue((out / "done1").exists())
        self.assertEqual(self.execute(work, "kinds"), 2)                      # the plan's job 0 is no measurement
        self.assertEqual(po.read_json(po.task_marker(work, "kinds", 0))["status"], "FAILED")
        mock = MockSlurm(self.base)
        log = mock.root / "scontrol.log"
        path = os.environ["PATH"]
        self.assertEqual(self.execute(work, "checkpoint", PATH=f"{mock.root / 'bin'}:{path}", MOCK_ROOT=mock.root,
                                      SLURM_JOB_ID=4242, SLURM_ARRAY_JOB_ID="", CM_MAX_CONTINUATIONS=2), 0)
        self.assertEqual(log.read_text().strip(), "requeue 4242")
        self.assertEqual(po.read_json(po.task_marker(work, "checkpoint", 0))["status"], "REQUEUED")
        for _ in range(2):
            self.execute(work, "checkpoint", PATH=f"{mock.root / 'bin'}:{path}", MOCK_ROOT=mock.root,
                         SLURM_JOB_ID=4242, CM_MAX_CONTINUATIONS=2)
        marker = po.read_json(po.task_marker(work, "checkpoint", 0))
        self.assertEqual((marker["status"], marker["exit_code"]), ("FAILED", 75))     # the budget is spent
        self.assertEqual(self.execute(work, "fails"), 3)
        self.assertEqual(po.read_json(po.task_marker(work, "fails", 0))["exit_code"], 3)

    def test_execute_restarts_non_resumable_runs_fresh_and_runs_deferred_plans(self):
        directory = self.base / "exp" / "gate"
        planner = ["{python}", "-c", f"import json,pathlib; d=pathlib.Path(r'{directory}'); d.mkdir(parents=True, "
                                     "exist_ok=True); (d/'plan.json').write_text(json.dumps({'jobs': [1, 2]}))"]
        runner = ["{python}", "-c", f"import pathlib; (pathlib.Path(r'{directory}')/'ARTIFACT_MANIFEST.json').write_text('x')"]
        deferred = self.base / "exp" / "deferred"
        defer = ["{python}", "-c", f"import json,pathlib; d=pathlib.Path(r'{deferred}'); d.mkdir(parents=True, "
                                   "exist_ok=True); (d/'plan.json').write_text(json.dumps({'jobs': [1, 2, 3]}))"]
        work = self.tiny_plan([
            {"id": "gate", "resource": "gpu", "dependencies": [], "tasks": [{"commands": [runner], "fresh": {
                "directory": str(directory), "complete": str(directory / "ARTIFACT_MANIFEST.json"),
                "keep": ["plan.json"], "plan": planner, "plan_file": str(directory / "plan.json")}}]},
            {"id": "plan.exact", "resource": "cpu", "dependencies": [], "tasks": [{"commands": []}],
             "plan_step": {"argv": defer, "plan_file": str(deferred / "plan.json"), "jobs": 2, "exact": True}},
            {"id": "plan.bound", "resource": "cpu", "dependencies": [], "tasks": [{"commands": []}],
             "plan_step": {"argv": defer, "plan_file": str(deferred / "plan.json"), "jobs": 5, "exact": False}}])
        directory.mkdir(parents=True)
        (directory / "plan.json").write_text("{}")
        (directory / "FAILURE.json").write_text("{}")                        # a failed earlier attempt
        self.assertEqual(self.execute(work, "gate"), 0)
        self.assertTrue((directory.parent / "gate.attempt-1" / "FAILURE.json").is_file())
        self.assertEqual(po.read_json(directory / "plan.json"), {"jobs": [1, 2]})      # planned again
        self.assertTrue((directory / "ARTIFACT_MANIFEST.json").is_file())
        self.assertEqual(self.execute(work, "plan.exact"), 2)                # 3 jobs, the array was sized for 2
        self.assertEqual(self.execute(work, "plan.bound"), 0)                # 3 jobs fit an upper bound of 5

    def test_batch_script_runs_a_node_with_the_site_settings(self):
        out = self.base / "written"
        work = self.tiny_plan([{"id": "note", "title": "write one file", "resource": "cpu", "dependencies": [],
                                "requeue": False, "dataset": None, "array": None, "tasks": [{"commands": [[
                                    "{python}", "-c", f"import os; open(r'{out}', 'w').write(os.environ['OMP_NUM_THREADS'])"]]}]}])
        script = work / "note.sbatch"
        script.write_text(po.batch_script(po.load_plan(work)["nodes"][0], work))
        slurm = MockSlurm(self.base)
        env = dict(slurm.env, CM_PYTHON=sys.executable, CM_BUILD_MODULE="", CM_PYTHON_MODULE="", CM_EXPECT_ARCH="",
                   SLURM_JOB_ID="77", SLURM_CPUS_PER_TASK="3")
        result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(out.read_text(), "3")
        self.assertEqual(po.read_json(po.task_marker(work, "note", 0))["job"], "77")

    # ------------------------------------------------------------------ collection and the publication map
    def test_collect_exports_verifies_and_maps_every_required_artifact(self):
        suite, plan = self.plan_examples("--only", "nominal_time", "sensor_parameters",
                                         executor=po._subprocess_executor)
        result = subprocess.run([sys.executable, "-B", "scripts/paper_outputs.py", "collect", "--work",
                                 str(self.base / "work")], cwd=PROJECT, capture_output=True, text=True, timeout=600)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        mapping = po.read_json(self.base / "work" / po.MAP_NAME)
        required = po.required_artifact_ids()
        self.assertEqual(sorted(k for k in mapping if not k.startswith("_")), sorted(required))
        self.assertEqual(mapping["records:nominal_time"], str(self.base / "work/records/nominal_time"))
        self.assertEqual(mapping["run:ogbn-arxiv"], str(suite / "runs/01_arxiv"))
        self.assertEqual(mapping["records:input_gate"], "")
        self.assertEqual(mapping["_awaiting"]["records:input_gate"], "outside --only")
        self.assertIn("_fixture", mapping)
        from scripts.records import load_specs, verify_set
        specs = load_specs(PROJECT / "records", PROJECT / "records/specs.json")
        for name in ("nominal_time", "sensor_parameters"):
            entry = verify_set(self.base / "work/records" / name, specs["sets"][name], specs)
            self.assertTrue(entry["synthetic"])
        for key, value in mapping.items():
            if not key.startswith("_") and value:
                self.assertTrue(Path(value).is_dir(), key)
        # the pack step of hpc/publish_paper_artifacts_arrhenius.sh reads exactly these entries
        packed = {k: v for k, v in mapping.items() if not k.startswith("_") and v}
        self.assertLessEqual(set(packed), set(required))
        collected = po.read_json(self.base / "work" / "collect.json")
        self.assertEqual(collected["exports"]["export.nominal_time"]["status"], "COMPLETE")
        again = subprocess.run([sys.executable, "-B", "scripts/paper_outputs.py", "collect", "--work",
                                str(self.base / "work")], cwd=PROJECT, capture_output=True, text=True, timeout=600)
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn("outputs present and verified", again.stdout)

    def test_a_planned_node_executes_and_collects_into_a_verified_record_set(self):
        """E11 is torch-free: the real sweep plan, the node's task, the sweep export and the map, end to end."""
        suite, plan = self.plan_examples("--only", "buffer_sensitivity", executor=po._subprocess_executor)
        work = self.base / "work"
        node = {n["id"]: n for n in plan["nodes"]}["buffer_sensitivity.run"]
        self.assertEqual(node["resource"], po.CPU)
        self.assertEqual(self.execute(work, "buffer_sensitivity.run"), 0)
        self.assertEqual(po.read_json(po.task_marker(work, "buffer_sensitivity.run", 0))["status"], "COMPLETE")
        result = subprocess.run([sys.executable, "-B", "scripts/paper_outputs.py", "collect", "--work", str(work)],
                                cwd=PROJECT, capture_output=True, text=True, timeout=600)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        mapping = po.read_json(work / po.MAP_NAME)
        self.assertEqual(mapping["records:buffer_sensitivity"], str(work / "records/buffer_sensitivity"))
        provenance = po.read_json(work / "records/buffer_sensitivity/provenance.json")
        self.assertIn("CPU; torch", provenance["hardware"])                # the runs' and the job's own records
        start, end = provenance["period"].split("/")
        self.assertLess(start, end)
        from scripts.records import load_specs, verify_set
        specs = load_specs(PROJECT / "records", PROJECT / "records/specs.json")
        self.assertTrue(verify_set(work / "records/buffer_sensitivity", specs["sets"]["buffer_sensitivity"],
                                   specs)["verified"])

    def test_the_complete_map_names_a_directory_for_every_required_artifact(self):
        plan = full_plan(self.base)
        results = {e["id"]: {"status": "COMPLETE"} for e in plan["exports"]}
        mapping = po.publication_map(plan, results, self.base / "work")
        required = po.required_artifact_ids()
        self.assertEqual(sorted(k for k in mapping if not k.startswith("_")), sorted(required))
        self.assertEqual([k for k in required if not mapping[k]], [])
        self.assertEqual(mapping["_awaiting"], {})
        self.assertEqual(mapping["run:Roman-empire"], str(self.base / "suite/runs/04_roman_empire"))
        self.assertEqual(mapping["campaign:budget"], str(self.base / "work/campaigns/budget"))
        self.assertEqual(mapping["campaign:confirmation"], str(self.base / "work/campaigns/confirmation"))
        self.assertEqual(mapping["environment:gpu"], str(self.base / "work/environment/gpu"))
        failed = dict(results, **{"export.input_gate": {"status": "FAILED", "reason": "exited with 2"}})
        mapping = po.publication_map(plan, failed, self.base / "work")
        self.assertEqual(mapping["records:input_gate"], "")
        self.assertEqual(mapping["_awaiting"]["records:input_gate"], "exited with 2")

    def test_hardware_and_period_are_read_from_the_runs_records(self):
        gate = self.base / "gate"
        gate.mkdir()
        po.write_json(gate / "environment.json", {"visible_gpus": ["NVIDIA GH200 96GB"], "torch": "2.12.1",
                                                  "python_version": "3.13.5"})
        po.write_json(gate / "experiment.json", {"started_utc": "2026-11-08T10:00:00+00:00"})
        po.write_json(gate / "results.json", {"finished_utc": "2026-11-09T11:00:00+00:00"})
        hardware, period = po.derive_provenance({"kind": "group_b", "dirs": [str(gate)]})
        self.assertEqual(hardware, "NVIDIA GH200 96GB; torch 2.12.1; python 3.13.5")
        self.assertEqual(period, "2026-11-08T10:00:00+00:00/2026-11-09T11:00:00+00:00")
        sweep = self.base / "sweep"
        po.write_json(sweep / "plan.json", {"parents": [{"members": [str(EXAMPLES / "arxiv")]}], "variants": []})
        hardware, period = po.derive_provenance({"kind": "sweep", "plan": str(sweep / "plan.json")})
        self.assertIn("CPU; torch", hardware)
        self.assertRegex(period, r"^\d{4}-\d\d-\d\dT.*/\d{4}-\d\d-\d\dT")
        campaign = self.base / "campaign"
        po.write_json(campaign / "studies/ogbn-arxiv/seed-0/split-0/environment.json",
                      {"visible_gpus": ["NVIDIA GH200 96GB"], "python_version": "3.13.5",
                       "packages": {"torch": "2.12.1+cu130"}})
        (campaign / "execution/ogbn-arxiv/seed-0/split-0").mkdir(parents=True)
        (campaign / "execution/ogbn-arxiv/seed-0/split-0/accounting-0.tsv").write_text(
            "Cluster|JobIDRaw|JobID|Submit|Start|End|State\nc|5|5|x|2026-11-10T01:00:00|2026-11-10T03:00:00|COMPLETED\n")
        self.assertEqual(po.derive_provenance({"kind": "budget", "campaign": str(campaign)}),
                         ("NVIDIA GH200 96GB; torch 2.12.1+cu130; python 3.13.5",
                          "2026-11-10T01:00:00/2026-11-10T03:00:00"))
        self.assertEqual(po.derive_provenance({"kind": "group_b", "dirs": [str(self.base / "empty")]}), (None, None))

    def test_budget_manifest_names_every_study_with_its_accounting_and_logs(self):
        studies = []
        for seed in (0, 1):
            study = self.base / "studies" / "arxiv" / f"seed-{seed}"
            evidence = study.with_name(f"{study.name}.job-{100 + seed}")
            evidence.mkdir(parents=True)
            (evidence / "process_exit_code.txt").write_text("0\n")
            (evidence / "job_id.txt").write_text(f"{100 + seed}\n")
            failed = study.with_name(f"{study.name}.job-{90 + seed}")
            failed.mkdir()
            (failed / "process_exit_code.txt").write_text("2\n")
            studies.append({"dataset": "ogbn-arxiv", "seed": seed, "split": 0, "parent": str(self.base / "parent"),
                            "study": str(study)})
        manifest = self.base / "campaign.json"
        po.write_budget_manifest({"manifest": str(manifest), "studies": studies, "seeds": list(range(12))},
                                 {"git_commit": "d" * 40, "dirty": False})
        value = po.read_json(manifest)
        self.assertEqual(value["schema"], "cm-arm-campaign-1")
        self.assertEqual(value["seed_ids"], list(range(12)))
        self.assertEqual(value["source"]["git_commit"], "d" * 40)
        self.assertEqual(len(value["source"]["tree_sha256"]), 64)
        self.assertEqual(value["studies"][1]["accounting"], [str(self.base / "studies/arxiv/seed-1.job-101/sacct.tsv")])
        self.assertEqual(value["studies"][0]["job_log"][0], str(self.base / "studies/arxiv/seed-0.job-100/stdout.log"))
        with self.assertRaises(po.PlanError):
            po.write_budget_manifest({"manifest": str(self.base / "other.json"), "studies": studies, "seeds": []},
                                     {"git_commit": "d" * 40, "dirty": True})

    # ------------------------------------------------------------------ the one command
    def test_hpc_wrapper_plans_in_a_job_submits_and_prints_the_publish_command(self):
        suite = make_suite(self.base)
        slurm = MockSlurm(self.base)
        work = self.base / "work"
        env = dict(slurm.env, CM_PYTHON=sys.executable, CM_BUILD_MODULE="", CM_PYTHON_MODULE="")
        result = subprocess.run(["bash", str(PROJECT / "hpc/produce_paper_outputs_arrhenius.sh"), str(suite),
                                 "--work", str(work), "--allow-fixture", "--negative-controls-dataset",
                                 "Peptides-func", "--only", "nominal_time", "environment:gpu"],
                                env=env, capture_output=True, text=True, timeout=600, cwd=self.base)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        submitted = slurm.submissions()
        planner = submitted[0]["args"]
        self.assertIn("--wait", planner)
        self.assertIn("--export=ALL,CM_PRODUCE_PLAN_JOB=1", planner)
        self.assertTrue((work / "plan.json").is_file())
        names = [Path(s["args"][-1]).name for s in submitted[1:]]
        self.assertEqual(names, ["environment.capture.sbatch", "collect.sbatch"])
        self.assertIn(f"publish_paper_artifacts_arrhenius.sh all {work / po.MAP_NAME}", result.stdout)
        again = subprocess.run(["bash", str(PROJECT / "hpc/produce_paper_outputs_arrhenius.sh"), str(suite),
                                "--work", str(work)], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(again.returncode, 2)
        self.assertIn("plan.json exists", again.stderr)


if __name__ == "__main__":
    unittest.main()
