"""The reproduction launcher end to end on a mock Slurm cluster.

Mock sbatch/squeue/scontrol/sacct/sinfo/srun record what the launcher asks for;
recorded jobs are then executed as Slurm would, with a fake GPU interpreter
(real python3 plus a stub torch) and a fake runtime that writes the sealed
records of a completed example run. No cluster, GPU or PyTorch is contacted.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

MOCKS = {
    "sbatch": r'''#!/bin/bash
n_file="$MOCK_ROOT/next_id"; n=$(( $(cat "$n_file" 2>/dev/null || echo 1000) + 1 )); echo "$n" > "$n_file"
python3 -c 'import json,sys,os; open(os.environ["MOCK_ROOT"]+"/sbatch.jsonl","a").write(json.dumps({"id":sys.argv[1],"args":sys.argv[2:],"env":{k:v for k,v in os.environ.items() if k.startswith("CM_")}})+"\n")' "$n" "$@"
if [[ " $* " == *" --parsable "* ]]; then echo "$n"; else echo "Submitted batch job $n"; fi
''',
    "squeue": '#!/bin/bash\ncat "$MOCK_ROOT/squeue.txt" 2>/dev/null || true\n',
    "scontrol": '#!/bin/bash\necho "$*" >> "$MOCK_ROOT/scontrol.log"\n'
                '[[ "${MOCK_REQUEUE_FAILS:-0}" == 1 && "$1" == requeue ]] && exit 1\nexit 0\n',
    "sacct": '#!/bin/bash\ncat "$MOCK_ROOT/sacct.psv" 2>/dev/null || true\n',
    "sinfo": '#!/bin/bash\necho gpu\n',
    "srun": '#!/bin/bash\nwhile [[ "$1" == --* ]]; do shift; done\nexec "$@"\n',
    "nvidia-smi": '#!/bin/bash\necho MOCK GPU\n',
}
FAKE_TORCH = '''__version__ = "2.12.1+cu130"
class _V: cuda = "13.0"
version = _V()
class _C:
    @staticmethod
    def is_available(): return True
    @staticmethod
    def get_device_name(i=0): return "MOCK GH200"
cuda = _C()
'''
FAKE_PYTHON = r'''#!/bin/bash
export PYTHONPATH="${MOCK_ROOT}/fake_site"
if [[ "${1:-}" == -m && "${2:-}" == pip ]]; then
  case "${3:-}" in freeze) echo "numpy==2.3.5" ;; *) echo "[mock pip] $*" ;; esac
  exit 0
fi
for a in "$@"; do
  if [[ "$a" == *run_cross_morse.py ]]; then
    if [[ " $* " == *" --gpu-health-check "* ]]; then
      [[ "${MOCK_GPU_FAIL:-0}" == 1 ]] && { echo "CUDA error: device is lost" >&2; exit 1; }
      echo '{"status": "PASS", "gpu": "MOCK GH200"}'; exit 0
    fi
    exec python3 "${MOCK_ROOT}/fake_runner.py" "$@"
  fi
done
exec python3 "$@"
'''
FAKE_RUNNER = r'''import json, os, shutil, sys
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index("--output-dir") + 1])
behaviour = os.environ.get("MOCK_RUNNER", "complete")
if behaviour == "checkpoint":
    (out / "interruption.json").write_text(json.dumps({"status": "CHECKPOINTED_INTERRUPTION"}))
    sys.exit(75)
if behaviour == "cudafail":
    print("RuntimeError: CUDA error: unspecified launch failure", file=sys.stderr)
    (out / "failure_summary.json").write_text(json.dumps({"status": "FAILED", "type": "RuntimeError", "reason": "CUDA error"}))
    sys.exit(1)
if behaviour == "crash":
    (out / "failure_summary.json").write_text(json.dumps({"status": "FAILED", "type": "ValueError", "reason": "bad input"}))
    sys.exit(1)
if behaviour == "noop":
    sys.exit(0)
example = Path(os.environ["MOCK_EXAMPLE"])
for name in ("run_state.json", "config.json", "environment.json", "policy_lock.json"):
    shutil.copy2(example / name, out / name)
for folder in ("final", "stages", "select", "confirm", "admit"):
    shutil.rmtree(out / folder, ignore_errors=True)
    shutil.copytree(example / folder, out / folder)
'''


class MockCluster:
    def __init__(self, root: Path):
        self.root = root
        self.mock = root / "mock"
        self.base = root / "base with spaces"
        (self.mock / "bin").mkdir(parents=True)
        (self.mock / "fake_site" / "torch").mkdir(parents=True)
        for name, text in MOCKS.items():
            path = self.mock / "bin" / name
            path.write_text(text)
            path.chmod(0o755)
        (self.mock / "fake_site" / "torch" / "__init__.py").write_text(FAKE_TORCH)
        (self.mock / "fake_runner.py").write_text(FAKE_RUNNER)
        venv = self.base / "code" / "venv_arrhenius_gh200" / "bin"
        venv.mkdir(parents=True)
        (venv / "python").write_text(FAKE_PYTHON)
        (venv / "python").chmod(0o755)
        (root / "home").mkdir()
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("CM_", "SLURM_", "MOCK_"))}
        self.env.update(PATH=str(self.mock / "bin") + os.pathsep + self.env.get("PATH", ""), HOME=str(root / "home"),
                        MOCK_ROOT=str(self.mock), CM_BASE=str(self.base), CM_ACCOUNT="fixture-account",
                        MOCK_EXAMPLE=str(PROJECT / "examples/runs/peptides_func"), USER="tester")

    def launcher(self, *args, script=PROJECT / "run_cross_morse.sh", env=None, cwd=PROJECT):
        return subprocess.run(["bash", str(script), *args], env=dict(self.env, **(env or {})), text=True,
                              capture_output=True, timeout=120, cwd=cwd)

    def submissions(self) -> list[dict]:
        path = self.mock / "sbatch.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []

    def execute(self, record: dict, task=None, **extra) -> subprocess.CompletedProcess:
        """Run one recorded job (one array task) as Slurm would."""
        args = record["args"]
        index = next(i for i, arg in enumerate(args) if arg.endswith("run_cross_morse.sh"))
        env = {k: v for k, v in self.env.items() if not k.startswith("CM_")}
        env.update(record["env"])
        env.update(CM_BUILD_MODULE="", CM_PYTHON_MODULE="", CM_EXPECT_ARCH="", CM_GPU_HEALTH_RETRIES="1",
                   SLURM_JOB_ID=record["id"] if task is None else f"{record['id']}{task:02d}", SLURM_CPUS_PER_TASK="16",
                   SLURMD_NODENAME="node7", SLURM_JOB_NAME="mock")
        if task is not None:
            env.update(SLURM_ARRAY_JOB_ID=record["id"], SLURM_ARRAY_TASK_ID=str(task))
        env.update({k: str(v) for k, v in extra.items()})
        return subprocess.run(["bash", args[index], *args[index + 1:]], env=env, text=True, capture_output=True, timeout=120)

    def execute_all(self, records, **extra):
        for record in records:
            tasks = range(10) if any(a.startswith("--array=") for a in record["args"]) else [None]
            for task in tasks:
                result = self.execute(record, task, **extra)
                assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


class ReproductionLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cm reproduction ")
        self.addCleanup(self.temporary.cleanup)
        self.cluster = MockCluster(Path(self.temporary.name))

    def reproduce(self, *datasets, profile="smoke", script=PROJECT / "run_cross_morse.sh"):
        env = {"CM_REPRODUCE_DATASETS": ",".join(datasets)} if datasets else {}
        result = self.cluster.launcher("reproduce", profile, script=script, env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        suites = sorted((self.cluster.base / "outputs").glob("CROSS_MORSE_REPRODUCTION_0*"))
        return suites[-1], result

    def test_check_reports_identity_configurations_and_the_plan_without_writing(self):
        result = self.cluster.launcher("check", "full")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("(the released runtime)", result.stdout)
        self.assertIn("0 problem(s)", result.stdout)
        for dataset in ("ogbn-arxiv", "ogbn-products", "Roman-empire", "Peptides-func"):
            self.assertIn(dataset, result.stdout)
        self.assertIn("4.8 per seed, 2.28 qualification", result.stdout)
        self.assertIn("Nothing was created or submitted.", result.stdout)
        self.assertFalse(self.cluster.base.joinpath("outputs").exists())
        self.assertEqual(self.cluster.submissions(), [])
        missing = self.cluster.launcher("check", "smoke", env={"CM_ACCOUNT": ""})
        self.assertEqual(missing.returncode, 2)
        self.assertIn("[FAIL] CM_ACCOUNT is empty", missing.stdout)

    def test_check_and_reproduce_refuse_a_modified_runtime(self):
        copy = Path(self.temporary.name) / "modified source"
        shutil.copytree(PROJECT, copy, ignore=shutil.ignore_patterns(".git", "examples", "tests", "docs", "__pycache__"))
        with (copy / "cross_morse" / "statistics.py").open("a") as stream:
            stream.write("\n# local edit\n")
        result = self.cluster.launcher("check", "smoke", script=copy / "run_cross_morse.sh", cwd=copy)
        self.assertEqual(result.returncode, 2)
        self.assertIn("differs from the released", result.stdout)
        result = self.cluster.launcher("reproduce", "smoke", script=copy / "run_cross_morse.sh", cwd=copy)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("released runtime unchanged", result.stderr)
        self.assertEqual(self.cluster.submissions(), [])
        allowed = self.cluster.launcher("reproduce", "smoke", script=copy / "run_cross_morse.sh", cwd=copy,
                                        env={"CM_ALLOW_SOURCE_CHANGE": "1", "CM_REPRODUCE_DATASETS": "Peptides-func"})
        self.assertEqual(allowed.returncode, 0, allowed.stderr)
        manifest = json.loads(next((self.cluster.base / "outputs").glob("CROSS_MORSE_REPRODUCTION_0*/000_MANIFEST/suite.json")).read_text())
        self.assertTrue(manifest["source"]["modified_runtime_allowed"])

    def test_reproduce_freezes_the_suite_and_submits_the_job_graph(self):
        suite, result = self.reproduce("ogbn-arxiv", "Roman-empire")
        manifest = json.loads((suite / "000_MANIFEST/suite.json").read_text())
        self.assertEqual(manifest["tag"], "CMR_0001")
        self.assertEqual([run["key"] for run in manifest["runs"]], ["01_arxiv", "04_roman_empire"])
        self.assertTrue(manifest["source"]["matches_release"])
        self.assertTrue((suite / "runs/01_arxiv/source/run_cross_morse.sh").is_file())
        self.assertTrue((suite / "runs/04_roman_empire/campaign.json").is_file())
        self.assertTrue((suite / "000_MANIFEST/WHERE_ARE_MY_FILES.txt").is_file())
        setup, arxiv, roman, report = self.cluster.submissions()
        self.assertEqual(setup["args"][-1], "setup")
        self.assertIn("--gpus=1", setup["args"])
        self.assertIn("--account=fixture-account", arxiv["args"])
        self.assertIn(f"--dependency=afterok:{setup['id']}", arxiv["args"])
        self.assertIn("--kill-on-invalid-dep=yes", arxiv["args"])
        self.assertEqual(arxiv["args"][-2:], ["resume", str(suite / "runs/01_arxiv")])
        self.assertEqual(arxiv["env"]["CM_PROJECT_DIR"], str(suite / "runs/01_arxiv/source"))
        self.assertEqual(arxiv["env"]["CM_RUN_KEY"], "01_arxiv")
        self.assertEqual(arxiv["env"]["CM_REPRODUCTION_SUITE"], str(suite))
        self.assertIn("--array=0-9", roman["args"])
        self.assertEqual(roman["args"][-3:], ["campaign", "smoke", str(suite / "runs/04_roman_empire")])
        self.assertIn(f"--dependency=afterany:{arxiv['id']}:{roman['id']}", report["args"])
        self.assertEqual(report["args"][-2:], ["finalize", str(suite)])
        jobs = (suite / "000_MANIFEST/job_ids.tsv").read_text().splitlines()
        self.assertEqual([line.split("\t")[0] for line in jobs[1:]], ["setup", "run", "run", "report"])
        self.assertIn("status", result.stdout)

    def test_dataset_wrapper_reproduces_only_its_dataset(self):
        suite, _ = self.reproduce(script=PROJECT / "run_cross_morse_peptides_func.sh")
        manifest = json.loads((suite / "000_MANIFEST/suite.json").read_text())
        self.assertEqual(manifest["datasets"], ["Peptides-func"])
        self.assertEqual(len(self.cluster.submissions()), 3)

    def test_complete_suite_runs_and_reports(self):
        suite, _ = self.reproduce("ogbn-proteins", "Roman-empire")
        (self.cluster.mock / "sacct.psv").write_text(
            "JobIDRaw|JobID|JobName|Partition|State|ExitCode|ElapsedRaw|Elapsed|Timelimit|AllocTRES|Submit|Start|End\n"
            "1002|1002|cmr0001_02_proteins|gpu|COMPLETED|0:0|7200|02:00:00|01:00:00|cpu=16,gres/gpu:gh200=1|x|x|x\n")
        self.cluster.execute_all(self.cluster.submissions())
        status = self.cluster.launcher("status", str(suite))
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertIn("2/2 runs complete", status.stdout)
        self.assertIn("10/10 splits complete", status.stdout)
        report = (suite / "REPRODUCTION_REPORT.md").read_text()
        self.assertIn("Runs complete | 2 of 2", report)
        self.assertIn("identical to the released runtime", report)
        self.assertIn("GPU-hours (Slurm accounting) | 2.0", report)
        machine = json.loads((suite / "SUITE_STATUS.json").read_text())
        self.assertEqual(machine["complete_runs"], 2)
        with zipfile.ZipFile(suite / "CMR_0001_RESULTS.zip") as archive:
            names = archive.namelist()
            self.assertIn("REPRODUCTION_REPORT.md", names)
            self.assertIn("runs/02_proteins/final/results_summary.json", names)
            self.assertFalse([name for name in names if name.endswith((".pt", ".npz", ".npy"))])
            self.assertIn("<redacted>", archive.read("000_MANIFEST/suite.json").decode())
            self.assertNotIn("000_MANIFEST/suite.env", names)
        registry = (suite / "000_MANIFEST/run_registry.tsv").read_text()
        self.assertIn("SETUP_COMPLETE", registry)
        self.assertIn("04_roman_empire/split_09\tCOMPLETED", registry)
        live = (self.cluster.base / "logs/CROSS_MORSE_LIVE.out").read_text()
        self.assertIn("RUN 02_proteins (ogbn-proteins)", live)
        self.assertIn("[REPORT]", live)

    def test_failure_classes_requeue_and_continuation(self):
        suite, _ = self.reproduce("ogbn-arxiv", "ogbn-proteins", "COCO-SP", "Peptides-func")
        setup, arxiv, proteins, coco, peptides, report = self.cluster.submissions()
        self.assertEqual(self.cluster.execute(setup).returncode, 0)
        scontrol = self.cluster.mock / "scontrol.log"
        # A checkpoint at the time limit requeues the same job.
        result = self.cluster.execute(arxiv, MOCK_RUNNER="checkpoint")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"requeue {arxiv['id']}", scontrol.read_text())
        terminal = json.loads((suite / "runs/01_arxiv/launcher/terminal.json").read_text())
        self.assertEqual((terminal["status"], terminal["failure_class"]), ("REQUEUED", "CONTINUATION"))
        # The requeued job resumes and completes; the earlier records are kept.
        result = self.cluster.execute(arxiv, SLURM_RESTART_COUNT=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((suite / "runs/01_arxiv/launcher/attempts" / f"{arxiv['id']}_r1_interruption.json").is_file())
        self.assertEqual(json.loads((suite / "runs/01_arxiv/launcher/terminal.json").read_text())["status"], "COMPLETE")
        # Where requeue is refused, a continuation job is submitted after this one.
        result = self.cluster.execute(proteins, MOCK_RUNNER="checkpoint", MOCK_REQUEUE_FAILS=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        continuation = self.cluster.submissions()[-1]
        self.assertIn(f"--dependency=afterany:{proteins['id']}", continuation["args"])
        self.assertIn("continuation\t02_proteins", (suite / "000_MANIFEST/job_ids.tsv").read_text())
        # With the continuation budget spent, the run is recorded as incomplete.
        result = self.cluster.execute(continuation, MOCK_RUNNER="checkpoint", CM_MAX_CONTINUATIONS=1, SLURM_RESTART_COUNT=1)
        self.assertEqual(result.returncode, 75)
        self.assertEqual(json.loads((suite / "runs/02_proteins/launcher/terminal.json").read_text())["failure_class"], "TIME_BUDGET")
        # A GPU failure excludes the node and requeues; a runtime error is final.
        scontrol.write_text("")
        result = self.cluster.execute(coco, MOCK_RUNNER="cudafail")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"update JobId={coco['id']} ExcNodeList=node7", scontrol.read_text())
        result = self.cluster.execute(coco, MOCK_GPU_FAIL=1, SLURM_RESTART_COUNT=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((suite / "runs/06_coco_sp/launcher/infrastructure_count").read_text().strip(), "2")
        result = self.cluster.execute(peptides, MOCK_RUNNER="crash")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads((suite / "runs/07_peptides_func/launcher/terminal.json").read_text())["failure_class"], "RUNTIME")
        result = self.cluster.execute(peptides, MOCK_RUNNER="noop", SLURM_RESTART_COUNT=1)
        self.assertEqual(result.returncode, 74)
        status = self.cluster.launcher("status", str(suite)).stdout
        self.assertIn("01_arxiv             ogbn-arxiv     COMPLETE", status)
        self.assertIn("TIME_BUDGET", status)
        self.assertIn("ARTIFACT_MISSING", status)
        registry = (suite / "000_MANIFEST/run_registry.tsv").read_text()
        for event in ("REQUEUE_CONTINUATION", "REQUEUE_INFRASTRUCTURE", "FAILED_RUNTIME", "FAILED_ARTIFACT_MISSING"):
            self.assertIn(event, registry)

    def test_report_waits_for_active_runs_and_resume_skips_finished_ones(self):
        suite, _ = self.reproduce("ogbn-arxiv", "Peptides-func")
        setup, arxiv, peptides, report = self.cluster.submissions()
        self.cluster.execute_all([setup, arxiv])
        (self.cluster.mock / "squeue.txt").write_text(f"{peptides['id']} PENDING\n{report['id']} RUNNING\n")
        result = self.cluster.execute(report)
        self.assertEqual(result.returncode, 0, result.stderr)
        deferred = self.cluster.submissions()[-1]
        self.assertEqual(deferred["args"][-2:], ["finalize", str(suite)])
        self.assertIn(f"--dependency=afterany:{peptides['id']}", deferred["args"])
        self.assertFalse((suite / "REPRODUCTION_REPORT.md").exists())
        resumed = self.cluster.launcher("reproduce", "resume", str(suite))
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertIn("[SKIP] 01_arxiv complete", resumed.stdout)
        self.assertIn("[SKIP] 07_peptides_func already queued", resumed.stdout)
        (self.cluster.mock / "squeue.txt").write_text("")
        resumed = self.cluster.launcher("reproduce", "resume", str(suite))
        self.assertIn("[RUN] 07_peptides_func", resumed.stdout)

    def test_site_env_prints_effective_settings_for_the_auxiliary_scripts(self):
        result = self.cluster.launcher("site-env", env={"CM_PARTITION": "gpu-a100", "CM_BUILD_MODULE": ""})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("export CM_PARTITION=gpu-a100\n", result.stdout)
        self.assertIn("export CM_BUILD_MODULE=''\n", result.stdout)
        self.assertIn("export CM_PYTHON_MODULE=GPU/Python/3.13.5-bare-gcc-2025b-eb\n", result.stdout)
        self.assertIn("export CM_TORCH_SPEC=torch==2.12.1+cu130\n", result.stdout)
        self.assertIn("export CM_PYTHON=", result.stdout)

    def test_help_and_no_mode_print_usage(self):
        for args in ((), ("--help",)):
            result = self.cluster.launcher(*args)
            self.assertEqual(result.returncode, 0)
            self.assertIn("bash run_cross_morse.sh reproduce", result.stdout)
            self.assertIn("SITE SETTINGS", result.stdout)


if __name__ == "__main__":
    unittest.main()
