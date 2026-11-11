#!/usr/bin/env python3
"""Check the repository against the availability text printed in the manuscript.

The journal registered a short availability statement (release/availability_statement.json,
checked by scripts/statement_check.py). The identified manuscript PDFs also print a longer
'Data and code availability' paragraph in the main text and in the Supplementary Information,
and two repository sentences in the main Methods. release/printed_availability.json holds
those paragraphs as extracted from the PDFs (bound to the SHA-256 identities of
paper/sources.json) and divides them into atomic claims. Each claim is evaluated here with the
vocabulary of the registered statement:

    fulfilled                  the repository (and its published outputs) provides it
    awaiting_measured_outputs  code, contracts and route exist; benchmark outputs are not yet
                               published or registered
    missing                    the repository cannot currently provide what the sentence says

    python -B scripts/printed_statement_check.py              # print the evaluation
    python -B scripts/printed_statement_check.py --json       # machine-readable evaluation
    python -B scripts/printed_statement_check.py --require-fulfilled

STATEMENT_COMPLIANCE.md renders this evaluation below the registered statement
(scripts/statement_check.py --write). Nothing here imports PyTorch, downloads or computes
results: symbols are located with the Python AST and files and manifests are inspected.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGISTRY = "release/printed_availability.json"
FULFILLED, AWAITING, MISSING = "fulfilled", "awaiting_measured_outputs", "missing"
LABEL = {FULFILLED: "Fulfilled", AWAITING: "Awaiting measured outputs", MISSING: "Missing"}
DATASETS = ("ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func")
STEMS = {"ogbn-arxiv": "arxiv", "ogbn-proteins": "proteins", "ogbn-products": "products",
         "Roman-empire": "roman_empire", "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp",
         "Peptides-func": "peptides_func"}
ZENODO_DOI = re.compile(r"10\.5281/zenodo\.[1-9][0-9]*\Z")
ANY_DOI = re.compile(r"10\.[0-9]{4,9}/\S+\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
PUBLICATION = ("bash hpc/produce_paper_outputs_arrhenius.sh <suite>, then "
               "bash hpc/publish_paper_artifacts_arrhenius.sh all <work>/paper-artifacts.map.json")
EVIDENCE_ROUTE = ("deposit each run archive and register its DOI with the evidence-registration route "
                  "(docs/EVIDENCE_REGISTRATION.md: make evidence-deposit, evidence-publish PUBLISH=1, evidence-register)")


class CheckError(ValueError):
    pass


def _json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


class Context:
    """Facts shared by several claims, computed once."""

    def __init__(self, root: Path):
        self.root = root
        self._makefile = None
        self._symbols: dict[str, set] = {}
        self._registered = None
        self._manifest = None

    # Makefile -------------------------------------------------------------------------
    @property
    def makefile(self) -> str:
        if self._makefile is None:
            path = self.root / "Makefile"
            self._makefile = path.read_text(encoding="utf-8") if path.is_file() else ""
        return self._makefile

    def recipe(self, target: str) -> str | None:
        """The recipe lines of a Makefile target, or None when the target is not defined."""
        lines = self.makefile.splitlines()
        for index, line in enumerate(lines):
            if re.match(rf"^{re.escape(target)}\s*:(?!=)", line):
                body = []
                for following in lines[index + 1:]:
                    if following.startswith("\t") or (body and body[-1].endswith("\\")):
                        body.append(following)
                    elif not following.strip():
                        break
                    else:
                        break
                return "\n".join([line, *body])
        return None

    def references(self, text: str, variable: str) -> bool:
        """``text`` uses $(variable) directly or through one Makefile variable that expands to it."""
        if f"$({variable})" in text:
            return True
        for name in re.findall(r"\$\(([A-Z_][A-Z0-9_]*)\)", text):
            definition = re.search(rf"^{name}\s*[:?]?=(.*)$", self.makefile, re.M)
            if definition and f"$({variable})" in definition.group(1):
                return True
        return False

    # Python symbols -------------------------------------------------------------------
    def symbols(self, relative: str) -> set:
        if relative not in self._symbols:
            path = self.root / relative
            names: set = set()
            if path.is_file():
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        names.add(node.name)
                    elif isinstance(node, ast.ClassDef):
                        names.add(node.name)
                        names.update(f"{node.name}.{item.name}" for item in node.body
                                     if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)))
                    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                        names.update(t.id for t in targets if isinstance(t, ast.Name))
            self._symbols[relative] = names
        return self._symbols[relative]

    def literal(self, relative: str, name: str):
        path = self.root / relative
        if not path.is_file():
            return None
        for node in ast.parse(path.read_text(encoding="utf-8"), filename=relative).body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            if any(isinstance(t, ast.Name) and t.id == name for t in targets) and node.value is not None:
                try:
                    return ast.literal_eval(node.value)
                except ValueError:
                    return None
        return None

    # Registered statement and release ---------------------------------------------------
    @property
    def registered(self) -> dict:
        if self._registered is None:
            from scripts import statement_check
            self._registered = {row["id"]: row for row in statement_check.evaluate(self.root)["items"]}
        return self._registered

    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            from scripts import paper_artifacts
            self._manifest = paper_artifacts.load_manifest(self.root / "release/paper_artifacts.json")
        return self._manifest

    def published(self) -> set:
        from scripts import paper_artifacts
        return {a["id"] for a in self.manifest["artifacts"]
                if a.get("status") == paper_artifacts.PUBLISHED and a.get("scientific_only_verification") is not False}

    def release_record_sets(self) -> list[str]:
        from scripts import paper_artifacts
        return sorted(a["id"].split(":", 1)[1] for a in paper_artifacts.required_artifacts(self.root)
                      if a["kind"] == "records")

    def example_runs(self) -> list[Path]:
        base = self.root / "examples" / "runs"
        return sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []


# ------------------------------------------------------------------------------------ checks
def result(status: str, where: str, remaining: str = "") -> dict:
    return {"status": status, "where": where, "remaining": remaining}


def missing_paths(ctx: Context, paths) -> list[str]:
    return [p for p in paths if not (ctx.root / p).exists()]


def check_paths(ctx, spec):
    absent = missing_paths(ctx, spec["paths"])
    where = ", ".join(f"`{p}`" for p in spec["paths"])
    return result(MISSING, where, "absent: " + ", ".join(absent)) if absent else result(FULFILLED, where)


def check_symbols(ctx, spec):
    absent = []
    for item in spec["symbols"]:
        relative, name = item.split("::", 1)
        if name not in ctx.symbols(relative):
            absent.append(item)
    files = sorted({s.split("::", 1)[0] for s in spec["symbols"]})
    where = "; ".join(f"`{f}`: " + ", ".join(s.split("::", 1)[1] for s in spec["symbols"] if s.startswith(f + "::"))
                      for f in files)
    return result(MISSING, where, "not defined: " + ", ".join(absent)) if absent else result(FULFILLED, where)


def check_action_registry(ctx, spec):
    from scripts._paper_common import parse_actions_from_core
    actions = parse_actions_from_core(ctx.root / "cross_morse" / "core.py")
    ids = [str(a["id"]) for a in actions]
    expected = [str(i) for i in range(spec["actions"])]
    expected_named = [f"A{i}" for i in range(spec["actions"])]
    where = f"`cross_morse/core.py`: ACTIONS ({len(actions)} registered actions A0–A{len(actions) - 1})"
    if ids not in (expected, expected_named):
        return result(MISSING, "`cross_morse/core.py`: ACTIONS", f"registry ids {ids} differ from A0–A{spec['actions'] - 1}")
    return result(FULFILLED, where)


def check_policy_families(ctx, spec):
    path = ctx.root / "cross_morse" / "policies.py"
    found: dict[str, bool] = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef) and node.name == "make_registry":
            for call in ast.walk(node):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "add"
                        and len(call.args) >= 3 and all(isinstance(a, ast.Constant) for a in call.args[:3])):
                    found[call.args[0].value] = bool(call.args[2].value)
    open_loop = sorted(r for r, flag in found.items() if flag)
    feedback = sorted(r for r, flag in found.items() if not flag)
    where = (f"`cross_morse/policies.py`: make_registry registers {len(found)} route families "
             f"({len(open_loop)} open loop, {len(feedback)} feedback)")
    if open_loop != sorted(spec["open_loop"]) or feedback != sorted(spec["feedback"]):
        return result(MISSING, "`cross_morse/policies.py`: make_registry",
                      f"open loop {open_loop} and feedback {feedback} differ from main Eq. (15)")
    return result(FULFILLED, where)


def check_lifecycle(ctx, spec):
    problems = [s for s in spec["sources"] if list(ctx.literal(s, "STAGES") or ()) != spec["stages"]]
    unsealed = []
    for run in ctx.example_runs():
        for stage in spec["stages"]:
            seal = _json(run / "stages" / f"{stage}.json", {})
            if seal.get("status") != "COMPLETE" or not HEX64.fullmatch(str(seal.get("manifest_sha256", ""))):
                unsealed.append(f"{run.name}/{stage}")
    where = (f"STAGES in {', '.join(f'`{s}`' for s in spec['sources'])}; "
             f"{len(ctx.example_runs())} bundled runs sealed at all {len(spec['stages'])} stages")
    if problems or unsealed or not ctx.example_runs():
        return result(MISSING, where, "; ".join(filter(None, [
            f"stage list differs in {', '.join(problems)}" if problems else "",
            f"unsealed stages: {', '.join(unsealed)}" if unsealed else "",
            "no bundled run demonstrates the sealed chain" if not ctx.example_runs() else ""])))
    return result(FULFILLED, where)


def check_datasets(ctx, spec):
    adapters = list(ctx.literal("cross_morse/datasets.py", "BENCHMARKS") or ())
    absent = [d for d in spec["datasets"] if d not in adapters]
    problems = [f"no adapter for {', '.join(absent)}"] if absent else []
    where = f"`cross_morse/datasets.py` (BENCHMARKS: {len(adapters)} official dataset adapters)"
    if spec.get("providers"):
        records = (_json(ctx.root / "data" / "sources.json", {}) or {}).get("datasets", [])
        sourced = {r.get("dataset") for r in records
                   if isinstance(r, dict) and str(r.get("source_documentation", "")).startswith("https://")
                   and r.get("provider_raw_data_bundled") is False}
        unsourced = [d for d in spec["datasets"] if d not in sourced]
        if unsourced:
            problems.append("no provider record in data/sources.json for " + ", ".join(unsourced))
        where += "; provider documentation and download resolution in `data/sources.json` (no provider data bundled)"
    return result(MISSING, where, "; ".join(problems)) if problems else result(FULFILLED, where)


def check_entry_points(ctx, spec):
    absent = []
    for stem in spec["stems"]:
        absent += missing_paths(ctx, [f"run_cross_morse_{stem}.py"])
        absent += missing_paths(ctx, [f"configs/{stem}_{p}.json" for p in spec["profiles"]])
    where = (f"`run_cross_morse_<dataset>.py` and `configs/<dataset>_{{{','.join(spec['profiles'])}}}.json` "
             f"for {len(spec['stems'])} datasets")
    return result(MISSING, where, "absent: " + ", ".join(absent)) if absent else result(FULFILLED, where)


def check_slurm(ctx, spec):
    absent = missing_paths(ctx, [spec["launcher"], spec["site_profile"]] + [f"run_cross_morse_{s}.sh" for s in spec["stems"]])
    launcher = ctx.root / spec["launcher"]
    text = launcher.read_text(encoding="utf-8") if launcher.is_file() else ""
    problems = ["absent: " + ", ".join(absent)] if absent else []
    if "sbatch" not in text:
        problems.append(f"{spec['launcher']} submits no Slurm job")
    if "arrhenius" not in text.lower():
        problems.append(f"{spec['launcher']} carries no Arrhenius site settings")
    where = f"`{spec['launcher']}` (SITE SETTINGS preset to Arrhenius), seven dataset wrappers, `{spec['site_profile']}`"
    return result(MISSING, where, "; ".join(problems)) if problems else result(FULFILLED, where)


def check_tests(ctx, spec):
    tests = sorted(ctx.root.glob(spec["pattern"]))
    targets = [t for t in spec["targets"] if ctx.recipe(t) is None]
    where = (f"`tests/` (at least {spec['minimum']} test modules); "
             + ", ".join(f"`make {t}`" for t in spec["targets"]))
    if len(tests) < spec["minimum"] or targets:
        return result(MISSING, where, f"{len(tests)} test modules; undefined targets: {', '.join(targets) or '—'}")
    return result(FULFILLED, where)


def check_make_targets(ctx, spec):
    problems = []
    for target, script in spec["targets"].items():
        recipe = ctx.recipe(target)
        if recipe is None:
            problems.append(f"`make {target}` is not defined")
        elif script not in recipe:
            problems.append(f"`make {target}` does not run {script}")
        elif spec.get("runs_root_variable") and not ctx.references(recipe, spec["runs_root_variable"]):
            problems.append(f"`make {target}` reads no completed run directory ({spec['runs_root_variable']})")
    where = ", ".join(f"`make {t}` ({s})" for t, s in spec["targets"].items())
    if spec.get("runs_root_variable"):
        where += f"; `{spec['runs_root_variable']}=` names a directory of completed runs"
    return result(MISSING, where, "; ".join(problems)) if problems else result(FULFILLED, where)


def check_experiments(ctx, spec):
    experiments = (_json(ctx.root / "release" / "experiment_code.json", {}) or {}).get("experiments", [])
    problems = []
    for label in spec["experiments"]:
        rows = [e for e in experiments if e.get("experiment") == label]
        if not rows:
            problems.append(f"{label} is not mapped in release/experiment_code.json")
            continue
        absent = missing_paths(ctx, rows[0].get("code", []))
        if absent:
            problems.append(f"{label}: absent code {', '.join(absent)}")
    problems += [f"absent: {p}" for p in missing_paths(ctx, spec.get("paths", []))]
    where = ", ".join(spec["experiments"]) + " in `release/experiment_code.json`"
    if spec.get("paths"):
        where += "; " + ", ".join(f"`{p}`" for p in spec["paths"])
    return result(MISSING, where, "; ".join(problems)) if problems else result(FULFILLED, where)


def check_exporter(ctx, spec):
    absent = missing_paths(ctx, spec["paths"])
    specs = _json(ctx.root / "records" / "specs.json", {}) or {}
    sets = specs.get("sets", {})
    untraced = [n for n, s in sets.items() if not s.get("raw") and not s.get("raw_files") and not s.get("columns")]
    where = (f"`scripts/export_records.py` writes {len(sets)} record contracts of `records/specs.json`; "
             "`scripts/records.py check` traces each value to its raw file")
    if absent or not sets:
        return result(MISSING, where, "; ".join(filter(None, ["absent: " + ", ".join(absent) if absent else "",
                                                            "no record contracts" if not sets else ""])))
    return result(FULFILLED, where)


def check_repository_url(ctx, spec):
    unlisted = [f for f in spec["files"] if (ctx.root / f).is_file()
                and spec["url"] not in (ctx.root / f).read_text(encoding="utf-8")]
    absent = missing_paths(ctx, spec["files"])
    where = f"{spec['url']} in " + ", ".join(f"`{f}`" for f in spec["files"])
    if absent or unlisted:
        return result(MISSING, where, "; ".join(filter(None, ["absent: " + ", ".join(absent) if absent else "",
                                                            "URL not named in " + ", ".join(unlisted) if unlisted else ""])))
    return result(FULFILLED, where)


def check_fingerprint(ctx, spec):
    problems = []
    if ctx.recipe(spec["target"]) is None:
        problems.append(f"`make {spec['target']}` is not defined")
    stages = ("prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate")
    for run in ctx.example_runs():
        identities = {(_json(run / "stages" / f"{s}.json", {}) or {}).get("source_sha256") for s in stages}
        if len(identities) != 1 or not all(isinstance(i, str) and HEX64.fullmatch(i) for i in identities):
            problems.append(f"{run.name}: stage seals do not carry one runtime identity")
    where = ("`make fingerprint`; `source_sha256` sealed in every stage record (bundled runs); "
             "publication re-verifies the seals of every reported run")
    if problems or not ctx.example_runs():
        return result(MISSING, where, "; ".join(problems) or "no bundled run demonstrates the sealed identity")
    return result(FULFILLED, where)


def check_registered_design(ctx, spec):
    problems = []
    keys = (("data_roles", "rule"), ("data_roles", "split_protocol"), ("selection", "outer_folds"),
            ("selection", "inner_folds"), ("controller", "crossfit_folds"), ("selection", "epsilon_r"),
            ("selection", "epsilon_eq"), ("admission", "epsilon_a"), ("admission", "clip"))
    for stem in spec["stems"]:
        for profile in spec["profiles"]:
            value = _json(ctx.root / "manifests" / "registered" / f"{stem}_{profile}.json")
            if value is None:
                problems.append(f"{stem}_{profile}: no registered manifest")
                continue
            absent = [".".join(k) for k in keys if (value.get(k[0]) or {}).get(k[1]) is None]
            if not value.get("training_seeds") or value.get("partition_seed") is None:
                absent.append("seeds")
            if absent:
                problems.append(f"{stem}_{profile}: no {', '.join(absent)}")
    where = (f"`manifests/registered/` ({len(spec['stems']) * len(spec['profiles'])} official-data configurations): "
             "split rule and protocol, partition and training seeds, outer/inner/cross-fitting folds, selection and "
             "admission margins; smoke configurations use synthetic data")
    return result(MISSING, where, "; ".join(problems)) if problems else result(FULFILLED, where)


def check_role_manifests(ctx, spec):
    problems = []
    examples = sorted((ctx.root / "manifests" / "examples").glob("*.json"))
    for path in examples:
        hashes = ((_json(path, {}) or {}).get("splits") or {}).get("role_hashes") or {}
        if not set(spec["roles"]) <= set(hashes):
            problems.append(f"{path.name}: role memberships {sorted(set(spec['roles']) - set(hashes))} not extracted")
    if not (ctx.root / "scripts" / "run_manifests.py").is_file() or not examples:
        problems.append("no role-membership extraction")
    if problems:
        return result(MISSING, "`scripts/run_manifests.py`, `manifests/`", "; ".join(problems))
    published = ctx.published()
    pending = [d for d in DATASETS if f"run:{d}" not in published
               or not (ctx.root / "manifests" / "published" / f"{STEMS[d]}.json").is_file()]
    where = (f"`scripts/run_manifests.py` extracts the seven role memberships with their hashes "
             f"(`manifests/examples/`: {len(examples)} runs; `manifests/published/`: reported runs)")
    if pending:
        return result(AWAITING, where, f"executed memberships of {len(pending)} reported runs ({PUBLICATION})")
    return result(FULFILLED, where)


def _npz_members(path: Path) -> set:
    try:
        with zipfile.ZipFile(path) as z:
            return {name[:-4] if name.endswith(".npy") else name for name in z.namelist()}
    except (OSError, zipfile.BadZipFile):
        return set()


def check_run_archive_contract(ctx, spec):
    required = {
        "prepared data and split indices": lambda r: (r / "prepared" / "splits.npz").is_file() and (r / "prepared" / "official_splits.npz").is_file(),
        "frozen checkpoints": lambda r: any((r / "train").glob("plant_seed*_best.pt")),
        "propensity-logged exploration trajectories": lambda r: any("propensity" in _npz_members(p) for p in (r / "explore").glob("trajectory_*.npz")),
        "candidate predictions": lambda r: any((r / "predictions").rglob("*.npy")) and any((r / "member_predictions").rglob("*.npy")),
        "final predictions": lambda r: (r / "final" / "qD_probabilities.npy").is_file() and (r / "final" / "test_predictions.npz").is_file(),
        "selection, confirmation and admission records": lambda r: all((r / s / "decision.json").is_file() for s in ("select", "confirm", "admit")),
        "stage manifests": lambda r: len(list((r / "stages").glob("*.json"))) == 8,
        "environment records": lambda r: (r / "environment.json").is_file(),
    }
    runs = ctx.example_runs()
    problems = [f"{run.name}: {part}" for run in runs for part, test in required.items() if not test(run)]
    if not (ctx.root / "scripts" / "archive_run.py").is_file():
        problems.append("scripts/archive_run.py is absent")
    where = (f"`scripts/archive_run.py` packs and verifies the sealed run directory; the {len(runs)} bundled runs "
             "contain every listed component")
    if problems or not runs:
        return result(MISSING, where, "; ".join(problems) or "no bundled run demonstrates the archive contents")
    return result(FULFILLED, where)


def check_runs_index_dois(ctx, spec):
    index = _json(ctx.root / "runs" / "index.json", {"runs": []}) or {"runs": []}
    pattern = ZENODO_DOI if spec["service"] == "zenodo" else ANY_DOI
    registered = {}
    for entry in index.get("runs", []):
        if (entry.get("profile") == "full" and entry.get("scientific_run") is True
                and isinstance(entry.get("deposit_doi"), str) and pattern.fullmatch(entry["deposit_doi"])):
            registered[entry.get("dataset")] = entry["deposit_doi"]
    pending = [d for d in DATASETS if d not in registered]
    kind = "Zenodo DOI" if spec["service"] == "zenodo" else "DOI"
    where = f"`runs/index.json` ({len(registered)} of {len(DATASETS)} datasets registered with a {kind}; written by `scripts/ingest_run.py`)"
    if pending:
        return result(AWAITING, where, f"{len(pending)} datasets without a registered full-run {kind}: {EVIDENCE_ROUTE}")
    return result(FULFILLED, where)


def check_records_index_dois(ctx, spec):
    index = _json(ctx.root / "records" / "index.json", {"sets": []}) or {"sets": []}
    sets = ctx.release_record_sets() if spec["sets"] == "release" else list(spec["sets"])
    pattern = ZENODO_DOI if spec["service"] == "zenodo" else ANY_DOI
    with_doi = {e.get("set") for e in index.get("sets", [])
                if isinstance(e.get("deposit_doi"), str) and pattern.fullmatch(e["deposit_doi"])}
    pending = [s for s in sets if s not in with_doi]
    kind = "Zenodo DOI" if spec["service"] == "zenodo" else "DOI"
    where = f"`records/index.json` ({len(sets) - len(pending)} of {len(sets)} record sets with a {kind})"
    if not pending:
        return result(FULFILLED, where)
    contract_has_doi = any("deposit_doi" in e for e in index.get("sets", []))
    if contract_has_doi:
        return result(AWAITING, where, f"{len(pending)} record sets without a registered {kind}")
    published = ctx.published()
    via_release = [s for s in sets if f"records:{s}" in published]
    return result(MISSING, where,
                  "the record contract of `records/index.json` has no deposit-DOI field; the record sets are "
                  "distributed as `records:*` assets of the release `paper-artifacts` (`release/paper_artifacts.json`"
                  + (f", {len(via_release)} published" if via_release else "") + "). Either add record-set deposit "
                  "registration to the record contract or state the release as the location of the measurement "
                  "records in the printed paragraph")


def check_make_paper(ctx, spec):
    registered = ctx.registered.get("one_command", {})
    recipe = ctx.recipe("paper")
    if recipe is None or "scripts/reconstruct_paper.py" not in recipe:
        return result(MISSING, "`make paper`", "the target does not run scripts/reconstruct_paper.py")
    where = ("`make paper` (`scripts/reconstruct_paper.py`): downloads and verifies the published archives, rebuilds "
             "every numbered display and compares each value with the printed one")
    return result(registered.get("status", MISSING), where, registered.get("remaining", ""))


def check_records_tables(ctx, spec):
    recipe = ctx.recipe("records-tables")
    problems = []
    if recipe is None or "scripts/records.py tables" not in recipe:
        problems.append("`make records-tables` does not run scripts/records.py tables")
    if not (ctx.root / "examples" / "records" / "index.json").is_file():
        problems.append("no traced example record set")
    if problems:
        return result(MISSING, "`make records-tables`", "; ".join(problems))
    sets = ctx.release_record_sets()
    published = ctx.published()
    local = {e.get("set") for e in (_json(ctx.root / "records" / "index.json", {"sets": []}) or {}).get("sets", [])}
    pending = [s for s in sets if f"records:{s}" not in published and s not in local]
    where = ("`make records-tables` (`scripts/records.py tables`): every value traced to its raw file; reads "
             "`records/` or the record sets that `make paper` verified under `ARTIFACTS_ROOT`")
    if pending:
        return result(AWAITING, where, f"{len(pending)} measurement-record sets unpublished ({PUBLICATION})")
    return result(FULFILLED, where)


def check_result_registry(ctx, spec):
    displays = (_json(ctx.root / "paper" / "display_manifest.json", {}) or {}).get("displays", [])
    from paper.build.displays import framework
    framework._load_generators()
    without = [d["id"] for d in displays if d["id"] not in framework.GENERATORS]
    schema = (_json(ctx.root / "paper" / "protocol" / "S38.json", {}) or {})
    problems = []
    if not displays or without:
        problems.append(f"{len(without)} displays without a generator" if displays else "no display inventory")
    if not schema:
        problems.append("no results-record schema (paper/protocol/S38.json)")
    if problems:
        return result(MISSING, "`paper/display_manifest.json`", "; ".join(problems))
    where = (f"`paper/display_manifest.json` binds each of the {len(displays)} numbered displays to its generator and "
             "inputs; reconstruction writes every value with its source record to `paper/source_data/`")
    registered = ctx.registered.get("source_data", {})
    if registered.get("status") != FULFILLED:
        return result(AWAITING if registered.get("status") != MISSING else MISSING, where,
                      registered.get("remaining") or "source data are generated from the published outputs")
    return result(FULFILLED, where)


def check_synthetic_separation(ctx, spec):
    problems = []
    for path in sorted((ctx.root / "manifests" / "examples").glob("*.json")):
        value = _json(path, {}) or {}
        if value.get("data_kind") != "synthetic" or value.get("is_benchmark_data") is not False or value.get("scientific_run"):
            problems.append(f"{path.name} is not marked as a synthetic software run")
    text = (ctx.root / "scripts" / "paper_artifacts.py").read_text(encoding="utf-8") \
        if (ctx.root / "scripts" / "paper_artifacts.py").is_file() else ""
    if "scientific_only" not in text:
        problems.append("publication does not refuse software fixtures")
    where = ("bundled runs are labelled synthetic (`manifests/examples/`), their tables carry a synthetic banner "
             "(`paper/generated/examples/`), and publication accepts only full scientific runs")
    return result(MISSING, where, "; ".join(problems)) if problems else result(FULFILLED, where)


CHECKS = {
    "paths": check_paths, "symbols": check_symbols, "action_registry": check_action_registry,
    "policy_families": check_policy_families, "lifecycle_stages": check_lifecycle, "datasets": check_datasets,
    "entry_points_configs": check_entry_points, "slurm": check_slurm, "tests": check_tests,
    "make_targets": check_make_targets, "experiments": check_experiments, "exporter": check_exporter,
    "repository_url": check_repository_url, "fingerprint": check_fingerprint,
    "registered_design": check_registered_design, "role_manifests": check_role_manifests,
    "run_archive_contract": check_run_archive_contract, "runs_index_dois": check_runs_index_dois,
    "records_index_dois": check_records_index_dois, "make_paper": check_make_paper,
    "records_tables": check_records_tables, "result_registry": check_result_registry,
    "synthetic_separation": check_synthetic_separation,
}
ORDER = {MISSING: 0, AWAITING: 1, FULFILLED: 2}


def evaluate(root: Path = ROOT, registry: dict | None = None, registered: dict | None = None) -> dict:
    """Evaluate every printed claim. ``registered`` reuses an evaluation of statement_check.evaluate."""
    registry = registry or _json(root / REGISTRY)
    if registry is None or registry.get("schema") != "cross-morse-printed-availability-1":
        raise CheckError(f"{REGISTRY} is absent or has another schema")
    ctx = Context(root)
    if registered is not None:
        ctx._registered = {row["id"]: row for row in registered["items"]}
    by_id = {claim["id"]: claim for claim in registry["claims"]}
    if len(by_id) != len(registry["claims"]):
        raise CheckError("duplicate claim identifiers")
    results: dict[str, dict] = {}

    def run(claim_id: str, stack=()) -> dict:
        if claim_id in results:
            return results[claim_id]
        if claim_id in stack:
            raise CheckError(f"circular claim reference at {claim_id}")
        check = by_id[claim_id]["check"]
        kind = check["type"]
        if kind == "same_as":
            base = run(check["claim"], (*stack, claim_id))
            value = {**base, "where": f"as {check['claim']}: {base['where']}"}
        elif kind == "all_of":
            parts = [run(c, (*stack, claim_id)) for c in check["claims"]]
            extra = check_paths(ctx, {"paths": check.get("paths", [])}) if check.get("paths") else result(FULFILLED, "")
            parts.append(extra)
            if check.get("generators"):
                parts.append({**ctx.registered.get("generators", result(MISSING, "", "generators not evaluated"))})
            worst = min(parts, key=lambda p: ORDER[p["status"]])["status"]
            remaining = "; ".join(f"{c}: {r['remaining']}" for c, r in zip([*check["claims"], "contracts", "generators"], parts)
                                  if r["status"] != FULFILLED and r["remaining"])
            value = result(worst, f"claims {check['claims'][0]}–{check['claims'][-1]}; run-archive contract "
                                  "(`scripts/archive_run.py`, `release/schemas/`); record contracts (`records/specs.json`); "
                                  "a generator for every numbered display (`paper/build/displays/`)", remaining)
        else:
            if kind not in CHECKS:
                raise CheckError(f"{claim_id}: unknown check type {kind}")
            value = CHECKS[kind](ctx, check)
        results[claim_id] = value
        return value

    for claim in registry["claims"]:
        run(claim["id"])
    rows = [{"id": c["id"], "paragraph": c["paragraph"], "text": c["text"], **results[c["id"]]} for c in registry["claims"]]
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in (FULFILLED, AWAITING, MISSING)}
    return {"schema": "cross-morse-printed-availability-compliance-1",
            "paragraphs": {k: {kk: v[kk] for kk in ("document", "pages", "heading", "sha256")}
                           for k, v in registry["paragraphs"].items()},
            "claims": rows, "counts": counts}


def render_section(evaluation: dict, registry: dict) -> list[str]:
    """Markdown lines appended to STATEMENT_COMPLIANCE.md below the registered statement."""
    counts = evaluation["counts"]
    lines = ["## The availability text printed in the manuscript", "",
             "The identified manuscript also prints a longer availability paragraph (main text, page 15; "
             "Supplementary Information, page 30) and two repository sentences in the main Methods. "
             "[`release/printed_availability.json`](release/printed_availability.json) holds the text and divides it "
             f"into {len(evaluation['claims'])} claims; `scripts/printed_statement_check.py` evaluates each one: "
             f"{counts[FULFILLED]} fulfilled, {counts[AWAITING]} awaiting measured outputs, {counts[MISSING]} missing.", ""]
    titles = {"main": "Main text, Data and code availability (page 15)",
              "methods": "Main text, Methods: Reported results and statistical power (pages 14–15)",
              "supplementary": "Supplementary Information, Data and code availability (page 30)"}
    for paragraph in ("main", "methods", "supplementary"):
        rows = [r for r in evaluation["claims"] if r["paragraph"] == paragraph]
        if not rows:
            continue
        lines += [f"### {titles[paragraph]}", "", "| Claim | Printed text | Status | Where | Remaining |", "|---|---|---|---|---|"]
        for row in rows:
            text = row["text"].replace("|", r"\|")
            lines.append(f"| {row['id']} | {text} | {LABEL[row['status']]} | {row['where'].replace('|', r'\|')} | "
                         f"{(row['remaining'] or '—').replace('|', r'\|')} |")
        lines.append("")
    if counts[MISSING]:
        lines += ["Claims marked missing describe a location that the repository's publication route does not "
                  "produce: the printed paragraph names the location, or the route adds it.", ""]
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-complete", action="store_true", help="exit 1 when any claim is missing")
    parser.add_argument("--require-fulfilled", action="store_true", help="exit 1 unless every claim is fulfilled")
    args = parser.parse_args(argv)
    try:
        evaluation = evaluate(ROOT)
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        print(f"Invalid printed-availability evidence: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(evaluation, indent=1, ensure_ascii=False))
    else:
        for row in evaluation["claims"]:
            print(f"{row['id']}  {LABEL[row['status']]:<27} {row['text'][:72]}"
                  + (f" — {row['remaining']}" if row["remaining"] else ""))
        c = evaluation["counts"]
        print(f"{c[FULFILLED]} fulfilled, {c[AWAITING]} awaiting measured outputs, {c[MISSING]} missing "
              f"of {len(evaluation['claims'])} printed claims")
    if (args.require_complete or args.require_fulfilled) and evaluation["counts"][MISSING]:
        return 1
    if args.require_fulfilled and evaluation["counts"][AWAITING]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
