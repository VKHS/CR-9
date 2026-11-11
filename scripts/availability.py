#!/usr/bin/env python3
"""Generate concise availability text from registered repository artifacts.

``python -B scripts/availability.py`` updates AVAILABILITY.md and the marked
README block, and synchronizes the derived registered-dataset list in
release/release_scope.json. It never enables affirmative original-data or
reproduction claims. ``--check`` detects stale text and validates affirmative
original-results claims against the complete scientific evidence audit.

Counts come from release/release_scope.json, runs/index.json, records/index.json
and the bundled example tree. Runtime identity comes from the computation-source
fingerprint. This tool does not modify manuscripts or create experimental records.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))
from scripts._paper_common import runtime_source_fingerprint  # noqa: E402
from scripts.records import summary_state as records_state  # noqa: E402
from cm_campaigns.common import write_bytes  # noqa: E402
DATASETS = ["ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func"]
STEMS = {"ogbn-arxiv": "arxiv", "ogbn-products": "products", "ogbn-proteins": "proteins", "Roman-empire": "roman_empire",
         "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp", "Peptides-func": "peptides_func"}
START, END = "<!-- availability:start -->", "<!-- availability:end -->"
# Run-derived table columns and related numerical plots; these mappings do not
# establish exact reproduction of a printed figure or its complete panel design.
# paper/display_manifest.json records the reviewed scope of all manuscript displays.
BUILT_DISPLAYS = ("the run-derived columns of main Tables 1–2 and Supplementary Tables S4, S12, S12c, S13–S15, S23 and S41; "
                  "numerical plots related to main Figs. 2c and 4d and Supplementary Fig. S9; "
                  "a partial internal-method map related to main Fig. 5a; an entry-loss depth diagnostic; "
                  "and stage-cost bars related to Table S41")
NOT_BUILT = ("Registry plots use transcribed table values and analytical scenarios; "
             "they are not a replay of benchmark observations. "
             "Auxiliary tables require their corresponding validated measurement-record sets. "
             "Numerical plot builders do not reproduce every printed panel or its exact artwork; "
             "paper/PAPER_SUPPORT.md states the display-specific scope.")


def _json(path: Path, default=None):
    if not path.is_file():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def state(root: Path = ROOT) -> dict:
    scope = _json(root / "release" / "release_scope.json")
    index = _json(root / "runs" / "index.json", {"schema": "cross-morse-runs-index-1", "runs": []})
    if not isinstance(index, dict) or index.get("schema") != "cross-morse-runs-index-1" or not isinstance(index.get("runs"), list):
        raise ValueError("runs/index.json must contain the registered runs-index schema and runs list")
    official = {}
    for entry in index.get("runs", []):
        if not isinstance(entry, dict):
            raise ValueError("runs/index.json entries must be objects")
        if entry.get("verified") is True and entry.get("scientific_run") is True and entry.get("profile") == "full":
            if entry.get("dataset") not in DATASETS or entry["dataset"] in official:
                raise ValueError("runs/index.json has an unknown or duplicate full-profile dataset registration")
            official[entry["dataset"]] = entry
    examples = [d for d in DATASETS if (root / "examples" / "runs" / STEMS[d] / "final" / "results_summary.json").is_file()]
    registered = sorted(p.name for p in (root / "manifests" / "registered").glob("*_full.json"))
    registered_pilot = sorted(p.name for p in (root / "manifests" / "registered").glob("*_pilot.json"))
    records = records_state(root / "records") if (root / "records" / "specs.json").is_file() else None
    tables_examples = (root / "paper" / "generated" / "examples" / "tables" / "T1.tex").is_file()
    tables_official = (root / "paper" / "generated" / "official" / "tables" / "T1.tex").is_file()
    campaigns = all((root / rel).is_file() for rel in ("cm_experiments/__init__.py", "cm_campaigns/__init__.py",
                                                       "scripts/run_upstream_baseline.py", "scripts/export_records.py"))
    example_index = _json(root / "examples" / "records" / "index.json", {"sets": []})
    example_records = [e["set"] for e in example_index.get("sets", []) if e.get("verified") and e.get("synthetic")]
    release = None
    if scope.get("result_distribution") == "repository_release":
        from scripts import paper_artifacts, statement_check
        release = {"statement": _json(root / "release" / "availability_statement.json"),
                   "manifest": paper_artifacts.load_manifest(root / "release" / "paper_artifacts.json"),
                   "compliance": statement_check.evaluate(root)}
    return {"campaigns": campaigns, "example_records": example_records, "scope": scope, "official": official, "examples": examples, "registered_manifests": registered,
            "release": release,
            "source_sha256": runtime_source_fingerprint(root),
            "registered_pilot_manifests": registered_pilot, "records": records,
            "tables_examples": tables_examples, "tables_official": tables_official,
            "missing_official": [d for d in DATASETS if d not in official]}


def _table_list(labels: list[str]) -> str:
    """'Supplementary Table S15', ... -> 'Supplementary Tables S6c, S15, ... and S46b' in numeric order."""
    import re as _re
    names = sorted({label.split()[-1] for label in labels},
                   key=lambda s: (int(_re.search(r"\d+", s).group()), s))
    return "Supplementary Tables " + (", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0])


def records_sentence(st: dict) -> str:
    """Name registered inputs and other contracts without deficit scorecards."""
    records = st.get("records")
    if not records:
        return "No auxiliary measurement-record contract is registered."
    verified, defined = set(records["verified"]), set(records["defined"])
    if verified:
        text = "Registered auxiliary measurement records: " + ", ".join(sorted(verified)) + "."
    else:
        text = ("Auxiliary measurement contracts and import tools are bundled; "
                "measured inputs are registered in `records/index.json` before reconstruction.")
    other = sorted(defined - verified)
    if other:
        text += " Other defined record contracts: " + ", ".join(other) + "."
    elif defined:
        text += " All defined sets are registered."
    return text


def example_records_sentence(st: dict) -> str:
    names = st.get("example_records") or []
    if not names:
        return ""
    return (f"Synthetic record examples: {len(names)} sets ({', '.join(sorted(names))}); "
            "`make example-records-check` traces their values to the saved files.")


def repository_release(st: dict) -> bool:
    """Measured outputs published with the repository's GitHub release, bound to the registered statement."""
    return (st.get("scope") or {}).get("result_distribution") == "repository_release" and st.get("release") is not None


STATUS_TEXT = {"fulfilled": "Fulfilled", "awaiting_measured_outputs": "Awaiting measured outputs", "missing": "Missing"}


def _gib(value: int) -> str:
    return f"{value / 2**30:.2f} GiB" if value else "—"


def render_repository_release(st: dict) -> str:
    release = st["release"]
    statement, manifest, compliance = release["statement"], release["manifest"], release["compliance"]
    counts = compliance["counts"]
    url = st["scope"].get("repository_url") or "https://github.com/VKHS/CROSS-MORSE"
    lines = ["# Data and code availability", "", "## Availability statement", "",
             "The paper's registered data-availability statement:", "", f"> {statement['registered_text']}", "",
             "The table below identifies the available resources and the measured outputs that await publication. "
             "[Statement compliance](STATEMENT_COMPLIANCE.md) records the checks for each item "
             "(`make statement-check`).", "",
             "| Statement item | Status | Where in the repository |", "|---|---|---|"]
    lines += [f"| {row['phrase']} | {STATUS_TEXT[row['status']]} | {row['where']} |" for row in compliance["items"]]
    published = [a for a in manifest["artifacts"] if a.get("status") == "published"]
    lines += ["", "## Measured outputs", "",
              f"The publication workflow distributes checkpoints, predictions, propensity-logged trajectories and "
              f"statistical outputs through the `{manifest['release_tag']}` release in "
              f"[GitHub Releases]({url}/releases); [`release/paper_artifacts.json`](release/paper_artifacts.json) lists every asset with "
              "the SHA-256 of each published part, archive and file. The status of each artifact appears below.", "",
              "| Artifact | Status | Files | Size | Checkpoints | Predictions | Trajectories | Statistical outputs |",
              "|---|---|---|---|---|---|---|---|"]
    for entry in manifest["artifacts"]:
        contents = entry.get("contents") or {}
        if entry.get("status") == "published":
            lines.append(f"| `{entry['id']}` | published | {entry.get('files', '—')} | {_gib(entry.get('bytes', 0))} | "
                         f"{contents.get('checkpoints', 0)} | {contents.get('predictions', 0)} | "
                         f"{contents.get('propensity_logged_trajectories', 0)} | {contents.get('statistical_outputs', 0)} |")
        else:
            lines.append(f"| `{entry['id']}` | awaiting measured outputs | — | — | — | — | — | — |")
    lines += ["", "## One command", "",
              "`make paper` downloads the published outputs (analysis tier; `PAPER_TIER=complete` adds lifecycle checkpoints "
              "and full prediction arrays), verifies every part, archive and file against the manifest, verifies each run's "
              "sealed eight-stage chain and each record set's trace to its raw files, rebuilds every numbered table and "
              "figure with its source data (`paper/build/displays/`) and compares every computed value with the value "
              "printed in the paper. It exits 0 only when every display is complete and every compared value agrees. "
              "See [Reconstruction](docs/RECONSTRUCTION.md).", "",
              "## Publication of the measured outputs", "",
              "`bash hpc/publish_paper_artifacts_arrhenius.sh` packs the reported runs, record sets and campaigns on the "
              "cluster, uploads them to the release, downloads them back for verification, and writes the commit-sized "
              "views: `manifests/published/` (executed role memberships and seeds), `environment/observed/` (execution "
              "environments) and `paper/source_data/` (the source data of every display). See "
              "[Publishing the measured outputs](docs/PUBLICATION.md).", "",
              "## Independent regeneration", "",
              "`bash run_cross_morse.sh reproduce` executes the registered lifecycles on any Slurm GPU cluster after its "
              "SITE SETTINGS block is edited; its report compares the new runs with the paper "
              "([HPC guide](docs/HPC.md)). Official datasets are obtained from their providers by the adapters. "
              f"The release also includes {len(st['examples'])} complete synthetic example runs for software "
              "verification without a GPU.", "",
              "## Source identity", "",
              f"Runtime source identity: `{st['source_sha256']}` (`make fingerprint`). Every published run records the "
              "runtime identity it was produced with; verification resolves it to this tree, a bundled snapshot or "
              "the run's own frozen source. Paper PDFs are not distributed here; [paper/sources.json](paper/sources.json) "
              "retains their hashes and page references, and "
              "[paper/reported_results.json](paper/reported_results.json) holds the printed values used only for "
              "comparison.", "",
              "Generated by `scripts/availability.py`; `make availability-check` verifies freshness and claim "
              "consistency.", ""]
    return "\n".join(lines)


def code_release(st: dict) -> bool:
    """A code release with execution tooling and separately declared example outputs."""
    return (st.get("scope") or {}).get("result_distribution") == "regenerate" and not st.get("official")


CONTENTS = (
    ("Reference implementation and the sealed eight-stage lifecycle", "`cross_morse/`, `run_cross_morse*.py`"),
    ("Reproduction launcher for Slurm GPU clusters and the seven dataset wrappers",
     "`run_cross_morse.sh`, `run_cross_morse_<dataset>.sh`; [HPC guide](docs/HPC.md)"),
    ("Smoke, pilot and full-profile configurations; split rules, seeds and folds",
     "`configs/`, `manifests/registered/`, `cross_morse/split_contract.py`"),
    ("Experiments outside the lifecycle: E3–E6 arms, retrained baselines, E17 timing, scheduler accounting",
     "`cm_experiments/`, `cm_campaigns/`, `hpc/`; [Experiments](docs/EXPERIMENTS.md)"),
    ("Curated paper-reported values and registered protocol constants",
     "`paper/reported_results.json`, `paper/registered_constants.json`"),
    ("Builders of the run-derived tables and figures; comparison of a run with the paper",
     "`paper/build/`, `scripts/build_paper.py`, `scripts/compare_with_manuscript.py`"),
    ("Analytical statements and supplementary arithmetic", "`cm_theory/`, `scripts/check_si_identities.py`"),
    ("Environment specifications and containers", "`environment/`, `Dockerfile`"),
    ("Typed artifact catalog: included files, provider data, generated outputs and registered archives",
     "[Artifact catalog](docs/ARTIFACT_CATALOG.md); `release/artifact_catalog.json`"),
)


def render_code_release(st: dict) -> str:
    scope = st["scope"]
    url = scope.get("repository_url") or "https://github.com/VKHS/CROSS-MORSE"
    code_doi = scope.get("code_archive_doi")
    paragraph = (
        "The CROSS-MORSE reference implementation, the seven dataset adapters with their configurations, the Slurm "
        "reproduction launcher, environment specifications, statistical analysis tools and reported-result references "
        f"are available at [{url}]({url}). The release includes **{len(st['examples'])} complete synthetic example runs** "
        "with checkpoints, predictions and exploration logs for software verification. Official datasets are obtained "
        "from their providers by the adapters. Original benchmark execution archives can be supplied through the "
        "run/record registration workflow; this code release does not package those archives. "
        "`bash run_cross_morse.sh reproduce` executes the registered dataset lifecycles on a Slurm GPU cluster; "
        "each run seals the runtime fingerprint, configuration, data-role membership, checkpoints and decision records "
        "it was produced with. The suite report rebuilds supported run-derived outputs and compares their quantities "
        "with the reported references. [The artifact catalog](docs/ARTIFACT_CATALOG.md) distinguishes included files, "
        "provider data, new execution products and registered experimental archives.")
    if code_doi:
        paragraph += f" The registered code-archive DOI is https://doi.org/{code_doi}; publication is checked separately."
    lines = ["# Data and code availability", "", "## Availability statement", "", paragraph, "",
             "## What the release contains", "", "| Component | Location |", "|---|---|"]
    lines += [f"| {name} | {where} |" for name, where in CONTENTS]
    lines += [f"| Complete synthetic example runs | {len(st['examples'])} under `examples/runs/` |", "",
              "## Execution and reconstruction", "",
              "Edit the SITE SETTINGS block of `run_cross_morse.sh` for the cluster (or select a profile with "
              "`CM_ENV_FILE`), then `bash run_cross_morse.sh check` verifies the settings and the runtime fingerprint "
              "and prints the job plan, and `bash run_cross_morse.sh reproduce` submits the lifecycle suite: one setup job, the "
              "eight-stage lifecycle of every dataset (Roman-empire on its ten official splits) and one report job. "
              "Runs that reach their time limit checkpoint and requeue themselves; a GPU or driver failure requeues "
              "the job on another node.",
              "", f"The report rebuilds {BUILT_DISPLAYS} with `scripts/build_paper.py`, and "
              "`scripts/compare_with_manuscript.py` places every compared quantity of a full-profile run beside the "
              "printed value (exact at the printed precision, inside the printed 95% interval, or outside). "
              "The experiments outside the lifecycle and their record exporters are run as described in "
              "[Experiments](docs/EXPERIMENTS.md); `make records-check` traces a record set to its raw files. "
              + NOT_BUILT,
              "", "`make reconstruct REPORT_ROOT=/abs/reports` assembles the supported numerical outputs and a "
              "manifest of all numbered displays. `make reconstruction-example REPORT_ROOT=/abs/reports` uses the "
              "bundled synthetic examples. Each report distinguishes generated columns, partial or related plots, "
              "and check-only references. See [Reconstruction](docs/RECONSTRUCTION.md).",
              "", "`make reproduce` and `make example-paper` replay the bundled synthetic runs without a GPU. "
              "The run index (`runs/index.json`) registers sealed archives packed with `scripts/archive_run.py`; "
              "the record index (`records/index.json`) registers measurement sets and their raw-file provenance. "
              "An original experiment and a new reproduction retain their actual source identities and an explicit "
              "relationship to the manuscript.",
              "", "## Source identity", "",
              f"Runtime source identity: `{st['source_sha256']}` (`make fingerprint`). `run_cross_morse.sh` checks it "
              "before a reproduction and every run records it with its environment. "
              "Paper PDFs are not distributed here; [paper/sources.json](paper/sources.json) retains their "
              "hashes and page references. Numerical comparison targets remain in "
              "[paper/reported_results.json](paper/reported_results.json).",
              "", "See [HPC and reproduction](docs/HPC.md) and [Getting started](docs/GETTING_STARTED.md). Generated by "
              "`scripts/availability.py`; `make availability-check` verifies freshness and claim consistency.", ""]
    return "\n".join(lines)


def render(st: dict) -> str:
    if repository_release(st):
        return render_repository_release(st)
    if code_release(st):
        return render_code_release(st)
    scope = st["scope"]
    url = scope.get("repository_url") or "https://github.com/VKHS/CROSS-MORSE"
    code_doi = scope.get("code_archive_doi")
    n = len(st["official"])
    paragraph = (
        "The CROSS-MORSE reference implementation, seven dataset adapters, configurations, "
        "environment specifications, tests and reproducibility tools are available at "
        f"[{url}]({url}). Official datasets are obtained from their providers and are not redistributed. "
        f"The release includes **{len(st['examples'])} complete synthetic example runs**. "
        "Registered experimental inputs are identified below by dataset or measurement-set name. "
    )
    if n == 0:
        paragraph += ("This checkout packages the implementation and synthetic examples; original benchmark "
                      "archives can be supplied through the separate registration workflow.")
    else:
        paragraph += "Agreement with the manuscript requires validated result bindings and the scientific evidence audit."
    if code_doi:
        paragraph += f" The registered code-archive DOI is https://doi.org/{code_doi}; publication is checked separately."
    lines = ["# Data and code availability", "", "## Availability statement", "", paragraph, "",
             "## Registered artifacts", ""]
    if n:
        lines.append("Registered full-profile dataset archives: " + ", ".join(sorted(st["official"])) + ".")
    else:
        lines.append("The code release provides execution and archive-registration tools; the run index is the source of registered experiment identities.")
    dois = [(dataset, entry["deposit_doi"]) for dataset, entry in sorted(st["official"].items()) if entry.get("deposit_doi")]
    if dois:
        lines.append("Registered DOI references: " + "; ".join(f"{dataset}: https://doi.org/{doi}" for dataset, doi in dois) + ".")
        without_doi = sorted(dataset for dataset, entry in st["official"].items() if not entry.get("deposit_doi"))
        if without_doi:
            lines.append("Archives with no registered DOI reference: " + ", ".join(without_doi) + ".")
    elif n:
        lines.append("No DOI references are recorded for the registered archives.")
    lines += ["", records_sentence(st), "",
              "Archive identities are recorded in [runs/index.json](runs/index.json); "
              "measured-record coverage is recorded in [records/index.json](records/index.json). "
              "Index registration does not itself verify public availability; `make fetch-deposits` checks the registered run deposits and their bytes. "
              "Synthetic examples exercise the software workflow and do not supply official benchmark measurements. "
              "[The artifact catalog](docs/ARTIFACT_CATALOG.md) separates bundled files, execution products and registered archives."]
    if st.get("example_records"):
        lines += ["", example_records_sentence(st)]
    lines += ["", "## Reconstruction scope", "",
              f"With validated input archives, `make tables figures` builds {BUILT_DISPLAYS} "
              "for the registered datasets. `make records-tables` builds the registered auxiliary sets. " + NOT_BUILT,
              "", "`make reproduce` and `make example-paper` use the bundled synthetic runs. "
              "`make compare-runs` checks reported outcomes; `make scientific-audit` additionally requires "
              "complete applicable archives, measurement records, model identities and result bindings.",
              "", "`make reconstruct` assembles the supported outputs into one hashed workspace and records the "
              "input and build scope of every numbered display. See [Reconstruction](docs/RECONSTRUCTION.md).",
              "", "## Source identity", "",
              f"Runtime source identity: `{st['source_sha256']}` (`make fingerprint`). "
              "Each run records its computation-source identity and environment. "
              "Paper PDFs are not distributed here; [paper/sources.json](paper/sources.json) retains their "
              "hashes and page references. Numerical comparison targets remain in "
              "[paper/reported_results.json](paper/reported_results.json).",
              "", "See [Getting started](docs/GETTING_STARTED.md) and [Run archives](docs/RUN_ARCHIVES.md) "
              "for execution and deposit commands. Generated by `scripts/availability.py`; "
              "`make availability-check` verifies freshness and claim consistency.", ""]
    return "\n".join(lines)


def readme_block(st: dict) -> str:
    """One coverage sentence and one link on the landing page."""
    if repository_release(st):
        compliance = st["release"]["compliance"]
        manifest = st["release"]["manifest"]
        published = sum(1 for a in manifest["artifacts"] if a.get("status") == "published")
        release_url = (manifest["release_url"] if published else
                       f"https://github.com/{manifest['repository']}/releases")
        return "\n".join([
            START,
            "**Reproduction resources:** Experiment code, configurations, artifact verification and reconstruction "
            "commands are included. [Statement compliance](STATEMENT_COMPLIANCE.md) identifies the published "
            "files and the measured outputs awaiting publication. `make paper` verifies the "
            "[release artifacts](" + release_url + "), rebuilds their displays and compares them with "
            "the paper; see [availability](AVAILABILITY.md).",
            END,
        ])
    if code_release(st):
        return "\n".join([
            START,
            "**Release:** implementation, configurations, reproduction launcher and reported-result references. "
            "`bash run_cross_morse.sh reproduce` executes the dataset lifecycles ([guide](docs/HPC.md)); "
            f"{len(st['examples'])} complete synthetic example runs replay without a GPU. "
            "See [availability](AVAILABILITY.md) and the [artifact catalog](docs/ARTIFACT_CATALOG.md).",
            END,
        ])
    return "\n".join([
        START,
        "**Release:** implementation, configurations, analysis tools and complete synthetic examples. "
        "Registered dataset archives and measurement sets are named in "
        "[availability and reconstruction scope](AVAILABILITY.md); the "
        "[artifact catalog](docs/ARTIFACT_CATALOG.md) maps files and producing commands.",
        END,
    ])


def update_readme(root: Path, st: dict) -> bool:
    path = root / "README.md"
    content = path.read_text(encoding="utf-8")
    if START not in content or END not in content:
        return False
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    new = pattern.sub(lambda _m: readme_block(st), content)
    if new != content:
        path.write_text(new, encoding="utf-8")
    return True


def release_flags(st: dict) -> dict:
    """In repository_release mode the affirmative scope flags follow the published manifest and statement check."""
    if not repository_release(st):
        return {}
    manifest, compliance = st["release"]["manifest"], st["release"]["compliance"]
    runs = [a for a in manifest["artifacts"] if a["kind"] == "run"]
    runs_published = bool(runs) and all(a.get("status") == "published" for a in runs)
    status = {row["id"]: row["status"] for row in compliance["items"]}
    return {"original_result_archives_included": runs_published,
            "original_trained_weights_included": runs_published and status.get("checkpoints") == "fulfilled",
            "one_command_original_tables_reproduction_claimed": status.get("one_command") == "fulfilled"}


def update_registered_scope(root: Path, st: dict) -> None:
    """Keep the derived registration list current without asserting public/scientific validation."""
    expected = sorted(st["official"])
    flags = release_flags(st)
    if st["scope"].get("full_official_dataset_runs_verified") == expected and all(
            st["scope"].get(k) == v for k, v in flags.items()):
        return
    scope = dict(st["scope"], full_official_dataset_runs_verified=expected, **flags)
    write_bytes(root / "release" / "release_scope.json",
                (json.dumps(scope, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"), replace=True)
    st["scope"] = scope


def _reproduction_evidence(root: Path, budget_campaign: Path | None,
                           confirmation_campaign: Path | None = None) -> dict:
    """Import the scientific validator only for an affirmative reproduction claim."""
    from scripts.scientific_evidence_audit import build_report
    return build_report(root, budget_campaign=budget_campaign, confirmation_campaign=confirmation_campaign)


def _registered_campaign(root: Path, scope: dict, key: str, override: Path | None) -> Path | None:
    """Resolve a registered repository path; explicit CLI paths may be external."""
    if override is not None or scope.get(key) is None:
        return override
    relative = scope[key]
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or ".." in Path(relative).parts):
        raise ValueError(f"{key} must be a repository-relative path")
    campaign = (root / relative).resolve()
    if not campaign.is_relative_to(root.resolve()):
        raise ValueError(f"{key} escapes the repository")
    return campaign


def check(root: Path, st: dict, *, budget_campaign: Path | None = None,
          confirmation_campaign: Path | None = None) -> int:
    problems = []
    expected = render(st)
    actual = (root / "AVAILABILITY.md").read_text(encoding="utf-8") if (root / "AVAILABILITY.md").is_file() else ""
    if actual != expected:
        problems.append("AVAILABILITY.md is stale; run `make availability`")
    readme = (root / "README.md").read_text(encoding="utf-8")
    if START in readme and END in readme:
        block = readme[readme.index(START): readme.index(END) + len(END)]
        if block != readme_block(st):
            problems.append("README.md availability block is stale; run `make availability`")
    else:
        problems.append("README.md lacks the availability markers")
    scope = st["scope"]
    if repository_release(st):
        for flag, value in release_flags(st).items():
            if scope.get(flag) is not value:
                problems.append(f"release_scope.json {flag} must be {str(value).lower()} for the published manifest; "
                                "run `make availability`")
        if scope.get("validated_runtime_source_sha256") not in (None, st["source_sha256"]):
            problems.append("release_scope.json validated_runtime_source_sha256 is not the fingerprint of this tree")
        for p in problems:
            print("CHECK FAILED:", p, file=sys.stderr)
        if problems:
            return 1
        print("availability statement matches the registered statement, the published manifest and the release scope")
        return 0
    all_official = set(st["official"]) == set(DATASETS)
    for flag in ("original_result_archives_included", "one_command_original_tables_reproduction_claimed"):
        if type(scope.get(flag, False)) is not bool:
            problems.append(f"release_scope.json {flag} must be a boolean")
    # Full-profile registration can describe new reruns rather than the paper's
    # original producing observations. Never force an affirmative original-data
    # or reproduction claim merely because all seven datasets are registered.
    if scope.get("original_result_archives_included") is True and not all_official:
        problems.append("release_scope.json original_result_archives_included claims complete archives without all seven datasets")
    if scope.get("one_command_original_tables_reproduction_claimed") is True:
        if not all_official:
            problems.append("Original-table reconstruction requires all seven official archives")
        else:
            try:
                budget_campaign = _registered_campaign(root, scope, "scientific_evidence_budget_campaign", budget_campaign)
                confirmation_campaign = _registered_campaign(root, scope, "scientific_evidence_confirmation_campaign", confirmation_campaign)
                audit = _reproduction_evidence(root, budget_campaign, confirmation_campaign)
                if not isinstance(audit, dict) or audit.get("publication_ready") is not True:
                    blockers = audit.get("blockers", []) if isinstance(audit, dict) else []
                    count = len(blockers) if isinstance(blockers, list) else "invalid"
                    problems.append("Original-table reconstruction requires publication_ready=true from the complete "
                                    f"scientific evidence audit ({count} outstanding evidence items)")
            except Exception as exc:
                # An unavailable dependency, invalid record or validator error
                # cannot silently establish scientific readiness.
                problems.append("Original-table reconstruction audit could not be completed: "
                                f"{type(exc).__name__}: {exc}")
    if scope.get("full_official_dataset_runs_verified") != sorted(st["official"]):
        problems.append("release_scope.json full_official_dataset_runs_verified disagrees with runs/index.json")
    if scope.get("validated_runtime_source_sha256") not in (None, st["source_sha256"]):
        problems.append("release_scope.json validated_runtime_source_sha256 is not the fingerprint of this tree")
    for p in problems:
        print("CHECK FAILED:", p, file=sys.stderr)
    if problems:
        return 1
    print("availability statement matches the current artifact indexes and release scope")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--budget-campaign", type=Path,
                        help="E3-E6 campaign for validating an affirmative reconstruction claim; alternatively "
                             "set scientific_evidence_budget_campaign to a repository-relative path in release_scope.json")
    parser.add_argument("--confirmation-campaign", type=Path,
                        help="Original S14 campaign for validating an affirmative reconstruction claim; alternatively "
                             "set scientific_evidence_confirmation_campaign to a repository-relative path in release_scope.json")
    args = parser.parse_args(argv)
    try:
        st = state(args.root)
        if args.check:
            return check(args.root, st, budget_campaign=args.budget_campaign, confirmation_campaign=args.confirmation_campaign)
        update_registered_scope(args.root, st)
        (args.root / "AVAILABILITY.md").write_text(render(st), encoding="utf-8")
        if not update_readme(args.root, st):
            print("warning: README.md has no availability markers; only AVAILABILITY.md was written", file=sys.stderr)
        print("wrote AVAILABILITY.md from the current artifact indexes and release scope")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
