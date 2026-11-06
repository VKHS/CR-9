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
    return {"campaigns": campaigns, "example_records": example_records, "scope": scope, "official": official, "examples": examples, "registered_manifests": registered,
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
    """Describe measured-record coverage without implying missing sets exist."""
    records = st.get("records")
    if not records:
        return "No auxiliary measurement-record contract is registered."
    verified, defined = set(records["verified"]), set(records["defined"])
    text = f"Auxiliary measurement records: **{len(verified)} of {len(defined)} sets**."
    if verified:
        text += " Registered: " + ", ".join(sorted(verified)) + "."
    missing = sorted(defined - verified)
    if missing:
        text += " Missing: " + ", ".join(missing) + "."
    else:
        text += " All defined sets are registered."
    return text


def example_records_sentence(st: dict) -> str:
    names = st.get("example_records") or []
    if not names:
        return ""
    return (f"Synthetic record examples: {len(names)} sets ({', '.join(sorted(names))}); "
            "`make example-records-check` traces their values to the saved files.")


def render(st: dict) -> str:
    scope = st["scope"]
    url = scope.get("repository_url") or "https://github.com/VKHS/CR-9"
    code_doi = scope.get("code_archive_doi")
    n = len(st["official"])
    records = st.get("records") or {"verified": [], "defined": []}
    paragraph = (
        "The CROSS-MORSE reference implementation, seven dataset adapters, configurations, "
        "environment specifications, tests and reproducibility tools are available at "
        f"[{url}]({url}). Official datasets are obtained from their providers and are not redistributed. "
        f"This release registers **{n} of {len(DATASETS)} official full-profile dataset archives** and "
        f"**{len(records['verified'])} of {len(records['defined'])} auxiliary measurement-record sets**; "
        f"it includes **{len(st['examples'])} complete synthetic example runs**. "
    )
    if n == 0:
        paragraph += "Reconstruction of the manuscript's benchmark results is not established by this release."
    else:
        paragraph += "Agreement with the manuscript requires validated result bindings and the scientific evidence audit."
    if code_doi:
        paragraph += f" The registered code-archive DOI is https://doi.org/{code_doi}; publication is checked separately."
    lines = ["# Data and code availability", "", "## Availability statement", "", paragraph, "",
             "## Registered artifacts", ""]
    if n:
        lines.append("Official datasets: " + ", ".join(sorted(st["official"])) + ".")
    if st["missing_official"]:
        lines.append("Missing official archives: " + ", ".join(st["missing_official"]) + ".")
    with_doi = sum(bool(entry.get("deposit_doi")) for entry in st["official"].values())
    dois = sorted({entry["deposit_doi"] for entry in st["official"].values() if entry.get("deposit_doi")})
    if with_doi:
        lines.append(f"Registered DOI references: {with_doi} of {n} registered datasets; " +
                     ", ".join(f"https://doi.org/{doi}" for doi in dois) + ".")
    elif n:
        lines.append("No DOI references are recorded for the registered archives.")
    lines += ["", records_sentence(st), "",
              "Archive identities are recorded in [runs/index.json](runs/index.json); "
              "measured-record coverage is recorded in [records/index.json](records/index.json). "
              "Index registration does not itself verify public availability; `make fetch-deposits` checks the registered run deposits and their bytes. "
              "Synthetic examples exercise the software workflow and do not supply official benchmark measurements."]
    if st.get("example_records"):
        lines += ["", example_records_sentence(st)]
    lines += ["", "## Reconstruction scope", "",
              f"With validated input archives, `make tables figures` builds {BUILT_DISPLAYS} "
              "for the registered datasets. `make records-tables` builds the registered auxiliary sets. " + NOT_BUILT,
              "", "`make reproduce` and `make example-paper` use the bundled synthetic runs. "
              "`make compare-runs` checks reported outcomes; `make scientific-audit` additionally requires "
              "complete applicable archives, measurement records, model identities and result bindings.",
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
    records = st.get("records") or {"verified": [], "defined": []}
    return "\n".join([
        START,
        f"**Registered:** {len(st['official'])} of {len(DATASETS)} official benchmark archives, "
        f"{len(records['verified'])} of {len(records['defined'])} auxiliary measurement sets, "
        f"and {len(st['examples'])} complete synthetic examples. "
        "See [availability and reconstruction scope](AVAILABILITY.md).",
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


def update_registered_scope(root: Path, st: dict) -> None:
    """Keep the derived registration list current without asserting public/scientific validation."""
    expected = sorted(st["official"])
    if st["scope"].get("full_official_dataset_runs_verified") == expected:
        return
    scope = dict(st["scope"], full_official_dataset_runs_verified=expected)
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
    print(f"availability statement matches the tree ({len(st['official'])}/7 official run archives registered)")
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
        print(f"wrote AVAILABILITY.md ({len(st['official'])}/7 official run archives registered)")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
