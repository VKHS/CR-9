#!/usr/bin/env python3
"""Generate an availability paragraph bounded by the registered repository contents.

DOI strings are described as registered references, not as proof of publication.
The reviewer workflow separately checks public record metadata and pinned bytes.
This generator does not claim that index registration establishes reproduction.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.availability import BUILT_DISPLAYS, DATASETS  # noqa: E402
from scripts.claims_ledger import registered_doi, registration_state  # noqa: E402

TARGET = ROOT / "paper" / "AVAILABILITY_STATEMENT.txt"


def render_code_release(st: dict) -> str:
    """Describe shipped code and generated products without asserting historical artifacts."""
    scope = st["scope"]
    url = scope.get("repository_url") or "https://github.com/VKHS/CR-9"
    code_doi = registered_doi(scope.get("code_archive_doi"))
    text = ("Data and code availability. The CROSS-MORSE implementation, seven dataset adapters and their configurations, "
            "split and seed design manifests, environment specifications, statistical analysis code, supported table "
            "and figure builders, and manuscript reference values are available at "
            f"{url}. The repository also provides a Slurm lifecycle launcher, E3–E6 campaign tools, a pinned-upstream "
            "baseline driver, timing/accounting tools and measurement-record exporters. ")
    if code_doi:
        text += f"The registered code-archive DOI reference is https://doi.org/{code_doi}. "
    text += ("Official datasets are obtained from their providers through the adapters. The launcher "
             "(bash run_cross_morse.sh reproduce) executes the registered dataset lifecycles, including Roman-empire's "
             "ten official splits. Each new run records its source/configuration identity, actual data-role membership, "
             "seeds, environment, checkpoints, predictions, exploration propensities and decision records; its report "
             "compares supported run-derived quantities with the paper. Auxiliary campaigns follow their separate "
             "documented protocols. make reconstruct assembles the supported numerical outputs and reports the "
             "input and build scope of every numbered display. The artifact catalog (docs/ARTIFACT_CATALOG.md) "
             "distinguishes bundled files, external-provider data, generated outputs and registered experiment archives; "
             "paper/PAPER_SUPPORT.md records display-specific coverage. Two complete synthetic example runs support "
             "software verification without benchmark data. This code release packages those examples; it does not "
             "package the original benchmark run archives. Original experiments and subsequent reproductions retain "
             "separate provenance and manuscript-relationship declarations.")
    return text + "\n"


def render(st: dict) -> str:
    if st["scope"].get("result_distribution") == "repository_release":
        # The statement registered with the journal is the paragraph; the repository is checked against it
        # (make statement-check), never the other way round.
        import json
        statement = json.loads((ROOT / "release" / "availability_statement.json").read_text(encoding="utf-8"))
        return "Data and code availability. " + statement["registered_text"] + "\n"
    if st["scope"].get("result_distribution") == "regenerate" and not st["official"]:
        return render_code_release(st)
    scope, official = st["scope"], st["official"]
    url = scope.get("repository_url") or "https://github.com/VKHS/CR-9"
    code_doi = registered_doi(scope.get("code_archive_doi"))
    records = st.get("records") or {"verified": [], "defined": []}
    registered = sorted(set(records["verified"]) & set(records["defined"]))
    dois = {dataset: registered_doi(official[dataset].get("deposit_doi")) for dataset in DATASETS if dataset in official}
    text = (f"Code and data. The CROSS-MORSE reference implementation, seven dataset adapters, configurations, "
            f"environment specifications, tests, experiment drivers and reconstruction tools are available at {url}. "
            "Official datasets are obtained from their providers through the adapters; the repository specifies "
            "the split rules and seeds. ")
    if code_doi:
        text += f"The registered code-archive DOI reference is https://doi.org/{code_doi}. "
    if official:
        text += "The run index registers full-profile dataset archives for " + ", ".join(sorted(official)) + ". "
    else:
        text += ("This checkout packages the implementation and synthetic examples; original benchmark archives "
                 "can be supplied through the separate registration workflow. ")
    references = [f"{dataset}: https://doi.org/{doi}" for dataset, doi in dois.items() if doi]
    if references:
        text += "Registered run-archive DOI references are " + "; ".join(references) + ". "
    without_doi = [dataset for dataset, doi in dois.items() if not doi]
    if without_doi:
        text += "No deposit DOI is registered for the indexed archives of " + ", ".join(without_doi) + ". "
    if registered:
        text += "Registered record sets are " + ", ".join(registered) + "; their summary values are checked against raw files by make records-check. "
    else:
        text += "Auxiliary record contracts and import tools are included; reconstruction uses separately supplied measured inputs. "
    text += (f"With the required validated archives, make paper reconstructs {BUILT_DISPLAYS}. "
             "Auxiliary record-table builders and groups without a dedicated display builder are identified in "
             "paper/CLAIMS.md. make review checks registered public run deposits against their pinned file hashes, "
             "rebuilds the supported displays and runs the scientific evidence audit for its enumerated core "
             "results and auxiliary records. Index flags and DOI references alone do not establish public "
             "availability or agreement with the manuscript; the review report records the checks and remaining "
             "limits. make reconstruct assembles supported outputs and reports the input/build scope of every "
             "numbered display; docs/ARTIFACT_CATALOG.md maps files, providers, execution products and registered "
             "archives. Bundled synthetic examples demonstrate the software workflow and are not benchmark measurements.")
    return text + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        text = render(registration_state(args.root, index_path=args.index, records_root=args.records))
        target = args.output or args.root / "paper" / "AVAILABILITY_STATEMENT.txt"
        if args.check:
            if not target.is_file() or target.read_text(encoding="utf-8") != text:
                print(f"CHECK FAILED: {target} is stale; run `make availability-statement`", file=sys.stderr)
                return 1
            print("manuscript availability paragraph matches registered contents")
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target} ({len(text.split())} words)")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
