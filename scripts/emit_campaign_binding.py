#!/usr/bin/env python3
"""Write the result binding of the ten-split Roman-empire campaign that the scientific audit requires.

    python -B scripts/emit_campaign_binding.py /abs/roman_campaign \
        --relationship exact_result_producing_run --output ../cm-evidence/roman_empire_full.json
    python -B scripts/emit_campaign_binding.py runs/roman_empire --relationship ... --check

``compare_with_manuscript.py --emit-binding`` writes single-run bindings and refuses a campaign.
The audit (scripts/scientific_evidence_audit.check_campaign_binding) binds Roman-empire through
the recomputed ten-split aggregate instead: aggregation ``paired_official_split_t95``, the ten
sealed split identities, and the virtual artifact ``derived/roman_split_aggregate.json``.  This
script writes that declaration from the sealed split runs:

  run_identities   {"dataset": "Roman-empire", "splits": [{split_index, source/config/unit sha256, dataset}, ...]}
  results          the paired loss and official-metric effects of main Table 1 (mean and interval ends)
                   and the qN/qC/qD official scores of Supplementary Table S15, at printed precision

Reported values come from paper/reported_results.json through the same resolved policy view and
tolerance rule (half the last printed digit) as the single-run emitter.  ``--check`` replays the
ten splits numerically and evaluates the declaration with the audit's own campaign checker; the
relationship to the manuscript is the author's explicit declaration.  Outputs are never written
inside the campaign or the source tree.  NumPy only; no PyTorch, checkpoint loading or network.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))
from scripts._paper_common import load_registered, load_reported, resolved_policy_view  # noqa: E402

DATASET = "Roman-empire"
ARTIFACT = "derived/roman_split_aggregate.json"
AGGREGATION = "paired_official_split_t95"
RELATIONSHIPS = ("exact_result_producing_run", "refactor_with_prediction_equivalence", "post_result_revision")
DEFAULT_IDENTIFIER = "CROSS-MORSE manuscript identified by SHA-256 in paper/sources.json"


class CampaignBindingError(ValueError):
    """The campaign or the declaration cannot support a campaign binding."""


def split_paths(campaign: Path) -> list[Path]:
    campaign = Path(campaign).resolve()
    paths = [campaign / f"split_{index:02d}" for index in range(10)]
    if any(path.is_symlink() for path in paths):
        raise CampaignBindingError("campaign split directories must be real directories, not symlinks")
    missing = [p.name for p in paths if not (p / "config.json").is_file()]
    if missing:
        raise CampaignBindingError(f"the campaign needs all ten split runs; missing: {', '.join(missing)}")
    extra = sorted(p.name for p in campaign.glob("split_*") if p.is_dir() and p not in paths)
    if extra:
        raise CampaignBindingError(f"unexpected split directories: {', '.join(extra)}")
    return paths


def split_identities(paths: list[Path]) -> list[dict]:
    """Sealed identities in the exact form check_campaign_binding reconstructs."""
    from cross_morse.review import NumericEvidence, canonical
    identities = []
    shared = set()
    for path in paths:
        evidence = NumericEvidence(path)
        config = evidence.json("config.json")
        if config.get("benchmark_dataset", config.get("dataset")) != DATASET:
            raise CampaignBindingError(f"{path.name} is not a Roman-empire run")
        if config.get("scientific_run") is not True or config.get("profile") != "full":
            raise CampaignBindingError(f"{path.name} is not a full-profile scientific run")
        if evidence.json("prepared/metadata.json").get("is_benchmark_data") is not True:
            raise CampaignBindingError(f"{path.name} does not identify benchmark observations")
        index = config["data"]["split_index"]
        if type(index) is not int or index not in range(10) or path.name != f"split_{index:02d}":
            raise CampaignBindingError(f"{path.name} does not carry its own official split index")
        common = copy.deepcopy(config)
        common["data"].pop("split_index")
        shared.add((canonical(common), evidence.identities["source_sha256"]))
        identities.append({"split_index": index, **evidence.identities})
    identities.sort(key=lambda row: row["split_index"])
    if [row["split_index"] for row in identities] != list(range(10)):
        raise CampaignBindingError("split directories do not carry split indices 0-9 exactly once")
    if len(shared) != 1:
        raise CampaignBindingError("the ten official splits must share source and non-split configuration")
    return identities


def _effect_half_unit(point, interval) -> float:
    from scripts.compare_with_manuscript import effect_half_unit
    return effect_half_unit(point, interval)


def build_binding(identities: list[dict], *, relationship: str, identifier: str = DEFAULT_IDENTIFIER,
                  tolerance_scale: float = 1.0, reported: dict | None = None, registered: dict | None = None) -> dict:
    if relationship not in RELATIONSHIPS:
        raise CampaignBindingError(f"--relationship must be one of {', '.join(RELATIONSHIPS)}")
    if not isinstance(identifier, str) or not identifier.strip():
        raise CampaignBindingError("a manuscript identifier is required")
    if (isinstance(tolerance_scale, bool) or not isinstance(tolerance_scale, (int, float))
            or not math.isfinite(tolerance_scale) or not 0 < tolerance_scale <= 1):
        raise CampaignBindingError("the tolerance scale must be positive and at most the printed half-unit (1)")
    if (not isinstance(identities, list) or len(identities) != 10
            or any(not isinstance(row, dict) or type(row.get("split_index")) is not int
                   or row.get("split_index") != i or row.get("dataset") != DATASET
                   or any(not isinstance(row.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", row[key])
                          for key in ("source_sha256", "config_sha256", "unit_sha256"))
                   for i, row in enumerate(identities))):
        raise CampaignBindingError("the binding needs the ten complete split identities in official order")
    reported = load_reported() if reported is None else reported
    registered = load_registered() if registered is None else registered
    view = resolved_policy_view(reported, DATASET, registered)
    if view["deployed_rule"] is None:
        raise CampaignBindingError("the resolved Roman-empire policy has no recorded outcome to bind")
    identity = view["comparator_identity"]
    results = []
    for pointer, field, interval_field in (("paired_qD_minus_qC_loss", "loss_effect_nats", "loss_effect_interval_nats"),
                                           ("paired_qD_minus_qC_metric_pp", "official_gain_pp", "official_gain_interval_pp")):
        bounds = [0.0, 0.0] if identity else view[interval_field]
        if bounds is None or len(bounds) != 2:
            raise CampaignBindingError(f"main Table 1 has no interval for {field}")
        tolerance = tolerance_scale * _effect_half_unit(view[field], bounds)
        for statistic, value in zip(("mean", "lower", "upper"), (view[field], *bounds)):
            entry = {"manuscript_location": f"Main Table 1, Roman-empire, ten official splits, {pointer}/{statistic}",
                     "artifact": ARTIFACT, "json_pointer": f"/{pointer}/{statistic}", "reported_value": value,
                     "absolute_tolerance": tolerance, "target_role": "served_policy",
                     "target_kind": "derived_comparator_identity" if identity else "author_reported_policy"}
            if not identity:
                suffix = field if statistic == "mean" else f"{interval_field}/{('lower', 'upper').index(statistic)}"
                entry["reference_pointer"] = f"/main_table_1_policy_and_deployment_ledger/{DATASET}/{suffix}"
            results.append(entry)
    block = reported["supplementary_table_S15_method_comparison"][DATASET]
    for name in ("qN", "qC", "qD"):
        row = view["scores"][name]
        if row.get("official_score") is None:
            raise CampaignBindingError(f"Supplementary Table S15 has no {name} score for Roman-empire")
        index = next(i for i, candidate in enumerate(block["rows"]) if candidate["method"] == row["method"])
        results.append({"manuscript_location": f"Supplementary Table S15, Roman-empire, {row['method']} official score (mean of ten splits)",
                        "artifact": ARTIFACT, "json_pointer": f"/models/{name}/metric_percent",
                        "reported_value": row["official_score"], "absolute_tolerance": tolerance_scale * 0.005,
                        "target_role": "served_policy", "target_kind": "author_reported_score",
                        "reference_pointer": f"/supplementary_table_S15_method_comparison/{DATASET}/rows/{index}/official_score"})
    return {"schema_version": "result-binding-1", "status": "author_completed", "aggregation": AGGREGATION,
            "manuscript_identifier": identifier, "relationship_to_manuscript": relationship,
            "run_identities": {"dataset": DATASET, "splits": identities}, "results": results,
            "generated_by": "scripts/emit_campaign_binding.py",
            "scope": "ten official splits on one shared graph; paired split-level Student-t descriptive interval"}


def check(paths: list[Path], binding: dict) -> dict:
    """Numerically replay every split and apply the audit's own campaign checks."""
    from scripts._source_snapshots import replay_results_exact
    from scripts.scientific_evidence_audit import binding_coverage, check_campaign_binding
    identities = split_identities(paths)
    expected = build_binding(identities, relationship=binding.get("relationship_to_manuscript"),
                             identifier=binding.get("manuscript_identifier"))
    actual = binding.get("results", [])
    expected_rows = {row["json_pointer"]: row for row in expected["results"]}
    if len(actual) != len(expected_rows) or len({row.get("json_pointer") for row in actual}) != len(expected_rows):
        raise CampaignBindingError("each required aggregate target must be bound exactly once")
    for row in actual:
        target = expected_rows.get(row.get("json_pointer"))
        if target is None or any(row.get(key) != target.get(key)
                                 for key in ("reported_value", "artifact", "target_role", "target_kind", "reference_pointer")):
            raise CampaignBindingError("campaign targets must identify the registered manuscript values and roles")
        tolerance = row.get("absolute_tolerance")
        if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                or not math.isfinite(tolerance) or not 0 <= tolerance <= target["absolute_tolerance"]):
            raise CampaignBindingError("campaign tolerance cannot exceed the printed half-unit")
    missing = binding_coverage(binding, campaign=True, dataset=DATASET, manuscript_root=ROOT)
    inspected = [{"numeric_replay": replay_results_exact(path, selection=True, root=ROOT)} for path in paths]
    result = check_campaign_binding(paths, inspected, binding)
    return {"status": result["status"] if not missing else "INCOMPLETE", "coverage_missing": missing,
            "results": [{k: r.get(k) for k in ("json_pointer", "reported_value", "actual_value", "absolute_tolerance", "agrees")}
                        for r in result["results"]],
            "aggregate": result["aggregate"]}


def output_path(path: Path, campaign: Path) -> Path:
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise CampaignBindingError(f"choose a new output file; existing paths are preserved: {path}")
    resolved = path.resolve()
    if resolved.is_relative_to(Path(campaign).resolve()) or resolved.is_relative_to(ROOT):
        raise CampaignBindingError("write the binding outside the campaign and the source tree, then install it "
                                   "with scripts/evidence_pipeline.py bind or copy it to paper/bindings/roman_empire_full.json")
    return resolved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("campaign", type=Path, help="campaign directory holding split_00 ... split_09")
    parser.add_argument("--relationship", choices=RELATIONSHIPS, required=True,
                        help="explicit declaration of the campaign's relationship to the manuscript")
    parser.add_argument("--manuscript-identifier", default=DEFAULT_IDENTIFIER)
    parser.add_argument("--tolerance-scale", type=float, default=1.0, help="positive fraction of the printed half-unit (at most 1)")
    parser.add_argument("--output", type=Path, help="new JSON file outside the campaign and the source tree")
    parser.add_argument("--check", action="store_true", help="replay the ten splits and evaluate the declaration")
    args = parser.parse_args(argv)
    try:
        paths = split_paths(args.campaign)
        binding = build_binding(split_identities(paths), relationship=args.relationship,
                                identifier=args.manuscript_identifier, tolerance_scale=args.tolerance_scale)
        if args.output:
            target = output_path(args.output, args.campaign)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as stream:
                json.dump(binding, stream, indent=2, allow_nan=False)
                stream.write("\n")
            print(f"wrote {target}", file=sys.stderr)
        else:
            print(json.dumps(binding, indent=2, allow_nan=False))
        if args.check:
            result = check(paths, binding)
            print(json.dumps(result, indent=2, allow_nan=False, default=float))
            return 0 if result["status"] == "MATCH" else 1
        return 0
    except (CampaignBindingError, OSError, ValueError, KeyError, IndexError, TypeError, StopIteration) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
