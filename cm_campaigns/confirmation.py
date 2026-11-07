"""Replay the manuscript's joint S14 confirmation family from original tests.

The heterogeneous original cluster, graph-bootstrap and small-sample tests are
preserved as hashed inputs. This auditor verifies paired effects and replays
Holm plus the registered point-margin/fallback decisions; it does NOT replace
those tests with one generic test or execute arbitrary supplied analysis code.

Two campaign origins are audited.  An imported campaign carries the original
test files; every requirement, including the admission links, is strict.  A
campaign that cm_campaigns/s14_builder.py builds from the sealed lifecycle runs
declares ``production.kind == "repository_built"``.  Its joint lock is fixed
after the lifecycle's own admission stages, so the admission-link conditions
(joint-lock governance, admission after the joint lock, lifecycle pre-admission
lock equal to the joint decision) are evaluated and reported in
``admission_findings`` with a ``post_hoc`` flag instead of being required, and
its tests may carry the single ``ensemble`` seed level (one observation per
unit).  Every other requirement is the same for both origins.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import math
from pathlib import Path

from .common import (canonical, confined, directory_inventory, read_json,
                     require, rows, sha256, unique, valid_hash)
from .numeric import holm

SCHEMA = "cm-s14-confirmation-1"
PROTOCOL_SCHEMA = "cm-s14-protocol-1"
ROLE_SCHEMA = "cm-s14-data-roles-1"
RESULT_SCHEMA = "cm-s14-original-test-1"
LOCK_SCHEMA = "cm-s14-joint-lock-1"
ROLES = ("fit", "explore", "select", "confirm", "compare", "admit", "test")
REPOSITORY_BUILT = "repository_built"      # production.kind of a campaign built by cm_campaigns/s14_builder.py
ENSEMBLE_SEED_LEVEL = "ensemble"           # its seed axis: the policies' E-member ensembles, one observation per unit
# This is the family in S14, not the selected local family emitted by one run.
FAMILY = {
    "arxiv/state": ("ogbn-arxiv", "state", "buffered_cluster", 32, "CE", 1, .002),
    "arxiv/radius": ("ogbn-arxiv", "radius", "buffered_cluster", 32, "CE", -1, .0015),
    "proteins/state": ("ogbn-proteins", "state", "buffered_cluster", 32, "mean_BCE", 1, .002),
    "proteins/radius": ("ogbn-proteins", "radius", "buffered_cluster", 32, "mean_BCE", -1, .0015),
    "products/state": ("ogbn-products", "state", "buffered_cluster", 64, "CE", 1, .002),
    "Roman/RH": ("Roman-empire", "mpc", "official_split", 10, "CE", -1, .002),
    "Pascal/RH": ("PascalVOC-SP", "mpc", "graph", 425, "CE", -1, .002),
    "COCO/RH": ("COCO-SP", "mpc", "graph", 5664, "CE", -1, .002),
    "Peptides/radius": ("Peptides-func", "radius", "graph", 544, "mean_BCE", -1, .0015),
}
DATASETS = tuple(dict.fromkeys(row[0] for row in FAMILY.values()))


def _method_family(kind, unit):
    if unit == "graph":
        return "graph_bootstrap"
    if unit == "official_split":
        return "paired_small_sample"
    return "paired_cluster" if kind == "state" else "paired_unit"


def _text(value, label):
    require(isinstance(value, str) and bool(value.strip()), f"{label}: nonempty text required")
    return value


def _number(value, label, *, low=None, high=None):
    require(type(value) in (float, int) and math.isfinite(value), f"{label}: finite JSON number required")
    require(low is None or value >= low, f"{label}: below {low}")
    require(high is None or value <= high, f"{label}: above {high}")
    return float(value)


def _stamp(value, label):
    _text(value, label)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label}: ISO timestamp required") from exc
    require(result.tzinfo is not None and result.utcoffset() is not None,
            f"{label}: explicit timezone required")
    return result


def _hash(value, label):
    require(valid_hash(value), f"{label}: SHA-256 required")
    return value


def _reference(root, ref, label):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, f"{label}: path/SHA-256 reference required")
    path = confined(root, ref["path"])
    require(sha256(path) == _hash(ref["sha256"], label), f"{label}: bytes differ from the original hash")
    return path


def _json_reference(root, ref, label):
    return read_json(_reference(root, ref, label))


def _keyed(items, key, expected, label):
    require(isinstance(items, list) and all(isinstance(item, dict) for item in items), f"{label}: object list required")
    values = [item.get(key) for item in items]
    require(all(isinstance(value, str) for value in values), f"{label}: string identities required")
    unique(values, label)
    require(set(values) == set(expected), f"{label}: exact registered family required; missing={sorted(set(expected)-set(values))}, unexpected={sorted(set(values)-set(expected))}")
    return dict(zip(values, items))


def _ids(value, label):
    require(isinstance(value, list) and value and all(isinstance(v, str) and v for v in value), f"{label}: explicit observation IDs required")
    unique(value, label)
    return set(value)


def _roles(root, ref, dataset, registered):
    data = _json_reference(root, ref, f"{dataset} role manifest")
    require(data.get("schema") == ROLE_SCHEMA and data.get("dataset") == dataset,
            f"{dataset}: incorrect role manifest identity")
    require(data.get("reserved_at_stage") == "prepare", f"{dataset}: data roles must be reserved at prepare")
    require(_stamp(data.get("reserved_at"), "reserved_at") <= registered,
            f"{dataset}: roles were reserved after the family")
    allocation = data.get("roles")
    require(isinstance(allocation, dict) and set(allocation) == set(ROLES), f"{dataset}: all seven separated roles required")
    seen = set()
    for role in ROLES:
        ids = _ids(allocation[role], f"{dataset}/{role}")
        require(not (ids & seen), f"{dataset}: observation roles overlap at {role}")
        seen.update(ids)
    units = data.get("confirmation_units")
    require(isinstance(units, dict) and units, f"{dataset}: confirmation unit membership required")
    assigned = set()
    for unit, ids in units.items():
        _text(unit, "unit ID")
        members = _ids(ids, f"{dataset}/{unit}")
        require(not assigned.intersection(members), f"{dataset}: confirmation observations occur in multiple units; use split-qualified IDs for Roman")
        assigned.update(members)
    require(assigned == set(allocation["confirm"]), f"{dataset}: confirmation units do not cover D_confirm exactly")
    # Graphs are units, not nodes or optimizer seeds, in the multigraph tests.
    if dataset in ("PascalVOC-SP", "COCO-SP", "Peptides-func"):
        require(all(ids == [unit] for unit, ids in units.items()), f"{dataset}: graph-unit IDs must be their reserved graph IDs")
    return data


def _paired_effect(path, expected_units, expected_seeds, weighting):
    data = rows(path, ("seed_id", "unit_id", "left_loss_nats", "right_loss_nats", "unit_weight"))
    cells, weights, seed_ids = {}, {}, set()
    for row in data:
        seed, unit = _text(row["seed_id"], "seed ID"), _text(row["unit_id"], "unit ID")
        require(unit in expected_units, "Paired loss unit is absent from D_confirm")
        key = (seed, unit)
        require(key not in cells, "Duplicate paired seed/unit record")
        try:
            left, right, weight = (float(row[k]) for k in ("left_loss_nats", "right_loss_nats", "unit_weight"))
        except (ValueError, TypeError) as exc:
            raise ValueError("Paired losses must be finite numeric CSV values") from exc
        require(all(math.isfinite(v) for v in (left, right, weight)) and left >= 0 and right >= 0 and weight > 0,
                "Losses must be finite/nonnegative and unit weights positive")
        require(unit not in weights or weight == weights[unit], "Unit weights change across optimization seeds")
        weights[unit] = weight
        cells[key] = left - right
        seed_ids.add(seed)
    require(set(seed_ids) == set(expected_seeds), "Paired observations omit or add prespecified optimization seeds")
    require(set(weights) == set(expected_units), "Missing D_confirm units in paired losses")
    if weighting["scheme"] == "equal_unit":
        require(len(set(weights.values())) == 1, "Protocol requires equal statistical-unit weights")
    else:
        require(weights == weighting["weights"], "Observed unit weights differ from prespecified fixed weights")
    require(len(cells) == len(seed_ids) * len(expected_units), "Incomplete crossed seed-by-unit pairing")
    total_weight = math.fsum(weights.values())
    # Seeds are an averaging axis, never extra independent graphs/cores/splits.
    effect = math.fsum(weights[unit] * math.fsum(cells[seed, unit] for seed in sorted(seed_ids)) / len(seed_ids)
                       for unit in sorted(expected_units)) / total_weight
    return {"effect_nats": effect, "unit_count": len(expected_units), "seed_count": len(seed_ids),
            "weighting": "equal_unit" if len(set(weights.values())) == 1 else "fixed_unit_weights"}


def _selection(root, item, dataset, first_test_time):
    selected = _json_reference(root, item["selection"], f"{dataset} selector")
    candidates = _json_reference(root, item["candidates"], f"{dataset} candidates")
    require(_stamp(item.get("selection_locked_at"), "selection_locked_at") < first_test_time,
            f"{dataset}: selection must be fixed before confirmation tests")
    require(isinstance(candidates, list) and candidates and all(isinstance(c, dict) for c in candidates),
            f"{dataset}: candidate registry required")
    hashes = [_hash(c.get("hash"), f"{dataset} candidate hash") for c in candidates]
    unique(hashes, f"{dataset} candidate hashes")
    for key in ("comparator_index", "provisional_index"):
        require(type(selected.get(key)) is int and 0 <= selected[key] < len(candidates), f"{dataset}: invalid {key}")
    eligible = selected.get("eligible_indices")
    require(isinstance(eligible, list) and all(type(i) is int and 0 <= i < len(candidates) for i in eligible),
            f"{dataset}: eligible candidate indices required")
    unique(eligible, f"{dataset} eligible indices")
    qc, provisional = hashes[selected["comparator_index"]], hashes[selected["provisional_index"]]
    comparator = candidates[selected["comparator_index"]]
    require(comparator.get("is_open_loop") is True and _route(comparator.get("route")) in
            ("nominal", "best_depth", "learned_stop", "phase", "fixed_sequence", "mpc_ol"),
            f"{dataset}: qC must identify a registered open-loop route")
    eligible_hashes = {hashes[i] for i in eligible}
    require(provisional == qc or provisional in eligible_hashes, f"{dataset}: provisional policy is outside Q_s")
    return {"qC": qc, "provisional": provisional, "eligible": eligible_hashes,
            "candidates": dict(zip(hashes, candidates))}


def _route(value):
    return {"one-step": "one_step", "MPC-RH": "mpc_rh", "MPC-OL": "mpc_ol", "fixed-sequence": "fixed_sequence",
            "best-depth": "best_depth", "learned-stop": "learned_stop"}.get(value, value)


def _policy_decision(dataset, selection, tested):
    qc, candidate = selection["qC"], selection["provisional"]
    required = [key for key, spec in FAMILY.items() if spec[0] == dataset]
    for key in required:
        row = tested[key]
        # A state intervention is destroyed-minus-intact; radius/RH is candidate-minus-predecessor.
        tested_candidate = row["right_policy_sha256"] if FAMILY[key][1] == "state" else row["left_policy_sha256"]
        require(tested_candidate in selection["candidates"], f"{dataset}: original confirmation target is absent from the frozen candidate ledger")
        if candidate != qc:
            require(tested_candidate == candidate, f"{dataset}: confirmation observations concern a different selected policy")
        row["tested_policy_selected"] = tested_candidate == candidate
        if FAMILY[key][1] == "state":
            target = selection["candidates"][tested_candidate]
            require(target.get("is_open_loop") is False and _route(target.get("route")) in ("one_step", "radius", "mpc_rh"),
                    f"{dataset}: state intervention must identify a registered feedback candidate, not the open-loop comparator")
    for key in required:
        kind = FAMILY[key][1]
        if kind in ("radius", "mpc"):
            expected, simpler = ("radius", "one_step") if kind == "radius" else ("mpc_rh", "mpc_ol")
            target = tested[key]["left_policy_sha256"]
            require(_route(selection["candidates"][target].get("route")) == expected,
                    f"{dataset}: tested selected policy has the wrong route family")
            predecessor = tested[key]["right_policy_sha256"]
            require(predecessor in selection["candidates"] and
                    _route(selection["candidates"][predecessor].get("route")) == simpler,
                    f"{dataset}: tested predecessor must be the registered {simpler} route")
    if candidate == qc:
        return {"comparator_hash": qc, "provisional_hash": candidate, "confirmed_hash": qc,
                "reason": "comparator selected; prespecified unselected contrasts remain in the joint family"}
    state = [tested[key] for key in required if FAMILY[key][1] == "state"]
    route = [tested[key] for key in required if FAMILY[key][1] in ("radius", "mpc")]
    if state and not all(row["retained"] for row in state):
        confirmed, reason = qc, "state confirmation failed; comparator returned"
    elif route and not route[0]["retained"]:
        predecessor = route[0]["right_policy_sha256"]
        require(predecessor in selection["candidates"], f"{dataset}: predecessor absent from registered candidate ledger")
        confirmed = predecessor if predecessor == qc or predecessor in selection["eligible"] else qc
        reason = "route confirmation failed; eligible predecessor" if confirmed != qc else "route confirmation failed; comparator returned"
    else:
        confirmed, reason = candidate, "all required joint-family contrasts retained"
    return {"comparator_hash": qc, "provisional_hash": candidate, "confirmed_hash": confirmed, "reason": reason}



def composite_policy_hash(split_policy_hashes):
    """Identity of a ten-split Roman policy; a split is not a new graph."""
    return hashlib.sha256(canonical({"split_policy_hashes": split_policy_hashes})).hexdigest()


def _run_bindings(root, entry, decision, selection, original_lock_ref, findings=None):
    """Original run bindings.  ``findings`` (a list) switches the comparison of each lifecycle pre-admission
    lock with the joint decision from a requirement to a reported finding (repository-built campaigns)."""
    dataset = entry["dataset"]
    bindings = entry.get("run_bindings")
    expected = list(range(10)) if dataset == "Roman-empire" else [None]
    require(isinstance(bindings, list) and all(isinstance(r, dict) for r in bindings) and len(bindings) == len(expected) and
            [r.get("split_index") for r in bindings] == expected,
            f"{dataset}: exact original run bindings required (all ten Roman official splits)")
    components = {}
    if dataset == "Roman-empire":
        for identity, candidate in selection["candidates"].items():
            parts = candidate.get("split_policy_hashes")
            require(isinstance(parts, list) and len(parts) == 10 and
                    all(isinstance(r, dict) and set(r) == {"split_index", "hash"} and type(r["split_index"]) is int for r in parts)
                    and [r["split_index"] for r in parts] == expected,
                    "Roman aggregate candidates must identify every original split policy")
            components[identity] = [_hash(row.get("hash"), "Roman split candidate hash") for row in parts]
            require(identity == composite_policy_hash(parts), "Roman aggregate policy hash is not its canonical split composition")
    output = []
    for binding in bindings:
        identities = {key: _hash(binding.get(key), f"{dataset}/{key}") for key in
                      ("source_sha256", "config_sha256", "unit_sha256")}
        selected = _json_reference(root, binding["selection"], f"{dataset} original run selector")
        candidates = _json_reference(root, binding["candidates"], f"{dataset} original run candidates")
        lock = _json_reference(root, binding["pre_admission_lock"], f"{dataset} original run lock")
        require(isinstance(candidates, list) and candidates and all(isinstance(row, dict) for row in candidates),
                "Original run candidate ledger required")
        hashes = [_hash(row.get("hash"), "Original candidate hash") for row in candidates]
        unique(hashes, "original run candidate hashes")
        for key in ("comparator_index", "provisional_index"):
            require(type(selected.get(key)) is int and 0 <= selected[key] < len(candidates), f"Invalid original {key}")
        qc, provisional = hashes[selected["comparator_index"]], hashes[selected["provisional_index"]]
        if dataset == "Roman-empire":
            split = binding["split_index"]
            wanted = {key: components[decision[key]][split] for key in ("comparator_hash", "provisional_hash", "confirmed_hash")}
            require(qc == wanted["comparator_hash"] and provisional == wanted["provisional_hash"],
                    "Roman aggregate selection differs from an original split selector")
            native = dict(zip(hashes, candidates))
            for identity, aggregate in selection["candidates"].items():
                part = native.get(components[identity][split])
                require(part is not None and _route(part.get("route")) == _route(aggregate.get("route"))
                        and part.get("is_open_loop") == aggregate.get("is_open_loop"),
                        "Roman aggregate route/open-loop identity differs from an original split candidate")
        else:
            wanted = decision
            require(binding["selection"]["sha256"] == entry["selection"]["sha256"] and
                    binding["candidates"]["sha256"] == entry["candidates"]["sha256"] and
                    binding["pre_admission_lock"]["sha256"] == original_lock_ref["sha256"],
                    f"{dataset}: joint confirmation is not bound to the original selector/lock bytes")
        agrees = lock.get("confirmed_hash") == wanted["confirmed_hash"] and lock.get("comparator_hash") == wanted["comparator_hash"]
        if findings is None:
            require(agrees and lock.get("config_sha256") == identities["config_sha256"],
                    f"{dataset}: original run predictor/configuration lock disagrees")
        else:
            require(lock.get("config_sha256") == identities["config_sha256"], f"{dataset}: original run configuration lock disagrees")
            findings.append({"split_index": binding["split_index"], "lifecycle_lock_agrees_with_joint_decision": agrees,
                             "lifecycle_comparator_hash": lock.get("comparator_hash"),
                             "lifecycle_confirmed_hash": lock.get("confirmed_hash"),
                             "joint_comparator_hash": wanted["comparator_hash"], "joint_confirmed_hash": wanted["confirmed_hash"]})
        require(wanted["confirmed_hash"] in hashes, f"{dataset}: confirmed policy missing from original candidates")
        output.append({"split_index": binding["split_index"], "identities": {"dataset": dataset, **identities},
                       "artifacts": {"select/decision.json": binding["selection"]["sha256"],
                                     "select/candidates.json": binding["candidates"]["sha256"],
                                     "admit/pre_admission_lock.json": binding["pre_admission_lock"]["sha256"]}})
    if dataset == "Roman-empire":
        aggregate = _json_reference(root, original_lock_ref, "Roman aggregate original lock")
        require(aggregate.get("split_lock_sha256") == {str(row["split_index"]): row["pre_admission_lock"]["sha256"] for row in bindings},
                "Roman original aggregate lock does not bind all ten split pre-admission locks")
    return output


def _role_identity(roles, units):
    return hashlib.sha256(canonical({"roles": {k: sorted(v) for k, v in roles.items()},
                                     "confirmation_units": {k: sorted(v) for k, v in units.items()}})).hexdigest()


def bind_official_runs(report, official_runs):
    """Match S14 inputs with already validated original run-stage manifests.

    This identity check is necessary in addition to a self-contained campaign
    audit. It does not silently turn copied or unrelated files into run evidence.
    """
    from cross_morse.review import NumericEvidence
    matched = {}
    for dataset, paths in official_runs.items():
        expected = report["official_run_bindings"][dataset]
        require(len(paths) == len(expected), f"{dataset}: original run count differs from S14 bindings")
        checked, observed_roles, observed_units = [], {r: [] for r in ROLES}, {}
        for path in paths:
            evidence = NumericEvidence(Path(path))
            config = evidence.json("config.json")
            metadata = evidence.json("prepared/metadata.json")
            require(config.get("dataset") == dataset and config.get("profile") == "full" and
                    config.get("scientific_run") is True and metadata.get("is_benchmark_data") is True,
                    f"{dataset}: synthetic/pilot runs cannot establish joint S14 linkage")
            split = config["data"]["split_index"] if dataset == "Roman-empire" else None
            row = next((r for r in expected if r["split_index"] == split), None)
            require(row is not None, f"{dataset}: S14 binding missing for original split {split}")
            require(row["identities"] == evidence.identities, f"{dataset}: S14 source/config/unit identity differs from the original official run")
            for artifact, expected_hash in row["artifacts"].items():
                evidence.file(artifact)
                require(evidence.sealed[artifact]["sha256"] == expected_hash,
                        f"{dataset}: S14 {artifact} differs from its original sealed run artifact")
            native_candidates = evidence.json("select/candidates.json")
            native_confirm = evidence.json("confirm/decision.json")
            native_lock = evidence.json("admit/pre_admission_lock.json")
            for index_key, hash_key in (("comparator_index", "comparator_hash"), ("confirmed_index", "confirmed_hash")):
                index = native_confirm.get(index_key)
                require(type(index) is int and 0 <= index < len(native_candidates),
                        f"{dataset}: invalid native confirmation policy index")
                require(native_candidates[index]["hash"] == native_lock.get(hash_key)
                        and native_lock.get(index_key) == index,
                        f"{dataset}: joint pre-admission lock differs from the actual native confirmation decision")
            seed_ids = set(map(str, config["training"]["seeds"]))
            # A repository-built test is a contrast of E-member ensembles (training seeds[:E]): one level.
            ensemble = report.get("production_kind") == REPOSITORY_BUILT and bool(seed_ids)
            require(all(set(test["seed_ids"]) == seed_ids or (ensemble and test["seed_ids"] == [ENSEMBLE_SEED_LEVEL])
                        for test in report["rows"] if test["dataset"] == dataset),
                    f"{dataset}: S14 optimization seed population differs from the original run")
            independent = dataset in ("PascalVOC-SP", "COCO-SP", "Peptides-func")
            splits = evidence.arrays("prepared/graph_splits.npz" if independent else "prepared/splits.npz")
            prefix = f"split={split}:" if dataset == "Roman-empire" else ""
            for role in ROLES:
                observed_roles[role].extend(prefix + str(int(i)) for i in splits[role])
            if dataset == "Roman-empire":
                observed_units[f"split={split}"] = [prefix + str(int(i)) for i in splits["confirm"]]
            elif independent:
                observed_units = {str(int(i)): [str(int(i))] for i in splits["confirm"]}
            else:
                clusters = evidence.arrays("prepared/clusters.npz")["cluster_ids"]
                for i in splits["confirm"]:
                    observed_units.setdefault(str(int(clusters[i])), []).append(str(int(i)))
            checked.append(split)
        unique(checked, f"{dataset} original split bindings")
        require(_role_identity(observed_roles, observed_units) == report["role_identities"][dataset],
                f"{dataset}: S14 data roles or confirmation-unit memberships differ from the prepared official run")
        matched[dataset] = {"runs": len(checked), "split_indices": checked}
    missing = sorted(set(DATASETS) - set(matched))
    return {"status": "PASS" if not missing else "PENDING", "datasets": matched, "missing_datasets": missing}


def _reference_match(actual, printed):
    if isinstance(printed, str) and printed.strip().startswith("<"):
        return actual < float(printed.strip()[1:].strip())
    if type(printed) not in (int, float):
        return False
    # Keep the target at its stored printed precision; never infer unrounded p.
    from scripts._paper_common import printed_half_unit
    return abs(actual - printed) <= printed_half_unit(printed) + 1e-14


def input_identity(root):
    """Digest every campaign file; reports should be written outside this folder."""
    return hashlib.sha256(canonical(directory_inventory(Path(root)))).hexdigest()


def check_campaign(campaign: Path, *, manuscript_root: Path) -> dict:
    root, manuscript_root = Path(campaign).resolve(), Path(manuscript_root).resolve()
    manifest = read_json(confined(root, "campaign.json"))
    require(manifest.get("schema") == SCHEMA and manifest.get("status") == "author_completed", "An author-completed S14 campaign is required; a template is not evidence")
    require(manifest.get("synthetic") is False and manifest.get("evidence_kind") == "original_confirmation_tests",
            "Synthetic, reconstructed-from-paper and sensitivity inputs cannot establish S14 evidence")
    sources = read_json(manuscript_root / "paper/sources.json")
    expected_sources = {key: value["sha256"] for key, value in sources["documents"].items()}
    require(manifest.get("manuscript_sha256") == expected_sources, "S14 campaign is bound to a different manuscript identity")
    production = manifest.get("production")
    repository_built = isinstance(production, dict) and production.get("kind") == REPOSITORY_BUILT
    if repository_built:
        # The builder and its production record travel with the campaign.
        _text(production.get("builder"), "production builder")
        _reference(root, production.get("provenance"), "production provenance")
        if production.get("plan") is not None:
            _reference(root, production["plan"], "production plan")
    protocol = _json_reference(root, manifest["protocol"], "prespecified S14 protocol")
    require(protocol.get("schema") == PROTOCOL_SCHEMA and protocol.get("family") == "Supplementary Table S14",
            "Incorrect confirmation protocol family")
    constants = read_json(manuscript_root / "paper/registered_constants.json")["confirmation_eq20_table_S11"]
    alpha = _number(protocol.get("alpha"), "alpha", low=0, high=1)
    require(alpha == constants["holm_alpha"] == .05, "Joint S14 alpha must be the registered 0.05")
    registered = _stamp(protocol.get("registered_at"), "protocol registered_at")
    specifications = _keyed(protocol.get("contrasts"), "contrast_id", FAMILY, "prespecified nine-test family")
    tests = _keyed(manifest.get("tests"), "contrast_id", FAMILY, "original nine-test outputs")
    datasets = _keyed(manifest.get("datasets"), "dataset", DATASETS, "seven dataset evidence bindings")
    role_maps = {dataset: _roles(root, item["role_manifest"], dataset, registered) for dataset, item in datasets.items()}
    tested, completed, started = {}, [], {}
    for key, expected in FAMILY.items():
        dataset, kind, unit, count, loss, direction, margin = expected
        spec, entry = specifications[key], tests[key]
        for field, value in (("dataset", dataset), ("kind", kind), ("statistical_unit", unit), ("unit_count", count),
                             ("loss", loss), ("beneficial_direction", direction), ("practical_margin_nats", margin), ("data_role", "D_confirm")):
            require(spec.get(field) == value and (field not in ("unit_count", "beneficial_direction") or type(spec[field]) is int), f"{key}: protocol {field} differs from S14")
        method = spec.get("method")
        require(isinstance(method, dict), f"{key}: original test method specification required")
        _text(method.get("name"), f"{key} original method")
        require(method.get("family") == _method_family(kind, unit), f"{key}: original method family differs from S14")
        require(method.get("tail") in ("less", "greater", "two_sided") and method.get("null") == "zero",
                f"{key}: original unadjusted test against zero must be identified")
        require(method["tail"] in ("two_sided", "greater" if direction > 0 else "less"),
                f"{key}: test tail opposes the registered beneficial direction")
        _text(method.get("command"), f"{key} original analysis command")
        for field in ("analysis_code", "analysis_config", "environment"):
            _reference(root, method[field], f"{key}/{field}")
        for field in ("left_policy_sha256", "right_policy_sha256"):
            _hash(spec.get(field), f"{key} prespecified {field}")
        expected_seeds = _ids(spec.get("seed_ids"), f"{key} prespecified seed IDs")
        weighting = spec.get("weighting")
        require(isinstance(weighting, dict) and weighting.get("scheme") in ("equal_unit", "fixed_unit_weights"),
                f"{key}: prespecified weighting required")
        if unit in ("graph", "official_split"):
            require(weighting["scheme"] == "equal_unit", f"{key}: graph and official-split estimands require equal unit weights")
        if weighting["scheme"] == "fixed_unit_weights":
            values = _json_reference(root, weighting["unit_weights"], f"{key} fixed unit weights")
            require(isinstance(values, dict) and set(values) == set(role_maps[dataset]["confirmation_units"]),
                    f"{key}: fixed weights must name all confirmation units")
            weighting = {"scheme": "fixed_unit_weights", "weights": {k: _number(v, "fixed unit weight", low=0) for k, v in values.items()}}
            require(all(v > 0 for v in weighting["weights"].values()), f"{key}: unit weights must be positive")
        result = _json_reference(root, entry["result"], f"{key} original test result")
        require(result.get("schema") == RESULT_SCHEMA and result.get("contrast_id") == key and result.get("dataset") == dataset,
                f"{key}: incorrect original test identity")
        require(result.get("evidence_kind") == "original_confirmation_test" and result.get("p_value_kind") == "unadjusted_against_zero"
                and result.get("numeric_precision") == "unrounded" and result.get("synthetic") is False,
                f"{key}: an original unrounded raw test is required, not a printed or locally adjusted P value")
        for field, value in (("data_role", "D_confirm"), ("statistical_unit", unit), ("unit_count", count),
                             ("loss", loss), ("method_name", method["name"]), ("method_family", method["family"]), ("tail", method["tail"]),
                             ("interval_confidence", .95)):
            require(result.get(field) == value, f"{key}: result {field} disagrees with the prespecified test")
        require(_ids(result.get("seed_ids"), f"{key} result seed IDs") == expected_seeds,
                f"{key}: original test seed IDs differ from its prespecified population")
        require(all(result.get(field) == spec[field] for field in ("left_policy_sha256", "right_policy_sha256")),
                f"{key}: original test policies differ from the prespecified contrast")
        if kind == "state":
            require(result.get("intervention_valid") is True and result.get("intervention_scope") == "controller_observations_only"
                    and result.get("left_policy_sha256") != result.get("right_policy_sha256"),
                    f"{key}: valid conditional controller-observation intervention must be identified")
        require(result.get("protocol_sha256") == manifest["protocol"]["sha256"], f"{key}: result not bound to the prespecified protocol")
        require(result.get("paired_losses_sha256") == entry["paired_losses"]["sha256"], f"{key}: original test input hash mismatch")
        require(result.get("role_manifest_sha256") == datasets[dataset]["role_manifest"]["sha256"], f"{key}: test role identity mismatch")
        start, finish = _stamp(result.get("started_at"), f"{key} start"), _stamp(result.get("completed_at"), f"{key} finish")
        require(registered < start <= finish, f"{key}: prespecification must precede the original tests")
        started[dataset] = min(start, started.get(dataset, start)); completed.append(finish)
        effect = _paired_effect(_reference(root, entry["paired_losses"], f"{key} paired observations"), role_maps[dataset]["confirmation_units"], expected_seeds, weighting)
        require(effect["unit_count"] == count, f"{key}: wrong number of independent units; seeds cannot increase n")
        observed = _number(result.get("effect_nats"), f"{key} effect")
        require(math.isclose(observed, effect["effect_nats"], rel_tol=1e-9, abs_tol=1e-12), f"{key}: effect does not reconstruct from the paired observations")
        p = _number(result.get("p_value_zero"), f"{key} original raw P", low=0, high=1)
        require(result.get("valid_interval") is True, f"{key}: original test interval is invalid")
        interval = result.get("interval_nats")
        require(isinstance(interval, list) and len(interval) == 2, f"{key}: original interval required")
        low, high = (_number(v, f"{key} interval") for v in interval)
        require(low <= observed <= high, f"{key}: original effect lies outside its interval")
        tested[key] = {"contrast_id": key, "dataset": dataset, **effect, "reconstructed_effect_nats": effect["effect_nats"],
                       "effect_nats": observed, "p_value_zero": p,
                       "method_name": method["name"], "method_family": method["family"], "tail": method["tail"], "statistical_unit": unit,
                       "loss": loss, "seed_ids": sorted(expected_seeds), "interval_nats": interval, "practical_margin_nats": margin,
                       "beneficial_direction": direction,
                       "left_policy_sha256": _hash(result.get("left_policy_sha256"), f"{key} left policy"),
                       "right_policy_sha256": _hash(result.get("right_policy_sha256"), f"{key} right policy")}
    adjusted = holm([tested[key]["p_value_zero"] for key in FAMILY])
    for key, p in zip(FAMILY, adjusted):
        item = tested[key]
        item["holm_p_value_zero"] = p
        item["point_beyond_margin"] = item["beneficial_direction"] * item["effect_nats"] > item["practical_margin_nats"]
        item["retained"] = p < alpha and item["point_beyond_margin"]
    selections = {dataset: _selection(root, entry, dataset, started[dataset]) for dataset, entry in datasets.items()}
    decisions = {dataset: _policy_decision(dataset, selections[dataset], tested) for dataset in datasets}
    # The historical lock binds the complete nine-test inputs before admission.
    # The auditor never creates or backdates that lock.
    digest_material = {"protocol": manifest["protocol"], "tests": manifest["tests"],
                       "datasets": [{k: v for k, v in entry.items() if k != "admission_link"} for entry in manifest["datasets"]]}
    digest = hashlib.sha256(canonical(digest_material)).hexdigest()
    lock = _json_reference(root, manifest["joint_decision_lock"], "pre-admission joint decision lock")
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("input_sha256") == digest, "Joint lock does not bind all nine original tests and selection/role inputs")
    locked_at = _stamp(lock.get("locked_at"), "joint lock time")
    require(max(completed) <= locked_at, "Joint decisions were locked before the original tests completed")
    require(lock.get("mechanism_retained") == {key: value["retained"] for key, value in tested.items()},
            "Locked mechanism decisions disagree with strict joint Holm plus point-margin replay")
    expected_policies = {dataset: {k: row[k] for k in ("comparator_hash", "provisional_hash", "confirmed_hash")}
                         for dataset, row in decisions.items()}
    require(lock.get("policies") == expected_policies, "Locked policies disagree with joint confirmation and Q_s-or-qC fallback")
    run_bindings, admission_findings = {}, {}
    for dataset, entry in datasets.items():
        link = _json_reference(root, entry["admission_link"], f"{dataset} admission link")
        if not repository_built:
            require(link.get("dataset") == dataset and link.get("joint_lock_sha256") == manifest["joint_decision_lock"]["sha256"],
                    f"{dataset}: admission did not use the joint decision lock")
            require(link.get("confirmed_hash") == decisions[dataset]["confirmed_hash"] and link.get("role_manifest_sha256") == entry["role_manifest"]["sha256"],
                    f"{dataset}: admission predictor or reserved roles differ from the joint lock")
            require(_stamp(link.get("admission_started_at"), f"{dataset} admission time") > locked_at,
                    f"{dataset}: admission labels were accessed before joint confirmation was fixed")
            original_lock = _json_reference(root, link["original_pre_admission_lock"], f"{dataset} original run lock")
            require(original_lock.get("confirmed_hash") == decisions[dataset]["confirmed_hash"] and original_lock.get("comparator_hash") == decisions[dataset]["comparator_hash"],
                    f"{dataset}: original executable lock differs from the joint decision")
            run_bindings[dataset] = _run_bindings(root, entry, decisions[dataset], selections[dataset], link["original_pre_admission_lock"])
            continue
        # Repository-built: identities stay strict; the admission's relation to the joint lock is reported.
        require(link.get("dataset") == dataset and link.get("role_manifest_sha256") == entry["role_manifest"]["sha256"],
                f"{dataset}: admission link names another dataset or other reserved roles")
        started = _stamp(link.get("admission_started_at"), f"{dataset} admission time")
        original_lock = _json_reference(root, link["original_pre_admission_lock"], f"{dataset} original run lock")
        lock_findings = []
        run_bindings[dataset] = _run_bindings(root, entry, decisions[dataset], selections[dataset],
                                              link["original_pre_admission_lock"], findings=lock_findings)
        lifecycle = (original_lock.get("confirmed_hash") == decisions[dataset]["confirmed_hash"]
                     and original_lock.get("comparator_hash") == decisions[dataset]["comparator_hash"]
                     and all(row["lifecycle_lock_agrees_with_joint_decision"] for row in lock_findings))
        admission_findings[dataset] = {
            "admission_started_at": link["admission_started_at"], "joint_lock_locked_at": lock["locked_at"],
            "admission_governed_by_joint_lock": link.get("joint_lock_sha256") == manifest["joint_decision_lock"]["sha256"],
            "admission_after_joint_lock": started > locked_at,
            "admission_predictor_is_joint_confirmed_policy": link.get("confirmed_hash") == decisions[dataset]["confirmed_hash"],
            "lifecycle_lock_agrees_with_joint_decision": lifecycle,
            "lifecycle_confirmed_hash": original_lock.get("confirmed_hash"),
            "joint_confirmed_hash": decisions[dataset]["confirmed_hash"], "run_locks": lock_findings}
    post_hoc = any(not row["admission_governed_by_joint_lock"] or not row["admission_after_joint_lock"]
                   for row in admission_findings.values())
    reference = read_json(manuscript_root / "paper/reported_results.json")["supplementary_table_S14_mechanism_contrasts_submitted_protocol"]
    comparisons = []
    for row in reference:
        actual = tested[row["contrast"]]
        for name, source in (("effect_nats", "effect"), ("p_value_zero", "raw_P"), ("holm_p_value_zero", "holm_P")):
            comparisons.append({"contrast_id": row["contrast"], "quantity": name, "actual": actual[name],
                                "reported": row[source], "agrees_at_printed_precision": _reference_match(actual[name], row[source])})
        for index, name in enumerate(("interval_lower_nats", "interval_upper_nats")):
            observed, printed = actual["interval_nats"][index], row["interval"][index]
            comparisons.append({"contrast_id": row["contrast"], "quantity": name, "actual": observed,
                                "reported": printed, "agrees_at_printed_precision": _reference_match(observed, printed)})
    matches = all(item["agrees_at_printed_precision"] for item in comparisons)
    limits = ["Raw P values retain the heterogeneous original test methods; archived analysis code/configuration are hashed and preserved, not executed by this auditor.",
              "Timestamp ordering and author evidence classifications are checked declarations, not independent proof of historical data access.",
              "Selector eligibility is read from original candidate ledgers; scientific-audit separately binds their exact hashes and identities to the ingested official run manifests. Full nested-selection replay and checkpoint re-execution remain separate checks."]
    if repository_built:
        limits.append("Repository-built campaign: its tests were produced from the sealed lifecycle runs by the builder named in production "
                      "(rerunnable from its plan); the admission-link conditions are reported in admission_findings and summarized by "
                      "post_hoc, not required, because the joint lock follows the lifecycle's own admission stages.")
    return {"schema": "cm-s14-audit-1", "status": "PASS" if matches else "REFERENCE_MISMATCH",
            "campaign": str(root), "input_identity": input_identity(root), "joint_inputs_sha256": digest,
            "family_size": 9, "alpha": alpha, "data_role": "D_confirm", "rows": list(tested.values()),
            "policy_decisions": decisions, "manuscript_comparisons": comparisons, "official_run_bindings": run_bindings,
            "role_identities": {dataset: _role_identity(data["roles"], data["confirmation_units"]) for dataset, data in role_maps.items()},
            "production_kind": REPOSITORY_BUILT if repository_built else "imported_original",
            "post_hoc": bool(repository_built and post_hoc), "admission_findings": admission_findings,
            "original_test_p_values_recomputed": False,
            "scope": "Original-test file integrity, prespecified identities, paired effect reconstruction, joint Holm and strict point-margin/fallback replay",
            "limits": limits}


def template(manuscript_root: Path) -> dict:
    """An incomplete declaration only: never include invented observations/P values."""
    sources = read_json(Path(manuscript_root) / "paper/sources.json")
    reference = lambda path: {"path": path, "sha256": None}
    protocol = {"schema": PROTOCOL_SCHEMA, "family": "Supplementary Table S14", "registered_at": None, "alpha": .05, "contrasts": []}
    for key, spec in FAMILY.items():
        dataset, kind, unit, count, loss, direction, margin = spec
        protocol["contrasts"].append({"contrast_id": key, "dataset": dataset, "kind": kind, "statistical_unit": unit,
                                      "unit_count": count, "loss": loss, "beneficial_direction": direction,
                                      "practical_margin_nats": margin, "data_role": "D_confirm", "seed_ids": None, "weighting": {"scheme": "equal_unit"},
                                      "left_policy_sha256": None, "right_policy_sha256": None,
                                      "method": {"name": None, "family": _method_family(kind, unit), "tail": None, "null": "zero", "command": None,
                                                 **{k: reference(f"code/{key.replace('/', '_')}/{k}.json" if k != "analysis_code" else f"code/{key.replace('/', '_')}/analysis.py")
                                                    for k in ("analysis_code", "analysis_config", "environment")}}})
    return {"schema": SCHEMA, "status": "template", "synthetic": None, "evidence_kind": None,
            "manuscript_sha256": {key: value["sha256"] for key, value in sources["documents"].items()},
            "protocol": reference("protocol.json"), "protocol_template": protocol,
            "tests": [{"contrast_id": key, "result": reference(f"raw/{key.replace('/', '_')}/test.json"),
                       "paired_losses": reference(f"raw/{key.replace('/', '_')}/paired.csv")} for key in FAMILY],
            "datasets": [{"dataset": dataset, "role_manifest": reference(f"roles/{dataset}.json"),
                          "selection": reference(f"selection/{dataset}/decision.json"), "candidates": reference(f"selection/{dataset}/candidates.json"),
                          "selection_locked_at": None, "admission_link": reference(f"admission/{dataset}.json"),
                          "run_bindings": [{"split_index": split, "source_sha256": None, "config_sha256": None, "unit_sha256": None,
                                            "selection": reference(f"original/{dataset}/{split}/select/decision.json"),
                                            "candidates": reference(f"original/{dataset}/{split}/select/candidates.json"),
                                            "pre_admission_lock": reference(f"original/{dataset}/{split}/admit/pre_admission_lock.json")}
                                           for split in (range(10) if dataset == "Roman-empire" else [None])]}
                         for dataset in DATASETS],
            "joint_decision_lock": reference("joint_decision_lock.json"),
            "instructions": "Attach the original tests, original reserved roles, original pre-admission locks and their hashes. Do not infer raw P values from printed adjusted values, synthesize observations, or backdate registration/locks. See docs/S14_CONFIRMATION.md."}
