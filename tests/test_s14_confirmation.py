"""Ephemeral protocol fixtures for joint S14 replay, never benchmark evidence."""
from __future__ import annotations
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from cm_campaigns.common import canonical, read_json, sha256
from cm_campaigns.confirmation import (FAMILY, LOCK_SCHEMA, RESULT_SCHEMA, ROLE_SCHEMA,
                                       check_campaign, input_identity, template, composite_policy_hash, bind_official_runs)
from scripts.audit_s14_confirmation import main

ROOT = Path(__file__).resolve().parents[1]


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class S14CampaignTests(unittest.TestCase):
    # The fixture's optimization seed population; ("ensemble",) is the single level of a repository-built campaign.
    SEEDS = ("0", "1")

    def setUp(self):
        self.lifecycle = {}       # dataset -> lifecycle pre-admission lock policies differing from the joint decision
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "campaign"
        self.root.mkdir()
        self.manuscript = self.base / "manuscript"
        for name in ("sources.json", "registered_constants.json", "reported_results.json"):
            target = self.manuscript / "paper" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "paper" / name, target)
        self.manifest = template(self.manuscript)
        self.protocol = self.manifest.pop("protocol_template")
        self.manifest.pop("instructions")
        self.manifest.update(status="author_completed", synthetic=False, evidence_kind="original_confirmation_tests")
        self.protocol["registered_at"] = "2020-01-02T00:00:00+00:00"
        for spec in self.protocol["contrasts"]:
            spec["seed_ids"] = list(self.SEEDS)
            spec["method"].update(name="original fixture test method, not a scientific result",
                                  tail="greater" if spec["beneficial_direction"] > 0 else "less", command="original_analysis --original-config")
            for key in ("analysis_code", "analysis_config", "environment"):
                spec["method"][key] = self.write(spec["method"][key]["path"], {"fixture_only": True, "kind": key})
        self.manifest["protocol"] = self.write("protocol.json", self.protocol)
        self.policies = {}
        self.roles = {}
        self.previous = {}
        self.targets = {}
        self.native_configs = {}
        for entry in self.manifest["datasets"]:
            dataset = entry["dataset"]
            spec = next(value for value in FAMILY.values() if value[0] == dataset)
            n = spec[3]
            units = [str(i) for i in range(n)]
            role = {"schema": ROLE_SCHEMA, "dataset": dataset, "reserved_at_stage": "prepare",
                    "reserved_at": "2020-01-01T00:00:00+00:00",
                    "roles": {r: [str(n+i)] for i, r in enumerate(("fit", "explore", "select", "compare", "admit", "test"))},
                    "confirmation_units": {u: [u] for u in units}}
            role["roles"]["confirm"] = units
            if dataset == "Roman-empire":
                role["roles"] = {r: [f"split={split}:{i}" for split in range(10)] for i, r in enumerate(role["roles"])}
                role["confirmation_units"] = {f"split={split}": [role["roles"]["confirm"][split]] for split in range(10)}
            entry["role_manifest"] = self.write(entry["role_manifest"]["path"], role)
            self.roles[dataset] = role
            qc, chosen, previous = [digest(dataset + tag) for tag in (":qC", ":chosen", ":previous")]
            route = "radius" if dataset in ("ogbn-arxiv", "ogbn-proteins", "Peptides-func") else "mpc_rh"
            candidates = [{"hash": qc, "route": "fixed_sequence", "is_open_loop": True}, {"hash": chosen, "route": route, "is_open_loop": False},
                          {"hash": previous, "route": "one_step" if route == "radius" else "mpc_ol", "is_open_loop": route != "radius"}]
            if dataset == "Roman-empire":
                for candidate in candidates:
                    candidate["split_policy_hashes"] = [{"split_index": i, "hash": digest(candidate["hash"]+str(i))} for i in range(10)]
                    candidate["hash"] = composite_policy_hash(candidate["split_policy_hashes"])
                qc, chosen, previous = [row["hash"] for row in candidates]
            self.previous[dataset] = previous
            self.targets[dataset] = chosen
            returned = dataset == "ogbn-products"
            entry["candidates"] = self.write(entry["candidates"]["path"], candidates)
            entry["selection"] = self.write(entry["selection"]["path"], {"comparator_index": 0, "provisional_index": 0 if returned else 1,
                                                                          "eligible_indices": [] if returned else [1]})
            entry["selection_locked_at"] = "2020-01-03T00:00:00+00:00"
            self.policies[dataset] = {"comparator_hash": qc, "provisional_hash": qc if returned else chosen,
                                      "confirmed_hash": qc if returned or dataset == "Peptides-func" else chosen}
        for spec in self.protocol["contrasts"]:
            dataset, kind = FAMILY[spec["contrast_id"]][:2]
            spec["left_policy_sha256"] = digest(dataset + ":destroyed") if kind == "state" else self.targets[dataset]
            spec["right_policy_sha256"] = self.targets[dataset] if kind == "state" else self.previous[dataset]
        self.manifest["protocol"] = self.write("protocol.json", self.protocol)
        source_rows = {row["contrast"]: row for row in read_json(self.manuscript / "paper/reported_results.json")["supplementary_table_S14_mechanism_contrasts_submitted_protocol"]}
        self.results = {}
        for entry in self.manifest["tests"]:
            key = entry["contrast_id"]
            dataset, kind, unit, count, loss, direction, margin = FAMILY[key]
            source = source_rows[key]
            p = .00001 if key == "COCO/RH" else .0001625 if key == "Pascal/RH" else source["raw_P"]
            self.write_losses(entry, source["effect"])
            chosen = self.targets[dataset]
            result = {"schema": RESULT_SCHEMA, "contrast_id": key, "dataset": dataset,
                      "synthetic": False, "evidence_kind": "original_confirmation_test", "p_value_kind": "unadjusted_against_zero",
                      "numeric_precision": "unrounded", "data_role": "D_confirm", "statistical_unit": unit,
                      "unit_count": count, "loss": loss, "seed_ids": list(self.SEEDS),
                      "intervention_valid": True, "intervention_scope": "controller_observations_only", "method_name": "original fixture test method, not a scientific result",
                      "tail": "greater" if direction > 0 else "less", "protocol_sha256": self.manifest["protocol"]["sha256"],
                      "paired_losses_sha256": entry["paired_losses"]["sha256"],
                      "role_manifest_sha256": next(row["role_manifest"]["sha256"] for row in self.manifest["datasets"] if row["dataset"] == dataset),
                      "started_at": "2020-01-04T00:00:00+00:00", "completed_at": "2020-01-05T00:00:00+00:00",
                      "effect_nats": source["effect"], "p_value_zero": p, "valid_interval": True, "interval_confidence": .95,
                      "method_family": next(s["method"]["family"] for s in self.protocol["contrasts"] if s["contrast_id"] == key),
                      "interval_nats": source["interval"],
                      "left_policy_sha256": digest(dataset + ":destroyed") if kind == "state" else chosen,
                      "right_policy_sha256": chosen if kind == "state" else self.previous[dataset]}
            self.results[key] = result
            entry["result"] = self.write(entry["result"]["path"], result)
        self.retained = {key: key not in ("products/state", "Peptides/radius") for key in FAMILY}
        self.locked_at = "2020-01-06T00:00:00+00:00"
        self.seal()

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical(data))
        return {"path": name, "sha256": sha256(path)}

    def write_losses(self, entry, effect):
        dataset = FAMILY[entry["contrast_id"]][0]
        path = self.root / entry["paired_losses"]["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["seed_id", "unit_id", "left_loss_nats", "right_loss_nats", "unit_weight"])
            for index, seed in enumerate(self.SEEDS):
                offset = 0.0 if len(self.SEEDS) == 1 else (.0001 if index else -.0001)
                for unit in self.roles[dataset]["confirmation_units"]:
                    writer.writerow([seed, unit, 1 + effect + offset, 1, 1])
        entry["paired_losses"] = {"path": entry["paired_losses"]["path"], "sha256": sha256(path)}

    def seal(self):
        # Protocol fixture copies have native byte identities. These temporary
        # files exercise validators only and are never deposited as observations.
        self.original_locks = {}
        from cross_morse.review import canonical as config_hash
        for entry in self.manifest["datasets"]:
            dataset = entry["dataset"]
            candidates = read_json(self.root / entry["candidates"]["path"])
            selected = read_json(self.root / entry["selection"]["path"])
            for binding in entry["run_bindings"]:
                split = binding["split_index"]
                config = {"dataset": dataset, "profile": "full", "scientific_run": True,
                          "data": {"split_index": split}, "training": {"seeds": [0, 1]}}
                self.native_configs[(dataset, split)] = config
                binding.update(source_sha256=digest("fixture-source"), config_sha256=config_hash(config),
                               unit_sha256=digest(dataset + str(split)))
                native_candidates = copy.deepcopy(candidates)
                policies = dict(self.policies[dataset], **self.lifecycle.get(dataset, {}))
                if dataset == "Roman-empire":
                    components = {row["hash"]: row["split_policy_hashes"][split]["hash"] for row in candidates}
                    for row in native_candidates:
                        row["hash"] = components[row["hash"]]
                        row.pop("split_policy_hashes")
                    policies = {key: components[value] for key, value in policies.items()}
                    binding["selection"] = self.write(binding["selection"]["path"], selected)
                    binding["candidates"] = self.write(binding["candidates"]["path"], native_candidates)
                else:
                    binding["selection"] = entry["selection"]
                    binding["candidates"] = entry["candidates"]
                binding["pre_admission_lock"] = self.write(binding["pre_admission_lock"]["path"],
                    {**policies, "config_sha256": binding["config_sha256"],
                     "comparator_index": next(i for i, row in enumerate(native_candidates) if row["hash"] == policies["comparator_hash"]),
                     "confirmed_index": next(i for i, row in enumerate(native_candidates) if row["hash"] == policies["confirmed_hash"])})
            self.original_locks[dataset] = (self.write(f"original_locks/{dataset}.json", {**self.policies[dataset],
                "split_lock_sha256": {str(row["split_index"]): row["pre_admission_lock"]["sha256"] for row in entry["run_bindings"]}})
                if dataset == "Roman-empire" else entry["run_bindings"][0]["pre_admission_lock"])
        inputs = {"protocol": self.manifest["protocol"], "tests": self.manifest["tests"],
                  "datasets": [{k: v for k, v in row.items() if k != "admission_link"} for row in self.manifest["datasets"]]}
        lock = {"schema": LOCK_SCHEMA, "input_sha256": hashlib.sha256(canonical(inputs)).hexdigest(),
                "locked_at": self.locked_at, "mechanism_retained": self.retained, "policies": self.policies}
        self.manifest["joint_decision_lock"] = self.write("joint_decision_lock.json", lock)
        for entry in self.manifest["datasets"]:
            dataset = entry["dataset"]
            original = self.original_locks[dataset]
            link = {"dataset": dataset, "joint_lock_sha256": self.manifest["joint_decision_lock"]["sha256"],
                    "confirmed_hash": self.policies[dataset]["confirmed_hash"], "role_manifest_sha256": entry["role_manifest"]["sha256"],
                    "admission_started_at": "2020-01-07T00:00:00+00:00", "original_pre_admission_lock": original}
            entry["admission_link"] = self.write(entry["admission_link"]["path"], link)
        self.write("campaign.json", self.manifest)

    def native_runs(self):
        """Minimal sealed fixtures for the binding layer, not full scientific runs."""
        from cross_morse.review import STAGES
        output = {}
        for entry in self.manifest["datasets"]:
            dataset = entry["dataset"]
            output[dataset] = []
            for binding in entry["run_bindings"]:
                split = binding["split_index"]
                root = self.base / "native" / dataset / str(split)
                root.mkdir(parents=True, exist_ok=True)
                output[dataset].append(root)
                config = self.native_configs[(dataset, split)]
                (root / "config.json").write_bytes(canonical(config))
                (root / "prepared").mkdir(exist_ok=True)
                (root / "prepared/metadata.json").write_bytes(canonical({"is_benchmark_data": True}))
                role = self.roles[dataset]
                roles = role["roles"]
                if dataset == "Roman-empire":
                    roles = {r: [i.split(":")[1] for i in ids if i.startswith(f"split={split}:")] for r, ids in roles.items()}
                arrays = {r: np.array(list(map(int, ids)), dtype=np.int64) for r, ids in roles.items()}
                independent = dataset in ("PascalVOC-SP", "COCO-SP", "Peptides-func")
                np.savez(root / ("prepared/graph_splits.npz" if independent else "prepared/splits.npz"), **arrays)
                if not independent and dataset != "Roman-empire":
                    clusters = np.full(max(int(i) for ids in roles.values() for i in ids) + 1, -1, dtype=np.int64)
                    for unit, ids in role["confirmation_units"].items():
                        clusters[list(map(int, ids))] = int(unit)
                    np.savez(root / "prepared/clusters.npz", cluster_ids=clusters)
                for key, name in (("selection", "select/decision.json"), ("candidates", "select/candidates.json"),
                                  ("pre_admission_lock", "admit/pre_admission_lock.json")):
                    target = root / name
                    target.parent.mkdir(exist_ok=True)
                    shutil.copy2(self.root / binding[key]["path"], target)
                original_lock = read_json(root / "admit/pre_admission_lock.json")
                (root / "confirm").mkdir(exist_ok=True)
                (root / "confirm/decision.json").write_bytes(canonical(
                    {key: original_lock[key] for key in ("comparator_index", "confirmed_index")}))
                def descriptor(path):
                    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
                artifacts = [descriptor(path) for path in sorted(root.rglob("*")) if path.is_file() and "provenance" not in path.parts]
                (root / "provenance/stages").mkdir(parents=True, exist_ok=True)
                previous = None
                for stage in STAGES:
                    record = {"stage": stage, "dataset": dataset,
                              **{key: binding[key] for key in ("source_sha256", "config_sha256", "unit_sha256")},
                              "parent_manifest": previous, "inputs": [], "outputs": artifacts if stage == "prepare" else []}
                    path = root / f"provenance/stages/{stage}.json"
                    path.write_bytes(canonical(record))
                    previous = descriptor(path)
        return output

    def change_result(self, key, **changes):
        entry = next(row for row in self.manifest["tests"] if row["contrast_id"] == key)
        self.results[key].update(changes)
        entry["result"] = self.write(entry["result"]["path"], self.results[key])
        self.seal()

    def test_complete_family_reconstructs_effects_and_returns_failed_peptides_to_comparator(self):
        result = check_campaign(self.root, manuscript_root=self.manuscript)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["family_size"], 9)
        self.assertFalse(result["original_test_p_values_recomputed"])
        peptides = next(row for row in result["rows"] if row["contrast_id"] == "Peptides/radius")
        self.assertAlmostEqual(peptides["holm_p_value_zero"], .07)
        self.assertFalse(peptides["retained"])
        self.assertEqual((peptides["unit_count"], peptides["seed_count"]), (544, 2))
        outcome = result["policy_decisions"]["Peptides-func"]
        self.assertEqual(outcome["confirmed_hash"], outcome["comparator_hash"])

    def test_comparator_return_does_not_remove_products_from_the_joint_family(self):
        self.manifest["tests"] = [row for row in self.manifest["tests"] if row["contrast_id"] != "products/state"]
        self.seal()
        with self.assertRaisesRegex(ValueError, "exact registered family"):
            check_campaign(self.root, manuscript_root=self.manuscript)

    def test_duplicate_contrast_cannot_replace_the_ninth_original_test(self):
        self.manifest["tests"][-1] = copy.deepcopy(self.manifest["tests"][0])
        self.seal()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            check_campaign(self.root, manuscript_root=self.manuscript)

    def test_local_adjustment_and_printed_inequalities_are_not_raw_original_tests(self):
        self.change_result("COCO/RH", p_value_kind="locally_Holm_adjusted")
        with self.assertRaisesRegex(ValueError, "original unrounded"):
            check_campaign(self.root, manuscript_root=self.manuscript)
        self.change_result("COCO/RH", p_value_kind="unadjusted_against_zero", p_value_zero="< 1e-4")
        with self.assertRaisesRegex(ValueError, "finite JSON number"):
            check_campaign(self.root, manuscript_root=self.manuscript)

    def test_pairwise_observations_must_reproduce_the_reported_effect(self):
        self.change_result("arxiv/state", effect_nats=.006)
        with self.assertRaisesRegex(ValueError, "does not reconstruct"):
            check_campaign(self.root, manuscript_root=self.manuscript)

    def test_roles_cannot_overlap_even_when_all_hashes_are_updated(self):
        entry = self.manifest["datasets"][0]
        role = self.roles[entry["dataset"]]
        role["roles"]["admit"] = role["roles"]["confirm"][:1]
        entry["role_manifest"] = self.write(entry["role_manifest"]["path"], role)
        self.seal()
        with self.assertRaisesRegex(ValueError, "roles overlap"):
            check_campaign(self.root, manuscript_root=self.manuscript)

    def test_changed_original_result_is_rejected_before_using_its_p_value(self):
        entry = self.manifest["tests"][0]
        self.write(entry["result"]["path"], {**self.results[entry["contrast_id"]], "p_value_zero": 1e-9})
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            check_campaign(self.root, manuscript_root=self.manuscript)

    def test_relabelled_peptides_success_cannot_override_holm_failure(self):
        self.retained["Peptides/radius"] = True
        self.seal()
        with self.assertRaisesRegex(ValueError, "Locked mechanism decisions"):
            check_campaign(self.root, manuscript_root=self.manuscript)

    def test_ineligible_predecessor_cannot_be_locked_as_the_served_policy(self):
        self.policies["Peptides-func"]["confirmed_hash"] = digest("Peptides-func:previous")
        self.seal()
        with self.assertRaisesRegex(ValueError, "Locked policies"):
            check_campaign(self.root, manuscript_root=self.manuscript)

    def test_joint_holm_at_exact_alpha_does_not_retain(self):
        self.change_result("Peptides/radius", p_value_zero=.025)
        result = check_campaign(self.root, manuscript_root=self.manuscript)
        row = next(row for row in result["rows"] if row["contrast_id"] == "Peptides/radius")
        self.assertEqual(row["holm_p_value_zero"], .05)
        self.assertFalse(row["retained"])
        self.assertEqual(result["status"], "REFERENCE_MISMATCH")

    def test_exact_practical_margin_does_not_retain(self):
        key = "Peptides/radius"
        entry = next(row for row in self.manifest["tests"] if row["contrast_id"] == key)
        self.write_losses(entry, -.0015)
        self.change_result(key, p_value_zero=.001, effect_nats=-.0015,
                           paired_losses_sha256=entry["paired_losses"]["sha256"])
        result = check_campaign(self.root, manuscript_root=self.manuscript)
        row = next(row for row in result["rows"] if row["contrast_id"] == key)
        self.assertFalse(row["point_beyond_margin"])
        self.assertFalse(row["retained"])

    def test_late_joint_decision_cannot_validate_earlier_admission(self):
        self.locked_at = "2020-01-08T00:00:00+00:00"
        self.seal()
        with self.assertRaisesRegex(ValueError, "before joint confirmation"):
            check_campaign(self.root, manuscript_root=self.manuscript)

    def test_original_test_method_tail_cannot_be_replaced_by_the_opposite_test(self):
        self.protocol["contrasts"][1]["method"]["tail"] = "greater"
        self.manifest["protocol"] = self.write("protocol.json", self.protocol)
        # Keep all earlier result bindings current to reach the direction check.
        for entry in self.manifest["tests"]:
            key = entry["contrast_id"]
            self.results[key]["protocol_sha256"] = self.manifest["protocol"]["sha256"]
            entry["result"] = self.write(entry["result"]["path"], self.results[key])
        self.seal()
        with self.assertRaisesRegex(ValueError, "test tail opposes"):
            check_campaign(self.root, manuscript_root=self.manuscript)

    def test_template_is_incomplete_and_has_no_manufactured_observations(self):
        data = template(self.manuscript)
        self.assertEqual(data["status"], "template")
        self.assertIsNone(data["tests"][0]["result"]["sha256"])
        self.write("campaign.json", data)
        with self.assertRaisesRegex(ValueError, "template is not evidence"):
            check_campaign(self.root, manuscript_root=self.manuscript)

    def test_nested_raw_file_change_changes_saved_input_identity(self):
        before = input_identity(self.root)
        p = self.root / self.manifest["tests"][0]["paired_losses"]["path"]
        p.write_text(p.read_text() + "\n")
        self.assertNotEqual(input_identity(self.root), before)

    def test_native_linkage_binds_all_roman_splits_and_source_roles(self):
        report = check_campaign(self.root, manuscript_root=self.manuscript)
        paths = self.native_runs()
        linked = bind_official_runs(report, paths)
        self.assertEqual(linked["status"], "PASS")
        self.assertEqual(linked["datasets"]["Roman-empire"]["runs"], 10)
        self.assertEqual(bind_official_runs(report, {})["status"], "PENDING")
        report["role_identities"]["ogbn-arxiv"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "data roles or confirmation-unit memberships"):
            bind_official_runs(report, paths)

    def test_native_linkage_rejects_different_selection_bytes(self):
        report = check_campaign(self.root, manuscript_root=self.manuscript)
        paths = self.native_runs()
        report["official_run_bindings"]["ogbn-arxiv"][0]["artifacts"]["select/decision.json"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "original sealed run artifact"):
            bind_official_runs(report, paths)

    def test_native_linkage_rejects_omitted_original_seed(self):
        report = check_campaign(self.root, manuscript_root=self.manuscript)
        paths = self.native_runs()
        report["rows"][0]["seed_ids"] = ["0"]
        with self.assertRaisesRegex(ValueError, "seed population differs"):
            bind_official_runs(report, paths)

    def test_native_confirmed_decision_cannot_disagree_with_a_joint_consistent_prelock(self):
        from cross_morse.review import STAGES
        report = check_campaign(self.root, manuscript_root=self.manuscript)
        paths = self.native_runs()
        root = paths["Peptides-func"][0]
        (root / "confirm/decision.json").write_bytes(canonical({"comparator_index": 0, "confirmed_index": 1}))
        previous = None
        for stage in STAGES:
            path = root / f"provenance/stages/{stage}.json"
            record = read_json(path)
            record["parent_manifest"] = previous
            for descriptor in record["outputs"]:
                file = root / descriptor["path"]
                descriptor.update(bytes=file.stat().st_size, sha256=sha256(file))
            path.write_bytes(canonical(record))
            previous = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
        with self.assertRaisesRegex(ValueError, "actual native confirmation decision"):
            bind_official_runs(report, paths)

    def test_prespecified_seeds_cannot_be_dropped_from_paired_rows(self):
        entry = self.manifest["tests"][0]
        path = self.root / entry["paired_losses"]["path"]
        lines = path.read_text().splitlines()
        path.write_text("\n".join([lines[0], *[line for line in lines[1:] if line.startswith("0,")]]) + "\n")
        entry["paired_losses"]["sha256"] = sha256(path)
        self.change_result(entry["contrast_id"], paired_losses_sha256=sha256(path))
        with self.assertRaisesRegex(ValueError, "prespecified optimization seeds"):
            check_campaign(self.root, manuscript_root=self.manuscript)

    def test_unequal_graph_weights_cannot_change_the_registered_estimand(self):
        entry = next(row for row in self.manifest["tests"] if row["contrast_id"] == "Peptides/radius")
        path = self.root / entry["paired_losses"]["path"]
        lines = [line.rsplit(",", 1)[0] + ",2" if line.split(",")[1] == "0" else line for line in path.read_text().splitlines()]
        path.write_text("\n".join(lines) + "\n")
        entry["paired_losses"]["sha256"] = sha256(path)
        self.change_result(entry["contrast_id"], paired_losses_sha256=sha256(path))
        with self.assertRaisesRegex(ValueError, "equal statistical-unit weights"):
            check_campaign(self.root, manuscript_root=self.manuscript)

    def test_selected_radius_cannot_be_relabelled_as_a_different_route(self):
        entry = next(row for row in self.manifest["datasets"] if row["dataset"] == "ogbn-arxiv")
        candidates = read_json(self.root / entry["candidates"]["path"])
        candidates[1]["route"] = "one_step"
        entry["candidates"] = self.write(entry["candidates"]["path"], candidates)
        self.seal()
        with self.assertRaisesRegex(ValueError, "wrong route family"):
            check_campaign(self.root, manuscript_root=self.manuscript)

    def test_products_state_target_is_prespecified_feedback_and_does_not_change_comparator(self):
        report = check_campaign(self.root, manuscript_root=self.manuscript)
        row = next(row for row in report["rows"] if row["contrast_id"] == "products/state")
        self.assertFalse(row["tested_policy_selected"])
        decision = report["policy_decisions"]["ogbn-products"]
        self.assertEqual(decision["confirmed_hash"], decision["comparator_hash"])
        entry = next(row for row in self.manifest["datasets"] if row["dataset"] == "ogbn-products")
        candidates = read_json(self.root / entry["candidates"]["path"])
        candidates[1]["is_open_loop"] = True
        entry["candidates"] = self.write(entry["candidates"]["path"], candidates)
        self.seal()
        with self.assertRaisesRegex(ValueError, "registered feedback candidate"):
            check_campaign(self.root, manuscript_root=self.manuscript)

    def test_roman_aggregate_cannot_relabel_native_routes(self):
        entry = next(row for row in self.manifest["datasets"] if row["dataset"] == "Roman-empire")
        binding = entry["run_bindings"][0]
        candidates = read_json(self.root / binding["candidates"]["path"])
        candidates[1]["route"] = "one_step"
        binding["candidates"] = self.write(binding["candidates"]["path"], candidates)
        inputs = {"protocol": self.manifest["protocol"], "tests": self.manifest["tests"],
                  "datasets": [{k: v for k, v in row.items() if k != "admission_link"} for row in self.manifest["datasets"]]}
        lock = read_json(self.root / "joint_decision_lock.json")
        lock["input_sha256"] = hashlib.sha256(canonical(inputs)).hexdigest()
        self.manifest["joint_decision_lock"] = self.write("joint_decision_lock.json", lock)
        for row in self.manifest["datasets"]:
            link = read_json(self.root / row["admission_link"]["path"])
            link["joint_lock_sha256"] = self.manifest["joint_decision_lock"]["sha256"]
            row["admission_link"] = self.write(row["admission_link"]["path"], link)
        self.write("campaign.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "Roman aggregate route/open-loop identity"):
            check_campaign(self.root, manuscript_root=self.manuscript)

    def test_declared_synthetic_campaign_is_refused(self):
        self.manifest["synthetic"] = True
        self.write("campaign.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            check_campaign(self.root, manuscript_root=self.manuscript)

    # -- repository-built campaigns (cm_campaigns/s14_builder.py) --------------------------------
    def relink(self, **changes):
        """Rewrite every admission link (outside the joint-lock digest) and the manifest."""
        for entry in self.manifest["datasets"]:
            link = read_json(self.root / entry["admission_link"]["path"])
            link.update(changes)
            entry["admission_link"] = self.write(entry["admission_link"]["path"], link)
        self.write("campaign.json", self.manifest)

    def declare_repository_built(self):
        self.manifest["production"] = {"kind": "repository_built", "builder": "cm_campaigns/s14_builder.py",
                                       "provenance": self.write("provenance.json", {"fixture_only": True})}
        self.write("campaign.json", self.manifest)

    def test_repository_built_campaign_reports_post_hoc_admission_instead_of_failing(self):
        # The lifecycle admitted before the joint lock and no joint lock governed it.
        self.relink(joint_lock_sha256=None, admission_started_at="2020-01-05T12:00:00+00:00")
        with self.assertRaisesRegex(ValueError, "admission did not use the joint decision lock"):
            check_campaign(self.root, manuscript_root=self.manuscript)          # imported: strict, unchanged
        self.declare_repository_built()
        result = check_campaign(self.root, manuscript_root=self.manuscript)
        self.assertEqual(result["status"], "PASS")                               # values agree with the printed ones
        self.assertEqual((result["production_kind"], result["post_hoc"]), ("repository_built", True))
        self.assertEqual(set(result["admission_findings"]), {row["dataset"] for row in self.manifest["datasets"]})
        for finding in result["admission_findings"].values():
            self.assertFalse(finding["admission_governed_by_joint_lock"])
            self.assertFalse(finding["admission_after_joint_lock"])
            self.assertTrue(finding["lifecycle_lock_agrees_with_joint_decision"])
        self.assertEqual(len(result["admission_findings"]["Roman-empire"]["run_locks"]), 10)
        self.assertTrue(any(limit.startswith("Repository-built campaign") for limit in result["limits"]))
        # A changed printed-value comparison is the only status distinction.
        self.change_result("Peptides/radius", p_value_zero=.025)
        self.relink(joint_lock_sha256=None, admission_started_at="2020-01-05T12:00:00+00:00")
        self.declare_repository_built()
        self.assertEqual(check_campaign(self.root, manuscript_root=self.manuscript)["status"], "REFERENCE_MISMATCH")

    def test_repository_built_campaign_reports_a_lifecycle_lock_that_differs_from_the_joint_decision(self):
        # Peptides' own confirm stage kept the radius; the joint family returns q_C.
        self.lifecycle["Peptides-func"] = {"confirmed_hash": self.targets["Peptides-func"]}
        self.seal()
        with self.assertRaisesRegex(ValueError, "original executable lock differs from the joint decision"):
            check_campaign(self.root, manuscript_root=self.manuscript)
        self.declare_repository_built()
        result = check_campaign(self.root, manuscript_root=self.manuscript)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["post_hoc"])
        peptides = result["admission_findings"]["Peptides-func"]
        self.assertFalse(peptides["lifecycle_lock_agrees_with_joint_decision"])
        self.assertEqual(peptides["lifecycle_confirmed_hash"], self.targets["Peptides-func"])
        self.assertEqual(peptides["joint_confirmed_hash"], self.policies["Peptides-func"]["comparator_hash"])
        self.assertFalse(peptides["run_locks"][0]["lifecycle_lock_agrees_with_joint_decision"])
        self.assertTrue(all(f["lifecycle_lock_agrees_with_joint_decision"] for d, f in result["admission_findings"].items()
                            if d != "Peptides-func"))

    def test_repository_built_identities_stay_strict(self):
        self.manifest["production"] = {"kind": "repository_built", "builder": "cm_campaigns/s14_builder.py"}
        self.write("campaign.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "production provenance"):
            check_campaign(self.root, manuscript_root=self.manuscript)
        self.declare_repository_built()
        entry = self.manifest["datasets"][0]
        link = read_json(self.root / entry["admission_link"]["path"])
        link["role_manifest_sha256"] = "0" * 64
        entry["admission_link"] = self.write(entry["admission_link"]["path"], link)
        self.write("campaign.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "another dataset or other reserved roles"):
            check_campaign(self.root, manuscript_root=self.manuscript)

    def test_ensemble_seed_level_links_official_runs_only_for_a_repository_built_campaign(self):
        fixture = S14CampaignTests("test_complete_family_reconstructs_effects_and_returns_failed_peptides_to_comparator")
        fixture.SEEDS = ("ensemble",)
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        imported = check_campaign(fixture.root, manuscript_root=fixture.manuscript)
        self.assertEqual(imported["rows"][0]["seed_count"], 1)
        paths = fixture.native_runs()
        with self.assertRaisesRegex(ValueError, "seed population differs"):
            bind_official_runs(imported, paths)
        fixture.declare_repository_built()
        report = check_campaign(fixture.root, manuscript_root=fixture.manuscript)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(bind_official_runs(report, paths)["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
