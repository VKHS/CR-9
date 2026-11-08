"""Published evidence must survive independent identity and completeness checks.

All files here are deliberately small software fixtures. They are not benchmark
observations or substitutes for the author-supplied artifacts.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

from scripts import statement_evidence as se
from scripts import paper_artifacts as pa
from scripts.reconstruct_paper import summarize


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def baseline_view(folder):
    from cm_campaigns.contract import EXTERNAL
    from cm_campaigns.baselines import STUDY_ARTIFACT_SCHEMA, study_paths
    from cm_experiments.artifacts import canonical
    studies = []
    for dataset, methods in EXTERNAL.items():
        for method in methods:
            stem = f"{dataset}/{method.replace(' ', '_')}"
            target = folder / stem
            plan = {"schema": "cm-upstream-study-1", "dataset": dataset, "method": method,
                    "source_commit": "a" * 40, "adapter_commit": "b" * 40,
                    "trials": [{"fixture_trial": i} for i in range(48)], "seed_ids": list(range(12)),
                    "selection_role": "internal_training_validation", "direction": "maximize",
                    "metric_unit": "native_0_to_1", "adapter_argv": ["python", "adapter.py"],
                    "official_splits": list(range(10)) if dataset == "Roman-empire" else [0]}
            plan_path = write(target / "plan.json", plan)
            adapter = target / "adapter.tar"
            with tarfile.open(adapter, "w", format=tarfile.PAX_FORMAT,
                              pax_headers={"comment": plan["adapter_commit"]}) as handle:
                source = b"# software fixture; no benchmark execution\n"
                info = tarfile.TarInfo("adapter.py")
                info.size = len(source)
                handle.addfile(info, io.BytesIO(source))
            files, fits = {}, []
            def add(name):
                files[name] = {"path": name, "bytes": 1, "sha256": "f" * 64, "mode": 0o644}
                return {key: files[name][key] for key in ("path", "bytes", "sha256")}
            prefixes = [f"hpo/trial-{i:03d}" for i in range(48)]
            for seed in plan["seed_ids"]:
                for split in plan["official_splits"]:
                    prefix = f"final/seed-{seed}/split-{split}"
                    prefixes.append(prefix)
                    add(f"{prefix}/development_selection.json")
                    add(f"{prefix}/adapter_output/receipt.json")
                    artifacts = {kind: add(f"{prefix}/adapter_output/{name}") for kind, name in
                                 (("predictions", "predictions.npz"), ("checkpoint", "model.pt"),
                                  ("split_manifest", "splits.json"), ("timing_trace", "timing.json"))}
                    fits.append({"seed": seed, "split": split, "artifacts": artifacts})
            for prefix in prefixes:
                for name in ("attempt.json", "plan.json", "configuration.json", "invocation.json",
                             "ARTIFACT_MANIFEST.json", "upstream_source.tar", "adapter_source.tar"):
                    add(f"{prefix}/{name}")
            manifest = write(target / "study_files.json", {
                "schema": STUDY_ARTIFACT_SCHEMA, "dataset": dataset, "method": method,
                "plan_sha256": canonical(plan), "source_commit": plan["source_commit"],
                "adapter_commit": plan["adapter_commit"], "hpo_attempts": 48, "selected_trial": 0,
                "final_fits": fits, "files": list(files.values())})
            locations = study_paths(dataset, method, plan["adapter_commit"])
            archive = {"file": locations["study_archive"], "sha256": "e" * 64, "bytes": 10240,
                       "manifest_file": locations["study_manifest"], "manifest_sha256": se.sha256(manifest)}
            upstream = write(target / "upstream.json", {
                "dataset": dataset, "method": method, "source_commit": plan["source_commit"],
                "adapter_commit": plan["adapter_commit"],
                "study_archive": archive,
                "adapter_archive": {"commit": plan["adapter_commit"], "sha256": se.sha256(adapter)}})
            studies.append({"dataset": dataset, "method": method, "source_commit": plan["source_commit"],
                            "adapter_commit": plan["adapter_commit"],
                            "study_archive": {**archive, "artifact": "records:baselines"},
                            **{key: {"file": path.relative_to(folder).as_posix(), "sha256": se.sha256(path)}
                               for key, path in (("plan", plan_path), ("upstream", upstream),
                                                 ("adapter_archive", adapter), ("study_manifest", manifest))}})
    path = folder / "INDEX.json"
    write(path, {"schema": pa.BASELINE_SCHEMA, "source": "records:baselines", "studies": studies})
    return path


class PublishedStudyChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = baseline_view(self.root)

    def change(self, operation):
        index = se.read_json(self.path)
        operation(index)
        write(self.path, index)

    def test_complete_study_contract_is_accepted(self):
        self.assertEqual(len(se.check_baseline_studies(self.path)["studies"]), 14)

    def test_fourteen_empty_study_objects_cannot_fulfil_availability(self):
        self.change(lambda index: index.update(studies=[{} for _ in range(14)]))
        with self.assertRaisesRegex(ValueError, "unexpected or duplicate"):
            se.check_baseline_studies(self.path)

    def test_repeated_dataset_method_does_not_replace_a_missing_study(self):
        self.change(lambda index: index["studies"].__setitem__(1, index["studies"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            se.check_baseline_studies(self.path)

    def test_even_rehashed_plan_must_have_all_distinct_hpo_configurations(self):
        index = se.read_json(self.path)
        entry = index["studies"][0]["plan"]
        path = self.root / entry["file"]
        plan = se.read_json(path)
        plan["trials"][-1] = plan["trials"][0]
        write(path, plan)
        entry["sha256"] = se.sha256(path)
        write(self.path, index)
        with self.assertRaisesRegex(ValueError, "must not be duplicated"):
            se.check_baseline_studies(self.path)

    def test_empty_adapter_archive_is_not_code(self):
        index = se.read_json(self.path)
        entry = index["studies"][0]
        archive = self.root / entry["adapter_archive"]["file"]
        with tarfile.open(archive, "w"):
            pass
        entry["adapter_archive"]["sha256"] = se.sha256(archive)
        upstream_path = self.root / entry["upstream"]["file"]
        upstream = se.read_json(upstream_path)
        upstream["adapter_archive"]["sha256"] = se.sha256(archive)
        write(upstream_path, upstream)
        entry["upstream"]["sha256"] = se.sha256(upstream_path)
        write(self.path, index)
        with self.assertRaisesRegex(ValueError, "no source files"):
            se.check_baseline_studies(self.path)

    def test_index_cannot_redirect_to_a_file_outside_its_view(self):
        self.change(lambda index: index["studies"][0]["plan"].update(file="../private.json"))
        with self.assertRaisesRegex(ValueError, "outside"):
            se.check_baseline_studies(self.path)

    def test_adapter_commit_must_match_plan_and_upstream(self):
        self.change(lambda index: index["studies"][0].update(adapter_commit="c" * 40))
        with self.assertRaisesRegex(ValueError, "adapter_commit"):
            se.check_baseline_studies(self.path)

    def test_study_archive_must_point_to_the_published_record_set(self):
        self.change(lambda index: index["studies"][0]["study_archive"].update(artifact="unregistered"))
        with self.assertRaisesRegex(ValueError, "released record set"):
            se.check_baseline_studies(self.path)

    def change_study_inventory(self, operation, dataset=None):
        index = se.read_json(self.path)
        entry = next(e for e in index["studies"] if dataset is None or e["dataset"] == dataset)
        path = self.root / entry["study_manifest"]["file"]
        inventory = se.read_json(path)
        operation(inventory)
        write(path, inventory)
        entry["study_manifest"]["sha256"] = se.sha256(path)
        entry["study_archive"]["manifest_sha256"] = se.sha256(path)
        upstream_path = self.root / entry["upstream"]["file"]
        upstream = se.read_json(upstream_path)
        upstream["study_archive"]["manifest_sha256"] = se.sha256(path)
        write(upstream_path, upstream)
        entry["upstream"]["sha256"] = se.sha256(upstream_path)
        write(self.path, index)

    def test_resealed_inventory_cannot_drop_checkpoint_bytes(self):
        self.change_study_inventory(lambda value: value.update(files=[item for item in value["files"]
                                      if item["path"] != value["final_fits"][0]["artifacts"]["checkpoint"]["path"]]))
        with self.assertRaisesRegex(ValueError, "final checkpoint"):
            se.check_baseline_studies(self.path)

    def test_roman_requires_every_final_seed_and_split(self):
        self.change_study_inventory(lambda value: value["final_fits"].pop(), dataset="Roman-empire")
        with self.assertRaisesRegex(ValueError, "omit registered seeds or official splits"):
            se.check_baseline_studies(self.path)

    def test_all_failed_and_pruned_hpo_attempts_must_still_be_retained(self):
        self.change_study_inventory(lambda value: value.update(files=[item for item in value["files"]
                                      if not item["path"].startswith("hpo/trial-047/")]))
        with self.assertRaisesRegex(ValueError, "original hpo attempts"):
            se.check_baseline_studies(self.path)

    def test_final_selection_and_receipt_are_required_in_the_original_inventory(self):
        for filename in ("development_selection.json", "adapter_output/receipt.json"):
            with self.subTest(file=filename):
                self.path = baseline_view(self.root)
                self.change_study_inventory(lambda value: value.update(files=[item for item in value["files"]
                                          if not item["path"].endswith("/" + filename)]))
                with self.assertRaisesRegex(ValueError, "original final fit is missing"):
                    se.check_baseline_studies(self.path)


class DerivedSourceChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.folder = self.root / "paper/source_data"
        self.manifest = {"artifacts": []}
        write(self.root / "release/paper_artifacts.json", self.manifest)
        (self.root / "scripts").mkdir()
        (self.root / "scripts/fixture.py").write_text("# software fixture\n", encoding="utf-8")
        from scripts.reconstruct import source_inventory
        self.builders = source_inventory(self.root)
        counts = {"exact": 1, "differs": 0, "not_comparable": 0, "no_reference": 0, "missing": 0}
        coverage = {"expected_values": 1, "covered_values": 1, "missing_values": 0, "complete": True}
        output = {"id": "display:fixture:Table 1", "display_ids": ["fixture:Table 1"],
                  "scope": "protocol", "coverage": "full_numerical_display", "content_complete": True,
                  "comparison_complete": True, "comparison_coverage": coverage, "comparison": counts,
                  "input_kind": "repository_sources", "files": ["displays/tables/T1.csv",
                      "displays/data/values.json", "displays/compare/T1.csv"]}
        self.result = {"schema": "cross-morse-reconstruction-1", "synthetic_examples": False,
                       "inputs": [],
                       "software_sources": dict(self.builders),
                       "display_generators": {"comparison_with_printed_values": counts,
                                              "comparison_coverage_complete": True},
                       "displays": [{"id": "fixture:Table 1", "full_numerical_display": True,
                                     "generated_outputs": [output["id"]], "comparison": counts,
                                     "content_complete": True, "comparison_complete": True,
                                     "comparison_coverage": coverage}], "outputs": [output]}
        self.patch = mock.patch.object(pa, "builder_source_inventory", return_value=self.builders, create=True)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        write(self.folder / "data/values.json", {"fixture_value": 1})
        (self.folder / "tables").mkdir()
        (self.folder / "tables/T1.csv").write_text("name,value\nfixture,1\n", encoding="utf-8")
        (self.folder / "compare").mkdir()
        (self.folder / "compare/T1.csv").write_text("group,path,computed,reported,status,note\nfixture,x,1,1,exact,\n",
                                                    encoding="utf-8")
        self.extra_workspace = {}
        self.make_snapshot()

    def make_snapshot(self, *, rehash_outputs=True):
        sources = {"displays/" + p.relative_to(self.folder).as_posix(): {
                      "sha256": se.sha256(p), "bytes": p.stat().st_size}
                   for p in self.folder.rglob("*") if p.is_file()
                   and p.relative_to(self.folder).parts[0] in ("tables", "data", "compare")}
        sources.update(self.extra_workspace)
        if rehash_outputs:
            for output in self.result["outputs"]:
                output["file_sha256"] = {name: sources[name]["sha256"] for name in output["files"] if name in sources}
        report_path = write(self.folder / "reports/reconstruction.json", self.result)
        inventory = {"files": {"reconstruction.json": {"sha256": se.sha256(report_path),
                                                        "bytes": report_path.stat().st_size}, **sources}}
        write(self.folder / "reports/FILES.json", inventory)
        summary = summarize(self.result, self.manifest)
        write(self.folder / "reports/status.json", summary)
        files = {p.relative_to(self.folder).as_posix(): se.sha256(p) for p in self.folder.rglob("*")
                 if p.is_file() and p.name != "INDEX.json"}
        index = {"schema": "cross-morse-source-data-1",
                 "artifact_manifest_sha256": se.sha256(self.root / "release/paper_artifacts.json"),
                 "artifacts": [], "files": files, "builder_sources": self.builders,
                 "software_sources": self.result["software_sources"], "reconstruction": summary,
                 "inputs_sha256": hashlib.sha256(json.dumps(self.result["inputs"], sort_keys=True).encode()).hexdigest(),
                 "reports": {key: {"path": name, "sha256": files[name]} for key, name in
                             (("reconstruction", "reports/reconstruction.json"),
                              ("workspace_files", "reports/FILES.json"), ("status", "reports/status.json"))}}
        write(self.folder / "INDEX.json", index)

    def check(self):
        return se.check_source_data(self.root, self.manifest, [self.result["displays"][0]["id"]])

    def test_source_data_is_bound_to_its_report_and_file_inventory(self):
        index, summary = self.check()
        self.assertEqual(summary, index["reconstruction"])

    def test_empty_inventory_with_complete_count_is_not_evidence(self):
        write(self.folder / "INDEX.json", {"files": {}, "reconstruction": {"complete": 90, "values_differ": 0}})
        with self.assertRaises(ValueError):
            self.check()

    def test_summary_cannot_claim_more_complete_displays_than_report(self):
        index = se.read_json(self.folder / "INDEX.json")
        index["reconstruction"]["complete"] = 90
        write(self.folder / "INDEX.json", index)
        with self.assertRaisesRegex(ValueError, "completion summary"):
            self.check()

    def test_source_data_changed_and_rehashed_in_index_still_fails_workspace_binding(self):
        path = write(self.folder / "data/values.json", {"fixture_value": 999})
        index = se.read_json(self.folder / "INDEX.json")
        index["files"]["data/values.json"] = se.sha256(path)
        write(self.folder / "INDEX.json", index)
        with self.assertRaisesRegex(ValueError, "reconstruction inventory"):
            self.check()

    def test_current_manifest_change_requires_regeneration(self):
        write(self.root / "release/paper_artifacts.json", {"artifacts": [], "changed": True})
        with self.assertRaisesRegex(ValueError, "published identity"):
            self.check()

    def test_changed_generator_requires_regeneration(self):
        pa.builder_source_inventory.return_value = {"fixture.py": "b" * 64}
        with self.assertRaisesRegex(ValueError, "tools or their registered inputs changed"):
            self.check()

    def test_unindexed_output_cannot_be_hidden(self):
        write(self.folder / "data/unindexed.json", {"value": 3})
        with self.assertRaisesRegex(ValueError, "inventory omits"):
            self.check()

    def test_duplicate_display_cannot_replace_another(self):
        self.result["displays"].append(copy.deepcopy(self.result["displays"][0]))
        self.make_snapshot()
        with self.assertRaisesRegex(ValueError, "exactly once"):
            self.check()

    def test_synthetic_examples_cannot_be_published_measurements(self):
        self.result["synthetic_examples"] = True
        self.make_snapshot()
        with self.assertRaisesRegex(ValueError, "synthetic example"):
            self.check()

    def test_completion_flags_without_generated_outputs_are_rejected(self):
        self.result["outputs"] = []
        self.make_snapshot()
        with self.assertRaisesRegex(ValueError, "completion flags have no generated display output"):
            self.check()

    def test_source_identity_must_match_current_software_even_when_rehashed_consistently(self):
        self.result["software_sources"] = {}
        self.make_snapshot()
        with self.assertRaisesRegex(ValueError, "current source inventory"):
            self.check()

    def test_full_table_requires_its_actual_table_csv(self):
        self.result["outputs"][0]["files"].remove("displays/tables/T1.csv")
        self.make_snapshot()
        with self.assertRaisesRegex(ValueError, "full table has no generated table CSV"):
            self.check()

    def test_a_header_without_source_rows_does_not_support_a_full_display(self):
        (self.folder / "tables/T1.csv").write_text("name,value\n", encoding="utf-8")
        self.make_snapshot()
        with self.assertRaisesRegex(ValueError, "empty or malformed source CSV"):
            self.check()

    def test_comparison_rows_cannot_disagree_with_a_rehashed_completion_report(self):
        path = self.folder / "compare/T1.csv"
        path.write_text(path.read_text().replace(",exact,", ",differs,"))
        self.make_snapshot()
        with self.assertRaisesRegex(ValueError, "comparison counts differ"):
            self.check()

    def test_global_comparison_totals_are_recomputed_from_csv_rows(self):
        self.result["display_generators"]["comparison_with_printed_values"] = {
            "exact": 99, "differs": 0, "not_comparable": 0, "no_reference": 0, "missing": 0}
        self.make_snapshot()
        with self.assertRaisesRegex(ValueError, "global comparison counts differ"):
            self.check()

    def test_output_hash_must_match_the_original_workspace_inventory(self):
        self.result["outputs"][0]["file_sha256"]["displays/tables/T1.csv"] = "0" * 64
        self.make_snapshot(rehash_outputs=False)
        with self.assertRaisesRegex(ValueError, "identity differs from workspace FILES"):
            self.check()

    def test_a_full_protocol_display_needs_nonempty_comparisons(self):
        (self.folder / "compare/T1.csv").unlink()
        self.result["outputs"][0]["files"].remove("displays/compare/T1.csv")
        for counts in (self.result["displays"][0]["comparison"], self.result["outputs"][0]["comparison"],
                       self.result["display_generators"]["comparison_with_printed_values"]):
            counts.update(exact=0)
        self.make_snapshot()
        with self.assertRaisesRegex(ValueError, "full numerical/protocol display has no actual comparisons"):
            self.check()

    def test_honest_partial_source_data_remains_valid_but_unfulfilled(self):
        display, output = self.result["displays"][0], self.result["outputs"][0]
        display["full_numerical_display"] = False
        output["coverage"] = "columns"
        for row in (display, output):
            row.update(content_complete=False, comparison_complete=False)
        self.result["display_generators"]["comparison_coverage_complete"] = False
        self.make_snapshot()
        _, summary = self.check()
        self.assertTrue(se.reconstruction_problems(summary, 1))

    def test_main_figure_one_illustration_can_have_no_comparison_csv(self):
        (self.folder / "compare/T1.csv").unlink()
        (self.folder / "tables/T1.csv").unlink()
        display, output = self.result["displays"][0], self.result["outputs"][0]
        display.update(id="main:Figure 1", generated_outputs=["display:main:Figure 1"])
        output.update(id="display:main:Figure 1", display_ids=["main:Figure 1"], scope="illustration",
                      files=["displays/data/values.json", "displays/figures/Fig1.svg"])
        for counts in (display["comparison"], output["comparison"],
                       self.result["display_generators"]["comparison_with_printed_values"]):
            counts.update(exact=0)
        self.extra_workspace = {"displays/figures/Fig1.svg": {"sha256": "f" * 64, "bytes": 10}}
        self.make_snapshot()
        _, summary = self.check()
        self.assertEqual(summary["complete"], 1)


class RunAndCompletionChecks(unittest.TestCase):
    def test_unknown_or_absent_comparison_fields_fail_completion(self):
        summary = {"displays": 90, "complete": 90, "incomplete": [], "values_exact": 800,
                   "values_differ": 0, "values_not_comparable": 0, "values_no_reference": 0,
                   "values_missing": 0, "comparison_coverage_complete": True, "display_content_incomplete": []}
        self.assertEqual(se.reconstruction_problems(summary, 90), [])
        for key in ("values_differ", "values_not_comparable", "values_no_reference", "values_missing"):
            with self.subTest(key=key):
                incomplete = dict(summary, **{key: 1})
                self.assertTrue(se.reconstruction_problems(incomplete, 90))
                incomplete.pop(key)
                self.assertTrue(se.reconstruction_problems(incomplete, 90))
        self.assertTrue(se.reconstruction_problems(dict(summary, values_exact=0), 90))
        self.assertTrue(se.reconstruction_problems(dict(summary, comparison_coverage_complete=False), 90))
        self.assertTrue(se.reconstruction_problems(dict(summary, display_content_incomplete=["main:Figure 2"]), 90))

    def test_environment_file_without_observed_values_is_not_enough(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / "arxiv.json", {"schema": "cross-morse-observed-environment-1", "dataset_run": "arxiv",
                                       "members": [{"run": "arxiv", "environment": None}]})
            with self.assertRaisesRegex(ValueError, "observed Python"):
                se.check_observed_environment(root, "arxiv", "ogbn-arxiv", {"identity": {}})

    def test_copied_membership_bytes_and_all_identities_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            array = root / "arxiv_splits.npz"
            array.write_bytes(b"software-fixture-memberships")
            identities = {"source_sha256": "a" * 64, "config_sha256": "b" * 64, "unit_sha256": se.sha256(array)}
            value = {"schema": "cross-morse-run-manifest-1", "dataset": "ogbn-arxiv", "profile": "full",
                     "scientific_run": True, "is_benchmark_data": True, "identities": identities,
                     "seeds": {"training_seeds": list(range(12))},
                     "splits": {"arrays": {"splits.npz": {"sha256": se.sha256(array), "bytes": array.stat().st_size}}},
                     "index_arrays_copied": ["arxiv_splits.npz"]}
            write(root / "arxiv.json", value)
            se.check_run_manifest(root, "arxiv", "ogbn-arxiv", {"identity": identities})
            array.write_bytes(b"modified-memberships")
            with self.assertRaisesRegex(ValueError, "published identity"):
                se.check_run_manifest(root, "arxiv", "ogbn-arxiv", {"identity": identities})
            array.write_bytes(b"software-fixture-memberships")
            changed = dict(identities, source_sha256="c" * 64)
            with self.assertRaisesRegex(ValueError, "source_sha256"):
                se.check_run_manifest(root, "arxiv", "ogbn-arxiv", {"identity": changed})


if __name__ == "__main__":
    unittest.main()
