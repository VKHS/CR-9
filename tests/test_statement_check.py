"""The registered availability statement is evaluated item by item from repository contents.

These tests use the repository's own statement, display inventory and generators,
and replace only the published-artifact manifest to exercise each state.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import paper_artifacts as pa  # noqa: E402
from scripts import statement_check as sc  # noqa: E402

CONTENTS = {"checkpoints": 40, "predictions": 12, "propensity_logged_trajectories": 900,
            "statistical_outputs": 30, "split_and_seed_manifests": 3, "environment": 1, "configurations": 2}


def published(manifest, *, contents=CONTENTS):
    manifest = copy.deepcopy(manifest)
    for entry in manifest["artifacts"]:
        entry["status"] = pa.PUBLISHED
        entry["contents"] = dict(contents) if entry["kind"] == "run" else {"statistical_outputs": 5}
        entry["identity"] = {}
        entry["relationship"] = "post_result_revision"
        entry["inventory_sha256"] = "a" * 64
        entry["scientific_only_verification"] = True
        entry["tiers"] = {"complete": {"sha256": "b" * 64}}
    manifest["status"] = pa.PUBLISHED
    return manifest


class StatementCheck(unittest.TestCase):
    def setUp(self):
        self.manifest = pa.load_manifest(ROOT / "release/paper_artifacts.json")

    def evaluate(self, manifest):
        with mock.patch.object(pa, "load_manifest", lambda path=pa.MANIFEST: manifest):
            return sc.evaluate(ROOT)

    def test_every_statement_item_is_evaluated(self):
        result = self.evaluate(self.manifest)
        statement = sc._json(ROOT / sc.STATEMENT)
        self.assertEqual([row["id"] for row in result["items"]], [i["id"] for i in statement["items"]])
        self.assertEqual(len(result["items"]), 12)
        self.assertTrue(all(row["status"] in (sc.FULFILLED, sc.AWAITING, sc.MISSING) for row in result["items"]))
        self.assertIn("propensity-logged trajectories", result["statement"])

    def test_unpublished_outputs_are_awaiting_not_fulfilled(self):
        awaiting = copy.deepcopy(self.manifest)
        for entry in awaiting["artifacts"]:
            entry["status"] = pa.AWAITING
        status = {row["id"]: row["status"] for row in self.evaluate(awaiting)["items"]}
        for key in ("checkpoints", "predictions", "trajectories", "statistical_outputs", "one_command"):
            self.assertEqual(status[key], sc.AWAITING, key)

    def test_published_outputs_fulfil_the_artifact_items(self):
        status = {row["id"]: row["status"] for row in self.evaluate(published(self.manifest))["items"]}
        for key in ("checkpoints", "predictions", "trajectories", "statistical_outputs"):
            self.assertEqual(status[key], sc.FULFILLED, key)

    def test_a_published_run_without_checkpoints_is_a_gap(self):
        manifest = published(self.manifest, contents=dict(CONTENTS, checkpoints=0))
        status = {row["id"]: row["status"] for row in self.evaluate(manifest)["items"]}
        self.assertEqual(status["checkpoints"], sc.MISSING)

    def test_declared_non_executable_parts_make_the_code_item_missing(self):
        experiments = copy.deepcopy(sc._json(ROOT / sc.EXPERIMENTS))
        experiments["experiments"][0]["not_executable"] = [{"display": "Table S99", "reason": "no configuration key"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment_code.json"
            path.write_text(json.dumps(experiments), encoding="utf-8")
            with mock.patch.object(sc, "EXPERIMENTS", str(path)):
                rows = {row["id"]: row for row in self.evaluate(published(self.manifest))["items"]}
        self.assertEqual(rows["experiment_code"]["status"], sc.MISSING)
        self.assertIn("Table S99 (no configuration key)", rows["experiment_code"]["remaining"])

    def _with_view(self, mutate=None):
        """The E12 declaration pointed at a temporary experiments/baselines/ view of fourteen studies."""
        from tests.test_statement_evidence import baseline_view
        experiments = copy.deepcopy(sc._json(ROOT / sc.EXPERIMENTS))
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        view = Path(tmp.name) / "experiments" / "baselines"
        baseline_view(view)
        index = json.loads((view / "INDEX.json").read_text())
        index.update(artifact_manifest_sha256=sc._sha(ROOT / "release/paper_artifacts.json"),
                     artifacts=[pa.artifact_identity(entry, "complete")
                                for entry in published(self.manifest)["artifacts"]],
                     files={p.relative_to(view).as_posix(): sc._sha(p) for p in view.rglob("*")
                            if p.is_file() and p.name != "INDEX.json"})
        (view / "INDEX.json").write_text(json.dumps(index), encoding="utf-8")
        if mutate:
            mutate(view)
        declared = [e for e in experiments["experiments"] if e.get("published_with_outputs")]
        self.assertEqual([e["experiment"] for e in declared], ["E12"])
        declared[0]["published_with_outputs"]["view"] = str(view / "INDEX.json")
        path = Path(tmp.name) / "experiment_code.json"
        path.write_text(json.dumps(experiments), encoding="utf-8")
        return path

    def test_code_and_configurations_inside_an_unpublished_artifact_await_it(self):
        rows = {row["id"]: row for row in self.evaluate(self.manifest)["items"]}
        for key in ("experiment_code", "configurations"):
            self.assertEqual(rows[key]["status"], sc.AWAITING, key)
            self.assertIn("records:baselines", rows[key]["remaining"])
        self.assertIn("adapter code", rows["experiment_code"]["remaining"])
        self.assertIn("48 HPO configurations", rows["configurations"]["remaining"])

    def test_published_study_plans_and_adapters_fulfil_code_and_configurations(self):
        path = self._with_view()
        with mock.patch.object(sc, "EXPERIMENTS", str(path)):
            status = {row["id"]: row["status"] for row in self.evaluate(published(self.manifest))["items"]}
        self.assertEqual(status["experiment_code"], sc.FULFILLED)
        self.assertEqual(status["configurations"], sc.FULFILLED)

    def test_a_changed_or_incomplete_view_is_a_gap(self):
        def change(view):
            index = json.loads((view / "INDEX.json").read_text())
            (view / index["studies"][3]["adapter_archive"]["file"]).write_bytes(b"changed")
        path = self._with_view(change)
        with mock.patch.object(sc, "EXPERIMENTS", str(path)):
            rows = {row["id"]: row for row in self.evaluate(published(self.manifest))["items"]}
        self.assertEqual(rows["experiment_code"]["status"], sc.MISSING)
        self.assertIn("adapter.tar: differs", rows["experiment_code"]["remaining"])

        def drop(view):
            index = json.loads((view / "INDEX.json").read_text(encoding="utf-8"))
            index["studies"] = index["studies"][:13]
            (view / "INDEX.json").write_text(json.dumps(index), encoding="utf-8")
        path = self._with_view(drop)
        with mock.patch.object(sc, "EXPERIMENTS", str(path)):
            rows = {row["id"]: row for row in self.evaluate(published(self.manifest))["items"]}
        self.assertEqual(rows["configurations"]["status"], sc.MISSING)
        self.assertIn("omits registered studies", rows["configurations"]["remaining"])

    def test_deleting_required_artifact_rows_does_not_shrink_the_statement(self):
        manifest = published(self.manifest)
        manifest["artifacts"] = [a for a in manifest["artifacts"] if a["id"] != "run:ogbn-arxiv"]
        result = self.evaluate(manifest)
        self.assertIn("run:ogbn-arxiv", result["awaiting_artifacts"])
        status = {r["id"]: r["status"] for r in result["items"]}
        self.assertEqual(status["checkpoints"], sc.AWAITING)

    def test_software_fixture_archives_cannot_fulfil_benchmark_items(self):
        manifest = published(self.manifest)
        next(a for a in manifest["artifacts"] if a["id"] == "run:ogbn-arxiv")["scientific_only_verification"] = False
        status = {r["id"]: r["status"] for r in self.evaluate(manifest)["items"]}
        self.assertEqual(status["checkpoints"], sc.MISSING)
        self.assertEqual(status["statistical_outputs"], sc.MISSING)

    def test_report_names_every_item(self):
        text = sc.render(self.evaluate(self.manifest))
        statement = sc._json(ROOT / sc.STATEMENT)
        for item in statement["items"]:
            self.assertIn(f"| {item['phrase']} |", text)


if __name__ == "__main__":
    unittest.main()
