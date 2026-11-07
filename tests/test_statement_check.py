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

    def test_report_names_every_item(self):
        text = sc.render(self.evaluate(self.manifest))
        statement = sc._json(ROOT / sc.STATEMENT)
        for item in statement["items"]:
            self.assertIn(f"| {item['phrase']} |", text)


if __name__ == "__main__":
    unittest.main()
