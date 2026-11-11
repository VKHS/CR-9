"""The one command, end to end: published outputs -> verified download -> every display.

The two bundled synthetic example runs are published to a local stand-in for the
GitHub release API under the names of two datasets; the command must download
them, verify them as analysis-tier runs, build the workspace and report the other
artifacts as awaiting publication. The values are synthetic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from scripts import paper_artifacts as pa  # noqa: E402
from scripts import reconstruct_paper  # noqa: E402
from tests.test_paper_artifacts import FakeGitHub  # noqa: E402


class OneCommandReconstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
        cls.server = FakeGitHub()
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        # Each test owns a release. A tampered asset in the preceding test must
        # not become an immutable published-asset conflict in the next one.
        self.server.release = None
        self.server.assets.clear()
        self.server.uploads = 0
        self.server.next_id = 100
        self.tmp = Path(tempfile.mkdtemp(prefix="cm-one-command-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.manifest = self.tmp / "paper_artifacts.json"
        pa.write_json(self.manifest, pa.planned_manifest(pa.blank_manifest()))

    def publish_examples(self):
        staged = self.tmp / "staged"
        for artifact_id, example in (("run:ogbn-arxiv", "arxiv"), ("run:Peptides-func", "peptides_func")):
            copy = self.tmp / "measured" / example
            shutil.copytree(ROOT / "examples/runs" / example, copy)
            pa.pack_one(artifact_id, copy, staged, relationship="exact_result_producing_run", scientific_only=False)
        client = pa.GitHub("VKHS/CROSS-MORSE", "test-token", self.server.base)
        return pa.publish(staged, self.manifest, client=client, download_base=f"{self.server.base}/download")

    def test_awaiting_outputs_are_reported_and_nothing_is_fabricated(self):
        code = reconstruct_paper.main(["--manifest", str(self.manifest), "--artifacts", str(self.tmp / "art"),
                                       "--output-dir", str(self.tmp / "paper")])
        self.assertEqual(code, 1)
        status = json.loads((self.tmp / "paper.status.json").read_text())
        self.assertIn("run:ogbn-arxiv", status["artifacts_awaiting_publication"])
        result = json.loads((self.tmp / "paper/reconstruction.json").read_text())
        self.assertFalse(any(d["full_numerical_display"] and d["id"].startswith("main:Table") for d in result["displays"]))

    def test_published_runs_are_fetched_verified_and_rebuilt(self):
        manifest = self.publish_examples()
        published = [a["id"] for a in manifest["artifacts"] if a["status"] == pa.PUBLISHED]
        self.assertEqual(sorted(published), ["run:Peptides-func", "run:ogbn-arxiv"])
        code = reconstruct_paper.main(["--manifest", str(self.manifest), "--artifacts", str(self.tmp / "art"),
                                       "--output-dir", str(self.tmp / "paper")])
        self.assertEqual(code, 1, "the other artifacts are still awaiting publication")
        self.assertFalse(list((self.tmp / "art/runs").rglob("*.pt")))
        result = json.loads((self.tmp / "paper/reconstruction.json").read_text())
        runs = [row for row in result["inputs"] if row["kind"] == "sealed_run"]
        self.assertEqual(len(runs), 2)
        self.assertTrue(all(row["verification"].startswith("analysis tier") for row in runs))
        self.assertTrue(result["synthetic_examples"])
        self.assertTrue(any(o["id"] == "run:T1" for o in result["outputs"]))
        from scripts.reconstruct import verify_workspace
        verify_workspace(self.tmp / "paper")

    def test_a_tampered_download_stops_the_command(self):
        self.publish_examples()
        name = sorted(n for n in self.server.assets if n.startswith("run-arxiv.analysis"))[0]
        data = bytearray(self.server.assets[name]["bytes"])
        data[-50] ^= 0x01
        self.server.assets[name]["bytes"] = bytes(data)
        code = reconstruct_paper.main(["--manifest", str(self.manifest), "--artifacts", str(self.tmp / "art"),
                                       "--output-dir", str(self.tmp / "paper")])
        self.assertEqual(code, 2)

    @staticmethod
    def complete_display_fixture():
        # Isolate the artifact gate from display arithmetic. This is a software
        # fixture, not a reconstruction of a scientific table.
        return {"displays": [{"id": "main:Table 1", "full_numerical_display": True,
                               "content_complete": True}],
                "display_generators": {"comparison_with_printed_values": {"exact": 1},
                    "comparison_coverage_complete": True, "errors": {}, "module_errors": {}}}

    def offline_gate(self, required, *, tier="complete", name="paper"):
        with mock.patch.object(pa, "required_artifacts", return_value=[{"id": key} for key in required]), \
             mock.patch.object(reconstruct_paper, "reconstruct", return_value=self.complete_display_fixture()):
            code = reconstruct_paper.main(["--manifest", str(self.manifest), "--offline", "--tier", tier,
                                           "--artifacts", str(self.tmp / "art"),
                                           "--output-dir", str(self.tmp / name)])
        return code, json.loads((self.tmp / f"{name}.status.json").read_text()) if code != 2 else None

    def test_complete_request_requires_complete_cached_bytes_and_accepts_upgrade(self):
        manifest = self.publish_examples()
        manifest["artifacts"] = [row for row in manifest["artifacts"] if row["status"] == pa.PUBLISHED]
        manifest["status"] = pa.PUBLISHED
        pa.write_json(self.manifest, manifest)
        required = [row["id"] for row in manifest["artifacts"]]
        pa.fetch(self.tmp / "art", manifest_path=self.manifest, tier="analysis")

        code, report = self.offline_gate(required)
        self.assertEqual(code, 1, "analysis descriptors cannot supply the omitted checkpoint/prediction files")
        materialized = report["artifact_materialization"]
        self.assertEqual(materialized["missing_local"], [])
        self.assertEqual({row["id"] for row in materialized["incomplete_tiers"]}, set(required))
        self.assertTrue(all(row["available_tier"] == "analysis" for row in materialized["incomplete_tiers"]))

        pa.fetch(self.tmp / "art", manifest_path=self.manifest, tier="complete")
        code, report = self.offline_gate(required, name="complete")
        self.assertEqual(code, 0)
        self.assertEqual(report["artifact_materialization"]["incomplete_tiers"], [])
        self.assertEqual(set(report["artifact_materialization"]["verified_tiers"].values()), {"complete"})
        self.assertEqual(self.offline_gate(required, tier="analysis", name="analysis")[0], 0,
                         "complete materialization also supplies the analysis request")

    def test_missing_published_environment_cannot_hide_behind_complete_displays(self):
        source = self.tmp / "environment"
        source.mkdir()
        (source / "observed_environment.json").write_text('{"software_fixture": true}\n')
        staged = self.tmp / "staged"
        pa.pack_one("environment:gpu", source, staged, relationship="post_result_revision", scientific_only=False)
        client = pa.GitHub("VKHS/CROSS-MORSE", "test-token", self.server.base)
        manifest = pa.publish(staged, self.manifest, client=client,
                              download_base=f"{self.server.base}/download")
        manifest["artifacts"] = [row for row in manifest["artifacts"] if row["id"] == "environment:gpu"]
        manifest["status"] = pa.PUBLISHED
        pa.write_json(self.manifest, manifest)
        code, report = self.offline_gate(["environment:gpu"])
        self.assertEqual(code, 1)
        self.assertEqual(report["artifacts_awaiting_publication"], [])
        self.assertEqual(report["artifact_materialization"]["missing_local"], ["environment:gpu"])

    def test_missing_manifest_row_or_changed_required_flag_cannot_shrink_contract(self):
        manifest = pa.blank_manifest()
        manifest["artifacts"] = []
        pa.write_json(self.manifest, manifest)
        code, report = self.offline_gate(["environment:gpu"])
        self.assertEqual(code, 1)
        self.assertEqual(report["artifact_materialization"]["not_registered"], ["environment:gpu"])
        self.assertEqual(report["artifacts_awaiting_publication"], ["environment:gpu"])
        manifest["artifacts"] = [{"id": "environment:gpu", "kind": "environment",
                                  "status": pa.AWAITING, "required": False}]
        pa.write_json(self.manifest, manifest)
        self.assertEqual(self.offline_gate(["environment:gpu"], name="flag")[0], 1)

    def test_corrupt_cached_bytes_remain_verification_errors(self):
        self.publish_examples()
        pa.fetch(self.tmp / "art", manifest_path=self.manifest, tier="analysis")
        path = self.tmp / "art/runs/arxiv/config.json"
        path.write_text(path.read_text() + " ")
        self.assertEqual(self.offline_gate(["run:ogbn-arxiv"])[0], 2)

    def test_optional_extra_artifacts_do_not_block_materialization(self):
        with mock.patch.object(pa, "required_artifacts", return_value=[]):
            report = reconstruct_paper.materialization_status(
                {"artifacts": [], "awaiting": ["environment:gpu"]},
                {"artifacts": [{"id": "environment:gpu", "required": False, "status": pa.PUBLISHED}]},
                "complete")
        self.assertEqual(report["missing_local"], [])
        self.assertEqual(report["not_published"], [])


if __name__ == "__main__":
    unittest.main()
