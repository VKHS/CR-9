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
        client = pa.GitHub("VKHS/CR-9", "test-token", self.server.base)
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


if __name__ == "__main__":
    unittest.main()
