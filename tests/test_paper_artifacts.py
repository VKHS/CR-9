"""Publication and retrieval of the paper's measured outputs through a GitHub release.

A local HTTP server stands in for the GitHub REST and upload endpoints, so the
whole path (pack, split, upload, re-publish, download, reassemble, verify,
unpack, analysis-tier verification) runs offline. The packed run is the bundled
synthetic example; it exercises the transport, not the paper's results.
"""
from __future__ import annotations

import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest import mock
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from scripts import paper_artifacts as pa  # noqa: E402

EXAMPLE_RUN = ROOT / "examples/runs/arxiv"
EXAMPLE_RECORDS = ROOT / "examples/records/nominal_time"


class FakeGitHub(http.server.ThreadingHTTPServer):
    """Release endpoints of one repository, held in memory."""

    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.release = None
        self.assets = {}          # name -> dict(id, bytes)
        self.uploads = 0
        self.next_id = 100

    @property
    def base(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, status, value=None, raw=None):
        body = raw if raw is not None else (json.dumps(value).encode() if value is not None else b"")
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/octet-stream" if raw is not None else "application/json")
        self.end_headers()
        self.wfile.write(body)

    def _asset(self, name):
        data = self.server.assets[name]
        return {"id": data["id"], "name": name, "size": len(data["bytes"]), "state": "uploaded",
                "digest": "sha256:" + hashlib.sha256(data["bytes"]).hexdigest(),
                "browser_download_url": f"{self.server.base}/download/{urllib.parse.quote(name)}"}

    def do_GET(self):
        server, path = self.server, urllib.parse.urlparse(self.path)
        if path.path == "/repos/VKHS/CR-9/releases/tags/paper-artifacts":
            return self._send(200, server.release) if server.release else self._send(404, {"message": "Not Found"})
        if path.path == "/repos/VKHS/CR-9/releases/1/assets":
            page = int(urllib.parse.parse_qs(path.query).get("page", ["1"])[0])
            names = sorted(server.assets)[(page - 1) * 100: page * 100]
            return self._send(200, [self._asset(n) for n in names])
        if path.path.startswith("/download/"):
            name = urllib.parse.unquote(path.path[len("/download/"):])
            return self._send(200, raw=server.assets[name]["bytes"]) if name in server.assets else self._send(404, {})
        self._send(404, {"message": "Not Found"})

    def do_POST(self):
        server, path = self.server, urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if path.path == "/repos/VKHS/CR-9/releases":
            request = json.loads(body)
            assert request["tag_name"] == "paper-artifacts" and not request["draft"]
            server.release = {"id": 1, "tag_name": "paper-artifacts",
                              "html_url": f"{server.base}/VKHS/CR-9/releases/tag/paper-artifacts",
                              "upload_url": f"{server.base}/uploads/repos/VKHS/CR-9/releases/1/assets{{?name,label}}"}
            return self._send(201, server.release)
        if path.path == "/uploads/repos/VKHS/CR-9/releases/1/assets":
            assert self.headers["Authorization"] == "Bearer test-token"
            name = urllib.parse.parse_qs(path.query)["name"][0]
            server.next_id += 1
            server.uploads += 1
            server.assets[name] = {"id": server.next_id, "bytes": body}
            return self._send(201, self._asset(name))
        self._send(404, {})

    def do_DELETE(self):
        server = self.server
        asset_id = int(self.path.rsplit("/", 1)[1])
        for name, data in list(server.assets.items()):
            if data["id"] == asset_id:
                del server.assets[name]
                return self._send(204)
        self._send(404, {})


REQUIRED = [{"id": "run:ogbn-arxiv", "kind": "run", "name": "ogbn-arxiv", "asset_stem": "run-arxiv",
             "statement_items": pa.STATEMENT_ITEMS["run"], "displays": ["main:Table 1"]},
            {"id": "records:nominal_time", "kind": "records", "name": "nominal_time",
             "asset_stem": "records-nominal-time", "statement_items": pa.STATEMENT_ITEMS["records"],
             "displays": ["supplementary:Table S23b"]},
            {"id": "environment:gpu", "kind": "environment", "name": "gpu", "asset_stem": "environment-gpu",
             "statement_items": pa.STATEMENT_ITEMS["environment"], "displays": []}]


class PaperArtifactTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
        cls.server = FakeGitHub()
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.server.release, self.server.assets, self.server.uploads = None, {}, 0
        self.tmp = Path(tempfile.mkdtemp(prefix="cm-artifacts-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.patch = mock.patch.object(pa, "required_artifacts", lambda root=pa.ROOT: [dict(r) for r in REQUIRED])
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.manifest = self.tmp / "paper_artifacts.json"
        pa.write_json(self.manifest, pa.planned_manifest(pa.blank_manifest()))

    def stage(self):
        staged = self.tmp / "staged"
        run = self.tmp / "measured/arxiv_run"
        shutil.copytree(EXAMPLE_RUN, run)
        records = self.tmp / "measured/records/nominal_time"
        shutil.copytree(EXAMPLE_RECORDS, records)
        environment = self.tmp / "measured/environment"
        environment.mkdir(parents=True)
        (environment / "nvidia-smi.txt").write_text("GH200 96GB\n")
        run_entry = pa.pack_one("run:ogbn-arxiv", run, staged, relationship="exact_result_producing_run",
                                part_bytes=64 * 1024, scientific_only=False)
        pa.pack_one("records:nominal_time", records, staged, relationship="exact_result_producing_run",
                    part_bytes=64 * 1024, scientific_only=False)
        pa.pack_one("environment:gpu", environment, staged, relationship="exact_result_producing_run",
                    scientific_only=False)
        return staged, run_entry

    def client(self):
        return pa.GitHub("VKHS/CR-9", "test-token", self.server.base)

    def test_awaiting_manifest_lists_every_required_artifact(self):
        manifest = pa.load_manifest(self.manifest)
        self.assertEqual(manifest["status"], pa.AWAITING)
        self.assertEqual([a["id"] for a in manifest["artifacts"]], [r["id"] for r in REQUIRED])
        self.assertTrue(all(a["status"] == pa.AWAITING for a in manifest["artifacts"]))
        report = pa.fetch(self.tmp / "dest", manifest_path=self.manifest)
        self.assertEqual(report["awaiting"], [r["id"] for r in REQUIRED])

    def test_pack_splits_the_analysis_and_complete_tiers(self):
        staged, entry = self.stage()
        complete, analysis = entry["tiers"]["complete"], entry["tiers"]["analysis"]
        self.assertGreater(len(complete["parts"]), 1)
        self.assertLess(analysis["bytes"], complete["bytes"])
        self.assertGreater(entry["contents"]["checkpoints"], 0)
        self.assertGreater(entry["contents"]["propensity_logged_trajectories"], 0)
        self.assertGreater(entry["contents"]["statistical_outputs"], 0)
        for part in complete["parts"]:
            self.assertLessEqual(part["bytes"], 64 * 1024)
            self.assertEqual(pa.sha256_file(staged / part["asset"]), part["sha256"])

    def test_publish_fetch_verify_and_republish(self):
        staged, _ = self.stage()
        manifest = pa.publish(staged, self.manifest, client=self.client(), download_base=f"{self.server.base}/download")
        self.assertEqual(manifest["status"], pa.PUBLISHED)
        uploads = self.server.uploads
        self.assertEqual(uploads, len(self.server.assets))
        pa.publish(staged, self.manifest, client=self.client(), download_base=f"{self.server.base}/download")
        self.assertEqual(self.server.uploads, uploads, "an identical re-publish uploads nothing")

        light = pa.fetch(self.tmp / "light", manifest_path=self.manifest, tier="analysis")
        self.assertFalse(light["awaiting"])
        run = self.tmp / "light/runs/arxiv"
        self.assertFalse(list(run.rglob("*.pt")), "the analysis tier carries no checkpoints")
        self.assertTrue((run / "select/unit_scores.npz").is_file())
        self.assertTrue(list((run / "explore").glob("trajectory_*.npz")))
        identities, info = pa.verify_analysis_run(run, pa.analysis_omissions(run))
        self.assertGreater(info["omitted"], 0)
        self.assertEqual(identities["source_sha256"], json.loads((EXAMPLE_RUN / "run_state.json").read_text())["source_sha256"])
        self.assertTrue((self.tmp / "light/records/index.json").is_file())

        full = pa.fetch(self.tmp / "full", manifest_path=self.manifest, tier="complete")
        self.assertFalse(full["awaiting"])
        self.assertTrue(list((self.tmp / "full/runs/arxiv").rglob("*.pt")))
        from scripts.artifact_inventory import verify_run_bytes
        verify_run_bytes(self.tmp / "full/runs/arxiv")
        again = pa.fetch(self.tmp / "full", manifest_path=self.manifest, tier="complete")
        self.assertTrue(all(row["reused"] for row in again["published"]))

    def test_a_changed_published_part_is_rejected(self):
        staged, _ = self.stage()
        pa.publish(staged, self.manifest, client=self.client(), download_base=f"{self.server.base}/download")
        name = sorted(n for n in self.server.assets if n.startswith("run-arxiv.tar.gz"))[0]
        corrupted = bytearray(self.server.assets[name]["bytes"])
        corrupted[100] ^= 0xFF
        self.server.assets[name]["bytes"] = bytes(corrupted)
        with self.assertRaisesRegex(pa.ArtifactError, "differs from the manifest"):
            pa.fetch(self.tmp / "bad", manifest_path=self.manifest, tier="complete", only=["run:ogbn-arxiv"])

    def test_analysis_subset_cannot_hide_a_statistical_output(self):
        staged, _ = self.stage()
        pa.publish(staged, self.manifest, client=self.client(), download_base=f"{self.server.base}/download")
        pa.fetch(self.tmp / "light", manifest_path=self.manifest, tier="analysis", only=["run:ogbn-arxiv"])
        run = self.tmp / "light/runs/arxiv"
        (run / "admit/decision.json").unlink()
        with self.assertRaisesRegex(pa.ArtifactError, "neither present nor a declared omission"):
            pa.verify_analysis_run(run, pa.analysis_omissions(run))

    def test_a_non_scientific_run_is_refused_for_publication(self):
        with self.assertRaisesRegex(pa.ArtifactError, "not a full scientific run"):
            pa.pack_one("run:ogbn-arxiv", EXAMPLE_RUN, self.tmp / "staged", relationship="exact_result_producing_run")

    def test_repacking_identical_inputs_gives_identical_assets(self):
        _, first = self.stage()
        again = self.tmp / "again"
        second = pa.pack_one("run:ogbn-arxiv", self.tmp / "measured/arxiv_run", again,
                             relationship="exact_result_producing_run", part_bytes=64 * 1024, scientific_only=False)
        self.assertEqual(first["tiers"], second["tiers"])

    def test_parts_reassemble_to_the_original_bytes(self):
        source = self.tmp / "blob.bin"
        source.write_bytes(os.urandom(300_000))
        digest = pa.sha256_file(source)
        parts = pa.split_parts(source, 100_000)
        self.assertEqual([p["bytes"] for p in parts], [100_000, 100_000, 100_000])
        joined = b"".join((self.tmp / p["asset"]).read_bytes() for p in parts)
        self.assertEqual(hashlib.sha256(joined).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
