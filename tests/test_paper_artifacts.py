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
import shlex
import subprocess
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
        if path.path == "/repos/VKHS/CROSS-MORSE/releases/tags/paper-artifacts":
            return self._send(200, server.release) if server.release else self._send(404, {"message": "Not Found"})
        if path.path == "/repos/VKHS/CROSS-MORSE/releases/1/assets":
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
        if path.path == "/repos/VKHS/CROSS-MORSE/releases":
            request = json.loads(body)
            assert request["tag_name"] == "paper-artifacts" and not request["draft"]
            server.release = {"id": 1, "tag_name": "paper-artifacts",
                              "html_url": f"{server.base}/VKHS/CROSS-MORSE/releases/tag/paper-artifacts",
                              "upload_url": f"{server.base}/uploads/repos/VKHS/CROSS-MORSE/releases/1/assets{{?name,label}}"}
            return self._send(201, server.release)
        if path.path == "/uploads/repos/VKHS/CROSS-MORSE/releases/1/assets":
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
        return pa.GitHub("VKHS/CROSS-MORSE", "test-token", self.server.base)

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
        staged, entry = self.stage()
        pa.publish(staged, self.manifest, client=self.client(), download_base=f"{self.server.base}/download")
        name = entry["tiers"]["complete"]["parts"][0]["asset"]
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

    def test_cached_fetched_bytes_are_reverified(self):
        staged, _ = self.stage()
        pa.publish(staged, self.manifest, client=self.client(), download_base=f"{self.server.base}/download")
        destination = self.tmp / "cached"
        pa.fetch(destination, manifest_path=self.manifest, tier="complete")
        (destination / "runs/arxiv/final/results_summary.json").write_text("{}")
        for check in (lambda: pa.fetch(destination, manifest_path=self.manifest, tier="complete"),
                      lambda: pa.verify_fetched(destination, manifest_path=self.manifest)):
            with self.assertRaisesRegex(pa.ArtifactError, "differs from its inventory"):
                check()

    def test_offline_use_rejects_a_published_directory_without_its_stamp(self):
        staged, _ = self.stage()
        pa.publish(staged, self.manifest, client=self.client(), download_base=f"{self.server.base}/download")
        destination = self.tmp / "offline"
        pa.fetch(destination, manifest_path=self.manifest)
        (destination / ".artifacts/run-arxiv.json").unlink()
        with self.assertRaisesRegex(pa.ArtifactError, "no managed download stamp"):
            pa.verify_fetched(destination, manifest_path=self.manifest)

    def test_publish_never_deletes_a_conflicting_existing_asset(self):
        staged, entry = self.stage()
        pa.publish(staged, self.manifest, client=self.client(), download_base=f"{self.server.base}/download")
        before = self.manifest.read_bytes()
        name = entry["tiers"]["complete"]["parts"][0]["asset"]
        original = self.server.assets[name]
        self.server.assets[name] = dict(original, bytes=b"X" * len(original["bytes"]))
        with self.assertRaisesRegex(pa.ArtifactError, "differs|collision"):
            pa.publish(staged, self.manifest, client=self.client(), download_base=f"{self.server.base}/download")
        self.assertEqual(self.server.assets[name]["bytes"], b"X" * len(original["bytes"]))
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_repack_into_the_same_stage_is_idempotent(self):
        staged, first = self.stage()
        second = pa.pack_one("run:ogbn-arxiv", self.tmp / "measured/arxiv_run", staged,
                             relationship="post_result_revision", part_bytes=64 * 1024, scientific_only=False)
        self.assertEqual(first["tiers"], second["tiers"])
        self.assertEqual(second["relationship"], "post_result_revision")

    def test_traceable_but_partial_records_do_not_pass_scientific_packing(self):
        from scripts import records
        source = self.tmp / "records/nominal_time"
        shutil.copytree(EXAMPLE_RECORDS, source)
        provenance = pa.read_json(source / "provenance.json")
        provenance["synthetic"] = False  # adversarial claim; no benchmark evidence is created
        pa.write_json(source / "provenance.json", provenance)
        records.write_manifest(source)
        with self.assertRaisesRegex(ValueError, "incomplete or unexpected dataset coverage"):
            pa.pack_one("records:nominal_time", source, self.tmp / "staged", relationship="post_result_revision")

    def test_empty_or_unsafe_published_descriptors_are_refused(self):
        staged, _ = self.stage()
        manifest = pa.publish(staged, self.manifest, client=self.client(), download_base=f"{self.server.base}/download")
        for change in (lambda value: value["artifacts"][0]["tiers"]["complete"].update(parts=[]),
                       lambda value: value["artifacts"][0]["tiers"]["complete"]["parts"][0].update(asset="../outside")):
            broken = json.loads(json.dumps(manifest))
            change(broken)
            with self.assertRaises(pa.ArtifactError):
                pa.validate_manifest(broken)

    def test_pack_requires_an_explicit_relationship_and_complete_map_when_requested(self):
        mapping = self.tmp / "map.json"
        pa.write_json(mapping, {"run:ogbn-arxiv": str(EXAMPLE_RUN)})
        with mock.patch("sys.stderr"):
            self.assertEqual(pa.main(["pack", "--map", str(mapping), "--output-dir", str(self.tmp / "staged")]), 2)
            self.assertEqual(pa.main(["pack", "--map", str(mapping), "--output-dir", str(self.tmp / "staged"),
                                      "--relationship", "post_result_revision", "--require-complete"]), 2)
        self.assertFalse((self.tmp / "staged").exists())

    def test_derive_binds_all_views_to_verified_assets_and_report_files(self):
        from scripts.reconstruct import write_file_manifest
        staged, _ = self.stage()
        pa.publish(staged, self.manifest, client=self.client(), download_base=f"{self.server.base}/download")
        fetched, checkout, workspace = self.tmp / "fetched", self.tmp / "checkout", self.tmp / "workspace"
        checkout.mkdir()
        pa.fetch(fetched, manifest_path=self.manifest, tier="complete")
        (workspace / "displays/data").mkdir(parents=True)
        (workspace / "displays/data/panel.json").write_text('{"value":1}')
        (workspace / "displays/data/panel.csv").write_text("value\n1\n")
        identities, _ = pa.verify_analysis_run(fetched / "runs/arxiv", {})
        run = fetched / "runs/arxiv"
        record_index = pa.read_json(fetched / "records/index.json")
        inputs = [{"kind": "sealed_run", "path": str(run), "identities": identities,
                   "stage_manifests": {p.name: pa.sha256_file(p) for p in (run / "provenance/stages").glob("*.json")}},
                  {"kind": "measurement_records", "path": str(fetched / "records/index.json"),
                   "sha256": pa.sha256_file(fetched / "records/index.json"), "sets": record_index["sets"]}]
        report = {"schema": "cross-morse-reconstruction-1", "label": "transport fixture", "inputs": inputs,
                  "software_sources": {}, "displays": [{"id": "fixture", "full_numerical_display": False}],
                  "display_generators": {"comparison_with_printed_values": {"no_reference": 1}}}
        pa.write_json(workspace / "reconstruction.json", report)
        write_file_manifest(workspace)
        pa.derive(fetched, checkout=checkout, workspace=workspace, manifest_path=self.manifest)
        index = pa.read_json(checkout / "paper/source_data/INDEX.json")
        self.assertEqual(index["artifact_manifest_sha256"], pa.sha256_file(self.manifest))
        self.assertEqual(index["reconstruction"]["values_no_reference"], 1)
        self.assertEqual(set(index["reports"]), {"reconstruction", "workspace_files", "status"})
        self.assertIn("data/panel.json", index["files"])
        for directory in ("paper/source_data", "manifests/published", "environment/observed"):
            view = checkout / directory
            value = pa.read_json(view / "INDEX.json")
            self.assertEqual(value["files"], pa.view_files(view))
            self.assertEqual({a["id"] for a in value["artifacts"]}, {r["id"] for r in REQUIRED})
        previous = (checkout / "paper/source_data/INDEX.json").read_bytes()
        # A self-consistent report cannot silently omit a fetched measured input.
        pa.write_json(workspace / "reconstruction.json", dict(report, inputs=inputs[1:]))
        write_file_manifest(workspace)
        with self.assertRaisesRegex(pa.ArtifactError, "complete fetched run set"):
            pa.derive(fetched, checkout=checkout, workspace=workspace, manifest_path=self.manifest)
        self.assertEqual((checkout / "paper/source_data/INDEX.json").read_bytes(), previous)
        pa.write_json(workspace / "reconstruction.json", report)
        write_file_manifest(workspace)
        (workspace / "displays/data/panel.csv").write_text("changed")
        with self.assertRaises(ValueError):
            pa.derive(fetched, checkout=checkout, workspace=workspace, manifest_path=self.manifest)
        self.assertEqual((checkout / "paper/source_data/INDEX.json").read_bytes(), previous)

    def test_parts_reassemble_to_the_original_bytes(self):
        source = self.tmp / "blob.bin"
        source.write_bytes(os.urandom(300_000))
        digest = pa.sha256_file(source)
        parts = pa.split_parts(source, 100_000)
        self.assertEqual([p["bytes"] for p in parts], [100_000, 100_000, 100_000])
        joined = b"".join((self.tmp / p["asset"]).read_bytes() for p in parts)
        self.assertEqual(hashlib.sha256(joined).hexdigest(), digest)


class BaselineStudyView(unittest.TestCase):
    """The E12 study plans and adapter archives of a published baselines set become files of the checkout."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.records = self.tmp / "fetched" / "records"
        self.checkout = self.tmp / "checkout"
        self.checkout.mkdir()
        base = self.records / "baselines"
        self.archive = base / "code/adapters/arxiv/AMP/adapter-0123456789ab.tar"
        self.archive.parent.mkdir(parents=True)
        self.archive.write_bytes(bytes(1024))
        study = base / "raw/arxiv/AMP"
        study.mkdir(parents=True)
        (study / "plan.json").write_text('{"schema": "cm-upstream-study-1", "method": "AMP"}\n', encoding="utf-8")
        self.upstream = {"dataset": "ogbn-arxiv", "method": "AMP", "plan": "raw/arxiv/AMP/plan.json",
                         "source_commit": "a" * 40, "upstream_repository": "https://example.org/amp.git",
                         "adapter_archive": {"file": "code/adapters/arxiv/AMP/adapter-0123456789ab.tar",
                                             "sha256": pa.sha256_file(self.archive), "commit": "0123456789ab" + "c" * 28}}
        self.write_upstream()

    def write_upstream(self):
        (self.records / "baselines/raw/arxiv/AMP/upstream.json").write_text(json.dumps(self.upstream), encoding="utf-8")

    def test_plans_and_adapter_archives_are_copied_byte_for_byte(self):
        self.assertEqual(pa.derive_baseline_studies(self.records, self.checkout), ["ogbn-arxiv/AMP"])
        view = self.checkout / "experiments/baselines"
        index = json.loads((view / "INDEX.json").read_text(encoding="utf-8"))
        self.assertEqual(index["schema"], pa.BASELINE_SCHEMA)
        (study,) = index["studies"]
        self.assertEqual((study["dataset"], study["method"]), ("ogbn-arxiv", "AMP"))
        for key in ("plan", "upstream", "adapter_archive"):
            path = view / study[key]["file"]
            self.assertEqual(pa.sha256_file(path), study[key]["sha256"])
        self.assertEqual((view / study["adapter_archive"]["file"]).read_bytes(), self.archive.read_bytes())

    def test_a_failed_refresh_retains_the_previous_complete_view(self):
        pa.derive_baseline_studies(self.records, self.checkout)
        index = self.checkout / "experiments/baselines/INDEX.json"
        before = index.read_bytes()
        self.archive.write_bytes(b"changed")
        with self.assertRaises(pa.ArtifactError):
            pa.derive_baseline_studies(self.records, self.checkout)
        self.assertEqual(index.read_bytes(), before)

    def test_no_baselines_set_writes_nothing(self):
        self.assertIsNone(pa.derive_baseline_studies(self.tmp / "absent", self.checkout))
        self.assertFalse((self.checkout / "experiments").exists())

    def test_a_changed_archive_or_an_escaping_path_is_refused(self):
        self.archive.write_bytes(b"changed")
        with self.assertRaisesRegex(pa.ArtifactError, "differs from its recorded SHA-256"):
            pa.derive_baseline_studies(self.records, self.checkout)
        self.archive.write_bytes(bytes(1024))
        self.upstream["plan"] = "../../outside.json"
        (self.records / "outside.json").write_text("{}", encoding="utf-8")
        self.write_upstream()
        with self.assertRaisesRegex(pa.ArtifactError, "is not a file inside"):
            pa.derive_baseline_studies(self.records, self.checkout)


class PublicationWrapper(unittest.TestCase):
    """Failure propagation through the real shell wrapper with a local interpreter stand-in."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cm publication shell ")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root, self.stage = self.base / "checkout", self.base / "publication"
        (self.root / "hpc").mkdir(parents=True)
        shutil.copyfile(ROOT / "hpc/publish_paper_artifacts_arrhenius.sh", self.root / "hpc/publish.sh")
        interpreter = self.base / "fake-python"
        interpreter.write_text("#!" + sys.executable + "\n" +
            "import json,os,sys\n" +
            "with open(os.environ['FAKE_LOG'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n" +
            "args=sys.argv[1:]\n" +
            "code=int(os.environ.get('FAKE_FETCH_RC','0')) if 'fetch' in args else " +
            "int(os.environ.get('FAKE_RECONSTRUCT_RC','0')) if 'scripts/reconstruct_paper.py' in args else 0\n" +
            "raise SystemExit(code)\n")
        interpreter.chmod(0o755)
        assignments = ["CM_BASE=" + shlex.quote(str(self.base)), "CM_PYTHON=" + shlex.quote(str(interpreter)),
                       "CM_BUILD_MODULE=", "CM_PYTHON_MODULE="]
        (self.root / "run_cross_morse.sh").write_text("#!/bin/bash\nprintf '%s\\n' " +
                                                     " ".join(shlex.quote(line) for line in assignments) + "\n")
        (self.stage / "staged").mkdir(parents=True)
        (self.stage / "staged/fixture.entry.json").write_text("{}")
        (self.stage / "fetched").mkdir()
        (self.stage / "fetched/retain.txt").write_text("previous managed download")
        self.log = self.base / "calls.jsonl"

    def run_publish(self, *, fetch=0, reconstruct=0):
        environment = dict(os.environ, CM_PUBLISH_ROOT=str(self.stage), CM_PUBLISH_COMMIT="0",
                           GITHUB_TOKEN="local-test-token", FAKE_LOG=str(self.log),
                           FAKE_FETCH_RC=str(fetch), FAKE_RECONSTRUCT_RC=str(reconstruct))
        process = subprocess.run(["bash", str(self.root / "hpc/publish.sh"), "publish"], env=environment,
                                 capture_output=True, text=True, timeout=30)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        return process, calls

    def test_transfer_failure_stops_before_reconstruction_and_preserves_cache(self):
        result, calls = self.run_publish(fetch=2)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(any("scripts/reconstruct_paper.py" in call or "derive" in call for call in calls))
        self.assertTrue((self.stage / "fetched/retain.txt").is_file())
        fetch = next(call for call in calls if "fetch" in call)
        self.assertEqual(fetch[fetch.index("--tier") + 1], "complete")

    def test_generator_error_stops_before_derived_views(self):
        result, calls = self.run_publish(reconstruct=2)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(any("derive" in call for call in calls))

    def test_comparison_difference_is_retained_in_a_derived_report(self):
        result, calls = self.run_publish(reconstruct=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(any("derive" in call for call in calls))
        self.assertIn("differences or incomplete display coverage", result.stdout)


if __name__ == "__main__":
    unittest.main()
