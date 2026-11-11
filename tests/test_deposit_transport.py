"""Offline regression tests for durable deposit and DOI-to-byte verification."""
from __future__ import annotations
from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import deposit_zenodo as deposit, fetch_deposits as fetch
from scripts.archive_run import pack, STAGES


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def row(root, rel):
    data = (root / rel).read_bytes()
    return {"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def make_run(root, dataset="Peptides-func", split=0, unsealed=False):
    config = {"benchmark_dataset": dataset, "dataset": "synthetic", "profile": "smoke",
              "scientific_run": False, "data": {"split_index": split}}
    dump(root / "config.json", config)
    dump(root / "prepared/metadata.json", {"benchmark_dataset": dataset, "is_benchmark_data": False})
    (root / "prepared/splits.npz").write_bytes(b"fixture split bytes")
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    ids = {"source_sha256": "a" * 64, "config_sha256": config_sha,
           "unit_sha256": row(root, "prepared/splits.npz")["sha256"]}
    dump(root / "run_state.json", {"status": "COMPLETE", "last_complete_stage": "evaluate", "dataset": "synthetic",
                                  "task": {"dataset": dataset}, "profile": "smoke", "scientific_run": False, **ids})
    parent = None
    for i, stage in enumerate(STAGES):
        outputs = [row(root, name) for name in ("config.json", "prepared/metadata.json", "prepared/splits.npz")] if i == 0 else []
        inputs = [{"path": "unsealed", "bytes": 0, "sha256": "b" * 64}] if unsealed and i == 1 else []
        rel = f"provenance/stages/{i:02d}_{stage}.json"
        dump(root / rel, {"stage": stage, "stage_index": i, **ids, "parent_manifest": parent,
                          "inputs": inputs, "outputs": outputs})
        parent = row(root, rel)


def make_archive(root):
    source = root / "source"
    make_run(source)
    archive = root / "peptides_func.tar.gz"
    manifest = pack(source, archive, campaign=False)
    return archive, manifest


class Response(io.BytesIO):
    def __init__(self, payload, url, status=200):
        super().__init__(json.dumps(payload).encode())
        self.url, self.status = url, status

    def geturl(self):
        return self.url


class FakeZenodo:
    """Stateful service: a create really creates a draft, including on lost responses."""
    def __init__(self, base=deposit.PRODUCTION):
        self.base, self.calls, self.drafts = base, [], {}
        self.pending, self.lose_create, self.bad_upload = False, False, None
        self.bad_public = False

    def __call__(self, request, timeout):
        method, url = request.get_method(), request.full_url
        self.calls.append((method, url, request.get_header("Authorization")))
        if method == "POST" and url == self.base + "/deposit/depositions":
            ident = 4242 + len(self.drafts)
            self.drafts[ident] = {"id": ident, "submitted": False, "state": "unsubmitted", "files": [],
                                  "links": {"bucket": self.base + f"/files/bucket-{ident}"}, "metadata": {}}
            if self.lose_create:
                self.lose_create = False
                raise OSError("response lost after server creates draft")
            return Response(self.drafts[ident], url, 201)
        if "/files/bucket-" in url:
            ident = int(url.split("/files/bucket-")[1].split("/")[0])
            data = bytearray()
            while True:
                chunk = request.data.read(65536)
                if not chunk:
                    break
                data.extend(chunk)
            from urllib.parse import unquote
            asset = {"key": unquote(url.rsplit("/", 1)[1]), "size": len(data), "checksum": "md5:" + hashlib.md5(data).hexdigest()}
            self.drafts[ident]["files"] = [asset]
            result = dict(asset)
            if self.bad_upload == "checksum":
                result["checksum"] = "md5:" + "0" * 32
            elif self.bad_upload:
                result.pop(self.bad_upload, None)
            return Response(result, url, 200)
        if "/records/" in url:
            if request.get_header("Authorization"):
                raise AssertionError("public verification must be anonymous")
            ident = int(url.rsplit("/", 1)[1])
            current = self.drafts[ident]
            result = {"id": ident, "doi": self.doi(ident), "is_published": current["submitted"],
                      "metadata": current["metadata"], "files": copy.deepcopy(current["files"])}
            if self.bad_public:
                result["files"][0]["checksum"] = "md5:" + "1" * 32
            return Response(result, url)
        ident = int(url.split("/deposit/depositions/")[1].split("/")[0])
        current = self.drafts[ident]
        if method == "PUT":
            current["metadata"] = json.loads(request.data)["metadata"]
            current["metadata"]["prereserve_doi"] = {"doi": self.doi(ident), "recid": ident}
        elif method == "POST" and url.endswith("/actions/publish"):
            if not self.pending:
                current.update(submitted=True, state="done", doi=self.doi(ident))
            return Response(current, url, 202)
        return Response(current, url)

    def doi(self, ident):
        return f"{'10.5072' if self.base == deposit.SANDBOX else '10.5281'}/zenodo.{ident}"

    def creates(self):
        return sum(method == "POST" and url.endswith("/deposit/depositions") for method, url, _ in self.calls)

    def publications(self):
        return sum(method == "POST" and url.endswith("/actions/publish") for method, url, _ in self.calls)


class DepositTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.file = self.root / "sample.tar.gz"
        self.file.write_bytes(b"test bytes" * 20000)
        self.metadata = {"title": "Fixture", "description": "Offline test", "publication_date": "2026-11-03",
                         "license": "cc-by-4.0", "prereserve_doi": True}
        self.fake = FakeZenodo()
        self.transport = deposit.Transport(deposit.PRODUCTION, "TEST_ONLY", opener=self.fake)

    def invoke(self, **kwargs):
        return deposit.deposit_file(self.transport, self.file, self.metadata, publish=kwargs.pop("publish", False), **kwargs)

    def test_draft_publish_resume_is_one_deposition_and_one_upload(self):
        draft = self.invoke()
        self.assertIsNone(draft["doi"])
        self.assertEqual(draft["reserved_doi"], "10.5281/zenodo.4242")
        self.assertFalse(draft["published"])
        published = self.invoke(publish=True)
        self.assertTrue(published["published"])
        self.assertEqual(published["doi"], draft["reserved_doi"])
        self.invoke(publish=True)
        self.assertEqual(self.fake.creates(), 1)
        self.assertEqual(self.fake.publications(), 1)
        self.assertEqual(sum("/files/bucket-" in url for _, url, _ in self.fake.calls), 1)
        self.assertNotIn("TEST_ONLY", deposit.receipt_path(self.file, deposit.PRODUCTION).read_text())

    def test_receipt_saved_before_upload_failure_can_resume_same_draft(self):
        for field in ("checksum", "size", "key"):
            with self.subTest(field=field):
                path = self.root / (field + ".tar.gz")
                path.write_bytes(b"known bytes")
                fake = FakeZenodo()
                fake.bad_upload = field
                client = deposit.Transport(deposit.PRODUCTION, "TEST_ONLY", opener=fake)
                with self.assertRaises(deposit.DepositError):
                    deposit.deposit_file(client, path, self.metadata, publish=False)
                receipt = json.loads(deposit.receipt_path(path, deposit.PRODUCTION).read_text())
                self.assertEqual(receipt["deposition_id"], 4242)
                fake.bad_upload = None
                deposit.deposit_file(client, path, self.metadata, publish=False)
                self.assertEqual(fake.creates(), 1)

    def test_uncertain_creation_requires_recovery_without_duplicate_post(self):
        self.fake.lose_create = True
        with self.assertRaises(OSError):
            self.invoke()
        with self.assertRaisesRegex(deposit.DepositError, "unknown"):
            self.invoke()
        self.assertEqual(self.fake.creates(), 1)
        result = self.invoke(deposition_id=4242)
        self.assertEqual(result["status"], "draft")
        self.assertEqual(self.fake.creates(), 1)

    def test_accepted_publication_is_not_published_until_verified(self):
        self.fake.pending = True
        pending = self.invoke(publish=True)
        self.assertEqual(pending["status"], "publishing")
        self.assertIsNone(pending["doi"])
        self.invoke(publish=True)
        self.assertEqual(self.fake.publications(), 1)
        self.fake.drafts[4242].update(submitted=True, state="done", doi=self.fake.doi(4242))
        self.assertTrue(self.invoke(publish=True)["published"])
        self.assertEqual(self.fake.publications(), 1)

    def test_service_isolation_and_untrusted_bucket_are_blocked(self):
        self.invoke()
        sandbox = FakeZenodo(deposit.SANDBOX)
        client = deposit.Transport(deposit.SANDBOX, "TEST_SANDBOX", opener=sandbox)
        result = deposit.deposit_file(client, self.file, self.metadata, publish=False)
        self.assertTrue(result["reserved_doi"].startswith("10.5072/"))
        self.assertNotEqual(deposit.receipt_path(self.file, deposit.PRODUCTION), deposit.receipt_path(self.file, deposit.SANDBOX))
        count = len(sandbox.calls)
        for url in ("https://zenodo.org/api/files/bucket-4242", "http://sandbox.zenodo.org/api/files/x", "https://evil.test/api/files/x"):
            with self.subTest(url=url), self.assertRaises(deposit.DepositError):
                client.upload(url, self.file)
        self.assertEqual(len(sandbox.calls), count)
        with self.assertRaisesRegex(deposit.DepositError, "different Zenodo"):
            deposit.deposit_file(client, self.file, self.metadata, publish=False, receipt=deposit.receipt_path(self.file, deposit.PRODUCTION))

    def test_redirects_and_bad_public_checksums_do_not_pass(self):
        with self.assertRaises(deposit.DepositError):
            deposit._NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://evil.test")
        self.fake.bad_public = True
        with self.assertRaisesRegex(deposit.DepositError, "checksum"):
            self.invoke(publish=True)
        saved = json.loads(deposit.receipt_path(self.file, deposit.PRODUCTION).read_text())
        self.assertFalse(saved["published"])
        self.assertIsNone(saved["doi"])

    def test_changed_metadata_or_file_refuses_resume(self):
        self.invoke()
        self.metadata["title"] = "Different material"
        with self.assertRaisesRegex(deposit.DepositError, "Metadata differs"):
            self.invoke()
        self.metadata["title"] = "Fixture"
        self.file.write_bytes(b"replacement")
        with self.assertRaisesRegex(deposit.DepositError, "Local file differs"):
            self.invoke()
        self.assertEqual(self.fake.creates(), 1)

    def test_archive_version_and_resume_are_stable_across_checkout_and_date(self):
        info = {"format": "run-archive", "kind": "run", "dataset": "Peptides-func", "profile": "smoke",
                "scientific_run": False, "files": 1, "run_source_sha256": "a" * 64}
        first = deposit.build_metadata("archive", archive=self.file, info=info, commit="a" * 40, license_id="cc-by-4.0")
        second = deposit.build_metadata("archive", archive=self.file, info=info, commit="b" * 40, license_id="cc-by-4.0")
        self.assertEqual(first["version"], "sha256-" + deposit.sha256_file(self.file))
        self.assertEqual(first, second)
        self.metadata = first
        self.invoke()
        self.metadata = dict(second, publication_date="2030-01-01")
        result = self.invoke(publish=True)
        self.assertEqual(result["metadata"]["publication_date"], first["publication_date"])
        self.assertEqual(self.fake.creates(), 1)
        code = deposit.build_metadata("code", archive=self.file, info={}, commit="b" * 40)
        self.assertEqual(code["version"], "b" * 40)

    def test_legacy_filesize_string_is_exact_and_modern_size_remains_typed(self):
        identity = deposit.file_identity(self.file)
        old = {"name": identity["filename"], "filesize": str(identity["bytes"]), "checksum": identity["md5"]}
        deposit._asset_matches(old, identity)
        for value in (True, float(identity["bytes"]), None, "", " " + str(identity["bytes"]), "1e9", "-1"):
            with self.subTest(value=value), self.assertRaises(deposit.DepositError):
                deposit._asset_matches(dict(old, filesize=value), identity)
        with self.assertRaises(deposit.DepositError):
            deposit._asset_matches({"key": identity["filename"], "size": str(identity["bytes"]), "checksum": "md5:" + identity["md5"]}, identity)
        with self.assertRaises(deposit.DepositError):
            deposit._asset_matches(dict(old, size=identity["bytes"] + 1), identity)

    def test_api_normalized_doi_matches_without_weakening_related_identity(self):
        expected = {"related_identifiers": [{"identifier": "https://doi.org/10.5281/zenodo.9",
                    "relation": "isSupplementTo", "resource_type": "software"}]}
        normalized = {"related_identifiers": [{"identifier": "10.5281/zenodo.9", "scheme": "doi",
                      "relation": "isSupplementTo", "resource_type": "software"}]}
        deposit._metadata_matches(normalized, expected)
        for key, value in (("identifier", "10.5281/zenodo.10"), ("relation", "isDerivedFrom"),
                           ("resource_type", "dataset")):
            changed = copy.deepcopy(normalized)
            changed["related_identifiers"][0][key] = value
            with self.subTest(field=key), self.assertRaises(deposit.DepositError):
                deposit._metadata_matches(changed, expected)
        expected_url = {"related_identifiers": [{"identifier": "https://github.com/VKHS/CROSS-MORSE", "relation": "isSupplementTo"}]}
        changed_url = {"related_identifiers": [{"identifier": "http://github.com/VKHS/CROSS-MORSE", "relation": "isSupplementTo"}]}
        with self.assertRaises(deposit.DepositError):
            deposit._metadata_matches(changed_url, expected_url)
        metadata = deposit.build_metadata("code", archive=self.file, info={}, commit="b" * 40,
                                          code_doi="10.5281/zenodo.9")
        self.assertEqual(metadata["related_identifiers"][-1]["identifier"], "10.5281/zenodo.9")

    def test_explicit_data_license_and_offline_plan(self):
        archive, _ = make_archive(self.root)
        info = deposit.describe_archive(archive)
        metadata = deposit.build_metadata("archive", archive=archive, info=info)
        self.assertNotIn("license", metadata)
        self.assertIn("software validation", metadata["description"])
        with self.assertRaisesRegex(deposit.DepositError, "license"):
            deposit.deposit_file(self.transport, archive, metadata, publish=False)
        with patch.dict("os.environ", {}, clear=True), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(deposit.main(["plan", str(archive)]), 0)
        self.assertEqual(json.loads(output.getvalue())["required_before_deposit"], ["--license"])
        self.assertEqual(self.fake.creates(), 0)


class ArchiveInspectionTests(unittest.TestCase):
    def test_sealed_archive_and_broken_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, _ = make_archive(root)
            self.assertEqual(deposit.describe_archive(archive)["dataset"], "Peptides-func")
            bad = root / "bad"
            make_run(bad, unsealed=True)
            bad_archive = root / "bad.tar.gz"
            pack(bad, bad_archive, campaign=False)
            with self.assertRaisesRegex(deposit.DepositError, "Unsealed"):
                deposit.describe_archive(bad_archive)

    def test_campaign_requires_exact_ten_real_split_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / "campaign"
            for i in range(9):
                make_run(campaign / f"split_{i:02d}", dataset="Roman-empire", split=i)
            bad = root / "partial.tar.gz"
            pack(campaign, bad, campaign=True)
            with self.assertRaisesRegex(deposit.DepositError, "ten splits"):
                deposit.describe_archive(bad)
            make_run(campaign / "split_09", dataset="Roman-empire", split=9)
            complete = root / "complete.tar.gz"
            pack(campaign, complete, campaign=True)
            self.assertEqual(deposit.describe_archive(complete)["kind"], "campaign")

    def test_code_snapshot_requires_clean_git_and_reuses_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            (repo / "README.md").write_text("source snapshot\n")
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], check=True)
            one, commit = deposit.code_snapshot(repo, root / "deposits")
            two, again = deposit.code_snapshot(repo, root / "deposits")
            self.assertEqual((one, commit), (two, again))
            (repo / "README.md").write_text("dirty\n")
            with self.assertRaisesRegex(deposit.DepositError, "clean"):
                deposit.code_snapshot(repo, root / "deposits")


class FakeDownloads:
    def __init__(self, archive):
        self.archive, self.downloads, self.metadata_calls = archive, 0, 0
        self.corrupt = False
        identity = deposit.file_identity(archive)
        self.metadata = {"id": 4242, "doi": "10.5281/zenodo.4242", "is_published": True,
                         "metadata": {"publication_date": "2026-11-03"},
                         "files": [{"key": archive.name, "size": identity["bytes"], "checksum": "md5:" + identity["md5"]}]}

    def get_metadata(self, doi):
        self.metadata_calls += 1
        return copy.deepcopy(self.metadata)

    def download(self, binding, target):
        self.downloads += 1
        if self.corrupt:
            target.write_bytes(b"corrupt download")
        else:
            shutil.copyfile(self.archive, target)


class FetchTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.archive, manifest = make_archive(self.root)
        self.runs = self.root / "runs"
        self.index = self.runs / "index.json"
        self.entry = {"dataset": "Peptides-func", "stem": "peptides_func", "path": "peptides_func", "kind": "run",
                      "profile": "smoke", "scientific_run": False, "archive_name": self.archive.name,
                      "archive_sha256": manifest["archive_sha256"], "archive_bytes": manifest["archive_bytes"],
                      "run_source_sha256": [manifest["source_sha256"]], "files": len(manifest["files"]),
                      "deposit_doi": "10.5281/zenodo.4242"}
        self.save_index()
        self.tools = FakeDownloads(self.archive)

    def save_index(self):
        dump(self.index, {"schema": "cross-morse-runs-index-1", "runs": [self.entry]})

    def run_fetch(self, **kwargs):
        options = dict(index_path=self.index, runs_root=self.runs, deposits_dir=self.root / "cache", datasets={"Peptides-func"},
                       local=[], status_only=False, force=False, tools=self.tools)
        options.update(kwargs)
        return fetch.run(**options)

    def test_fetch_and_reuse_verify_doi_every_time_without_rewriting_index(self):
        original = self.index.read_bytes()
        first = self.run_fetch()
        self.assertEqual(first["entries"][0]["status"], "INGESTED")
        self.assertTrue(first["public_verification"]["complete"])
        second = self.run_fetch()
        self.assertEqual(second["entries"][0]["status"], "PRESENT")
        self.assertEqual(self.tools.downloads, 1)
        self.assertEqual(self.tools.metadata_calls, 2)
        self.assertEqual(self.index.read_bytes(), original)

    def test_local_copy_still_requires_published_checksum(self):
        self.tools.metadata["files"][0]["checksum"] = "md5:" + "0" * 32
        report = self.run_fetch(local=[self.root])
        self.assertEqual(report["failed"], 1)
        self.assertFalse((self.runs / "peptides_func").exists())
        self.assertEqual(self.tools.downloads, 0)

    def test_corrupt_cache_and_existing_run_are_preserved_until_verified_repair(self):
        self.run_fetch()
        altered = self.runs / "peptides_func/config.json"
        altered.write_text("altered local file")
        report = self.run_fetch()
        self.assertEqual(report["failed"], 1)
        self.assertEqual(altered.read_text(), "altered local file")
        repaired = self.run_fetch(force=True)
        self.assertEqual(repaired["entries"][0]["status"], "INGESTED")
        cached = self.root / "cache" / self.entry["archive_sha256"] / self.archive.name
        cached.write_bytes(b"preserve conflicting cache")
        report = self.run_fetch(force=True)
        self.assertEqual(report["failed"], 1)
        self.assertEqual(cached.read_bytes(), b"preserve conflicting cache")

    def test_corrupt_partial_download_is_cleaned_without_touching_index(self):
        self.tools.corrupt = True
        report = self.run_fetch()
        self.assertEqual(report["failed"], 1)
        cache = self.root / "cache" / self.entry["archive_sha256"]
        self.assertFalse((cache / self.archive.name).exists())
        self.assertFalse(list(cache.glob(".cm-fetch-*")))

    def test_index_path_injection_and_source_substitution_fail(self):
        self.entry["path"] = "../elsewhere"
        self.save_index()
        with self.assertRaisesRegex(fetch.FetchError, "canonical"):
            self.run_fetch()
        self.entry["path"] = "peptides_func"
        self.entry["run_source_sha256"] = ["f" * 64]
        self.save_index()
        self.assertEqual(self.run_fetch()["failed"], 1)
        self.assertFalse((self.runs / "peptides_func").exists())

    def test_status_missing_doi_and_empty_index_cannot_claim_complete(self):
        report = self.run_fetch(status_only=True)
        self.assertEqual(report["entries"][0]["status"], "REGISTERED")
        self.assertFalse(report["public_verification"]["complete"])
        self.assertEqual(self.tools.metadata_calls, 0)
        self.entry["deposit_doi"] = None
        self.save_index()
        report = self.run_fetch(local=[self.root])
        self.assertEqual(report["missing_doi"], 1)
        self.assertFalse(report["public_verification"]["complete"])
        dump(self.index, {"schema": "cross-morse-runs-index-1", "runs": []})
        report = self.run_fetch(datasets=None)
        self.assertEqual(set(report["missing_datasets"]), set(deposit.DATASETS))
        self.assertFalse(report["public_verification"]["complete"])


if __name__ == "__main__":
    unittest.main()
