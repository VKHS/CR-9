"""The availability text printed in the manuscript is evaluated claim by claim.

The registry (release/printed_availability.json) binds the main, Methods and Supplementary
paragraphs to the PDF identities of paper/sources.json. These tests use the repository's own
contents and replace only the indexes or symbols a claim depends on. NumPy only; no PyTorch.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import printed_statement_check as psc  # noqa: E402
from scripts import statement_check as sc  # noqa: E402

REGISTRY = json.loads((ROOT / psc.REGISTRY).read_text(encoding="utf-8"))
DATASETS = psc.DATASETS


def with_json(overrides: dict):
    """Patch the checker's JSON reader for the given repository-relative paths."""
    original = psc._json

    def reader(path, default=None):
        relative = Path(path).resolve().relative_to(ROOT).as_posix() if Path(path).is_absolute() else str(path)
        if relative in overrides:
            return copy.deepcopy(overrides[relative])
        return original(path, default)
    return mock.patch.object(psc, "_json", reader)


class Registry(unittest.TestCase):
    def test_paragraphs_are_bound_to_the_registered_manuscript_identities(self):
        sources = json.loads((ROOT / "paper/sources.json").read_text(encoding="utf-8"))["documents"]
        for key, paragraph in REGISTRY["paragraphs"].items():
            self.assertEqual(paragraph["sha256"], sources[paragraph["document"]]["sha256"], key)
            self.assertTrue(paragraph["text"].strip())

    def test_claim_text_is_printed_text(self):
        for claim in REGISTRY["claims"]:
            paragraph = REGISTRY["paragraphs"][claim["paragraph"]]["text"]
            if "[...]" in claim["text"]:
                for piece in claim["text"].split("[...]"):
                    self.assertIn(piece.strip(), paragraph, claim["id"])
            elif claim["id"] not in ("M27",):
                self.assertIn(claim["text"], paragraph, claim["id"])

    def test_every_sentence_of_the_printed_paragraphs_is_covered(self):
        main = REGISTRY["paragraphs"]["main"]["text"]
        for phrase in ("make tables, make figures", "make fingerprint", "runs/index.json", "records/index.json",
                       "make paper", "make records-tables", "nine policy families", "eight-stage lifecycle"):
            self.assertIn(phrase, main)
            self.assertTrue(any(phrase in c["text"] for c in REGISTRY["claims"]), phrase)


class Evaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registered = sc.evaluate(ROOT)
        cls.evaluation = psc.evaluate(ROOT, registered=cls.registered)
        cls.status = {c["id"]: c["status"] for c in cls.evaluation["claims"]}

    def test_every_claim_is_evaluated_once(self):
        self.assertEqual([c["id"] for c in self.evaluation["claims"]], [c["id"] for c in REGISTRY["claims"]])
        self.assertEqual(sum(self.evaluation["counts"].values()), len(REGISTRY["claims"]))

    def test_code_contracts_and_commands_are_fulfilled_by_the_checkout(self):
        for claim in ("M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M10", "M11", "M12", "M13",
                      "M14", "M15", "M16", "M17", "M18", "M19", "M20", "M21", "M22", "M23", "M24", "M26", "R02",
                      "S02", "S07"):
            self.assertEqual(self.status[claim], psc.FULFILLED, claim)

    def test_published_outputs_await_publication(self):
        for claim in ("M25", "M29", "M30", "S05", "S06"):
            self.assertEqual(self.status[claim], psc.AWAITING, claim)

    def test_run_archive_dois_await_registration(self):
        self.assertEqual(self.status["M27"], psc.AWAITING)
        self.assertEqual(self.status["S03"], psc.AWAITING)

    def test_record_dois_without_a_contract_field_are_missing(self):
        self.assertEqual(self.status["M28"], psc.MISSING)
        self.assertEqual(self.status["S04"], psc.MISSING)

    def test_registered_zenodo_dois_fulfil_the_run_archive_claims(self):
        runs = {"schema": "cross-morse-runs-index-1",
                "runs": [{"dataset": d, "profile": "full", "scientific_run": True,
                          "deposit_doi": f"10.5281/zenodo.{1000 + i}"} for i, d in enumerate(DATASETS)]}
        with with_json({"runs/index.json": runs}):
            result = psc.evaluate(ROOT, registered=self.registered)
        status = {c["id"]: c["status"] for c in result["claims"]}
        self.assertEqual(status["M27"], psc.FULFILLED)
        self.assertEqual(status["S03"], psc.FULFILLED)

    def test_a_non_zenodo_doi_does_not_fulfil_the_zenodo_sentence(self):
        runs = {"runs": [{"dataset": d, "profile": "full", "scientific_run": True,
                          "deposit_doi": f"10.1234/example.{i}"} for i, d in enumerate(DATASETS)]}
        with with_json({"runs/index.json": runs}):
            result = psc.evaluate(ROOT, registered=self.registered)
        status = {c["id"]: c["status"] for c in result["claims"]}
        self.assertEqual(status["M27"], psc.AWAITING)
        self.assertEqual(status["S03"], psc.FULFILLED)

    def test_record_deposit_dois_fulfil_the_named_record_sets(self):
        sets = ["baselines", "budget_matched", "factorials", "negative_controls", "ensemble_grid", "inserted_site",
                "cross_backbone"]
        index = {"sets": [{"set": s, "deposit_doi": f"10.5281/zenodo.{2000 + i}"} for i, s in enumerate(sets)]}
        with with_json({"records/index.json": index}):
            result = psc.evaluate(ROOT, registered=self.registered)
        status = {c["id"]: c["status"] for c in result["claims"]}
        self.assertEqual(status["M28"], psc.FULFILLED)
        self.assertEqual(status["S04"], psc.AWAITING)  # the SI names every record set the displays read

    def test_an_undefined_symbol_makes_its_claim_missing(self):
        registry = copy.deepcopy(REGISTRY)
        claim = next(c for c in registry["claims"] if c["id"] == "M01")
        claim["check"]["symbols"].append("cross_morse/core.py::NoSuchPlant")
        result = psc.evaluate(ROOT, registry=registry, registered=self.registered)
        row = next(c for c in result["claims"] if c["id"] == "M01")
        self.assertEqual(row["status"], psc.MISSING)
        self.assertIn("NoSuchPlant", row["remaining"])
        s01 = next(c for c in result["claims"] if c["id"] == "S01")
        self.assertEqual(s01["status"], psc.MISSING)  # the SI sentence depends on M01

    def test_a_different_policy_family_set_is_missing(self):
        registry = copy.deepcopy(REGISTRY)
        claim = next(c for c in registry["claims"] if c["id"] == "M07")
        claim["check"]["feedback"] = ["one_step", "radius"]
        result = psc.evaluate(ROOT, registry=registry, registered=self.registered)
        self.assertEqual(next(c for c in result["claims"] if c["id"] == "M07")["status"], psc.MISSING)

    def test_same_as_and_all_of_follow_their_references(self):
        self.assertEqual(self.status["S02"], self.status["M23"])
        self.assertEqual(self.status["S05"], self.status["M29"])
        self.assertEqual(self.status["S06"], self.status["M30"])
        self.assertEqual(self.status["S07"], self.status["R02"])

    def test_circular_references_are_refused(self):
        registry = copy.deepcopy(REGISTRY)
        registry["claims"].append({"id": "X1", "paragraph": "main", "text": "x", "check": {"type": "same_as", "claim": "X2"}})
        registry["claims"].append({"id": "X2", "paragraph": "main", "text": "x", "check": {"type": "same_as", "claim": "X1"}})
        with self.assertRaises(psc.CheckError):
            psc.evaluate(ROOT, registry=registry, registered=self.registered)


class Report(unittest.TestCase):
    def test_the_compliance_page_carries_both_texts_and_is_current(self):
        registered = sc.evaluate(ROOT)
        printed = psc.evaluate(ROOT, registered=registered)
        text = sc.render(registered, printed)
        for claim in printed["claims"]:
            self.assertIn(f"| {claim['id']} |", text)
        self.assertIn("## The availability text printed in the manuscript", text)
        self.assertEqual((ROOT / sc.REPORT).read_text(encoding="utf-8"), text,
                         "STATEMENT_COMPLIANCE.md is not current: python -B scripts/statement_check.py --write")

    def test_rendering_without_the_printed_text_keeps_the_registered_report(self):
        text = sc.render(sc.evaluate(ROOT))
        self.assertNotIn("printed in the manuscript", text)
        self.assertIn("| figure/table generators |", text)


if __name__ == "__main__":
    unittest.main()
