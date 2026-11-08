"""The scientific-consistency ledger and its audit of original runs (NumPy only; no PyTorch)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import consistency_audit as ca  # noqa: E402
from tests.test_displays_run_audits import Fixture, lifecycle_files  # noqa: E402
from paper.build.displays.testing import fake_run  # noqa: E402

EXAMPLES = ROOT / "examples" / "runs"


def as_runs(views: dict, benchmark: bool = True) -> dict:
    out = {}
    for dataset, view in views.items():
        for member in view.members():
            config = json.loads((member.root / "config.json").read_text(encoding="utf-8"))
            out.setdefault(dataset, []).append({"path": member.root, "benchmark": benchmark, "config": config,
                                                "split": None})
    return out


class Ledger(unittest.TestCase):
    def test_ledger_is_well_formed_and_its_page_is_current(self):
        ledger = ca.load_ledger(ROOT)
        self.assertEqual([i["id"] for i in ledger["items"]], [f"K{k}" for k in range(1, 9)])
        checks = {i["check"] for i in ledger["items"]} - {None}
        self.assertEqual(checks, set(ca.CHECKS))
        self.assertEqual((ROOT / ca.DOC).read_text(encoding="utf-8"), ca.render(ledger) + "\n",
                         "docs/CONSISTENCY.md is not current: make consistency")

    def test_no_reference_value_is_carried_by_the_ledger(self):
        # The ledger states questions and rules; printed targets stay in paper/reported_results.json.
        text = (ROOT / ca.LEDGER).read_text(encoding="utf-8")
        self.assertNotIn("$value", text)


class SelectionFamily(unittest.TestCase):
    def test_bundled_runs_were_sealed_with_every_executed_configuration(self):
        report = ca.audit(EXAMPLES)
        k1 = next(i for i in report["items"] if i["id"] == "K1")
        self.assertEqual(k1["status"], "structure only")
        for row in k1["rows"]:
            self.assertIsNone(row["recorded_family"])
            self.assertEqual(row["reproduced_by"], ["executed"])
        # Treated as reported runs, records of the executed family call for a correction (ledger K1).
        runs = ca.discover(EXAMPLES)
        for members in runs.values():
            for member in members:
                member["benchmark"] = True
        self.assertEqual(ca.selection_family(runs, {})["status"], "needs correction")

    def test_runs_of_the_current_runtime_follow_algorithm_s4(self):
        fixture = Fixture(self)
        views = {d: v for d, v in fixture.runs().items() if d in ("ogbn-arxiv", "PascalVOC-SP")}
        result = ca.selection_family(as_runs(views), {})
        self.assertEqual(result["status"], "consistent")
        for row in result["rows"]:
            self.assertEqual(row["recorded_family"], "feasible")
            self.assertIn("feasible", row["reproduced_by"])


class Decisions(unittest.TestCase):
    def test_complete_ledger_rule_is_replayed_on_the_three_datasets(self):
        fixture = Fixture(self)
        views = fixture.runs()
        result = ca.complete_ledger_selection(as_runs(views), {})
        self.assertEqual({r["dataset"] for r in result["rows"]}, set(ca.LEDGER_DATASETS))
        self.assertEqual(len([r for r in result["rows"] if r["dataset"] == "Roman-empire"]), 2)
        self.assertTrue(all(r["agrees"] for r in result["rows"]))
        expected = ("consistent" if all(r["registered_rule"] == "MPC-RH" for r in result["rows"])
                    else "needs correction")
        self.assertEqual(result["status"], expected)

    def test_peptides_confirmation_against_the_frozen_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = lifecycle_files("Peptides-func", 7)
            candidates = files["select/candidates.json"]
            radius = next(j for j, c in enumerate(candidates) if c["route"] == "radius")
            comparator = 0

            def run(name, confirmed, holm, passes):
                files["confirm/decision.json"] = {
                    "provisional_index": radius, "confirmed_index": confirmed, "comparator_index": comparator,
                    "alpha": 0.05, "downgrades": [] if confirmed == radius else [{"from": radius, "to": comparator}],
                    "contrasts": [{"candidate_index": radius, "holm_p_value_zero": holm, "passes": passes}]}
                view = fake_run(Path(tmp) / name, "Peptides-func", files)
                return {"Peptides-func": [{"path": view.root, "benchmark": True, "config": files["config.json"],
                                           "split": None}]}

            confirmed = ca.peptides_confirmation(run("held", radius, 0.03, True), {})
            self.assertEqual((confirmed["status"], confirmed["rows"][0]["verdict"]), ("consistent", "consistent"))
            returned = ca.peptides_confirmation(run("returned", comparator, 0.070, False), {})
            self.assertEqual((returned["status"], returned["rows"][0]["verdict"]), ("needs correction", "returns qC"))
            overridden = ca.peptides_confirmation(run("overridden", radius, 0.070, False), {})
            self.assertEqual(overridden["rows"][0]["verdict"], "contradicts the map")

    def test_entry_gates_and_seed_hierarchy(self):
        fixture = Fixture(self)
        runs = as_runs({"ogbn-arxiv": fixture.runs()["ogbn-arxiv"]})
        self.assertEqual(ca.entry_gates(runs, {})["status"], "consistent")
        examples = ca.discover(EXAMPLES)
        hierarchy = ca.seed_hierarchy(examples, {})
        self.assertEqual(hierarchy["status"], "structure only")
        for row in hierarchy["rows"]:
            self.assertEqual(row["controller_fits_per_profile"], {"full": 1, "lean": 1})
            self.assertEqual(row["per_seed_controller_packages"], [])

    def test_command_line(self):
        self.assertEqual(ca.main(["--check"]), 0)
        self.assertEqual(ca.main(["--runs-root", str(EXAMPLES)]), 0)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "audit.json"
            ca.main(["--runs-root", str(EXAMPLES), "--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["benchmark_runs"], 0)
            self.assertEqual([i["id"] for i in report["items"]], [f"K{k}" for k in range(1, 9)])


if __name__ == "__main__":
    unittest.main()
