"""Availability statements distinguish unavailable, partial, and registered evidence."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.availability import DATASETS, END, NOT_BUILT, START, check, main, records_sentence, render, update_readme


def fixture_state(official=(), *, doi=True):
    official = list(official)
    return {
        "scope": {"repository_url": "https://example.org/cross-morse", "code_archive_doi": None,
                  "original_result_archives_included": False,
                  "one_command_original_tables_reproduction_claimed": False,
                  "full_official_dataset_runs_verified": sorted(official)},
        "source_sha256": "a" * 64,
        "official": {
            name: {"deposit_doi": "10.5281/zenodo.123" if doi else None}
            for name in official
        },
        "missing_official": [name for name in DATASETS if name not in official],
        "examples": list(DATASETS),
        "example_records": ["nominal_time"],
        "registered_manifests": [],
        "registered_pilot_manifests": [],
        "campaigns": True,
        "tables_examples": True,
        "tables_official": bool(official),
        "records": {
            "defined": ["baselines", "budget_matched"],
            "verified": [],
            "tables": {
                "baselines": ["Supplementary Table S15b"],
                "budget_matched": ["Supplementary Table S46b"],
            },
        },
    }


class EvidenceAvailabilityTests(unittest.TestCase):
    def test_zero_archives_and_records_do_not_assert_deposits(self):
        text = render(fixture_state())
        self.assertNotIn("0 of 7", text)
        self.assertNotIn("0 of 2", text)
        self.assertIn("original benchmark archives can be supplied through the separate registration workflow", text)
        self.assertNotIn("Registered DOI references:", text)
        self.assertNotIn("https://doi.org/", text)
        self.assertIn("do not supply official benchmark measurements", text)

    def test_partial_archives_name_registered_datasets_and_limit_reconstruction(self):
        text = render(fixture_state(["ogbn-arxiv"]))
        self.assertNotIn("1 of 7", text)
        self.assertIn("Registered full-profile dataset archives: ogbn-arxiv.", text)
        self.assertNotIn("Registered full-profile dataset archives: ogbn-proteins", text)
        self.assertIn("for the registered datasets", text)
        self.assertIn("Agreement with the manuscript requires validated result bindings", text)
        self.assertIn("Registered DOI references: ogbn-arxiv:", text)
        self.assertIn("https://doi.org/10.5281/zenodo.123", text)

    def test_all_registered_archives_without_dois_do_not_claim_public_deposits(self):
        text = render(fixture_state(DATASETS, doi=False))
        self.assertNotIn("7 of 7", text)
        self.assertIn("Registered full-profile dataset archives:", text)
        self.assertIn("No DOI references are recorded", text)
        self.assertNotIn("Registered DOI references:", text)
        self.assertNotIn("https://doi.org/", text)
        self.assertIn("validated result bindings and the scientific evidence audit", text)

    def test_partial_dois_are_not_described_as_covering_every_archive(self):
        st = fixture_state(DATASETS, doi=False)
        st["official"]["ogbn-arxiv"]["deposit_doi"] = "10.5281/zenodo.123"
        text = render(st)
        self.assertIn("Registered DOI references: ogbn-arxiv: https://doi.org/10.5281/zenodo.123.", text)
        self.assertIn("Archives with no registered DOI reference: COCO-SP", text)
        self.assertNotIn("Registered DOI references: 7 of 7", text)

    def test_complete_registered_deposits_keep_a_positive_accurate_statement(self):
        st = fixture_state(DATASETS)
        st["records"]["verified"] = list(st["records"]["defined"])
        text = render(st)
        self.assertIn("Registered DOI references: COCO-SP:", text)
        self.assertNotIn("Archives with no registered DOI reference:", text)
        self.assertIn("https://doi.org/10.5281/zenodo.123", text)
        self.assertIn("All defined sets are registered.", text)
        self.assertNotIn("Missing official archives:", text)
        self.assertNotIn("Missing: baselines", text)
        self.assertIn("Synthetic examples exercise the software workflow", text)

    def test_partial_measurement_records_do_not_claim_missing_tables(self):
        st = fixture_state()
        st["records"]["verified"] = ["baselines"]
        text = records_sentence(st)
        self.assertNotIn("1 of 2", text)
        self.assertIn("Registered auxiliary measurement records: baselines.", text)
        self.assertIn("Other defined record contracts: budget_matched.", text)
        self.assertNotIn("All defined sets are registered.", text)
        self.assertIn("registered auxiliary sets", render(st))

    def test_figure_statement_labels_transcribed_inputs_and_limits_replay_scope(self):
        self.assertIn("transcribed table values and analytical scenarios", NOT_BUILT)
        self.assertIn("not a replay of benchmark observations", NOT_BUILT)
        self.assertNotIn("regenerate every figure", NOT_BUILT)
        text = render(fixture_state())
        self.assertIn("Paper PDFs are not distributed", text)
        self.assertIn("hashes and page references", text)
        self.assertNotIn("docs/EVIDENCE.md", text)
        self.assertNotIn("superseded", text)

    def test_code_archive_doi_is_distinct_from_missing_result_deposits(self):
        st = fixture_state()
        st["scope"]["code_archive_doi"] = "10.5281/zenodo.456"
        text = render(st)
        self.assertIn("registered code-archive DOI is https://doi.org/10.5281/zenodo.456", text)
        self.assertIn("original benchmark archives can be supplied through the separate registration workflow", text)
        self.assertNotIn("Registered DOI references:", text)


class ReproductionClaimTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_statement(self, st):
        (self.root / "AVAILABILITY.md").write_text(render(st))
        (self.root / "README.md").write_text(f"# Test\n\n{START}\nstale\n{END}\n")
        update_readme(self.root, st)

    def test_all_registered_archives_allow_conservative_false_flags_without_scientific_audit(self):
        st = fixture_state(DATASETS)
        self.write_statement(st)
        with patch("scripts.availability._reproduction_evidence") as audit:
            self.assertEqual(check(self.root, st), 0)
            audit.assert_not_called()

    def test_complete_index_is_insufficient_for_affirmative_reproduction(self):
        st = fixture_state(DATASETS)
        st["scope"]["one_command_original_tables_reproduction_claimed"] = True
        self.write_statement(st)
        with patch("scripts.availability._reproduction_evidence", return_value={
            "publication_ready": False, "blockers": [{"code": "missing_auxiliary_set"},
                                                       {"code": "manuscript_numbers_unlinked"}],
        }) as audit:
            self.assertEqual(check(self.root, st), 1)
            audit.assert_called_once_with(self.root, None, None)

    def test_affirmative_reproduction_fails_closed_if_audit_cannot_run(self):
        st = fixture_state(DATASETS)
        st["scope"]["one_command_original_tables_reproduction_claimed"] = True
        self.write_statement(st)
        with patch("scripts.availability._reproduction_evidence", side_effect=ImportError("missing audit dependency")):
            self.assertEqual(check(self.root, st), 1)

    def test_affirmative_reproduction_requires_positive_complete_audit(self):
        st = fixture_state(DATASETS)
        st["scope"]["one_command_original_tables_reproduction_claimed"] = True
        self.write_statement(st)
        with patch("scripts.availability._reproduction_evidence", return_value={"publication_ready": True}) as audit:
            self.assertEqual(check(self.root, st), 0)
            audit.assert_called_once_with(self.root, None, None)

    def test_affirmative_reproduction_rejects_missing_archives_before_audit(self):
        st = fixture_state(["ogbn-arxiv"])
        st["scope"]["one_command_original_tables_reproduction_claimed"] = True
        self.write_statement(st)
        with patch("scripts.availability._reproduction_evidence") as audit:
            self.assertEqual(check(self.root, st), 1)
            audit.assert_not_called()

    def test_registered_budget_campaign_is_passed_to_complete_audit(self):
        st = fixture_state(DATASETS)
        st["scope"].update(one_command_original_tables_reproduction_claimed=True,
                           scientific_evidence_budget_campaign="runs/budget_campaign")
        self.write_statement(st)
        with patch("scripts.availability._reproduction_evidence", return_value={"publication_ready": True}) as audit:
            self.assertEqual(check(self.root, st), 0)
            audit.assert_called_once_with(self.root, self.root / "runs/budget_campaign", None)

    def test_registered_confirmation_and_budget_campaigns_reach_scientific_audit(self):
        st = fixture_state(DATASETS)
        st["scope"].update(one_command_original_tables_reproduction_claimed=True,
                           scientific_evidence_budget_campaign="runs/budget_campaign",
                           scientific_evidence_confirmation_campaign="runs/s14_campaign")
        self.write_statement(st)
        with patch("scripts.scientific_evidence_audit.build_report", return_value={"publication_ready": True}) as audit:
            self.assertEqual(check(self.root, st), 0)
            audit.assert_called_once_with(self.root, budget_campaign=self.root / "runs/budget_campaign",
                                           confirmation_campaign=self.root / "runs/s14_campaign")

    def test_missing_confirmation_evidence_cannot_support_an_affirmative_claim(self):
        st = fixture_state(DATASETS)
        st["scope"].update(one_command_original_tables_reproduction_claimed=True,
                           scientific_evidence_budget_campaign="runs/budget_campaign")
        self.write_statement(st)
        with patch("scripts.scientific_evidence_audit.build_report", return_value={
                "publication_ready": False, "confirmation_family": {"status": "PENDING"},
                "blockers": [{"code": "confirmation_campaign_missing"}]}) as audit:
            self.assertEqual(check(self.root, st), 1)
            audit.assert_called_once_with(self.root, budget_campaign=self.root / "runs/budget_campaign",
                                           confirmation_campaign=None)

    def test_cli_confirmation_campaign_overrides_registered_path(self):
        st = fixture_state(DATASETS)
        st["scope"].update(one_command_original_tables_reproduction_claimed=True,
                           scientific_evidence_budget_campaign="runs/budget_campaign",
                           scientific_evidence_confirmation_campaign="runs/registered_s14")
        self.write_statement(st)
        with tempfile.TemporaryDirectory() as external:
            supplied = Path(external) / "s14"
            with patch("scripts.availability.state", return_value=st), patch(
                    "scripts.scientific_evidence_audit.build_report", return_value={"publication_ready": True}) as audit:
                self.assertEqual(main(["--root", str(self.root), "--check", "--confirmation-campaign", str(supplied)]), 0)
                audit.assert_called_once_with(self.root, budget_campaign=self.root / "runs/budget_campaign",
                                               confirmation_campaign=supplied)

    def test_campaign_registration_is_only_read_for_an_affirmative_reconstruction_claim(self):
        st = fixture_state(DATASETS)
        st["scope"].update(scientific_evidence_budget_campaign="../unused_budget",
                           scientific_evidence_confirmation_campaign="../unused_confirmation")
        self.write_statement(st)
        with patch("scripts.availability._reproduction_evidence") as audit:
            self.assertEqual(check(self.root, st), 0)
            audit.assert_not_called()

    def test_unsafe_confirmation_campaigns_fail_before_scientific_audit(self):
        for relative in ("../outside", "/outside", "runs/../outside", "", False):
            with self.subTest(path=relative):
                st = fixture_state(DATASETS)
                st["scope"].update(one_command_original_tables_reproduction_claimed=True,
                                   scientific_evidence_confirmation_campaign=relative)
                self.write_statement(st)
                with patch("scripts.availability._reproduction_evidence") as audit:
                    self.assertEqual(check(self.root, st), 1)
                    audit.assert_not_called()

    def test_registered_campaign_symlink_escapes_are_refused_for_both_kinds(self):
        with tempfile.TemporaryDirectory() as external:
            (self.root / "outside_link").symlink_to(external, target_is_directory=True)
            for key in ("scientific_evidence_budget_campaign", "scientific_evidence_confirmation_campaign"):
                with self.subTest(key=key):
                    st = fixture_state(DATASETS)
                    st["scope"].update(one_command_original_tables_reproduction_claimed=True, **{key: "outside_link/campaign"})
                    self.write_statement(st)
                    with patch("scripts.availability._reproduction_evidence") as audit:
                        self.assertEqual(check(self.root, st), 1)
                        audit.assert_not_called()

    def test_unsafe_registered_campaign_is_rejected(self):
        st = fixture_state(DATASETS)
        st["scope"].update(one_command_original_tables_reproduction_claimed=True,
                           scientific_evidence_budget_campaign="../outside")
        self.write_statement(st)
        with patch("scripts.availability._reproduction_evidence") as audit:
            self.assertEqual(check(self.root, st), 1)
            audit.assert_not_called()

    def test_truthy_strings_are_not_accepted_as_boolean_claims(self):
        st = fixture_state(DATASETS)
        st["scope"]["one_command_original_tables_reproduction_claimed"] = "false"
        self.write_statement(st)
        self.assertEqual(check(self.root, st), 1)


if __name__ == "__main__":
    unittest.main()
