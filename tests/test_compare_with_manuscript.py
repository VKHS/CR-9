"""scripts/compare_with_manuscript.py on a synthetic run directory (standard library only).

The fixture imitates the sealed decision records of a completed run with values
copied from the manuscript's arxiv row, so that every quantity should reach the
'exact' or 'match' level; a second fixture perturbs the effect inside its
interval and must be reported as 'interval', never 'exact'.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from scripts import compare_with_manuscript as cmp
from scripts._paper_common import load_reported, resolved_policy_view


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1), encoding="utf-8")


def seal_fixture(root: Path) -> None:
    """Seal deliberately fabricated JSON for I/O tests, never benchmark evidence."""
    config = json.loads((root / "config.json").read_text())
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    outputs = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in sorted(root.rglob("*.json")) if "provenance" not in p.relative_to(root).parts]
    parent = None
    for stage in cmp.STAGES:
        p = root / "provenance/stages" / f"{stage}.json"
        write(p, {"stage": stage, "source_sha256": "a" * 64, "config_sha256": digest,
                  "unit_sha256": "c" * 64, "dataset": config["dataset"], "parent_manifest": parent,
                  "inputs": [], "outputs": outputs if stage == "prepare" else []})
        parent = {"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def arxiv_run(root: Path, *, loss_mean: float = -0.0045, metric_mean: float = 0.07, admitted: bool = True) -> Path:
    reported = load_reported()
    t1 = reported["main_table_1_policy_and_deployment_ledger"]["ogbn-arxiv"]
    s15 = {r["method"]: r for r in reported["supplementary_table_S15_method_comparison"]["ogbn-arxiv"]["rows"]}
    candidates = [
        {"route": "nominal", "profile": "none", "ensemble": 4, "is_open_loop": True, "cost": 4.0, "candidate_id": "n4", "hash": "h0"},
        {"route": "fixed_sequence", "profile": "none", "ensemble": 4, "is_open_loop": True, "cost": 4.0, "candidate_id": "fs4", "hash": "h1"},
        {"route": "one_step", "profile": "full", "ensemble": 4, "is_open_loop": False, "cost": 4.5, "candidate_id": "os4", "hash": "h2"},
        {"route": "radius", "profile": "full", "ensemble": 4, "kappa": 1.0, "is_open_loop": False, "cost": 4.7, "candidate_id": "r4", "hash": "h3"},
        {"route": "mpc_rh", "profile": "full", "ensemble": 4, "horizon": 8, "is_open_loop": False, "cost": 5.3, "candidate_id": "rh4", "hash": "h4"},
    ]
    write(root / "config.json", {"dataset": "ogbn-arxiv", "profile": "full", "scientific_run": True})
    write(root / "run_state.json", {"source_sha256": "a" * 64, "config_sha256": "b" * 64, "dataset": "ogbn-arxiv"})
    write(root / "provenance" / "stages" / "evaluate.json", {"stage": "evaluate", "source_sha256": "a" * 64, "config_sha256": "b" * 64, "unit_sha256": "c" * 64, "dataset": "ogbn-arxiv"})
    write(root / "select" / "candidates.json", candidates)
    write(root / "select" / "decision.json", {"comparator_index": 1, "provisional_index": 3, "eligible_indices": [2, 3, 4, 2, 3, 4, 2][:7],
                                              "upper": [0.0, 0.0, -0.00143, -0.00338, -0.00317], "joint_feasible": [True] * 5, "max_radius": 0.0032})
    write(root / "confirm" / "decision.json", {"confirmed_index": 3, "comparator_index": 1, "contrasts": [
        {"kind": "state", "candidate_index": 3, "mean": 0.00574, "units": 32, "holm_p_value_zero": 0.0098, "passes": True},
        {"kind": "radius", "candidate_index": 3, "mean": -0.00286, "units": 32, "holm_p_value_zero": 0.015, "passes": True}]})
    write(root / "admit" / "decision.json", {"passed": admitted, "loss": {"mean": -0.0046, "upper": -0.0008, "units": 32}, "comparator_index": 1, "confirmed_index": 3,
                                             "deployed_index": 3 if admitted else 1})
    write(root / "policy_lock.json", {"comparator_index": 1, "deployed_index": 3 if admitted else 1})
    write(root / "final" / "results_summary.json", {
        "dataset": "ogbn-arxiv", "admitted": admitted,
        "models": {"qD": {"metric": s15["radius (deployed)"]["official_score"] / 100}, "qC": {"metric": s15["best fixed sequence (qC)"]["official_score"] / 100},
                   "qN": {"metric": s15["full-depth plant"]["official_score"] / 100}},
        "paired_qD_minus_qC_loss": {"mean": loss_mean, "lower": t1["loss_effect_interval_nats"][0], "upper": t1["loss_effect_interval_nats"][1]},
        "paired_qD_minus_qC_metric_pp": {"mean": metric_mean, "lower": t1["official_gain_interval_pp"][0], "upper": t1["official_gain_interval_pp"][1]},
        "deployed": candidates[3 if admitted else 1], "comparator": candidates[1]})
    seal_fixture(root)
    return root


def confirmation_fixture(root: Path, dataset: str) -> tuple[cmp.RunReader, dict]:
    """Fabricated I/O fixture for S14 targeting, not an experiment or full result match."""
    arxiv_run(root)
    reference = next(row for row in load_reported()["supplementary_table_S14_mechanism_contrasts_submitted_protocol"]
                     if row["dataset"] == dataset)
    write(root / "config.json", {"dataset": dataset, "profile": "full", "scientific_run": True})
    summary = json.loads((root / "final/results_summary.json").read_text())
    summary["dataset"] = dataset
    write(root / "final/results_summary.json", summary)
    decision = json.loads((root / "confirm/decision.json").read_text())
    kind = "radius" if reference["contrast"].endswith("/radius") else "mpc_rh_vs_ol"
    passed = dataset != "Peptides-func"
    decision["contrasts"] = [{"kind": kind, "candidate_index": 3, "units": reference["n"],
                              "mean": reference["effect"], "lower": reference["interval"][0],
                              "upper": reference["interval"][1], "holm_p_value_zero": 0.07 if dataset == "Peptides-func" else 0.0013,
                              "passes": passed}]
    decision["confirmed_index"] = 3 if passed else 1
    write(root / "confirm/decision.json", decision)
    admission = json.loads((root / "admit/decision.json").read_text())
    admission.update(passed=passed, confirmed_index=decision["confirmed_index"], deployed_index=decision["confirmed_index"])
    write(root / "admit/decision.json", admission)
    write(root / "policy_lock.json", {"comparator_index": 1, "deployed_index": decision["confirmed_index"]})
    summary["admitted"] = passed
    write(root / "final/results_summary.json", summary)
    seal_fixture(root)
    reader = cmp.RunReader(root)
    reader.verify()
    return reader, reference


def peptides_identity_run(root: Path) -> cmp.RunReader:
    """Fabricated sealed I/O fixture with qD exactly equal to qC, not benchmark evidence."""
    confirmation_fixture(root, "Peptides-func")
    view = resolved_policy_view(load_reported(), "Peptides-func")
    candidates = json.loads((root / "select/candidates.json").read_text())
    summary = json.loads((root / "final/results_summary.json").read_text())
    summary.update(deployed=candidates[1], comparator=candidates[1],
        paired_qD_minus_qC_loss={"mean": 0., "lower": 0., "upper": 0.},
        paired_qD_minus_qC_metric_pp={"mean": 0., "lower": 0., "upper": 0.},
        models={name: {"metric": row["official_score"], "metric_percent": row["official_score"] * 100}
                for name, row in view["scores"].items()})
    write(root / "final/results_summary.json", summary)
    admission = json.loads((root / "admit/decision.json").read_text())
    admission["empirical_bernstein"] = {"units": view["admission_J"], "mean": 0., "upper": view["admission_U_a"]}
    write(root / "admit/decision.json", admission)
    seal_fixture(root)
    reader = cmp.RunReader(root)
    reader.verify()
    return reader


class CompareWithManuscriptTests(unittest.TestCase):
    def test_result_producing_values_reach_exact_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = arxiv_run(Path(tmp) / "run")
            comparison = cmp.compare(cmp.RunReader(run), load_reported())
            self.assertEqual(comparison["dataset"], "ogbn-arxiv")
            self.assertEqual(comparison["levels"]["outside"], 0)
            self.assertEqual(comparison["levels"]["differs"], 0)
            by_name = {row["quantity"]: row for row in comparison["rows"]}
            self.assertEqual(by_name["deployed rule"]["status"], "match")
            self.assertEqual(by_name["comparator qC route and reported settings"]["status"], "match")
            self.assertEqual(by_name["global admission G_admit"]["status"], "match")
            self.assertEqual(by_name["paired CE effect qD - qC (nats), mean"]["status"], "exact")
            self.assertEqual(by_name["official accuracy gain qD - qC (pp), mean"]["status"], "exact")
            self.assertEqual(by_name["official test score of qD (radius (deployed))"]["status"], "exact")
            self.assertEqual(by_name["|Q_s| (eligible feedback configurations)"]["status"], "match")
            self.assertEqual(by_name["U_min over feasible configurations"]["status"], "exact")
            self.assertEqual(by_name["held-out state contrast (state), mean"]["status"], "exact")
            self.assertEqual(by_name["input fallback fraction (g_input)"]["status"], "not produced by this source")

    def test_retraining_values_inside_the_interval_are_not_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = arxiv_run(Path(tmp) / "run", loss_mean=-0.0052, metric_mean=0.11)
            comparison = cmp.compare(cmp.RunReader(run), load_reported())
            by_name = {row["quantity"]: row for row in comparison["rows"]}
            self.assertEqual(by_name["paired CE effect qD - qC (nats), mean"]["status"], "interval")
            self.assertEqual(by_name["official accuracy gain qD - qC (pp), mean"]["status"], "interval")

    def test_peptides_final_identity_matches_resolved_policy_and_preserves_arm_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = peptides_identity_run(Path(tmp) / "run")
            result = cmp.compare(run, load_reported())
            rows = {row["quantity"]: row for row in result["rows"]}
            self.assertEqual(rows["deployed rule"]["status"], "match")
            self.assertEqual(rows["global admission G_admit"]["status"], "match")
            self.assertEqual(rows["paired BCE effect qD - qC (nats), mean"]["status"], "exact")
            self.assertEqual(rows["official AP gain qD - qC (pp), mean"]["status"], "exact")
            score = rows["official test score of qD (best fixed sequence (qC))"]
            self.assertEqual(score["reported_value"], .714)
            self.assertEqual(score["status"], "exact")
            self.assertEqual(result["policy_reference"]["origin"], "derived_comparator_identity")
            self.assertEqual(len(result["unbound_evaluated_arm_targets"]), 7)

    def test_radius_effect_cannot_pass_as_final_qc_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = peptides_identity_run(Path(tmp) / "run")
            path = run.root / "final/results_summary.json"
            summary = json.loads(path.read_text())
            summary["paired_qD_minus_qC_loss"].update(mean=-.00303, lower=-.0047, upper=-.0013)
            write(path, summary)
            seal_fixture(run.root)
            reader = cmp.RunReader(run.root); reader.verify()
            rows = {row["quantity"]: row for row in cmp.compare(reader, load_reported())["rows"]}
            self.assertEqual(rows["paired BCE effect qD - qC (nats), mean"]["status"], "outside")
            self.assertEqual(rows["paired loss effect, lower 95% end"]["status"], "outside")

    def test_peptides_binding_has_nine_identity_targets_and_separate_unavailable_arm_targets(self):
        from scripts.check_result_binding import check_binding
        with tempfile.TemporaryDirectory() as tmp:
            run = peptides_identity_run(Path(tmp) / "run")
            comparison = cmp.compare(run, load_reported())
            target = Path(tmp) / "binding.json"
            cmp.emit_binding(run, load_reported(), comparison, target, "post_result_revision", "fixture", 1.0)
            binding = json.loads(target.read_text())
            self.assertEqual(len(binding["results"]), 9)
            self.assertEqual(len(binding["unavailable_evaluated_arm_targets"]), 7)
            for row in binding["results"]:
                self.assertEqual(row["target_role"], "served_policy")
                if row["json_pointer"].startswith("/paired_"):
                    self.assertEqual(row["reported_value"], 0)
                    self.assertEqual(row["absolute_tolerance"], 0)
                    self.assertEqual(row["target_kind"], "derived_comparator_identity")
            for name in ("qN", "qC", "qD"):
                self.assertIn(f"/models/{name}/metric_percent", {row["json_pointer"] for row in binding["results"]})
            self.assertEqual(check_binding(run.root, binding)["status"], "MATCH")

    def test_multigraph_confirmation_targets_current_s14_not_s14b_sensitivity(self):
        for dataset in ("PascalVOC-SP", "COCO-SP", "Peptides-func"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as tmp:
                run, target = confirmation_fixture(Path(tmp) / "run", dataset)
                reported = load_reported()
                # Force the analytical row to disagree, so accidental substitution
                # cannot pass merely because S14 and S14b currently share means.
                for row in reported["supplementary_table_S14b_separated_data_sensitivity"]:
                    row.update(effect=-0.8, interval=[-0.9, -0.7], J_new=99999, holm_P=0.001)
                result = cmp.compare(run, reported)
                confirmation = [row for row in result["rows"] if row["section"] == "Table S14"]
                mean = next(row for row in confirmation if row["quantity"].endswith(", mean"))
                units = next(row for row in confirmation if row["quantity"].endswith("statistical units"))
                self.assertEqual(mean["reported_value"], target["effect"])
                self.assertEqual(mean["status"], "exact")
                self.assertEqual(units["reported_value"], target["n"])
                self.assertEqual(units["status"], "match")
                self.assertTrue(all(row["manuscript_location"] == "Supplementary Table S14, D_confirm graphs"
                                    for row in confirmation))
                self.assertTrue(all(row["evidence_kind"] == "author_reported_measurements" for row in confirmation))
                if dataset == "Peptides-func":
                    holm = next(row for row in confirmation if "Holm-adjusted" in row["quantity"])
                    self.assertEqual(holm["reported_value"], 0.07)
                    self.assertEqual(holm["run_value"], 0.07)
                    self.assertIn("passes: False", holm["quantity"])
                    self.assertEqual(holm["status"], "informational")

    def test_confirmation_units_must_match_the_current_role_count(self):
        for units, expected in ((2331, "differs"), (None, "missing"), (True, "differs")):
            with self.subTest(units=units), tempfile.TemporaryDirectory() as tmp:
                run, _ = confirmation_fixture(Path(tmp) / "run", "Peptides-func")
                path = run.root / "confirm/decision.json"
                decision = json.loads(path.read_text())
                decision["contrasts"][0]["units"] = units
                write(path, decision)
                seal_fixture(run.root)
                run = cmp.RunReader(run.root)
                run.verify()
                result = cmp.compare(run, load_reported())
                row = next(row for row in result["rows"] if row["quantity"] == "held-out radius contrast, statistical units")
                self.assertEqual(row["reported_value"], 544)
                self.assertEqual(row["status"], expected)

    def test_confirmation_target_requires_the_dataset_and_contrast_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, _ = confirmation_fixture(Path(tmp) / "run", "PascalVOC-SP")
            path = run.root / "confirm/decision.json"
            decision = json.loads(path.read_text())
            # An arxiv-style radius contrast is not the Pascal RH contrast.
            decision["contrasts"][0].update(kind="radius", mean=-0.00286, units=32)
            write(path, decision)
            seal_fixture(run.root)
            run = cmp.RunReader(run.root)
            run.verify()
            result = cmp.compare(run, load_reported())
            self.assertFalse(any(row["section"] == "Table S14" for row in result["rows"]))

    def test_rejected_admission_is_reported_as_a_difference_for_an_admitted_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = arxiv_run(Path(tmp) / "run", admitted=False)
            comparison = cmp.compare(cmp.RunReader(run), load_reported())
            by_name = {row["quantity"]: row for row in comparison["rows"]}
            self.assertEqual(by_name["deployed rule"]["status"], "differs")
            self.assertEqual(by_name["global admission G_admit"]["status"], "differs")
            self.assertGreater(comparison["levels"]["differs"], 0)

    def test_emit_binding_writes_an_author_completed_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = arxiv_run(Path(tmp) / "run")
            out = Path(tmp) / "reports" / "binding.json"
            code = cmp.main([str(run), "--emit-binding", str(out), "--relationship", "exact_result_producing_run", "--output", str(Path(tmp) / "reports" / "comparison.json")])
            self.assertEqual(code, 0)
            binding = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(binding["status"], "author_completed")
            self.assertEqual(binding["relationship_to_manuscript"], "exact_result_producing_run")
            self.assertEqual(binding["run_identities"], cmp.RunReader(run).identities())
            pointers = {e["json_pointer"] for e in binding["results"]}
            self.assertIn("/paired_qD_minus_qC_loss/mean", pointers)
            self.assertIn("/models/qD/metric_percent", pointers)
            self.assertTrue((Path(tmp) / "reports" / "comparison.md").is_file())

    def test_run_figure_renders(self):
        try:
            import matplotlib  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib/numpy are not installed")
        from scripts import plot_run_figures
        with tempfile.TemporaryDirectory() as tmp:
            run = arxiv_run(Path(tmp) / "run")
            self.assertEqual(plot_run_figures.main([str(run), "--output-dir", str(Path(tmp) / "figures")]), 0)
            self.assertTrue((Path(tmp) / "figures" / "run_run_ogbn-arxiv.png").is_file())

    def test_run_plot_rejects_modified_summary_before_creating_output(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")
        from scripts import plot_run_figures
        with tempfile.TemporaryDirectory() as tmp:
            run = arxiv_run(Path(tmp) / "run")
            write(run / "final/results_summary.json", {"admitted": True})
            out = Path(tmp) / "figures"
            with self.assertRaises(ValueError):
                plot_run_figures.main([str(run), "--output-dir", str(out)])
            self.assertFalse(out.exists())

    def test_synthetic_run_plot_does_not_load_paper_values(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")
        from scripts import plot_run_figures
        with tempfile.TemporaryDirectory() as tmp:
            run = arxiv_run(Path(tmp) / "run")
            write(run / "config.json", {"dataset": "synthetic", "benchmark_dataset": "ogbn-arxiv", "scientific_run": False})
            seal_fixture(run)
            with patch.object(plot_run_figures, "load_reported", side_effect=AssertionError("Paper scores must not be loaded")):
                self.assertEqual(plot_run_figures.main([str(run), "--output-dir", str(Path(tmp)/"plain")]), 0)
            with self.assertRaises(ValueError):
                plot_run_figures.main([str(run), "--output-dir", str(Path(tmp)/"overlay"), "--compare-manuscript"])
            self.assertFalse((Path(tmp)/"overlay").exists())

    def test_run_plot_preserves_existing_files_and_symlink_targets(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")
        from scripts import plot_run_figures
        with tempfile.TemporaryDirectory() as tmp:
            run = arxiv_run(Path(tmp)/"run")
            out = Path(tmp)/"figures"
            out.mkdir()
            target = out/"run_run_ogbn-arxiv.png"
            target.write_bytes(b"prior figure")
            with self.assertRaises(ValueError):
                plot_run_figures.main([str(run), "--output-dir", str(out)])
            self.assertEqual(target.read_bytes(), b"prior figure")
            target.unlink()
            evidence = run/"final/results_summary.json"
            before = evidence.read_bytes()
            target.symlink_to(evidence)
            with self.assertRaises(ValueError):
                plot_run_figures.main([str(run), "--output-dir", str(out)])
            self.assertEqual(evidence.read_bytes(), before)

    def test_run_plot_rejects_duplicate_output_names_before_writing(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")
        from scripts import plot_run_figures
        with tempfile.TemporaryDirectory() as tmp:
            one = arxiv_run(Path(tmp)/"one/run")
            two = arxiv_run(Path(tmp)/"two/run")
            out = Path(tmp)/"figures"
            with self.assertRaisesRegex(ValueError, "same figure name"):
                plot_run_figures.main([str(one), str(two), "--output-dir", str(out)])
            self.assertFalse(out.exists())

    def test_output_inside_the_run_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = arxiv_run(Path(tmp) / "run")
            code = cmp.main([str(run), "--output", str(run / "comparison.json")])
            self.assertEqual(code, 2)

    def test_incomplete_input_cannot_pass_or_emit_a_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = arxiv_run(Path(tmp) / "run")
            (run / "provenance/stages/train.json").unlink()
            out = Path(tmp) / "binding.json"
            self.assertEqual(cmp.main([str(run), "--emit-binding", str(out), "--relationship", "exact_result_producing_run"]), 2)
            self.assertFalse(out.exists())

    def test_config_only_is_not_a_successful_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            write(root / "config.json", {"dataset": "ogbn-arxiv", "scientific_run": True})
            self.assertEqual(cmp.main([str(root)]), 2)

    def test_modified_summary_is_rejected_before_writing_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = arxiv_run(Path(tmp) / "run")
            p = run / "final/results_summary.json"
            data = json.loads(p.read_text())
            data["paired_qD_minus_qC_loss"]["mean"] = -0.1
            write(p, data)
            out = Path(tmp) / "comparison.json"
            self.assertEqual(cmp.main([str(run), "--output", str(out)]), 2)
            self.assertFalse(out.exists())

    def test_synthetic_task_fixture_is_not_benchmark_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = arxiv_run(Path(tmp) / "run")
            write(run / "config.json", {"dataset": "synthetic", "benchmark_dataset": "ogbn-arxiv", "scientific_run": False})
            seal_fixture(run)
            self.assertEqual(cmp.main([str(run)]), 2)

    def test_existing_outputs_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = arxiv_run(Path(tmp) / "run")
            out = Path(tmp) / "comparison.json"
            out.write_text("preserve this report")
            self.assertEqual(cmp.main([str(run), "--output", str(out)]), 2)
            self.assertEqual(out.read_text(), "preserve this report")

    def test_binding_cannot_modify_run_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = arxiv_run(Path(tmp) / "run")
            out = run / "binding.json"
            self.assertEqual(cmp.main([str(run), "--emit-binding", str(out), "--relationship", "exact_result_producing_run"]), 2)
            self.assertFalse(out.exists())

    def test_binding_requires_the_authors_explicit_relationship(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = arxiv_run(Path(tmp) / "run")
            out = Path(tmp) / "binding.json"
            self.assertEqual(cmp.main([str(run), "--emit-binding", str(out)]), 2)
            self.assertFalse(out.exists())

    def test_nonfinite_comparisons_and_binding_tolerances_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cmp.level(value, 0.1, 0.01)
        with tempfile.TemporaryDirectory() as tmp:
            run = arxiv_run(Path(tmp) / "run")
            out = Path(tmp) / "binding.json"
            self.assertEqual(cmp.main([str(run), "--emit-binding", str(out), "--relationship", "post_result_revision",
                                       "--binding-tolerance-scale", "nan"]), 2)
            self.assertFalse(out.exists())

    def test_one_roman_split_is_only_a_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = arxiv_run(Path(tmp) / "run")
            write(run / "config.json", {"dataset": "Roman-empire", "scientific_run": True, "data": {"split_index": 0}})
            result = cmp.compare(cmp.RunReader(run), load_reported())
            self.assertFalse(result["paper_comparison_eligible"])
            self.assertTrue(all(row["status"] == "diagnostic only" for row in result["rows"]))

    def test_markdown_escapes_table_delimiters(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = arxiv_run(Path(tmp) / "run")
            result = cmp.compare(cmp.RunReader(run), load_reported())
            self.assertIn("\\|Q_s\\|", cmp.markdown(result))

    def test_printed_trailing_zeros_do_not_create_false_exact_matches(self):
        self.assertEqual(cmp.effect_half_unit(0.60, [0.38, 0.82]), 0.005)
        self.assertEqual(cmp.level(0.64, 0.60, cmp.effect_half_unit(0.60, [0.38, 0.82]), [0.38, 0.82]), "interval")
        self.assertEqual(cmp.level(0.14, 0.10, cmp.effect_half_unit(0.26, [0.10, 0.42])), "outside")

    def test_reported_horizon_is_part_of_comparator_settings(self):
        status, _ = cmp.comparator_status({"route": "mpc_ol", "ensemble": 8, "horizon": 1}, "MPC-OL H = 16", 8)
        self.assertEqual(status, "differs")
        status, _ = cmp.comparator_status({"route": "mpc_ol", "ensemble": 8, "horizon": 16}, "MPC-OL H = 16", 8)
        self.assertEqual(status, "match")
        status, _ = cmp.comparator_status({"route": "phase", "ensemble": 2, "sequence": [0]*8+[1]*24+[2]*32}, "phase boundaries 8/32", 2)
        self.assertEqual(status, "informational")

    def test_empty_sealed_decisions_are_not_complete_evidence(self):
        for relative, value in (("select/decision.json", None), ("confirm/decision.json", {}),
                                ("admit/decision.json", {}), ("policy_lock.json", {}), ("select/candidates.json", [])):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                run = arxiv_run(Path(tmp) / "run")
                write(run / relative, value)
                seal_fixture(run)
                self.assertEqual(cmp.main([str(run)]), 2)

    def test_nested_output_paths_fail_before_any_output_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = arxiv_run(Path(tmp) / "run")
            out = Path(tmp) / "new/report.json"
            binding = out / "binding.json"
            self.assertEqual(cmp.main([str(run), "--output", str(out), "--emit-binding", str(binding),
                                      "--relationship", "post_result_revision"]), 2)
            self.assertFalse(out.exists())
            self.assertFalse(out.with_suffix(".md").exists())

    def test_malformed_model_records_are_rejected_before_writing_reports(self):
        for name, value in ((None, None), ("qN", None), ("qC", []), ("qD", "invalid")):
            with self.subTest(model=name, value=value), tempfile.TemporaryDirectory() as tmp:
                run = arxiv_run(Path(tmp) / "run")
                summary_path = run / "final/results_summary.json"
                summary = json.loads(summary_path.read_text())
                if name is None:
                    summary["models"] = value
                else:
                    summary["models"][name] = value
                write(summary_path, summary)
                seal_fixture(run)
                out = Path(tmp) / "comparison.json"
                self.assertEqual(cmp.main([str(run), "--output", str(out)]), 2)
                self.assertFalse(out.exists())
                self.assertFalse(out.with_suffix(".md").exists())

    def test_roman_campaign_writes_an_aggregate_report(self):
        # The split summarizer is tested separately; this checks its reporting API.
        report = load_reported()
        t1 = report["main_table_1_policy_and_deployment_ledger"]["Roman-empire"]
        s15 = report["supplementary_table_S15_method_comparison"]["Roman-empire"]
        aggregate = {"scientific_run": True,
            "paired_qD_minus_qC_loss": dict(zip(("mean", "lower", "upper"), [t1["loss_effect_nats"], *t1["loss_effect_interval_nats"]])),
            "paired_qD_minus_qC_accuracy_pp": dict(zip(("mean", "lower", "upper"), [t1["official_gain_pp"], *t1["official_gain_interval_pp"]])),
            "models": {name: {"accuracy_percent": {"mean": row["official_score"]}} for name, row in (
                ("qN", next(r for r in s15["rows"] if r["method"] == "full-depth plant")),
                ("qC", next(r for r in s15["rows"] if "(qC" in r["method"])),
                ("qD", next(r for r in s15["rows"] if r["family"] == "CROSS-MORSE deployment")))}}
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / "campaign"
            campaign.mkdir()
            out = Path(tmp) / "report.json"
            with patch("scripts.summarize_roman_splits.summarize", return_value=aggregate):
                self.assertEqual(cmp.main([str(campaign), "--roman-campaign", "--output", str(out)]), 0)
            result = json.loads(out.read_text())
            self.assertEqual(result["comparison_scope"], "ten-split campaign aggregate")
            self.assertTrue(result["paper_comparison_eligible"])
            self.assertTrue(out.with_suffix(".md").is_file())


if __name__ == "__main__":
    unittest.main()
