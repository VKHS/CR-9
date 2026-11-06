"""An actual model-count disagreement cannot pass the manuscript evidence gate."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cross_morse.review import STAGES, canonical, digest
from scripts import compare_registered_runs, model_inventory
from scripts._paper_common import DATASETS, ROOT, STEMS, load_registered
from scripts.scientific_evidence_audit import validate_model_inventory


class ModelParityTests(unittest.TestCase):
    def record(self, count):
        return {"checkpoints": [{"named_tensor_elements": count, "tensors": [{"name": "fixture", "elements": count}]}]}

    def test_arxiv_reference_count_cannot_validate_reported_3_1m(self):
        with self.assertRaisesRegex(ValueError, "disagrees with Table S5"):
            validate_model_inventory(self.record(1_138_216), 3.1)

    def test_rounding_tolerance_is_the_printed_unit(self):
        self.assertEqual(validate_model_inventory(self.record(3_140_000), 3.1)["status"], "PASS")
        with self.assertRaises(ValueError):
            validate_model_inventory(self.record(3_160_000), 3.1)

    def test_half_unit_upper_boundary_rounds_to_the_next_printed_count(self):
        for target, lower, upper in ((3.1, 3_050_000, 3_150_000), (3.52, 3_515_000, 3_525_000)):
            with self.subTest(target=target):
                self.assertEqual(validate_model_inventory(self.record(lower), target)["status"], "PASS")
                self.assertEqual(validate_model_inventory(self.record(upper - 1), target)["status"], "PASS")
                with self.assertRaisesRegex(ValueError, "disagrees with Table S5"):
                    validate_model_inventory(self.record(upper), target)

    def test_inventory_cannot_pad_total_without_named_tensors(self):
        wrong = self.record(1_138_216)
        wrong["checkpoints"][0]["named_tensor_elements"] = 3_100_000
        with self.assertRaisesRegex(ValueError, "differs from named"):
            validate_model_inventory(wrong, 3.1)

    def typed_record(self):
        return {"checkpoints": [{"named_tensor_elements": 3_350_000,
                                 "parameter_count": 3_100_000, "buffer_elements": 250_000,
                                 "optimization_active_parameter_count": 3_100_000,
                                 "tensors": [{"name": "weight", "kind": "parameter", "elements": 3_100_000},
                                             {"name": "running_state", "kind": "buffer", "elements": 250_000}]}]}

    def test_typed_buffers_are_separate_from_s5_predictor_parameters(self):
        result = validate_model_inventory(self.typed_record(), 3.1)
        self.assertEqual(result["counts"], [3_100_000])
        self.assertEqual(result["buffer_elements"], [250_000])
        frozen = self.typed_record()
        frozen["checkpoints"][0]["optimization_active_parameter_count"] = 0
        self.assertEqual(validate_model_inventory(frozen, 3.1)["counts"], [3_100_000])

    def test_wrong_typed_totals_or_partial_kinds_are_rejected(self):
        for corruption in ("parameter_count", "buffer_elements", "optimization_active_parameter_count", "partial", "unknown"):
            with self.subTest(corruption=corruption):
                inventory = self.typed_record()
                row = inventory["checkpoints"][0]
                if corruption == "partial":
                    del row["tensors"][0]["kind"]
                elif corruption == "unknown":
                    row["tensors"][0]["kind"] = "ignored"
                else:
                    row[corruption] += 1
                with self.assertRaises(ValueError):
                    validate_model_inventory(inventory, 3.1)

    def test_legacy_inventory_cannot_claim_a_buffer_deduction_without_typed_tensors(self):
        inventory = self.typed_record()
        row = inventory["checkpoints"][0]
        for tensor in row["tensors"]:
            del tensor["kind"]
        with self.assertRaisesRegex(ValueError, "kind for every tensor"):
            validate_model_inventory(inventory, 3.1)
        for field in ("parameter_count", "buffer_elements", "optimization_active_parameter_count"):
            del row[field]
        with self.assertRaisesRegex(ValueError, "disagrees with Table S5"):
            validate_model_inventory(inventory, 3.1)


def sealed_model_fixture(root, *, dataset="ogbn-arxiv", count=3_100_000, inventory_sha=None):
    """Fabricated contract fixture; checkpoint bytes are deliberately not a model."""
    root = Path(root)
    (root / "train").mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / f"configs/{STEMS[dataset]}_full.json").read_text())
    (root / "config.json").write_text(json.dumps(config))
    checkpoint = root / "train/plant_seed1_best.pt"
    checkpoint.write_bytes(b"MODEL INVENTORY TEST FIXTURE; not a serialized model")
    reference = load_registered()["backbones_source_status"]["released_full_profiles"][dataset]
    inventory = {**model_inventory.configured_architecture(config), "configuration": config["model"],
                 "input_dim": reference["input_dim"], "num_outputs": reference["num_outputs"],
                 "checkpoints": [{"path": "train/plant_seed1_best.pt", "sha256": inventory_sha or digest(checkpoint),
                                  "named_tensor_elements": count, "parameter_count": count, "buffer_elements": 0,
                                  "parameter_components": {"fixture": count},
                                  "tensors": [{"name": "fixture", "shape": [count], "kind": "parameter", "elements": count}]}]}
    (root / "train/model_inventory.json").write_text(json.dumps(inventory))
    (root / "provenance").mkdir(exist_ok=True)
    (root / "provenance/frozen_models.json").write_text(json.dumps({
        "captured_after_stage": "train",
        "checkpoints": {"train/plant_seed1_best.pt": {"artifact": {
            "path": "train/plant_seed1_best.pt", "bytes": checkpoint.stat().st_size, "sha256": digest(checkpoint)}}}}))
    reseal_model_fixture(root)
    return root


def reseal_model_fixture(root):
    """Rebuild only a fabricated test fixture's seals after a deliberate mutation."""
    config = json.loads((root / "config.json").read_text())

    def descriptor(path):
        return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}

    previous = None
    for stage in STAGES:
        output = ([root / "config.json"] if stage == "prepare" else
                  sorted((root / "train").iterdir()) + [root / "provenance/frozen_models.json"] if stage == "train" else [])
        manifest = root / "provenance/stages" / f"{stage}.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"stage": stage, "source_sha256": "a" * 64,
                                       "config_sha256": canonical(config), "unit_sha256": "c" * 64,
                                       "dataset": config["dataset"], "parent_manifest": previous, "inputs": [],
                                       "outputs": [descriptor(path) for path in output]}))
        previous = descriptor(manifest)


class RegisteredModelInventoryTests(unittest.TestCase):
    def test_same_total_cannot_hide_wrong_architecture_or_tensor_metadata(self):
        corruptions = {
            "configuration": lambda record: record["configuration"].update(encoder_hidden=1041),
            "widths": lambda record: record["architecture_widths"].update(encoder_hidden=1041),
            "class": lambda record: record.update(architecture="HeterophilyPlant"),
            "dimensions": lambda record: record.update(input_dim=129),
            "components": lambda record: record["checkpoints"][0].update(parameter_components={"other": 3_100_000}),
            "shape": lambda record: record["checkpoints"][0]["tensors"][0].update(shape=[3_100_001]),
        }
        for name, mutate in corruptions.items():
            with self.subTest(corruption=name), tempfile.TemporaryDirectory() as directory:
                root = sealed_model_fixture(directory)
                path = root / "train/model_inventory.json"
                record = json.loads(path.read_text())
                mutate(record)
                path.write_text(json.dumps(record))
                reseal_model_fixture(root)
                result = compare_registered_runs.check_model_inventories(root, "ogbn-arxiv", 3.1)
                self.assertEqual(result["status"], "FAIL", result)

    def test_sealed_inventory_and_checkpoint_bytes_are_checked_without_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = sealed_model_fixture(Path(directory) / "run")
            with patch.dict("sys.modules", {"torch": None}):
                result = compare_registered_runs.check_model_inventories(root, "ogbn-arxiv", 3.1)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["runs"][0]["counts"], [3_100_000])
            self.assertTrue(result["runs"][0]["checkpoint_bytes_verified"])
            self.assertTrue(result["runs"][0]["checkpoint_inventory_complete"])
            self.assertEqual(result["runs"][0]["checkpoint_count"], 1)

    def test_default_width_arxiv_count_cannot_pass_the_s5_comparison(self):
        # The plant built with the default widths (Table S5d, 1,138,216
        # parameters) is not the 3.1 M benchmark plant of Table S5.
        with tempfile.TemporaryDirectory() as directory:
            root = sealed_model_fixture(Path(directory) / "run", count=1_138_216)
            result = compare_registered_runs.check_model_inventories(root, "ogbn-arxiv", 3.1)
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("1,138,216 disagrees with Table S5", result["runs"][0]["issue"])

    def test_released_full_profile_count_passes_the_s5_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = sealed_model_fixture(Path(directory) / "run", count=3_116_632)
            result = compare_registered_runs.check_model_inventories(root, "ogbn-arxiv", 3.1)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["runs"][0]["counts"], [3_116_632])

    def test_missing_or_modified_inventory_is_rejected(self):
        for mode in ("missing", "modified"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = sealed_model_fixture(Path(directory) / "run")
                path = root / "train/model_inventory.json"
                if mode == "missing":
                    path.unlink()
                else:
                    path.write_bytes(path.read_bytes() + b" ")
                result = compare_registered_runs.check_model_inventories(root, "ogbn-arxiv", 3.1)
                self.assertEqual(result["status"], "FAIL")
                self.assertIn("train/model_inventory.json", result["runs"][0]["issue"])

    def test_checkpoint_bytes_and_inventory_identity_must_both_match(self):
        for mode in ("changed_bytes", "wrong_identity"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = sealed_model_fixture(Path(directory) / "run",
                                            inventory_sha="d" * 64 if mode == "wrong_identity" else None)
                if mode == "changed_bytes":
                    (root / "train/plant_seed1_best.pt").write_bytes(b"changed checkpoint")
                result = compare_registered_runs.check_model_inventories(root, "ogbn-arxiv", 3.1)
                self.assertEqual(result["status"], "FAIL")
                self.assertIn("train/plant_seed1_best.pt", result["runs"][0]["issue"])

    def test_an_omitted_sealed_plant_checkpoint_cannot_pass(self):
        for frozen_baseline_includes_extra in (False, True):
            with self.subTest(frozen_baseline_includes_extra=frozen_baseline_includes_extra), tempfile.TemporaryDirectory() as directory:
                root = sealed_model_fixture(Path(directory) / "run")
                extra = root / "train/plant_seed17_best.pt"
                extra.write_bytes(b"SECOND SEALED TEST PLANT")
                if frozen_baseline_includes_extra:
                    path = root / "provenance/frozen_models.json"
                    baseline = json.loads(path.read_text())
                    baseline["checkpoints"]["train/plant_seed17_best.pt"] = {"artifact": {
                        "path": "train/plant_seed17_best.pt", "bytes": extra.stat().st_size, "sha256": digest(extra)}}
                    path.write_text(json.dumps(baseline))
                reseal_model_fixture(root)
                result = compare_registered_runs.check_model_inventories(root, "ogbn-arxiv", 3.1)
                self.assertEqual(result["status"], "FAIL")
                self.assertIn("missing=['train/plant_seed17_best.pt']", result["runs"][0]["issue"])

    def test_duplicate_or_substituted_checkpoint_inventory_is_rejected(self):
        for mode in ("duplicate", "substitute"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = sealed_model_fixture(Path(directory) / "run")
                path = root / "train/model_inventory.json"
                inventory = json.loads(path.read_text())
                if mode == "duplicate":
                    inventory["checkpoints"].append(dict(inventory["checkpoints"][0]))
                else:
                    # A validly sealed recovery checkpoint cannot replace the
                    # frozen best checkpoint in the parameter inventory.
                    substitute = root / "train/plant_seed1_last.pt"
                    substitute.write_bytes(b"SEALED RECOVERY TEST CHECKPOINT")
                    inventory["checkpoints"][0].update(path="train/plant_seed1_last.pt", sha256=digest(substitute))
                path.write_text(json.dumps(inventory))
                reseal_model_fixture(root)
                result = compare_registered_runs.check_model_inventories(root, "ogbn-arxiv", 3.1)
                self.assertEqual(result["status"], "FAIL")
                self.assertIn("Duplicate" if mode == "duplicate" else "unexpected=['train/plant_seed1_last.pt']",
                              result["runs"][0]["issue"])

    def test_frozen_baseline_must_cover_the_same_trained_checkpoint_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = sealed_model_fixture(Path(directory) / "run")
            path = root / "provenance/frozen_models.json"
            baseline = json.loads(path.read_text())
            baseline["checkpoints"] = {}
            path.write_text(json.dumps(baseline))
            reseal_model_fixture(root)
            result = compare_registered_runs.check_model_inventories(root, "ogbn-arxiv", 3.1)
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("Frozen-model baseline membership", result["runs"][0]["issue"])

    def test_roman_checks_every_split_and_rejects_one_count_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(10):
                sealed_model_fixture(root / f"split_{index:02d}", dataset="Roman-empire")
            result = compare_registered_runs.check_model_inventories(root, "Roman-empire", 3.1)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual([row["run"] for row in result["runs"]], [f"split_{i:02d}" for i in range(10)])
            sealed_model_fixture(root / "split_09", dataset="Roman-empire", count=1_138_216)
            result = compare_registered_runs.check_model_inventories(root, "Roman-empire", 3.1)
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["runs"][-1]["status"], "FAIL")

    def test_inventory_failure_makes_the_command_fail_even_when_numbers_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = sealed_model_fixture(Path(directory) / "run", count=1_138_216)
            row = {"dataset": "ogbn-arxiv", "path": root, "stem": "arxiv"}
            stdout = io.StringIO()
            with patch.object(compare_registered_runs, "inspect_index", return_value=([row], [])), \
                 patch.object(compare_registered_runs.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
                 contextlib.redirect_stdout(stdout):
                exit_code = compare_registered_runs.main(["--output-dir", str(Path(directory) / "reports")])
            report = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertEqual(report["status"], "INCOMPLETE_OR_INVALID")
            self.assertEqual(report["results"][0]["exit_code"], 0)
            self.assertEqual(report["results"][0]["model_inventory"]["status"], "FAIL")
            self.assertEqual(report["model_inventory_verified_datasets"], 0)
            saved = json.loads((Path(report["report_directory"]) / "coverage.json").read_text())
            self.assertEqual(saved, report)


class ModelInventoryReferenceTests(unittest.TestCase):
    def setUp(self):
        self.registered = load_registered()

    def test_source_widths_are_checked_against_the_executable_configuration(self):
        changed = copy.deepcopy(self.registered)
        changed["backbones_source_status"]["released_full_profiles"]["ogbn-arxiv"]["architecture_widths"]["encoder_hidden"] += 1
        with self.assertRaisesRegex(ValueError, "widths differ"):
            model_inventory.reference_profiles(changed)

    def test_invalid_width_and_site_configurations_fail_before_instantiation(self):
        for dataset, key, value in (("ogbn-arxiv", "head_hidden", None), ("ogbn-arxiv", "encoder_hidden", True),
                                    ("ogbn-arxiv", "block_mlp_layers", 2.0), ("PascalVOC-SP", "depth", 127),
                                    ("PascalVOC-SP", "depth", 2), ("Peptides-func", "depth", 63)):
            with self.subTest(dataset=dataset, key=key, value=value):
                config = json.loads((ROOT / f"configs/{STEMS[dataset]}_full.json").read_text())
                config["model"][key] = value
                with self.assertRaises(ValueError):
                    model_inventory.configured_architecture(config)

    def test_exact_seven_dataset_coverage_is_required_for_both_records(self):
        self.assertEqual(set(model_inventory.reference_profiles(self.registered)), set(DATASETS))
        for name in ("backbones_table_S5", "released_full_profiles"):
            for mode in ("missing", "extra"):
                with self.subTest(record=name, mode=mode):
                    changed = copy.deepcopy(self.registered)
                    rows = changed[name] if name == "backbones_table_S5" else changed["backbones_source_status"][name]
                    if mode == "missing":
                        rows.pop("Peptides-func")
                    else:
                        rows["unknown"] = copy.deepcopy(next(iter(rows.values())))
                    with self.assertRaisesRegex(ValueError, "exactly the seven"):
                        model_inventory.render_parameter_table(changed)

    def test_documented_dimensions_must_match_adapter_and_task_contracts(self):
        for dataset, field, wrong in (("Peptides-func", "input_dim", 9),
                                      ("ogbn-products", "input_dim", 128),
                                      ("ogbn-proteins", "num_outputs", 40)):
            with self.subTest(dataset=dataset, field=field):
                changed = copy.deepcopy(self.registered)
                changed["backbones_source_status"]["released_full_profiles"][dataset][field] = wrong
                with self.assertRaisesRegex(ValueError, "dimensions differ"):
                    model_inventory.reference_profiles(changed)

    def test_profile_defaults_use_real_or_synthetic_dimensions(self):
        for dataset, expected in (("Peptides-func", (174, 10)), ("ogbn-proteins", (8, 112)),
                                  ("ogbn-products", (100, 47))):
            for profile in ("full", "pilot"):
                config = json.loads((ROOT / f"configs/{STEMS[dataset]}_{profile}.json").read_text())
                self.assertEqual(model_inventory.resolve_dimensions(config, self.registered), expected)
        smoke = json.loads((ROOT / "configs/peptides_func_smoke.json").read_text())
        self.assertEqual(model_inventory.resolve_dimensions(smoke, self.registered), (12, 4))
        custom = {"dataset": "ogbn-proteins", "profile": "custom"}
        with self.assertRaisesRegex(ValueError, "require explicit"):
            model_inventory.resolve_dimensions(custom, self.registered)
        self.assertEqual(model_inventory.resolve_dimensions(custom, self.registered, 7, 3), (7, 3))
        with self.assertRaisesRegex(ValueError, "positive integers"):
            model_inventory.resolve_dimensions(custom, self.registered, 0, 3)

    def test_document_check_detects_staleness_and_write_preserves_surrounding_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "DATASETS.md"
            original = f"Keep this introduction.\n{model_inventory.BEGIN}\nstale\n{model_inventory.END}\nKeep this ending.\n"
            path.write_text(original)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
                result = model_inventory.main(["--check-docs", "--docs-path", str(path)])
            self.assertEqual(result, 1)
            self.assertIn("Stale model parameter table", errors.getvalue())
            self.assertEqual(path.read_text(), original)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(model_inventory.main(["--write-docs", "--docs-path", str(path)]), 0)
                self.assertEqual(model_inventory.main(["--check-docs", "--docs-path", str(path)]), 0)
            updated = path.read_text()
            self.assertTrue(updated.startswith("Keep this introduction.\n"))
            self.assertTrue(updated.endswith("Keep this ending.\n"))
            for dataset in DATASETS:
                self.assertEqual(updated.count(f"| {dataset} |"), 1)
            changed = copy.deepcopy(self.registered)
            products = changed["backbones_source_status"]["released_full_profiles"]["ogbn-products"]
            products["parameter_count"] += 1
            products["parameter_components"]["head"] += 1
            self.assertNotEqual(updated, model_inventory.document_with_parameter_table(updated, changed))
            products["parameter_components"]["head"] -= 1  # components no longer sum to the count
            with self.assertRaisesRegex(ValueError, "sum to the count"):
                model_inventory.document_with_parameter_table(updated, changed)

    def test_document_mode_runs_with_site_packages_disabled(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/model_inventory.py"), "--check-docs"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_invalid_adapter_source_fails_cleanly_without_instantiation(self):
        for error in (ValueError("missing adapter contract"), SyntaxError("invalid adapter source")):
            with self.subTest(error=type(error).__name__), \
                 patch("scripts.check_paper_constants.source_dimensions", side_effect=error), \
                 contextlib.redirect_stderr(io.StringIO()) as output:
                self.assertEqual(model_inventory.main(["--check-docs"]), 2)
            self.assertIn("ERROR:", output.getvalue())

    def test_missing_or_duplicate_document_markers_are_errors(self):
        for text in ("unmarked", model_inventory.END + model_inventory.BEGIN,
                     model_inventory.BEGIN * 2 + model_inventory.END):
            with self.assertRaisesRegex(ValueError, "one ordered"):
                model_inventory.document_with_parameter_table(text, self.registered)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is needed to instantiate plants")
class InstantiatedModelInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registered = load_registered()
        cls.report = model_inventory.all_full_inventory(cls.registered)
        cls.rows = {row["dataset"]: row for row in cls.report["profiles"]}

    def test_all_seven_actual_inventories_match_source_records_and_table_s5(self):
        self.assertEqual(set(self.rows), set(DATASETS))
        self.assertTrue(self.report["source_records_match"])
        self.assertTrue(self.report["table_S5_counts_agree"])
        for dataset, row in self.rows.items():
            with self.subTest(dataset=dataset):
                self.assertTrue(row["source_record_matches"])
                self.assertTrue(row["manuscript_reference"]["count_agrees_at_printed_precision"])
                self.assertEqual(row["parameter_count"], sum(item["elements"] for item in row["parameters"]))
                self.assertEqual(row["parameter_count"], sum(row["parameter_components"].values()))
                self.assertEqual(row["parameter_components"], row["source_record"]["parameter_components"])
                self.assertEqual(row["architecture_widths"], row["source_record"]["architecture_widths"])
                self.assertEqual(row["trainable_parameter_count"], row["parameter_count"])
                self.assertEqual(row["buffer_elements"], 0)
                for identity in ("config_sha256", "config_file_sha256", "source_sha256"):
                    self.assertRegex(row[identity], r"^[0-9a-f]{64}$")
        arxiv = self.rows["ogbn-arxiv"]
        self.assertEqual(arxiv["parameter_count"], 3_116_632)
        self.assertEqual(arxiv["manuscript_reference"]["parameters_M"], 3.1)
        self.assertEqual(self.rows["Peptides-func"]["parameter_count"], 5_114_130)
        self.assertEqual(self.rows["COCO-SP"]["parameter_count"] - self.rows["PascalVOC-SP"]["parameter_count"], 60 * 641)

    def test_default_widths_reproduce_the_table_s5d_released_default_plant(self):
        config = json.loads((ROOT / "configs/arxiv_full.json").read_text())
        row = model_inventory.inventory(model_inventory.default_width_config(config), 128, 40)
        self.assertEqual(row["parameter_count"], 1_138_216)
        self.assertEqual(row["architecture_widths"]["encoder_hidden"], 512)
        self.assertEqual(row["architecture_widths"]["head_layers"], 1)

    def test_two_layer_head_keeps_the_head_prefix_for_provenance_seals(self):
        names = [item["name"] for item in self.rows["PascalVOC-SP"]["parameters"]]
        self.assertIn("head.hidden.weight", names)
        self.assertIn("head.output.bias", names)
        self.assertEqual(self.rows["PascalVOC-SP"]["parameter_components"]["head"], 128 * 640 + 640 + 640 * 21 + 21)
        self.assertEqual([item["name"] for item in self.rows["ogbn-arxiv"]["parameters"] if item["name"].startswith("head")],
                         ["head.weight", "head.bias"])
        expected_classes = {"ogbn-arxiv": "FrozenPlant", "ogbn-products": "FrozenPlant",
                            "ogbn-proteins": "EdgeWeightedPlant", "Roman-empire": "HeterophilyPlant",
                            "PascalVOC-SP": "DesignatedSitePlant", "COCO-SP": "DesignatedSitePlant",
                            "Peptides-func": "DesignatedSitePlant"}
        self.assertEqual({name: row["architecture"] for name, row in self.rows.items()}, expected_classes)

    def test_task_semantics_and_smoke_dimensions_reach_the_model_constructor(self):
        for dataset, level in (("ogbn-proteins", "node"), ("Peptides-func", "graph")):
            row = self.rows[dataset]
            self.assertEqual(row["task"]["task_type"], "multilabel")
            self.assertEqual(row["task"]["prediction_level"], level)
        config = json.loads((ROOT / "configs/peptides_func_smoke.json").read_text())
        import cross_morse.core as core
        with patch.object(core, "build_plant", wraps=core.build_plant) as build:
            row = model_inventory.inventory(config, *model_inventory.resolve_dimensions(config, self.registered))
        self.assertEqual(build.call_args.args[:2], (12, 4))
        self.assertEqual(build.call_args.args[3].prediction_level, "graph")
        self.assertEqual(build.call_args.args[3].task_type, "multilabel")
        self.assertEqual(row["task"]["num_outputs"], 4)

    def test_parameter_trainable_and_buffer_counts_are_not_conflated(self):
        import torch
        model = torch.nn.Module()
        model.register_parameter("learned", torch.nn.Parameter(torch.zeros(2, 3)))
        model.register_parameter("frozen", torch.nn.Parameter(torch.zeros(4), requires_grad=False))
        model.register_buffer("running", torch.zeros(7))
        config = json.loads((ROOT / "configs/arxiv_full.json").read_text())
        with patch("cross_morse.core.build_plant", return_value=model):
            row = model_inventory.inventory(config, 128, 40)
        self.assertEqual(row["parameter_count"], 10)
        self.assertEqual(row["trainable_parameter_count"], 6)
        self.assertEqual(row["buffer_elements"], 7)
        self.assertEqual([item["name"] for item in row["buffers"]], ["running"])

    def test_actual_dimension_or_parameter_or_buffer_drift_is_reported(self):
        for field in ("input_dim", "num_outputs", "parameter_count", "buffer_elements"):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.rows["ogbn-arxiv"])
                changed[field] += 1
                result = model_inventory.compare_source_record(changed, self.registered)
                self.assertFalse(result["source_record_matches"])
                self.assertIn(field, result["source_record_mismatches"])

    def test_wrong_recorded_count_fails_cli_and_default_widths_are_descriptive(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = model_inventory.main(["--expect-parameters", "3116632"])
        self.assertEqual(result, 0)
        self.assertTrue(json.loads(output.getvalue())["manuscript_reference"]["count_agrees_at_printed_precision"])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = model_inventory.main(["--default-widths", "--expect-parameters", "1138216"])
        self.assertEqual(result, 0)
        self.assertFalse(json.loads(output.getvalue())["manuscript_reference"]["count_agrees_at_printed_precision"])
        changed = copy.deepcopy(self.registered)
        changed["backbones_source_status"]["released_full_profiles"]["Peptides-func"]["parameter_count"] += 1
        changed["backbones_source_status"]["released_full_profiles"]["Peptides-func"]["parameter_components"]["head"] += 1
        with patch.object(model_inventory, "load_registered", return_value=changed), contextlib.redirect_stdout(io.StringIO()) as output:
            result = model_inventory.main(["--all-full"])
        self.assertEqual(result, 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "SOURCE_RECORD_MISMATCH")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(model_inventory.main(["--expect-parameters", "1"]), 1)

    def test_matching_source_counts_cannot_hide_an_s5_mismatch_from_cli(self):
        changed = copy.deepcopy(self.registered)
        changed["backbones_table_S5"]["Peptides-func"]["parameters_M"] = 5.2
        with patch.object(model_inventory, "load_registered", return_value=changed), contextlib.redirect_stdout(io.StringIO()) as output:
            result = model_inventory.main(["--all-full"])
        report = json.loads(output.getvalue())
        self.assertTrue(report["source_records_match"])
        self.assertFalse(report["table_S5_counts_agree"])
        self.assertEqual(report["status"], "TABLE_S5_COUNT_MISMATCH")
        self.assertEqual(result, 1)

    def test_default_width_cli_still_requires_its_own_recorded_count(self):
        changed = copy.deepcopy(self.registered)
        changed["backbones_source_status"]["default_width_plant"]["ogbn-arxiv"] += 1
        with patch.object(model_inventory, "load_registered", return_value=changed), contextlib.redirect_stdout(io.StringIO()) as output:
            result = model_inventory.main(["--default-widths"])
        self.assertFalse(json.loads(output.getvalue())["default_source_record_matches"])
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
