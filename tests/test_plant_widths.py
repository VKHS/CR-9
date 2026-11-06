"""The width settings of the full profiles: counts, schema compatibility and execution."""
import importlib.util
import io
import json
import unittest

from scripts._paper_common import DATASETS, ROOT, STEMS, load_registered, printed_half_unit

torch_available = importlib.util.find_spec("torch") is not None
if torch_available:
    import torch
    import torch.nn.functional as F
    from cross_morse.core import DesignatedSitePlant, FrozenPlant, GraphOperator, HeterophilyPlant, build_plant
    from cross_morse.tasks import task_spec
    from cross_morse.training import native_forward


def _chain_graph(num_nodes, graph_ptr=None):
    boundaries = [0, num_nodes] if graph_ptr is None else graph_ptr.tolist()
    sources, targets = [], []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        sources += list(range(start, end - 1)) + list(range(start + 1, end))
        targets += list(range(start + 1, end)) + list(range(start, end - 1))
    return GraphOperator(torch.tensor([sources, targets], dtype=torch.long), num_nodes, "cpu", graph_ptr=graph_ptr)


def _site_model(backbone, **overrides):
    options = dict(encoder_layers=1, dropout=0.0, control_depth=8, native_layers=2,
                   backbone=backbone, attention_heads=2, head_hidden=5)
    if backbone == "gps":
        options.update(feedforward_hidden=24)
    else:
        options.update(block_mlp_hidden=20, block_mlp_layers=3,
                       task=task_spec("Peptides-func", {"num_outputs": 3}))
    options.update(overrides)
    return DesignatedSitePlant(4, 8, 3, **options)


def _legacy_schema(backbone):
    """Tensor schema before optional widths; independent of the new constructors."""
    shapes = {}
    def linear(prefix, inputs, outputs):
        shapes[prefix + ".weight"] = (outputs, inputs)
        shapes[prefix + ".bias"] = (outputs,)
    def norm(prefix):
        shapes[prefix + ".weight"] = shapes[prefix + ".bias"] = (8,)
    layers = 4 if backbone == "shared_transport" else 1
    for layer in range(layers):
        linear(f"encoder.{layer}", 4 if layer == 0 else 8, 8)
    linear("head", 8, 3)
    if backbone == "shared_transport":
        linear("shared", 8, 8)
        norm("norm")
    for layer in range(0 if backbone == "shared_transport" else 2):
        prefix = f"blocks.{layer}"
        if backbone == "linear_gine":
            shapes[prefix + ".epsilon"] = ()
            linear(prefix + ".first", 8, 8)
            linear(prefix + ".second", 8, 8)
            norm(prefix + ".norm")
        else:
            linear(prefix + ".local_first", 8, 8)
            linear(prefix + ".local_second", 8, 8)
            shapes[prefix + ".attention.in_proj_weight"] = (24, 8)
            shapes[prefix + ".attention.in_proj_bias"] = (24,)
            linear(prefix + ".attention.out_proj", 8, 8)
            linear(prefix + ".ff_first", 8, 16)
            linear(prefix + ".ff_second", 16, 8)
            for name in ("local_norm", "attention_norm", "ff_norm"):
                norm(prefix + "." + name)
    return shapes


@unittest.skipUnless(torch_available, "PyTorch is needed to instantiate plants")
class WidthSchemaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(314159)

    def test_default_widths_keep_the_previous_tensor_schema(self):
        for backbone in ("shared_transport", "gps", "linear_gine"):
            with self.subTest(backbone=backbone):
                model = (FrozenPlant(4, 8, 3, dropout=0.0) if backbone == "shared_transport" else
                         DesignatedSitePlant(4, 8, 3, encoder_layers=1, dropout=0.0, control_depth=8,
                                             native_layers=2, backbone=backbone, attention_heads=2))
                shapes = _legacy_schema(backbone)
                self.assertEqual({name: tuple(value.shape) for name, value in model.state_dict().items()}, shapes)
                # Loading uses an independently specified old schema, not a
                # state dictionary produced by another new-width constructor.
                old_state = {name: torch.randn(shape) for name, shape in shapes.items()}
                archive = io.BytesIO()
                torch.save({"model": old_state}, archive)
                archive.seek(0)
                model.load_state_dict(torch.load(archive, weights_only=True)["model"], strict=True)
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, old_state[name]), name)

    def test_changed_widths_require_their_own_checkpoint_schema(self):
        original = FrozenPlant(4, 8, 3)
        widened = FrozenPlant(4, 8, 3, encoder_hidden=12, head_hidden=6)
        with self.assertRaises(RuntimeError):
            widened.load_state_dict(original.state_dict(), strict=True)

    def test_encoder_width_and_two_layer_head_execute(self):
        model = FrozenPlant(4, 8, 3, encoder_layers=3, encoder_hidden=12, head_hidden=6, dropout=0.0)
        self.assertEqual([tuple(layer.weight.shape) for layer in model.encoder], [(12, 4), (12, 12), (8, 12)])
        self.assertEqual(sorted(name for name in model.state_dict() if name.startswith("head")),
                         ["head.hidden.bias", "head.hidden.weight", "head.output.bias", "head.output.weight"])
        self.assertEqual(model.architecture_widths(), {"hidden_dim": 8, "encoder_layers": 3, "encoder_hidden": 12,
                                                       "head_hidden": 6, "head_layers": 2})
        model.freeze()
        logits = model(torch.randn(5, 4), _chain_graph(5), depth=3)
        self.assertEqual(tuple(logits.shape), (5, 3))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(sum(p.numel() for p in model.head.parameters()), 8 * 6 + 6 + 6 * 3 + 3)
        hetero = HeterophilyPlant(4, 8, 3, encoder_hidden=10, dropout=0.0).freeze()
        self.assertEqual(tuple(hetero(torch.randn(5, 4), _chain_graph(5), depth=2).shape), (5, 3))

    def test_three_layer_gine_block_and_wide_feedforward_execute(self):
        gine = _site_model("linear_gine")
        self.assertEqual(tuple(gine.blocks[0].middle.weight.shape), (20, 20))
        self.assertEqual(gine.architecture_widths()["block_mlp_layers"], 3)
        gine.freeze()
        graph = _chain_graph(6, graph_ptr=torch.tensor([0, 3, 6]))
        logits = gine(torch.randn(6, 4), graph, depth=8)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertTrue(torch.isfinite(logits).all())
        gps = _site_model("gps").freeze()
        self.assertEqual(tuple(gps.blocks[0].ff_first.weight.shape), (24, 8))
        self.assertEqual(gps.architecture_widths()["feedforward_hidden"], 24)
        logits = gps(torch.randn(6, 4), _chain_graph(6, graph_ptr=torch.tensor([0, 3, 6])), depth=8)
        self.assertEqual(tuple(logits.shape), (6, 3))
        self.assertTrue(torch.isfinite(logits).all())

    def test_invalid_widths_are_rejected(self):
        with self.assertRaises(ValueError):
            FrozenPlant(4, 8, 3, encoder_hidden=0)
        with self.assertRaises(ValueError):
            FrozenPlant(4, 8, 3, head_hidden=-1)
        with self.assertRaises(ValueError):
            DesignatedSitePlant(4, 8, 3, encoder_layers=1, control_depth=8, native_layers=2,
                                backbone="linear_gine", block_mlp_layers=4)
        with self.assertRaises(ValueError):
            DesignatedSitePlant(4, 8, 3, encoder_layers=1, control_depth=8, native_layers=2,
                                backbone="gps", attention_heads=2, feedforward_hidden=0)

    def test_widths_are_not_silently_truncated_or_ignored(self):
        base = {"hidden_dim": 8, "encoder_layers": 3, "depth": 8, "backbone": "shared_transport"}
        for key in ("encoder_hidden", "head_hidden", "feedforward_hidden", "block_mlp_hidden", "block_mlp_layers"):
            for value in (True, 2.0, 2.5, "12"):
                with self.subTest(key=key, value=value):
                    config = dict(base, **{key: value})
                    if key == "feedforward_hidden":
                        config.update(backbone="gps", native_layers=2, attention_heads=2)
                    elif key.startswith("block_mlp"):
                        config.update(backbone="linear_gine", native_layers=2)
                    with self.assertRaises(ValueError):
                        build_plant(4, 3, config)
        for extra in ({"feedforward_hidden": 24}, {"block_mlp_hidden": 20},
                      {"block_mlp_layers": 3}, {"block_mlp_layers": 2.0}):
            with self.subTest(ignored_setting=extra), self.assertRaises(ValueError):
                build_plant(4, 3, dict(base, **extra))
        with self.assertRaises(ValueError):
            FrozenPlant(4, 8, 3, encoder_layers=1, encoder_hidden=12)
        for heads in (0, True, 2.5):
            with self.subTest(attention_heads=heads), self.assertRaises(ValueError):
                _site_model("gps", attention_heads=heads)

    def test_wide_blocks_preserve_nominal_and_inserted_site_semantics(self):
        graph = _chain_graph(6, torch.tensor([0, 3, 6]))
        for backbone in ("gps", "linear_gine"):
            with self.subTest(backbone=backbone):
                model = _site_model(backbone).freeze()
                h0 = model.encode(torch.randn(6, 4))
                native, adapted = h0, h0
                for site in range(8):
                    old = adapted
                    native = model.native_step(native, h0, graph, site=site)
                    adapted = model.step(adapted, h0, graph, 0, site=site)
                    self.assertTrue(torch.equal(native, adapted), (backbone, site))
                    if not model.is_native_site(site):
                        self.assertIs(adapted, old)
                inserted = next(site for site in range(8) if not model.is_native_site(site))
                model.reset_counters(); graph.reset_counters()
                model.step(adapted, h0, graph, 1, site=inserted)
                self.assertEqual(model.counters["dense_updates"], 0)
                self.assertEqual(model.counters["dense_multiply_adds"], 0)
                self.assertGreater(graph.sparse_multiply_adds, 0)

    def test_nonlinear_graph_head_pools_before_decoding_and_graphs_stay_separate(self):
        graph = _chain_graph(6, torch.tensor([0, 3, 6]))
        gine = _site_model("linear_gine").freeze()
        h = torch.randn(6, 8)
        pooled = torch.stack((h[:3].mean(0), h[3:].mean(0)))
        expected = gine.head["output"](F.relu(gine.head["hidden"](pooled)))
        torch.testing.assert_close(gine.logits(h, graph), expected)
        for backbone in ("gps", "linear_gine"):
            with self.subTest(backbone=backbone):
                model = _site_model(backbone).freeze()
                x = torch.randn(6, 4)
                original = model(x, graph, 8)
                changed = x.clone(); changed[3:] += 50
                altered = model(changed, graph, 8)
                first_rows = 1 if backbone == "linear_gine" else 3
                self.assertTrue(torch.equal(original[:first_rows], altered[:first_rows]))

    def test_widened_parameters_have_gradients_without_unused_padding(self):
        graph = _chain_graph(6, torch.tensor([0, 3, 6]))
        models = [FrozenPlant(4, 8, 3, encoder_hidden=12, head_hidden=6, dropout=0., activation="gelu"),
                  HeterophilyPlant(4, 8, 3, encoder_hidden=12, head_hidden=6, dropout=0., activation="gelu"),
                  _site_model("gps", activation="gelu"), _site_model("linear_gine", activation="gelu")]
        for model in models:
            with self.subTest(backbone=model.backbone):
                output = native_forward(model, torch.randn(6, 4), graph, model.control_depth or 3)
                F.mse_loss(output, torch.randn_like(output)).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_q0_optimizes_both_head_layers_without_updating_native_blocks(self):
        model = _site_model("linear_gine", activation="gelu")
        graph = _chain_graph(6, torch.tensor([0, 3, 6]))
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        params = list(model.encoder.parameters()) + list(model.head.parameters())
        optimizer = torch.optim.SGD(params, lr=.05)
        logits = native_forward(model, torch.randn(6, 4), graph, depth=0)
        F.binary_cross_entropy_with_logits(logits, torch.tensor([[0., 1., 0.], [1., 0., 1.]])).backward()
        optimizer.step()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                if name.startswith(("encoder.", "head.")):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertFalse(torch.equal(parameter, before[name]))
                else:
                    self.assertIsNone(parameter.grad)
                    self.assertTrue(torch.equal(parameter, before[name]))
        model.freeze()
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_work_counts_include_actual_encoder_head_and_block_widths(self):
        graph = _chain_graph(6, torch.tensor([0, 3, 6]))
        # Six node rows, or two graph rows at the Peptides head. Counts below
        # are independent dimension formulas, including GPS attention Q/K/V,
        # output projection, and both graph-local attention matrix products.
        encoder = 6 * (4*12 + 12*12 + 12*8)
        node_head = 6 * (8*6 + 6*3)
        cases = [
            (FrozenPlant(4, 8, 3, encoder_layers=3, encoder_hidden=12, head_hidden=6, dropout=0.),
             3, encoder + 3*6*8*8 + node_head, 3),
            (HeterophilyPlant(4, 8, 3, encoder_layers=3, encoder_hidden=12, head_hidden=6, dropout=0.),
             3, encoder + 3*2*6*8*8 + node_head, 3),
            (_site_model("gps"), 8,
             6*4*8 + 2*(2*6*8*8 + 4*6*8*8 + 2*(3*3 + 3*3)*8 + 2*6*8*24) + 6*(8*5 + 5*3), 2),
            (_site_model("linear_gine"), 8,
             6*4*8 + 2*6*(8*20 + 20*20 + 20*8) + 2*(8*5 + 5*3), 2),
        ]
        for model, depth, expected, updates in cases:
            with self.subTest(backbone=model.backbone):
                model.freeze(); model.reset_counters(); graph.reset_counters()
                model(torch.randn(6, 4), graph, depth)
                self.assertEqual(model.counters["dense_multiply_adds"], expected)
                self.assertEqual(model.counters["dense_updates"], updates)
                self.assertEqual(graph.sparse_multiply_adds, 8*8*updates)
                previous = model.counters["dense_multiply_adds"]
                model.logits(torch.randn(6, 8), graph)
                rows = 2 if model.task.prediction_level == "graph" else 6
                self.assertEqual(model.counters["dense_multiply_adds"] - previous,
                                 rows*(8*model.head_hidden + model.head_hidden*3))


@unittest.skipUnless(torch_available, "PyTorch is needed to instantiate plants")
class FullProfileCountTests(unittest.TestCase):
    def test_full_profile_parameter_counts_agree_with_table_s5_rounding(self):
        registered = load_registered()
        released = registered["backbones_source_status"]["released_full_profiles"]
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                config = json.loads((ROOT / f"configs/{STEMS[dataset]}_full.json").read_text(encoding="utf-8"))
                row = released[dataset]
                spec = task_spec(dataset, {"num_outputs": row["num_outputs"]})
                model = build_plant(row["input_dim"], row["num_outputs"], config["model"], spec)
                count = sum(p.numel() for p in model.parameters())
                self.assertEqual(count, row["parameter_count"])
                target = registered["backbones_table_S5"][dataset]["parameters_M"]
                self.assertLessEqual(abs(count - target * 1_000_000), printed_half_unit(target) * 1_000_000)
                self.assertEqual(model.architecture_widths(), row["architecture_widths"])
                self.assertEqual(model.hidden_dim, registered["backbones_table_S5"][dataset]["hidden"])
                self.assertEqual(model.task, spec)


if __name__ == "__main__":
    unittest.main()
