"""Dense mathematical references against actual sparse operators, sensors and sites.

These are small synthetic CPU executions, including native/A0 identity under
fp32 and bf16 autocast. They do not replace checkpoint-specific E1 evidence or
benchmark reproduction. The class is explicitly skipped without PyTorch.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from cm_theory import operators, spectral  # noqa: E402
from cm_theory._util import complete_bipartite, finish_adjacency, laplacian_eigensystem, random_graph, random_walk  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is needed for the sparse runtime operators")
class RuntimeParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from cross_morse import core
        cls.torch, cls.core = torch, core
        rng = np.random.default_rng(17)
        weighted = np.zeros((6, 6))
        for i, j, weight in ((0, 1, .25), (1, 2, 1.5), (2, 3, 2.), (3, 0, .5), (0, 4, .75)):
            weighted[i, j] = weighted[j, i] = weight
        cls.fixtures = []
        for name, raw, is_weighted in (("unweighted", random_graph(14, rng, density=.35), False),
                                       ("weighted_with_isolate", weighted, True),
                                       ("periodic", complete_bipartite(3, 4), False)):
            rows, cols = np.nonzero(raw)
            graph = core.GraphOperator(torch.as_tensor(np.stack([rows, cols]), dtype=torch.long), len(raw),
                                       edge_weight=torch.as_tensor(raw[rows, cols], dtype=torch.float32) if is_weighted else None,
                                       graph_ptr=torch.tensor([0, len(raw)]))
            # The isolated vertex is added by the actual runtime, independently
            # of the dense reference's isolated-vertex convention.
            cls.fixtures.append((name, graph, finish_adjacency(raw),
                                 rng.normal(size=(len(raw), 4)).astype(np.float32),
                                 rng.normal(size=(len(raw), 4)).astype(np.float32)))

    def _designated(self, backbone):
        with self.torch.random.fork_rng(devices=[]):
            self.torch.manual_seed(123)
            return self.core.DesignatedSitePlant(input_dim=3, hidden_dim=4, num_classes=3,
                                                encoder_layers=1, dropout=0., native_layers=2,
                                                control_depth=8 if backbone == "gps" else 6,
                                                backbone=backbone, attention_heads=2).freeze()

    def test_registry_literal_equals_the_runtime_tuple(self):
        literal = operators.load_registry()
        runtime = self.core.ACTIONS
        self.assertEqual(len(literal), len(runtime))
        for a, b in zip(literal, runtime):
            self.assertEqual((a.id, a.name, a.rho, a.nu, a.alpha, a.eta, a.order, a.probability, a.mode),
                             (b.id, b.name, b.rho, b.nu, b.alpha, b.eta, b.order, b.probability, b.mode))

    def test_transport_and_kernel_match_dense_references(self):
        for name, graph, adjacency, states, _ in self.fixtures:
            P = random_walk(adjacency)
            h = self.torch.as_tensor(states)
            for a in operators.transport_actions():
                for runtime, reference in ((self.core.apply_transport, operators.transport_matrix),
                                           (self.core.apply_kernel, operators.kernel_matrix)):
                    with self.subTest(graph=name, action=a.id, operator=runtime.__name__):
                        graph.reset_counters()
                        actual = runtime(graph, h, a.id).double().numpy()
                        np.testing.assert_allclose(actual, reference(P, a) @ states, rtol=2e-5, atol=2e-5)
                        products = a.order if a.eta is not None else 1
                        self.assertEqual(graph.counters["random_walk_products"], products)
                        self.assertEqual(graph.sparse_products, products)
                        self.assertEqual(graph.sparse_multiply_adds, products * graph.num_edges * states.shape[1])

    def test_runtime_registry_functions_match_the_operator_polynomial(self):
        for a in operators.transport_actions():
            self.assertAlmostEqual(self.core.executed_hops(a.id), operators.executed_range(a), places=12)
            self.assertAlmostEqual(self.core.identity_mass(a.id), operators.identity_mass(a), places=12)
            self.assertAlmostEqual(self.core.executed_identity_coefficient(a.id), operators.identity_coefficient(a), places=12)

    def test_operator_response_matches(self):
        for name, _, adjacency, _, _ in self.fixtures:
            lam, _ = laplacian_eigensystem(adjacency)
            eigenvalues = self.torch.as_tensor(lam, dtype=self.torch.float64)
            for a in operators.transport_actions():
                with self.subTest(graph=name, action=a.id):
                    runtime = self.core.action_operator_response(a.id, eigenvalues).numpy()
                    np.testing.assert_allclose(runtime, spectral.action_response(a, lam), atol=1e-12)

    def test_chebyshev_windows_match_the_numpy_port(self):
        runtime = self.core.chebyshev_window_coefficients().double().numpy()
        np.testing.assert_allclose(runtime, spectral.chebyshev_window_coefficients(), atol=1e-6)

    def test_measurement_bank_filter_matches_dense_windows_and_reuses_products(self):
        for name, graph, adjacency, states, _ in self.fixtures:
            for degree in (1, 8):
                coeff = spectral.chebyshev_window_coefficients(degree=degree)
                for bands in ((0, 2, 5), (1, 3, 4)):
                    with self.subTest(graph=name, degree=degree, bands=bands):
                        bank = self.core.MeasurementBank(graph, states.shape[1], bands=bands, degree=degree)
                        graph.reset_counters()
                        bank.reset_counters()
                        actual = bank._filter(self.torch.as_tensor(states))
                        for b, filtered in zip(bands, actual):
                            expected = spectral.window_matrix(adjacency, coeff[b]) @ states
                            np.testing.assert_allclose(filtered.double().numpy(), expected, rtol=1e-5, atol=2e-6)
                        self.assertEqual(graph.counters["symmetric_products"], degree)
                        self.assertEqual(graph.sparse_products, degree)
                        self.assertEqual(bank.counters["spectral_products"], degree)

    def test_measurement_bank_powers_alignment_and_weighted_frame_match_dense_reference(self):
        _, graph, adjacency, states, source = self.fixtures[1]
        mask = np.arange(len(states)) % 2 == 0
        coeff = spectral.chebyshev_window_coefficients()
        for sketch_dim in (0, 2):
            with self.subTest(sketch_dim=sketch_dim):
                bank = self.core.MeasurementBank(graph, states.shape[1], sketch_dim=sketch_dim, seed=91,
                                                mask=self.torch.as_tensor(mask))
                h, h0 = self.torch.as_tensor(states), self.torch.as_tensor(source)
                bank.prepare(h0, h0[:, :3])
                graph.reset_counters()
                bank.reset_counters()
                measured = bank.measure(h, h[:, :3], depth=0, max_depth=4).double().numpy()
                projection = np.eye(states.shape[1]) if bank.projection is None else bank.projection.double().numpy()
                X, X0 = states @ projection, source @ projection
                weighted = np.sqrt(adjacency.sum(axis=1))[:, None] * X
                filtered = [(spectral.window_matrix(adjacency, coeff[b]) @ X)[mask] for b in bank.bands]
                source_filtered = [(spectral.window_matrix(adjacency, coeff[b]) @ X0)[mask] for b in bank.bands]
                weighted_filtered = [(spectral.window_matrix(adjacency, coeff[b]) @ weighted)[mask] for b in bank.bands]
                powers = np.array([np.sum(value ** 2) for value in filtered])
                weighted_powers = np.array([np.sum(value ** 2) for value in weighted_filtered])
                alignments = np.array([spectral.alignment(current, initial, 1e-12)
                                       for current, initial in zip(filtered, source_filtered)])
                expected = np.r_[powers / (powers.sum() + 1e-12), alignments,
                                  weighted_powers / (weighted_powers.sum() + 1e-12)]
                names = ([f"band_power_{b}" for b in bank.bands] + [f"source_alignment_{b}" for b in bank.bands]
                         + [f"degree_weighted_power_{b}" for b in bank.bands])
                np.testing.assert_allclose([measured[bank.names.index(name)] for name in names], expected,
                                           rtol=3e-5, atol=3e-6)
                # Full sensing filters both ordinary and degree-weighted states;
                # the global Dirichlet feature costs one further sparse product.
                self.assertEqual(bank.counters["spectral_products"], 16)
                self.assertEqual(graph.sparse_products, 17)
                self.assertEqual(bank.counters["spectral_acquisitions"], 1)
                self.assertFalse(bank.metadata()["cached_spectral_values_are_certificates"])

    def test_inserted_site_bypass_and_nonlazy_formula_use_actual_plant_step(self):
        _, graph, adjacency, states, source = self.fixtures[1]
        P = random_walk(adjacency)
        h, h0 = self.torch.as_tensor(states), self.torch.as_tensor(source)
        for backbone in ("gps", "linear_gine"):
            model = self._designated(backbone)
            site = next(i for i in range(model.control_depth) if not model.is_native_site(i))
            for a in operators.transport_actions():
                with self.subTest(backbone=backbone, action=a.id):
                    graph.reset_counters()
                    model.reset_counters()
                    actual = model.step(h, h0, graph, a.id, site=site)
                    if a.id == 0:
                        self.assertIs(actual, h)
                        products = 0
                    else:
                        products = a.order if a.eta is not None else 1
                    expected = operators.inserted_site(states, source, P, a)
                    np.testing.assert_allclose(actual.double().numpy(), expected, rtol=2e-5, atol=2e-5)
                    self.assertEqual(graph.sparse_products, products)
                    self.assertEqual(model.counters["dense_updates"], 0)
                    self.assertEqual(model.counters["dense_multiply_adds"], 0)

    def test_native_and_a0_trajectories_are_identical_in_fp32_and_cpu_bf16(self):
        _, graph, _, states, _ = self.fixtures[1]
        with self.torch.random.fork_rng(devices=[]):
            self.torch.manual_seed(21)
            models = [cls(3, 4, 3, encoder_layers=1, dropout=0.).freeze()
                      for cls in (self.core.FrozenPlant, self.core.EdgeWeightedPlant, self.core.HeterophilyPlant)]
        models += [self._designated(name) for name in ("gps", "linear_gine")]
        for model in models:
            depth = model.control_depth or 4
            for bf16 in (False, True):
                with self.subTest(backbone=model.backbone, precision="bf16" if bf16 else "fp32"), \
                        self.torch.no_grad(), self.torch.autocast("cpu", dtype=self.torch.bfloat16, enabled=bf16):
                    h0 = model.encode(self.torch.as_tensor(states[:, :3]))
                    native = adapted = h0
                    for site in range(depth):
                        native = model.native_step(native, h0, graph, site=site)
                        adapted = model.step(adapted, h0, graph, 0, site=site)
                        self.torch.testing.assert_close(adapted, native, rtol=0., atol=0.)
                    self.torch.testing.assert_close(model.logits(adapted), model.logits(native), rtol=0., atol=0.)
                    self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))

    def test_hybrid_reset_assignments_do_not_transport_or_apply_a_dense_update(self):
        _, graph, _, states, source = self.fixtures[1]
        h, h0 = self.torch.as_tensor(states), self.torch.as_tensor(source)
        safe = h * .37
        shared = self.core.FrozenPlant(3, 4, 3, encoder_layers=1, dropout=0.).freeze()
        for model in (shared, self._designated("gps"), self._designated("linear_gine")):
            sites = [None] if model.control_depth is None else [
                next(i for i in range(model.control_depth) if model.is_native_site(i)),
                next(i for i in range(model.control_depth) if not model.is_native_site(i))]
            for site in sites:
                for action, last_safe, expected in ((8, safe, h0), (9, safe, safe), (9, None, h0)):
                    with self.subTest(backbone=model.backbone, site=site, action=action, source_default=last_safe is None):
                        graph.reset_counters()
                        model.reset_counters()
                        actual = model.step(h, h0, graph, action, last_safe=last_safe, site=site)
                        self.torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
                        self.assertNotEqual(actual.data_ptr(), expected.data_ptr())
                        self.assertEqual(graph.sparse_products, 0)
                        self.assertEqual(model.counters["dense_updates"], 0)
                        self.assertEqual(model.counters["resets"], 1)

    def test_recovery_modes_are_not_silently_treated_as_transport(self):
        _, graph, _, states, _ = self.fixtures[0]
        h = self.torch.as_tensor(states)
        for action in (8, 9):
            for operation in (self.core.apply_kernel, self.core.apply_transport):
                with self.subTest(action=action, operation=operation.__name__), self.assertRaises(ValueError):
                    operation(graph, h, action)
            with self.assertRaises(ValueError):
                self.core.action_operator_response(action, self.torch.tensor([0., 1., 2.]))


if __name__ == "__main__":
    unittest.main()
