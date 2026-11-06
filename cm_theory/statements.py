"""The analytical statements of the manuscript and where each one lives in this repository.

One :class:`Statement` per definition, lemma, proposition, theorem and corollary
of the main text, the Supplementary Lemma 3 (work bound of Algorithm S3) and the
equations whose arithmetic the lifecycle executes (Eqs. (5)-(7), (10), (19),
(32)-(33), the Note S19 cost model and the Note S24 variance hierarchy). The
fields name the reference implementation in :mod:`cm_theory`, the runtime code
that executes or relies on the statement, the numerical checks of
:mod:`cm_theory.verification` and the tests. ``docs/THEORY.md`` is generated
from this table by ``scripts/verify_theory.py --write-doc``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Statement:
    key: str
    label: str
    title: str
    source: str
    claim: str
    theory: tuple[str, ...]
    runtime: tuple[str, ...]
    checks: tuple[str, ...]
    limits: str


STATEMENTS: tuple[Statement, ...] = (
    Statement(
        "definition-1", "Definition 1", "Adapter-instrumented frozen GNN and supported model class",
        "main Section 2; Supplementary Tables S5b-S5c",
        "Registry action A0 reproduces the native aggregation (0.3 I + 0.7 P on the shared plants); at an inserted "
        "site V_a(H, R) = (1 - tau_a) H + tau_a [(1 - alpha_a) T_a H + alpha_a R] with tau_A0 = 0 a literal bypass, "
        "and every V_a preserves a common node state (affine weights sum to one).",
        ("cm_theory.operators.nominal_shared_matrix", "cm_theory.operators.inserted_site"),
        ("cross_morse/core.py: FrozenPlant.native_step / FrozenPlant.step (E1 identity)",
         "cross_morse/core.py: DesignatedSitePlant.step (inserted sites)"),
        ("definition1_nominal_identity", "definition1_inserted_sites"),
        "Operator level only; the E1 tolerance test of the trained plants (Table S5c) needs the run archives."),
    Statement(
        "lemma-1", "Lemma 1", "Executed transport validity",
        "main Eq. (8); Supplementary Notes S8-S14",
        "If every kernel is nonnegative and row stochastic, w lies in the simplex and 0 <= nu <= 1, then "
        "T = (1 - nu) I + nu sum_m w_m K_m is nonnegative and row stochastic; every registered action is a "
        "polynomial in P with nonnegative coefficients summing to one.",
        ("cm_theory.operators.mixture_transport", "cm_theory.operators.operator_polynomial",
         "cm_theory.operators.transport_matrix"),
        ("cross_morse/core.py: apply_kernel, apply_transport, ACTIONS",),
        ("lemma1_random_mixtures", "lemma1_registered_polynomials"),
        "Floating-point tolerance 1e-11 on row sums."),
    Statement(
        "lemma-2", "Lemma 2", "Mass-corrected truncated resolvent",
        "main Eqs. (9), (21)",
        "For 0 < eta <= 1 both K_eta = eta [I - (1 - eta) P]^{-1} and K~_{eta,T} are nonnegative and row "
        "stochastic, and ||K_eta - K~_{eta,T}||_inf <= 2 (1 - eta)^{T+1}; exactly, "
        "K_eta - K~_{eta,T} = (1 - eta)^{T+1} (P^{T+1} K_eta - I).",
        ("cm_theory.operators.truncated_resolvent", "cm_theory.operators.exact_resolvent",
         "cm_theory.operators.resolvent_truncation_error"),
        ("cross_morse/core.py: apply_kernel (on-line powers P H, ..., P^T H)",),
        ("lemma2_resolvent_bound",),
        "Dense references on graphs of up to a few dozen nodes, including periodic (bipartite) random walks."),
    Statement(
        "eq-10", "Eq. (10)", "Executed range, reach and identity quantities of the registry",
        "main Eq. (10); Supplementary Tables S6 and S6b",
        "h_a = nu [(1 - rho) + rho sum_t t eta (1 - eta)^t], the identity mass (1 - eta)^{T+1} and the identity "
        "coefficient (1 - nu) + nu rho [eta + (1 - eta)^{T+1}] equal the moments of the executed operator "
        "polynomial; the maximum reach is T for a resolvent action and one otherwise.",
        ("cm_theory.operators.executed_range", "cm_theory.operators.identity_coefficient",
         "cm_theory.operators.identity_mass", "cm_theory.operators.maximum_reach"),
        ("cross_morse/core.py: executed_hops, identity_mass, executed_identity_coefficient",
         "scripts/_paper_common.py (standard-library restatement)"),
        ("eq10_registry_moments",),
        "Table S6b values themselves are checked against the registry by scripts/check_paper_constants.py."),
    Statement(
        "proposition-1", "Proposition 1", "Deterministic replay equivalence",
        "main Section 2; Supplementary Notes S8-S14",
        "For a fixed graph, input, frozen model and deterministic sensors and policy, the realized closed-loop "
        "action sequence replayed open loop yields identical hidden states and predictions.",
        ("cm_theory.control.closed_loop", "cm_theory.control.open_loop"),
        ("cross_morse/policies.py: execute_policy (feedback routes and stored sequences)",),
        ("proposition1_replay",),
        "Exact (bitwise) equality on dense float64 plants; GPU nondeterminism is outside the statement."),
    Statement(
        "proposition-2", "Proposition 2", "Fixed-mode finite-depth perturbation",
        "main Eq. (22); Supplementary Notes S8-S14 and Table S22",
        "e_{l+1} <= beta_l [e_l + M_l eps_l^op] + R_l e_l^ref with beta_l = L_LN (1 - alpha_l) L_phi ||W_l||_2 and "
        "M_l = ||H~_l||_{inf,2}; the product-sum bound follows by unrolling, and the frozen-head CE-with-logits "
        "change per node is at most sqrt(2) ||W_head||_2 e_L.",
        ("cm_theory.stability.trajectory_tube", "cm_theory.stability.unrolled_bound",
         "cm_theory.stability.logit_loss_bound"),
        ("cross_morse/core.py: FrozenPlant.step (retention update of Eq. (8))",),
        ("proposition2_tube", "proposition2_logit_bound"),
        "Checked on the layer-normalization-free member of the class (L_LN = L_phi = 1 with ReLU); the local "
        "layer-normalization factor is a separate assumption."),
    Statement(
        "proposition-3", "Proposition 3", "Action stability",
        "main Eq. (23); Supplementary Table S21",
        "If every score is L_J-Lipschitz in the state and the gap between the best and second-best supported "
        "actions is m(s), the selected action is unchanged whenever 2 L_J ||s - s~|| < m(s).",
        ("cm_theory.stability.score_gap", "cm_theory.stability.stable_by_gap"),
        ("cross_morse/scoring.py: action_scores",),
        ("proposition3_gap_stability",),
        "The factor two is shown to be tight for fixed action sets. Runtime use additionally requires invariant support, "
        "recovery, budget masks, site type and search decisions; these are not certified by this scalar helper."),
    Statement(
        "theorem-1", "Theorem 1", "Finite-family supported selection",
        "main Eqs. (19), (24)-(25); Supplementary Algorithm S4 and Table S20",
        "For a declared candidate domain and its feasible subset Q_feas, on the simultaneous "
        "coverage event every member of Q_s (hence q_M) satisfies Delta(q) < -eps_R and "
        "Delta(q) <= min_{Q_feas} Delta + eps_eq + 2 r_max, unless the comparator is returned.",
        ("cm_theory.selection.eq19_select", "cm_theory.selection.theorem1_guarantee",
         "cm_theory.selection.max_t_radii"),
        ("cross_morse/statistics.py: nested_selection, _max_t_interval",),
        ("theorem1_selection", "theorem1_runtime_parity"),
        "Simulated families with Gaussian estimation error and max-t radii; coverage itself is a property of "
        "Algorithm S4 on the real units. The reference defaults to Eq. (3)'s feedback-only domain and also checks an "
        "all-feasible diagnostic, using the same family for selection, coverage and the guarantee. The candidate-domain "
        "contract across Eqs. (3), (18)-(19) and Table S11 remains unresolved. The current runtime uses feedback-only "
        "eligibility; its terminal rule is compared with the reference on synthetic nested-selection outputs, including "
        "metric/resource exclusions and failed observations. This is not a coverage proof for the real bootstrap."),
    Statement(
        "theorem-2", "Theorem 2", "Independent-graph clipped-loss admission",
        "main Eqs. (26)-(27); Supplementary Table S13",
        "For i.i.d. clipped paired differences in [-B, B], Pr(mu_clip <= U_a) >= 1 - delta_a with "
        "U_a = D_bar + sqrt(2 S^2 log(2/delta_a)/J) + 14 B log(2/delta_a)/(3 (J - 1)); the clipped fraction obeys "
        "f B^2 <= (J - 1) S^2/J + D_bar^2 deterministically.",
        ("cm_theory.admission.admission_upper", "cm_theory.admission.clipped_fraction_bound"),
        ("cross_morse/statistics.py: empirical_bernstein, admission",),
        ("theorem2_coverage", "theorem2_clipping_identity", "theorem2_runtime_parity"),
        "Monte Carlo coverage on bounded distributions (two-point, uniform, frequent small gains with rare large harms); the i.i.d. graph "
        "assumption is part of the statement."),
    Statement(
        "corollary-1", "Corollary 1", "Lifecycle error budget",
        "main Section 2; Supplementary Note S16",
        "On E_s and E_a, probability at least 1 - delta_s - delta_a, either q_D = q_C or the selection clause of "
        "Theorem 1 and mu_clip(q~_M) <= U_a <= -eps_a hold together; the confirmation map adds no term.",
        ("cm_theory.selection.eq19_select", "cm_theory.admission.admission_upper"),
        ("cross_morse/statistics.py: nested_selection, confirmation, admission",),
        ("corollary1_lifecycle",),
        "Simulated lifecycles with independent selection and admission samples; per dataset only. The reference "
        "declares feedback_only scope and checks Theorem 1 on the selector's same effective family."),
    Statement(
        "corollary-2", "Corollary 2", "Minimum admission sample size",
        "main Eq. (28); Supplementary Note S16",
        "U_a <= -eps_a requires J >= 1 + 14 B log(2/delta_a)/(3 (|D_bar| - eps_a)); equality is possible at zero "
        "sample variance, correcting the manuscript's strict-bound edge case. The numerator constant is "
        "2.045 nats at B = 0.10, delta_a = 0.025.",
        ("cm_theory.admission.minimum_admission_threshold", "cm_theory.admission.minimum_admission_units"),
        ("cross_morse/statistics.py: empirical_bernstein",),
        ("corollary2_necessity",),
        "Printed counts are compared with the registry by scripts/check_paper_constants.py and "
        "scripts/check_si_identities.py."),
    Statement(
        "proposition-4", "Proposition 4", "Value of state information",
        "main Section 2; Supplementary Note S16",
        "Every policy whose realized sequences lie in the supported family satisfies Delta(pi) >= Delta_oracle, "
        "with equality only if it attains the per-input oracle almost everywhere; Delta(q_D)/Delta_oracle lies in "
        "[0, 1] when Delta(q_D) <= 0.",
        ("cm_theory.selection.oracle_gap", "cm_theory.selection.policy_gap"),
        ("cross_morse/statistics.py: global_candidate_oracle", "cross_morse/training.py: audit_training"),
        ("proposition4_oracle",),
        "Finite families of sequence losses."),
    Statement(
        "proposition-5", "Proposition 5", "Equal-compute break-even under the ensemble law",
        "main Eq. (29); Supplementary Tables S26 and S26b",
        "Under l(E) = l_inf + D_inf/E, q_D improves on the comparator at equal compute iff "
        "|Delta| > D_inf o/(E (1 + o)).",
        ("cm_theory.economics.break_even_effect", "cm_theory.economics.fit_ensemble_law"),
        ("scripts/records.py (Table S26b record builder)",),
        ("proposition5_break_even", "proposition5_fit"),
        "The law itself is an assumption that E14 measures."),
    Statement(
        "definition-2", "Definition 2", "Deployment exchange rate and amortization horizon",
        "main Eq. (30); Table 2; Supplementary Table S23b",
        "lambda* = |Delta| / (E C(q_D) - C(q_C)); a deployer pricing work at lambda prefers q_D iff lambda < lambda*, "
        "and qualification is amortized to a share theta of the added work after "
        "M* = 3600 C_qual / [theta (E C(q_D) - C(q_C)) t_nom] inputs.",
        ("cm_theory.economics.exchange_rate", "cm_theory.economics.amortization_horizon",
         "cm_theory.economics.net_benefit"),
        ("scripts/_paper_common.py: exchange_rate_mnats, amortization_thousand_input_seconds",),
        ("definition2_economics",),
        "Identities; the printed Table 2 values are checked by scripts/check_paper_constants.py."),
    Statement(
        "proposition-6", "Proposition 6", "Band-limited sensitivity and alignment identity",
        "main Eqs. (6.1)-(6.2); Supplementary Note S17 parts (i)-(v); Figure S18c",
        "||(T_a - T_a')X||_F <= [sum_b s_aa'b^2 P_b(X)]^{1/2} in the symmetric frame; the row-stochastic "
        "realization adds d_min^{-1/2}; windows satisfy |P_b - ||G_b X||^2| <= eps_b ||X||^2; retention and "
        "reference terms add as in (iv'); the alignment identity (6.2) is exact at the registered eps; the "
        "Chebyshev recurrence equals the dense window and a Gaussian sketch has relative SD <= sqrt(2/k).",
        ("cm_theory.spectral.displacement_bound", "cm_theory.spectral.window_error",
         "cm_theory.spectral.alignment_identity_residual", "cm_theory.spectral.window_recurrence",
         "cm_theory.spectral.sketch_relative_sd"),
        ("cross_morse/core.py: chebyshev_window_coefficients, MeasurementBank._filter, action_operator_response",),
        ("proposition6_band_bound", "proposition6_windows", "proposition6_update_and_head",
         "proposition6_retention_terms", "proposition6_alignment_and_frame", "proposition6_sketch",
         "proposition6_reach_score"),
        "Exact eigendecompositions of small graphs; global spectral grid bounds include a derivative-based remainder "
        "and one-sided window boundaries. Float64 references are not interval-arithmetic certificates. The tightness "
        "ratio of Figure S18c is a separate empirical audit."),
    Statement(
        "proposition-7", "Proposition 7", "Model-free one-step regret under support qualification",
        "main Eqs. (7.1)-(7.2); Supplementary Tables S7, S28 and S28b",
        "On the simultaneous accuracy event Delta(pi_one) <= min_a Delta(a) + 2 max_a e(a) <= 2 max_a e(a) when A0 "
        "is supported; e(n) = 2B sqrt(log(2|A|/delta)/(2n)) (Hoeffding) and the empirical-Bernstein radius e_B(n) "
        "cover simultaneously; ESS >= 64 forces omega_max <= 0.125.",
        ("cm_theory.admission.hoeffding_radius", "cm_theory.admission.bernstein_radius",
         "cm_theory.admission.weight_cap_from_ess"),
        ("cross_morse/learning.py: _support (ESS and maximum normalized weight per cell)",
         "cross_morse/scoring.py: action_scores (one-step argmin)"),
        ("proposition7_regret", "proposition7_radius_coverage", "proposition7_constants", "eq7_support_identity"),
        "Uniform weights and independent logged units, as the statement requires; self-normalized weights are "
        "excluded by the statement itself."),
    Statement(
        "proposition-8", "Proposition 8", "Approximate search regret",
        "main Section 2; Supplementary Note S20",
        "If sup_a |J^(a) - J(a)| <= e_H and the search returns a zeta_H-approximate minimizer of J^, then "
        "J(a^) - min_a J(a) <= 2 e_H + zeta_H.",
        ("cm_theory.stability.search_regret_bound",),
        ("cross_morse/policies.py: beam_plan",),
        ("proposition8_search",),
        "Random finite sequence families; e_H is a supremum, not a per-step RMSE."),
    Statement(
        "proposition-9", "Proposition 9", "Conditional action stability and selective sensing",
        "main Section 2; Supplementary Note S23 and Algorithm S3",
        "If the gap between the two smallest scores at the predicted state exceeds 2 L_J r and the supported set "
        "is invariant on the ball, the minimizing action is the same at every state of the ball.",
        ("cm_theory.stability.certified_ball",),
        ("cm_experiments/budget.py: Certificate.allows_skip",),
        ("proposition9_certified_ball",),
        "Region containment and support, recovery, budget and search invariance are premises. The check only tests "
        "fixed finite families of linear scores; it does not generate a trained-runtime certificate."),
    Statement(
        "lemma-3", "Lemma 3 (Supplementary)", "Work bound of Algorithm S3",
        "Supplementary Algorithm S3, Lemma 3 (Note S22) and Figure S11",
        "With declared upper bounds on every operation, a feasible budget W >= C_qC(x), W_C >= C_qC(x) and "
        "C_pred <= W_C, the invariant w + W_C <= W holds after every paid operation and every branch that returns "
        "a prediction uses at most W; the infeasible branch pays nothing.",
        ("cm_theory.budget.algorithm_s3_reference", "cm_theory.budget.lemma3_invariants"),
        ("cm_experiments/budget.py: run_budget_episode, WorkLedger",),
        ("lemma3_work_bound", "lemma3_reference_agreement"),
        "Synthetic backends whose observed work never exceeds declared bounds. Four terminal statuses, paid-operation "
        "order, executed actions, acquisition/recovery counts and work are compared with the reference. Certificate "
        "premises are supplied by the synthetic cases; no learned-controller certificate or full benchmark path is established."),
    Statement(
        "proposition-10", "Proposition 10", "Budget-matched decision boundary",
        "main Section 2 and Figure 6; Supplementary Note S25 and Table S47",
        "(i) At a common ceiling q_pred beats q_FB iff phi > 1 and q_B iff phi > r_B; (ii) with ensemble spend the "
        "crossovers become 1 - b(o - o_dec)/g and r_B - [b(o - o_dec) - b(o - o_B)]/g; (iii) under priced work "
        "feedback is preferred iff lambda < (1 - phi) g / [EC(q_FB) - EC(q_pred)].",
        ("cm_theory.economics.decision_boundary", "cm_theory.economics.priced_crossover"),
        ("cm_experiments/ (E3-E6 arms)",),
        ("proposition10_boundaries",),
        "The additive ensemble-spend model of (ii) is a stated sensitivity assumption."),
    Statement(
        "proposition-11", "Proposition 11", "Comparator-relative advantage decomposition",
        "main Eq. (31); Supplementary Note S21",
        "V^pi_0 - V^C_0 = E_pi sum_l A^C_l(h_l, a_l) for every policy executing from the same initial history.",
        ("cm_theory.control.advantage_identity",),
        ("cross_morse/learning.py: ActionCritic (targets the nominal-relative quantity of Eq. (13))",),
        ("proposition11_advantage",),
        "Exact enumeration of finite-horizon processes with history-dependent stochastic policies. The current "
        "ActionCritic targets nominal-relative Eq. (13), not the comparator-relative advantage of Eq. (31)."),
    Statement(
        "eq-5-6", "Eqs. (5)-(6)", "Training-only entry contract and log-depth-integrated loss",
        "main Eqs. (5)-(6); Supplementary Tables S3-S4",
        "DIL_log d is the trapezoid rule in log2 depth normalized by log2(d_K/d_1); the six gates of Eq. (5) with "
        "eps_plant = 0.004, tau_PDD = 1.0, tau_DIL = 0.020, tau_sup = 0.80 and eps_headroom = 0.003.",
        ("cm_theory.contract.log_depth_integrated_loss", "cm_theory.contract.entry_contract"),
        ("cross_morse/training.py: audit_training",),
        ("eq6_dil",),
        "The Table S4 decisions are replayed by scripts/check_si_identities.py."),
    Statement(
        "eq-32-33", "Eqs. (32)-(33)", "Expected-cost identity and branch accounting",
        "main Eqs. (32)-(33); Supplementary Note S22 and Table S23",
        "E C = C_g + p0 C_C,0 + (1 - p0)[(1 - p1) C_S + p1 (C_A + C_C,1)] reduces to "
        "(1 - p_fallback) C_controller + p_fallback C_qC for the deployed rules (C_g = 0, p1 = 0).",
        ("cm_theory.economics.expected_cost", "cm_theory.economics.expected_cost_general"),
        ("cm_experiments/budget.py: WorkLedger",),
        ("eq33_branch_accounting",),
        "Expectation over mutually exclusive branches, checked against simulated branch frequencies."),
    Statement(
        "note-s19", "Note S19", "Action cost model and admissible truncation order",
        "Supplementary Note S19; Tables S6c and S44",
        "m(T) = (1 + r + s + f_res r (T - 1))/(1 + r); T <= 1 + [m (1 + r) - (1 + r + s)]/(f_res r); full-state "
        "degree-8 sensing costs 8r/(1 + r) of a nominal step; the S44 tail model is C_qC [1 + o_fix + f v] with "
        "v = r (T - 1)/(1 + r).",
        ("cm_theory.budget.per_member_multiplier", "cm_theory.budget.admissible_truncation_order",
         "cm_theory.budget.sensing_share", "cm_theory.budget.tail_scenarios"),
        ("cross_morse/core.py: GraphOperator counters (sparse products and multiply-adds)",),
        ("note_s19_cost_model",),
        "A floating-point work model; memory traffic and wall time are outside it."),
    Statement(
        "note-s24", "Note S24", "Two resampling sources and inference for the comparison",
        "main Section 2 and Methods; Supplementary Note S24 and Tables S14b, S42",
        "Var D_bar = sigma_u^2/J + sigma_v^2/S + sigma_e^2/(J S); the balanced two-way mean squares give "
        "Var^_unbiased(D_bar) = (MS_graph + MS_seed - MS_e)/(J S); nonnegative reporting estimates are separate. "
        "Minimum detectable effects are 2.80 SE (two-sided 0.05) "
        "and 3.34 SE at the first Holm step of four; equivalence uses two one-sided tests at the margin.",
        ("cm_theory.inference.variance_components", "cm_theory.inference.minimum_detectable_effect",
         "cm_theory.inference.tost", "cm_theory.inference.holm_adjust"),
        ("cm_campaigns/numeric.py: paired_effect, holm", "cross_morse/statistics.py: holm_adjust"),
        ("note_s24_variance_components", "methods_power_and_equivalence"),
        "Balanced designs; the crossed seed-by-unit bootstrap of the campaign is a separate estimator."),
)

_BY_KEY = {s.key: s for s in STATEMENTS}


def statement(key: str) -> Statement:
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(f"unknown statement {key!r}; known: {sorted(_BY_KEY)}") from None
