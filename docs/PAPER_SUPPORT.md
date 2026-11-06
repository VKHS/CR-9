# Paper support and scientific acceptance criteria

[Method](METHOD.md) · [Analytical statements](THEORY.md) · [Claims ledger](../paper/CLAIMS.md) · [Evidence registration](EVIDENCE_REGISTRATION.md)

The objective is an inspectable chain from each scientific claim to its
assumptions, executed method, original observations and reconstruction command.
The registered archive and record counts describe deposited material. They do
not establish that every theorem, experiment or manuscript sentence is supported.
Current counts come from [AVAILABILITY.md](../AVAILABILITY.md); this document is an
acceptance contract, not a second results registry or a declaration of completion.

## What each check establishes

| Layer | Executable entry point | Interpretation |
|---|---|---|
| Reference mathematics | `make theory-full` | Seeded numerical attempts to falsify the enumerated implications under generated hypotheses; numerical agreement is not a proof |
| Actual implementation | `make test` | Includes sparse PyTorch/reference parity, actual observer calls, inserted-site identity, statistical decision and budget-accounting checks |
| Printed arithmetic | `make si-report` | Recomputes enumerated relationships from printed inputs with explicit precision; cannot establish how those inputs were measured |
| Source identity | `make fingerprint`, `make verify-manifest` | Runtime identity and complete release-file integrity; historical observations retain the source that produced them |
| Original experiments | `make scientific-audit` | Replays and inspects the registered core archives, required auxiliary records, support/selector/budget invariants and enumerated manuscript bindings |
| Reproduction by a reader | `make review REQUIRE_EVIDENCE=1 ...`, then `make readiness ...` | Checks selected public deposits, fresh results, comparisons, PDF identities and report freshness within the documented audit scope |

A mathematical bound requires its hypotheses. A software test requires the
actual implementation it claims to test. A measured result requires observations.
An analytical sensitivity retains its scenario label even when its arithmetic
matches a reported central value.

## Decisions that must be explicit in the manuscript

### 1. Eligible selector family and predecessor map

`cross_morse.statistics.nested_selection` implements the Eq. (3) family:
feedback candidates together with the frozen open-loop comparator `qC`. It
records other open-loop contrasts but excludes those candidates from `Q_s`.
All feasibility tests, simultaneous upper limits, the near-best band, the
simplicity order and the selection guarantee must refer to the same family.

The reference function `cm_theory.selection.eq19_select` defaults to
`candidate_scope="feedback_only"`. Its explicit `all_feasible` option is a
diagnostic reading of the less explicit domain in Eqs. (18)–(19). On the printed
representative rows, admitting all open-loop candidates changes some choices.
Those rows are not a substitute for the full nested selection ledger.

Suggested clarification, conditional on the Eq. (3) interpretation:

> In Eqs. (18)–(19), the feasible family consists of the registered feedback
> candidates and the selected comparator. Other open-loop rows are diagnostic.
> Confirmation may retain a predecessor only if that exact configuration belongs
> to the selected eligible set or is the comparator.

This has a concrete consequence for S11: an MPC-OL predecessor different from
`qC` cannot become eligible under the current selector domain. Resolve that
wording against the original experiment configuration. Changing the scientific
candidate family requires a new declared protocol and new selection/confirmation
evidence; it cannot be inferred from an attractive representative row.

### 2. Confirmation, evaluated arms and deployed policies

The runtime uses strict Holm significance `P < alpha` and a point estimate
strictly beyond the registered practical margin. The decision checker and
canonical registry view use the same numeric boundary. A printed bound
`P < alpha` establishes that inequality; a numeric `P = alpha` does not.

S14 and the registered manuscript protocol specify nine prespecified mechanism
contrasts across datasets. The runtime `_stage_confirm` freezes the selected
route and eligible simpler predecessors on one dataset and records that local
scope in `confirm/contrast_registry.json`. Its local Holm family is separate from
the cross-dataset experiment.

The strict joint gate is `make confirmation-audit CONFIRMATION_CAMPAIGN=...`.
It preserves the original heterogeneous unrounded test outputs, reconstructs
paired effects and replays all nine tests jointly, with strict point margins and
the `Q_s`-or-`qC` fallback. `make scientific-audit` additionally checks source,
configuration, units, original selector/pre-admission artifacts, prepared roles
and optimization seeds against all seven validated official archives, including
every Roman split. Its saved reports and reviewer/readiness checks bind every
campaign file by hash. See [S14_CONFIRMATION.md](S14_CONFIRMATION.md) for the
incomplete import template and complete evidence contract.

This gate does not recompute the heterogeneous original raw P values or execute
supplied analysis code. Original code, settings and environments are preserved
and hashed for their separate reproduction. A comparator-return run can have no
local contrasts, while S14 still includes products/state. That original test
must identify its prespecified registered feedback target; the report marks it
unselected and keeps the comparator decision fixed. Synthetic observations,
printed adjusted P values and S14b sensitivity calculations cannot supply the
missing original tests. The joint decisions must precede admission; a new lock
cannot be backdated to turn a local family into the historical joint experiment.

The transcribed Peptides radius contrast has Holm `P = 0.070` at `alpha = 0.05`.
Its printed one-step upper limit also fails the registered improvement threshold.
The canonical lifecycle therefore returns `qC` on those inputs. The radius
measurements can still be reported as evaluated-arm results, with their own
observations and qualification, but cannot be labelled the admitted output of
the frozen lifecycle. A later admission calculation cannot repair confirmation.
See [paper/README.md](../paper/README.md) and `make decision-check`.

Suggested results wording:

> The evaluated Peptides radius arm did not satisfy the registered confirmation
> rule. Its arm-level measurements are reported separately; the frozen lifecycle
> returns the comparator under the available confirmation and eligibility inputs.

The final manuscript must either use that distinction consistently or provide
the different original records supporting a corrected, explicitly identified
decision. The repository preserves the printed numerical targets in both cases.

### 3. Formal algorithm, executed variant and statistical scope

Algorithm S3 contains certificate branches. The standard E6 study executes
bounded planning with scheduled acquisition and records zero certificate skips
unless a compatible certificate callback is supplied. Name the executed variant
in S46/S46b and preserve the event ledger. A per-input work bound, an expected
work ceiling and synchronized wall time are different quantities.

Independent-graph admission concerns the declared clipped risk difference for
a policy fixed before admission. Dependent cores on the three OGB graphs and
Roman's overlapping masks remain descriptive units. Twelve optimization seeds
do not create twelve independent graphs. Official AP, ROC-AUC and macro-F1
require their actual aggregation/evaluator in every relevant resample; averaging
graph-level scores is not automatically the official benchmark estimand.

Parameter-count agreement with S5 identifies a configuration check. Original
architecture and checkpoint identity still require the archived model inventory,
named tensors, frozen weights, operator construction and nominal-route checks.

## Evidence needed for E1–E20

These rows describe closure criteria. A record-set name denotes an existing
container for applicable measurements, not an assertion that the set has a
dedicated implementation for every listed experiment. Where the current archive
or record schema lacks an output, preserve the original analysis inputs and
script and extend the existing evidence machinery with explicit coverage.

| Experiment | Existing implementation or check | Evidence needed to close the paper claim |
|---|---|---|
| **E1 — Native/inserted identity** | `cross_morse/core.py`; runtime parity and plant tests | State/prediction identity on the actual paper plants, checkpoint/dtype/site inventory; inserted A0 literal bypass and direct recovery semantics |
| **E2 — Sequence predictability** | Reference information bounds and registered S30/S33/S47 targets | Held-out predicted/teacher sequences, initial-only features, sequence accuracy and paired retained benefit; keep the fitted model separate from analytical retention scenarios |
| **E3 — Initial-input prediction** | `cm_experiments/`; `budget_matched` records | Frozen teacher, fitted predictor, development selection, initial-only inference, paired `D_compare` predictions and actual charged work |
| **E4 — Distillation** | `cm_experiments/`; `budget_matched` records | Training-only teacher trajectories, student fit/selection, permitted inputs at deployment, paired outcomes and separate fitting/execution costs |
| **E5 — Predicted-state replanning** | `cm_experiments/`; `budget_matched` records | Predicted-state trajectories and model/search calls; absence of fresh forbidden observations; measured work and paired outcomes under the declared ceiling |
| **E6 — Budget-aware control** | `cm_experiments/budget.py`; reference/runtime episode parity | Executed variant, every paid operation, terminal prediction/restart reserve, cap/ceiling, infeasible and boundary outcomes; original campaign observations |
| **E7 — Spectral ablation** | Sensor implementation and analytical perturbation checks | Matched Full/deleted/scrambled features, capacity/work, seeds, unit predictions and selected-policy outcomes for S17 |
| **E8 — Observer alternatives** | `MeasurementBank`; observer and factorial reference checks | Raw/EMA/global/conditioned/history alternatives with identical or explicitly different stopping domains. A read-only observer on the same trajectory must preserve the nominal output |
| **E9 — Small-instance search** | Proposition 8 reference checks | Complete supported sequence family at horizon at most four, exact objectives, model errors and beam outputs; distinguish model approximation from search suboptimality |
| **E10 — Stage separation and confirmation** | Role, provenance, confirmation, admission and policy-lock checks | Immutable role IDs, fitted-object hashes, complete Holm family, clipping sensitivity, graph/core vectors and exact fallback; no comparison-driven promotion |
| **E11 — Dependence sensitivity** | Dependent-core and Roman campaign summaries | Core construction/counts/buffers, isolation alternatives, all masks and raw paired results for each S27 sensitivity; retain descriptive status |
| **E12 — External baselines** | Pinned upstream harness; `baselines` records | All fourteen declared dataset/method pairs, upstream and adapter revisions, all 48 HPO attempts, final seeds, predictions, checkpoints and scheduler records |
| **E13 — Inserted-site attribution** | Plant identity checks; `inserted_site` record contract | Every protocol-defined native/inserted open-loop/feedback comparison, nominal identity, matched work, paired predictions and source-derived attribution share |
| **E14 — Ensemble grid** | `ensemble_grid` contract; analytical loss-law calculations | Actual distinct members at E = 1, 2, 4, 8, 12, 16, 20; losses, official metrics, uncertainty, fit diagnostics and all additional checkpoint costs |
| **E15 — Complete selection replay** | `nested_selection`; reference terminal-rule parity; scientific audit | Every candidate/configuration, support/feasibility mask, simultaneous interval, nested comparator resample and tie order; explicit domain and full-ledger reconciliation with representative rows |
| **E16 — Crossed uncertainty** | `cm_theory/inference.py`; Note S24 arithmetic | Seed-by-unit arrays and ensemble memberships; graph/seed/residual components and registered resampling; distinguish nonnegative reporting estimates from signed moment estimates |
| **E17 — Cost and timing** | `cm_campaigns/timing.py`, scheduler capture; timing/baseline records | Raw synchronized repetitions, device/dtype/batch, sensing/action events, actual maxima and HPO/training/qualification allocations with an unambiguous denominator |
| **E18 — Spectral-bound tightness** | Dense/sparse operator parity; global numerical remainder bound | Actual operator/state/band powers, omitted-band contribution, approximation errors and constants; independently bounded displacement versus observation, with hypotheses checked |
| **E19 — Support qualification** | Logged-propensity support audit and selection masks | One common weight vector for minimum propensity, ESS and maximum normalized weight; raw counts/cell identity and propagation of unsupported cells through all later decisions |
| **E20 — Full/Lean profiles** | Profile-specific sensors, controller and cost records | Actual sketch size/acquisition period, observer cadence, spectral age, fresh cheap features, action/site traces and paired cost/benefit under the same declared evaluator |

Each measured row needs an unambiguous key: dataset, backbone/checkpoint or
ensemble, exact policy, execution profile, data role, statistical unit and
evaluator. Preserve unrounded values, pairing/unit counts, raw file paths and
SHA-256, source/configuration identity, analysis command/version, environment and
the work/timing denominator. Summary agreement without these inputs does not
reconstruct an experiment.

## Precision and interpretation notes from the current document review

The review concerns a 16-page main PDF with SHA-256
`7c4b5145035d5873057a510a84077ee54ea9dbb2af49e2c7cb7306305a9a15e2`
and a 38-page supplementary PDF with SHA-256
`120ce46ec1b4e4a14000c90d0f7ef64bad4c6b3d7a3800739b8153703675552e`.
These are the current document identities registered in `paper/sources.json`.
The byte-bound review of all 34 numerical groups is recorded in
`paper/transcription_check.json` as an AI-assisted contextual review requested
by the author. It records source correspondence and interpretation within its
stated scope; it does not verify experimental measurements or attest that an
author personally reviewed each cell.

| Location | Resolution needed |
|---|---|
| Corollary 2 | State the general necessary non-strict sample-size bound; make the strict version conditional on positive variance. The printed COCO inputs give a threshold about 341.26, so the integer requirement is at least 342, equivalently greater than 341. It is a necessary condition, not a sufficient sample-size guarantee |
| Note S24 | Under the stated variance formula including the graph term, implied seed SDs round to 0.0076 / 0.0064 / 0.0029. State any different estimator or component convention explicitly |
| S16b interaction wording | The central SO contrast is -0.000125. Four-decimal rounding gives -0.0001, but a literal absolute bound of 0.0001 needs qualification |
| Note S17 | The central calculation gives 2e(74) about 0.08048. Nearest-thousandth rounding gives 0.080; 0.081 is defensible as an explicitly conservative upward bound |
| S6c residual caption | The printed residual is computed after subtracting the sensing term. The caption must match that subtraction and identify where resolvent work is counted |
| Peptides boundary wording | Both Peptides and products are close to the headroom boundary; products is closer on the printed central values. This is a wording choice, not an invalid gate calculation |

`make si-report` emits advisory or informational rows for the numerical and
boundary-reading issues above. The S6c caption is a separate manual source
finding; the existing constants checker verifies its sensing-only arithmetic.
The SI report also distinguishes a value printed in the source from an unprinted parameter
inferred to match a table. The inferred open-loop share near 0.20 and decoder
overhead near 0.03 need explicit provenance before being treated as measured inputs.



## Closing the missing-evidence gap

1. **Discover the existing output tree.** Use `make evidence-discover` with an
   explicit export root. Review rejected and duplicate paths, complete missing
   Roman masks, and identify the original producing source. Discovery proposes
   paths; it never establishes the manuscript relationship or supplies a DOI.
2. **Preflight original completed runs.** Resolve actual model, role, support,
   selection and result mismatches. Preserve failed outcomes. Declare whether a
   run is the original result-producing experiment or a new reproduction.
3. **Import the ten record sets and raw analysis inputs.** Use the existing
   exporters/mappings, original campaigns and scheduler files. Extend missing
   experiment coverage explicitly instead of manufacturing summary rows from
   `reported_results.json`.
4. **Rebuild and reconcile.** Regenerate supported displays, inspect all
   manuscript differences, complete result bindings, resolve the scientific
   wording above, and contextually review the final PDFs before registration.
5. **Verify the public release.** Check public version-DOI bytes, run the
   evidence-required reviewer workflow with those exact inputs, and check its
   freshness with `make readiness`. Report any E1–E20 obligations outside that
   automated gate separately.

The [registration guide](EVIDENCE_REGISTRATION.md) contains the commands. Genuine
unexpected or negative outcomes are scientifically informative. The acceptance
criterion is faithful execution and transparent evidence for the reported claim,
not agreement manufactured by changing seeds, thresholds or recorded targets.
