# Executable choices

The [method guide](../docs/METHOD.md) describes the algorithm and supported model
classes. This page records choices needed to identify an execution. Protocol
values and page references live in [registered_constants.json](registered_constants.json)
and [sources.json](sources.json); printed outcomes live in
[reported_results.json](reported_results.json).

## Model and observer

| Component | Executable choice | Run identity |
|---|---|---|
| Backbone | `cross_morse/core.py::build_plant` defines encoders, native updates, heads and parameter-free inserted sites. The [seven-dataset inventory](../docs/DATASETS.md#model-parameter-inventory) records explicit widths, parameter scope, Table S5 count agreement and compatibility defaults. Archived configurations and checkpoint identities identify the executed model. | `train/model_inventory.json`, `architecture_widths`, model configuration, checkpoint hashes and nominal-route check |
| Prediction head | Transport full profiles use a linear head; GPS and linear-GINE full profiles use a frozen two-layer MLP. Its Lipschitz readout factor depends on both layer spectral norms, activation and any pooling; the single-matrix linear-head specialization applies only to linear heads. | Head parameters, activation, task readout and model configuration |
| Transport | Supported native sites expose the declared nonnegative linear aggregation. Proteins uses the mean edge-feature weight; Peptides uses `1 + mean(bond categorical codes)`. Nonlinear GINE messages and arbitrary attention require a separate compatible adapter. | Native/inserted-site map and prepared edge-weight metadata |
| Spectral observer | Six equal-interval, degree-8 Chebyshev windows with Jackson damping; retained bands `[0, 2, 5]`. Cheap support coordinates update at every site; spectral values cache between scheduled acquisitions. | Window coefficients, acquisition cadence, support and action traces |
| Large graphs | Products uses exact full-graph sparse propagation and dense calls chunked at 65,536 rows. Proteins uses degree-rank strata and two-hop buffered cores for descriptive statistics. | Execution configuration, unit membership, memory and work records |

Readout Lipschitz constants are analytical assumptions; the runtime does not
compute or certify them.

## Selection and evaluation

Training units use cumulative floor cuts at 0.60, 0.80, 0.90 and 0.95, producing
`fit`, `explore`, `select`, `confirm` and `compare` roles in 60/20/10/5/5
proportions. Official validation is reserved for admission; official test follows
policy lock. Independent graphs are the multigraph statistical units.
Transductive cores and Roman's ten official splits support descriptive analyses.

The entry oracle uses the best evaluated complete candidate at `E = 1`.
Selection records the full feasible candidate ledger. The eligible family is
**feedback candidates plus the selected comparator**: `nested_selection` excludes
other open-loop candidates from `Q_s`, while keeping their contrasts as diagnostics.
This is the Eq. (3) domain. The near-best band, practical margin and simplicity
order are applied inside that domain. Representative rows in S12/S12c provide
reference values; the executed decision uses the full ledger recorded in the run.
The manuscript should explicitly carry the same domain into Eqs. (18)–(19), the
associated guarantee and S11's predecessor map. See [Paper support](../docs/PAPER_SUPPORT.md)
for the unresolved wording and the separately labelled all-feasible diagnostic.

Confirmation precedes admission and uses the prespecified Holm test (`P < 0.05`)
together with the beneficial point-estimate margin. A subsequent admission
calculation does not change a failed confirmation decision. Each run seals its
thresholds, test family, role identities and policy lock. The learned-stop
baseline uses a nearest-template predictor of depth from initial measurements.

The runtime freezes a local family consisting of the selected route and its
eligible predecessor closure on one dataset (`confirm/contrast_registry.json`).
The manuscript's S14 protocol instead specifies nine contrasts jointly across
datasets. The SI checker reconstructs that printed joint adjustment; the local
runtime alone does not execute it. Original campaign-level raw tests and a
prespecified joint replay are required to establish the S14 experiment and its
resulting decisions before admission. The separate [joint S14 checker](../docs/S14_CONFIRMATION.md) now
verifies that original family, replays Holm and the frozen decision map, and
binds it to the original run records. It preserves the archived heterogeneous
raw-test outputs and does not infer them from local adjusted P values.

## Experiment and cost records

E3 initial-input prediction, E4 distillation, E5 no-observation replanning and E6
budget-aware control use the reserved `compare` role, a declared teacher and a
common work ceiling. The current E6 executor implements bounded model-call
planning with scheduled spectral acquisition. The formal Algorithm S3 also
defines certificate branches; the standard runtime records zero certificate
skips unless a compatible certificate implementation is explicitly supplied.

External baselines record their source revision, trials, seeds and predictions.
Operation counts and synchronized wall time are separate measurements. Timing
records include device, precision, batch, warmup, repetitions and synchronization;
branch counts define expected cost. Assurance cost, observed maxima and modelled
tail scenarios retain separate labels. Scheduler records identify whether
GPU-hours cover a checkpoint, HPO search or complete campaign.

Use the [archive guide](../docs/RUN_ARCHIVES.md) for artifact schemas and the
[experiment guide](../docs/EXPERIMENTS.md) for executable commands.
