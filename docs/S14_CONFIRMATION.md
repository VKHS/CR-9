# Original S14 confirmation and joint Holm replay

The nine contrasts in Supplementary Table S14 (page 10) form one cross-dataset
confirmation family. `make confirmation-audit` checks their original inputs and
replays the joint Holm correction, strict practical margins and policy decisions.
`make scientific-audit` additionally binds those inputs to the validated original
run archives. Both checks are required for a positive scientific review.

The implementation is [cm_campaigns/confirmation.py](../cm_campaigns/confirmation.py).
Its source targets are the reviewed [reported results](../paper/reported_results.json),
[registered constants](../paper/registered_constants.json) and
[document identities](../paper/sources.json). The individual runtime's
`confirm/decision.json` applies its declared local family. That result cannot
stand in for the cross-dataset S14 family.

The campaign (`campaign:confirmation`) has two origins. The repository builds it
from the seven sealed lifecycle runs (build path, next section), or an author
imports the original test files (import path, after it). The same auditor
validates both; the differences are listed under each path.

## Build path: from the sealed lifecycle runs

[cm_campaigns/s14_builder.py](../cm_campaigns/s14_builder.py) computes the nine
tests with the released runtime's own confirm-stage code and writes them in the
campaign format below. Run it outside the checkout:

```bash
python -B scripts/build_s14_campaign.py plan --parent ../cm-runs/arxiv-full --parent ../cm-runs/proteins-full \
    --parent ../cm-runs/products-full --parent ../cm-runs/roman_empire-full \
    --parent ../cm-runs/pascalvoc_sp-full --parent ../cm-runs/coco_sp-full \
    --parent ../cm-runs/peptides_func-full --output ../s14-build
python -B scripts/build_s14_campaign.py run --output ../s14-build [--job K] --device cuda --trust-local-checkpoints
python -B scripts/build_s14_campaign.py assemble --output ../s14-build --campaign ../s14-campaign
python -B scripts/audit_s14_confirmation.py ../s14-campaign --output ../s14-audit.json
```

The Roman-empire parent is the campaign directory of its ten split runs. Every
parent must be complete (all eight stages sealed).

- **plan** reads no labels. It fixes each dataset's target policy from the
  selection record: the provisional policy of Eq. (19) when the run selected a
  feedback route; when it selected q_C, the feedback candidate with the lowest
  D_select upper limit U (`confirmation_analysis.strongest_feedback`, as in
  Table S14's products row). The pairs are: state, the matched-shuffle state
  control of the target minus the target; radius, the radius minus its
  registered one-step predecessor; RH, MPC-RH (`replanning_pair`) minus its
  MPC-OL predecessor. `plan` registers `protocol.json` with its registration
  time, the policy hashes, the confirmation units and weights, the methods and
  the hashed test code under `<output>/registration/`. It refuses a missing
  dataset, an incomplete run, unit counts other than those of the table below,
  and a selection that a row cannot test (e.g. Pascal selecting a radius).
- **run** executes one job per run and source. A *read* job serves a target
  that belongs to the run's confirmation closure (`confirm/contrast_registry.json`):
  the run's sealed `confirm/paired_contrasts.json` row is the contrast. Its two
  sides are rescored from the sealed D_confirm predictions and must reproduce
  the sealed per-unit differences and weights exactly; its statistics are the
  sealed `confirm/decision.json` row, replayed with
  `cross_morse.statistics.confirmation`. A *compute* job (the target is not in
  the closure, e.g. products, which selected q_C) executes the contrast on
  D_confirm with the mechanism-contrast machinery of
  `cm_experiments/confirmation_analysis.py`. Each Roman split chooses its source
  separately.
- **assemble** writes the campaign, then records the auditor's verdict in
  `<output>/campaign_check.json`. A test without a valid measurement is refused,
  never substituted.

Statistics. Cluster and graph rows use the confirm stage's paired whole-unit
bootstrap (cluster weights are the scored-sample counts; graphs are equal): the
effect is the weighted mean, the 95% interval the 2.5% and 97.5% bootstrap
percentiles, P the one-sided null-centred bootstrap P against zero in the
beneficial direction, and P vs margin the runtime's one-sided bootstrap P
against the margin. Roman-empire uses d_k, the D_confirm loss of split k under
MPC-RH minus MPC-OL. Its test is the one-sided paired Student t test of the ten
d_k (9 df) with the two-sided 95% t interval over splits.

Seed level. The runtime's contrast is a contrast of E-member ensemble
predictions, one paired observation per unit. A repository-built test
therefore carries the single seed level `ensemble`; seeds are never resampled
and never counted as units.

Protocol position. On complete runs the joint lock is fixed after the
lifecycle's own admission stages. With the released runtime it cannot precede
them: the joint lock binds each run's `admit/pre_admission_lock.json`, and the
admit stage writes that file in the same call that opens the D_admit labels.
The campaign therefore declares `production.kind: repository_built` and records
the admission as it happened. Its admission links name no governing joint lock
when admission came first, and they compare the lifecycle's own pre-admission
lock (the run's local confirmation family) with the joint decision.

What the auditor does differently for `production.kind: repository_built`:

- The admission-link conditions are evaluated and reported in
  `admission_findings`, per dataset (and per Roman split lock), instead of being
  required: joint-lock governance, admission after the joint lock, admission
  predictor and lifecycle lock equal to the joint decision. `post_hoc: true`
  summarizes that some admission was not governed by, or did not follow, the
  joint lock. The link's dataset and role identities, the run-binding identities
  and every other check stay strict.
- `production.builder` and the `production.provenance` reference are required.
- `bind_official_runs` accepts the single `ensemble` seed level.
- The status is `PASS`, or `REFERENCE_MISMATCH` when values differ from the
  printed S14 at printed precision; findings never change it.

The display generators (Table S14, Figures 2, 3 and S9) use any campaign the
auditor validates, whatever its printed agreement. The comparison step reports
differences from the printed values. The E10 runner
(`scripts/run_confirmation_analysis.py confirmation`) runs its per-graph,
per-seed estimate over the parent run's registered training seeds when the
campaign carries the `ensemble` level.

## Import path: original test files

Create an incomplete import template outside the checkout, attach the original
files, and audit the resulting directory:

```bash
python -B scripts/audit_s14_confirmation.py --write-template ../s14-template.json
python -B scripts/audit_s14_confirmation.py ../original-s14 --output ../s14-audit.json
make confirmation-audit CONFIRMATION_CAMPAIGN=../original-s14 REPORT_ROOT=../review
make scientific-audit CONFIRMATION_CAMPAIGN=../original-s14 BUDGET_CAMPAIGN=../original-e3-e6 REPORT_ROOT=../review
make review REQUIRE_EVIDENCE=1 CONFIRMATION_CAMPAIGN=../original-s14 BUDGET_CAMPAIGN=../original-e3-e6 REPORT_ROOT=../review
make readiness CONFIRMATION_CAMPAIGN=../original-s14 BUDGET_CAMPAIGN=../original-e3-e6 REPORT_ROOT=../review
```

Supply the other run, record and PDF arguments described in
[REVIEWER_GUIDE.md](REVIEWER_GUIDE.md). The template is deliberately incomplete:
its status is `template`, original hashes and test values are absent, and it is
refused as evidence. There is no generated benchmark-observation example to copy.
The test suite uses temporary validator fixtures explicitly labelled as fixtures.

For `scripts/availability.py --check`, pass `--confirmation-campaign` and
`--budget-campaign`, or register repository-relative paths as
`scientific_evidence_confirmation_campaign` and
`scientific_evidence_budget_campaign` in `release/release_scope.json`. These
inputs are consulted only when `one_command_original_tables_reproduction_claimed`
is explicitly `true`; ordinary conservative availability statements do not run
the scientific audit. Registered paths cannot traverse or resolve through a
symlink outside the repository. Explicit CLI paths can identify external evidence.
Neither registration nor a supplied path enables an affirmative claim: the
complete scientific audit must return `publication_ready: true`.

A standalone `PASS` means the supplied campaign's integrity, paired effects,
joint decisions and printed S14 values agree. The report always includes
`original_test_p_values_recomputed: false`. Original cluster tests, graph
bootstraps and Roman small-sample tests retain their own code, configuration,
environment and original unrounded outputs. This auditor hashes that code but
does not execute it or substitute a generic statistical test. Full regeneration
of each original test remains a separate review of the preserved analysis.
For a repository-built campaign that code is the builder, and its tests are
regenerated by running the build path again on the same sealed runs. The report
also carries `production_kind`, `post_hoc` and `admission_findings`.

The scientific audit only gives the confirmation gate `PASS` when all seven
validated official archives also match the campaign. Missing archives leave the
gate `PENDING`; inconsistent archives or synthetic inputs cause failure. A local
software run, an S14b sensitivity calculation or a copied printed table cannot
close this gate. The final review also requires its other evidence gates.

## Exact family and decision rule

| Contrast ID | Statistical unit | Required units | Loss | Beneficial sign | Strict practical margin |
|---|---|---:|---|---|---:|
| `arxiv/state` | Buffered cluster | 32 | CE nats | Positive | 0.002 |
| `arxiv/radius` | Buffered cluster | 32 | CE nats | Negative | 0.0015 |
| `proteins/state` | Buffered cluster | 32 | Mean BCE nats | Positive | 0.002 |
| `proteins/radius` | Buffered cluster | 32 | Mean BCE nats | Negative | 0.0015 |
| `products/state` | Buffered cluster | 64 | CE nats | Positive | 0.002 |
| `Roman/RH` | Official split | 10 | CE nats | Negative | 0.002 |
| `Pascal/RH` | Confirmation graph | 425 | CE nats | Negative | 0.002 |
| `COCO/RH` | Confirmation graph | 5,664 | CE nats | Negative | 0.002 |
| `Peptides/radius` | Confirmation graph | 544 | Mean BCE nats | Negative | 0.0015 |

The nine original unadjusted tests against zero are pooled once. A mechanism is
retained only if its Holm-adjusted P is **strictly below 0.05** and its point
effect is **strictly beyond its signed practical margin**. These conditions do
not constitute a test against the margin or an equivalence test. P values against
the margin in S14 are separate results and are outside this gate's replay.

State contrasts are controller-observation destruction minus intact feedback;
radius and RH contrasts are candidate minus the registered one-step or MPC-OL
predecessor. Every test identifies its two original policy hashes in the
prespecified protocol. The intact state target must be a registered feedback
candidate. When selection already returns `qC`, an unselected, prespecified
feedback target can remain in the nine-test family, as in products. The report
marks it `tested_policy_selected: false` and the served policy stays `qC`.
This does not make an open-loop comparator state-dependent.

For a selected feedback policy, a failed state test returns `qC`. A failed
radius/RH test can return its exact registered predecessor only when that
predecessor belongs to `Q_s` or is `qC`; otherwise it returns `qC`. Route identities
are verified. The gate covers this S14 map. It does not claim to replay additional
Full/Lean contrasts or replace the complete nested-selection audit.

The supplied paper reports Peptides Holm P = 0.070. That radius contrast fails
confirmation; an admission calculation cannot repair it. Its evaluated-arm
measurements remain reported separately. Original unrounded values are necessary
for numerical replay: rounded or inequality-form values such as COCO's
`< 1e-4` are not numerical raw tests. In particular, deriving convenient raw
P values from printed adjusted values is not an evidence-import operation.

## Campaign files

Every reference is an exact object `{"path": "relative/path", "sha256": "..."}`.
Paths remain inside the campaign, with no traversal or symlinks. JSON rejects
nonfinite numbers and duplicate keys. All nested campaign files contribute to
the saved input identity, so keep generated reports outside that directory.

### Manifest and protocol

`campaign.json` uses schema `cm-s14-confirmation-1`, status `author_completed`,
`synthetic: false`, and `evidence_kind: original_confirmation_tests`. Its
`manuscript_sha256` must match the registered main and supplementary PDFs. It
references the protocol, all nine tests, seven dataset bindings, and the joint
decision lock. The template provides the exact field names and expected IDs.
A repository-built campaign also carries `production`: `kind: repository_built`,
`builder`, and references to its `provenance.json` and public `code/plan.json`.

Move the template's `protocol_template` object into the referenced `protocol.json`
and complete it from the original registration. Its schema is
`cm-s14-protocol-1`; `family` is `Supplementary Table S14`. It declares the actual
registration time, alpha, each contrast's unit/count/loss/sign/margin, two policy
hashes, exact optimization seed IDs, weighting and original method. Required
method families are `paired_cluster` for state cluster tests, `paired_unit` for
the cluster radius tests, `paired_small_sample` for Roman, and `graph_bootstrap`
for the three multigraph contrasts. Each method records its original name,
tail, zero null, command, and hashed analysis code/configuration/environment.

Graph and official-split means use equal unit weights. A buffered-cluster test
can use equal weights or `fixed_unit_weights`; the latter references a JSON map
of every unit to its original positive weight. The raw rows must use exactly
those weights on every seed. This records the original weighting convention;
it does not permit optimizing weights after seeing confirmation results.

### Original tests and paired losses

Each `tests` entry references its original normalized test record and paired CSV.
The JSON schema is `cm-s14-original-test-1`. Required identities include
`contrast_id`, `dataset`, `data_role: D_confirm`, statistical unit/count, loss,
method name/family/tail, seed IDs and both policy hashes. It binds the protocol,
paired CSV and role manifest by hash. It declares `synthetic: false`,
`evidence_kind: original_confirmation_test`,
`p_value_kind: unadjusted_against_zero`, and `numeric_precision: unrounded`.
Original start/completion times, numeric `effect_nats`, numeric `p_value_zero`,
`interval_confidence: 0.95`, `valid_interval: true`, and a two-element
`interval_nats` are required. State records also declare
`intervention_valid: true` and
`intervention_scope: controller_observations_only`.

The paired CSV columns are:

```text
seed_id,unit_id,left_loss_nats,right_loss_nats,unit_weight
```

It contains the complete declared seed-by-unit array with no repeated or missing
cells. Losses are finite and nonnegative; weights are finite and positive.
The reconstructed effect averages optimization seeds within each unit and then
averages units with the declared weights. Seeds are never counted as extra
independent graphs, cores or splits. The effect must agree with the original
unrounded result. Effects, interval endpoints, original P and pooled Holm P are
also compared with the reviewed printed targets at their recorded precision.

### Prepared roles and original run bindings

Each dataset references a `cm-s14-data-roles-1` manifest, with its dataset,
`reserved_at_stage: prepare`, original reservation time, the seven disjoint roles
`fit`, `explore`, `select`, `confirm`, `compare`, `admit`, `test`, and exact
`confirmation_units` membership. The unit memberships partition confirmation
observations exactly.

For graph datasets, observation and unit IDs are decimal graph IDs as strings;
each confirmation graph maps to itself. For OGB, observation IDs are decimal
sample IDs and unit keys are the prepared cluster IDs. Roman uses observation
IDs `split=0:123` and unit keys `split=0` through `split=9`; split qualification
preserves distinct official masks without calling ten masks independent graphs.
The scientific audit reconstructs these memberships from the sealed prepared
splits/clusters and checks the exact role identity. It also checks optimization
seed IDs against the original training configuration. A repository-built
campaign's single `ensemble` level (one observation per unit) is accepted.

Dataset bindings reference the original `select/decision.json`,
`select/candidates.json`, selection lock time and admission link. Every
`run_bindings` item includes `source_sha256`, `config_sha256`, `unit_sha256`,
`split_index`, and references to the original selection, candidates and
`admit/pre_admission_lock.json`. Non-Roman datasets have one item with
`split_index: null`; Roman requires ten ordered items, indices 0 through 9.
These copied artifacts must match their native sealed archive bytes. The native
`confirm/decision.json` indices must also resolve to the same comparator and
confirmed hashes in the pre-admission lock, so an unused or discrepant lock
cannot stand in for the actual confirmed policy.

Roman aggregate candidate identities contain exactly ten
`split_policy_hashes` entries, each with only `split_index` and `hash`. Compute
the aggregate identity with `composite_policy_hash` in the module. Each original
split's selected/comparator/confirmed hashes and each candidate's route and
open-loop identity must agree with those components. The aggregate original lock
also records `split_lock_sha256`, mapping string indices to the ten native lock
hashes. This aggregate is an identity wrapper around original records, not a new
historical computation.

### Joint lock and admission

The `cm-s14-joint-lock-1` record declares its actual `locked_at`, all nine
`mechanism_retained` booleans and the seven comparator/provisional/confirmed
policy-hash triples. Its `input_sha256` is SHA-256 of
`cm_campaigns.common.canonical` applied to this object:

```python
{
    "protocol": manifest["protocol"],
    "tests": manifest["tests"],
    "datasets": [
        {k: v for k, v in row.items() if k != "admission_link"}
        for row in manifest["datasets"]
    ],
}
```

Each admission link identifies the dataset, joint-lock hash, confirmed policy,
role-manifest hash, actual admission start time and original pre-admission lock.
The original lock's comparator/confirmed/configuration identities must agree.
The required ordering is roles reserved before registration, registration and
selection fixed before tests, all tests complete before the joint lock, and
admission started after that lock. For a repository-built campaign, the
admission conditions of this paragraph are reported in `admission_findings`
(and `post_hoc`) instead of required, as described under the build path. All
other conditions apply unchanged.

These are checked historical declarations and file identities. Hashes and
self-reported timestamps alone do not independently prove when labels were
accessed or whether the original programs ran. Preserve contemporaneous logs
and original provenance; do not invent missing records or backdate a new wrapper.
If original material is unavailable, keep the scientific gate pending and
retain the paper's measurements as source-reported values.
