# Numerical supplementary display builders

`scripts/build_supplement.py` produces tables, figures and their numerical CSV
sources from verified factorial records, comparator ensemble-grid records, the
original joint S14 confirmation campaign and a strictly replayed budget campaign.
It adds executable quantitative outputs beyond the generic record listing and
the six existing run-summary plots. It does not read measured cells from
`paper/reported_results.json` to manufacture an experiment. The S14 validator
uses that registry only to compare independently supplied original outputs with
the reviewed printed targets.

The original records are currently absent from the public registration. Running
this command with `records/index.json` therefore reports the missing inputs and
creates no substitute observations. A successful build means the supplied inputs
passed the stated checks and the listed outputs were produced; it is not a claim
that all 90 manuscript displays have been reconstructed. Use the overall
[reconstruction report](RECONSTRUCTION.md) for display-by-display coverage and
the [scientific audit](S14_CONFIRMATION.md) for original-run linkage.

## Run the builder

Use the repository's locked environment and choose a fresh output directory:

```bash
python -B scripts/build_supplement.py \
  --records-index records/index.json \
  --confirmation-campaign /archive/original-s14 \
  --budget-campaign /archive/executed-budget-campaign \
  --output-dir reports/supplement-build
```

All three input flags are optional. Omit an unavailable campaign instead of
pointing at an empty placeholder. A supplied missing, malformed, synthetic or
incomplete campaign is an error. An empty valid records index is an absence of
evidence, reported under `skipped`. A registered factorial or ensemble set must
be complete; missing cells or ensemble sizes cannot silently become partial
numerical results.

Outputs must be outside every input evidence directory. The builder preserves
existing files by refusing a nonempty output directory. Source symlinks inside
record/campaign archives are rejected by the underlying integrity validators.
The builder reads original analysis files but does not execute their scripts or
retrain models. Matplotlib uses the noninteractive Agg backend; no display server
is required.

The Python interface used by the repository reconstruction command is:

```python
from paper.build.supplement import build_supplement

report = build_supplement(
    "reports/supplement-build",
    records_index="records/index.json",
    confirmation_campaign=None,
    budget_campaign=None,
)
```

`manuscript_root=` (CLI `--root`) selects the checkout containing the registered
record and manuscript contracts. It does not replace the running builder code.

## Inputs and their scientific scope

### Complete eight-cell factorials

Register the `factorials` set using the existing
[measurement-record contract](../records/README.md) and `scripts/records.py`.
Every numeric summary value must resolve to the value in a hashed original raw
file. The index must carry the current manifest hash and `verified: true`.
Explicit synthetic/example/planning/advisory classifications are refused.
The builder requires both `ogbn-arxiv` and `PascalVOC-SP`, with each of the cells
`000, 100, 010, 001, 110, 101, 011, 111` and all six quantities:

| Quantity | Required interpretation |
| --- | --- |
| `ld_integrated_loss` | Finite nonnegative log-depth-integrated CE in nats |
| `interval_low`, `interval_high` | Original 95% cell interval in nats, containing the mean |
| `official_metric` | Accuracy/F1 in percentage points, not a 0–1 fraction |
| `activity` | Action activity as a 0–1 fraction |
| `relative_time` | Positive time ratio with an explicit, fixed unit/denominator within each dataset |

Relative times retain their original denominator in the output. The builder does
not renormalize Pascal's first cell to one: Supplementary Table S16b prints 1.08
for that cell, whereas S16 explicitly describes a cell-000 denominator. That
distinction must be documented in the original records instead of being erased
by plotting code. These are the factorial tables on supplementary pages 12–13.

For each nonempty subset of factors, two point-contrast conventions are emitted.
The standard effect-coded contrast is the signed sum of all eight cell means
divided by four. The baseline-anchored finite difference holds factors outside
the contrast at zero. These conventions differ for interactions; both columns
are explicitly named. No interaction confidence interval is calculated from
marginal cell intervals. The manuscript's contrast intervals use the full joint
resampling covariance, which the current record schema does not contain.

### Original comparator ensemble grid

The `ensemble_grid` set requires six admitted datasets and the exact grid
`E = 1, 2, 4, 8, 12, 16, 20`, each with original `loss_nats`. The six datasets are
arxiv, proteins, Roman-empire, PascalVOC-SP, COCO-SP and Peptides-func; products is
not an admitted feedback regime in this contract.

The builder fits `loss(E) = l_inf + D_inf / E` by ordinary least squares and
emits the actual points, fitted values, residuals and maximum absolute residual.
Negative coefficients or asymptotes remain visible as diagnostic failures of the
model's intended interpretation, rather than being clipped or discarded.

Losses alone do not determine the printed `D_inf` confidence band, plant-loss
ratio `kappa`, expected/assurance break-even costs, feedback response as a
function of E, or additional-checkpoint GPU-hours. Those columns of S26b
(supplementary page 18) remain outside this builder's coverage. The plots are
comparator fit diagnostics and are not labeled as the full main Figure 4b or
Supplementary Figure S3.

### Original joint S14 confirmation

The complete [S14 campaign contract](S14_CONFIRMATION.md) is validated before
rendering. It requires all nine prespecified original tests, complete seed-by-unit
paired losses, prepared data roles, frozen policy identities, original method
code/configuration/environment hashes and the pre-admission joint decision lock.
The table and forest preserve original unrounded P values and intervals, replay
paired effects, and recompute one nine-test Holm family with the strict
point-margin decision. They do not substitute a t-test for the heterogeneous
original procedures or reinterpret printed adjusted P values as raw tests.

The resulting table includes original raw P, joint Holm P, signed effect,
original interval, statistical unit/count, seed count, signed point margin,
retention and whether the tested policy was selected. Unselected fixed-family
feedback tests remain explicitly identified. Official-run linkage is separately
checked by the scientific audit against every native run manifest; rendering
this campaign alone does not establish that linkage or attest historical timing.

This output covers the zero-null and decision columns of S14 and the state/RH
forest quantities related to main Figure 3a–b. Tests against the practical margin,
equivalence tests, Figure 3c action entropy, Figure 3d action origins and the S14b
`J_old`/`J_new` sensitivity in Supplementary Figure S9 require other original
inputs. No sensitivity interval is invented from a printed sample count.

### Strict 24-arm budget campaign

The existing `cm_campaigns.arm_campaign.check_campaign(..., strict=True)` must
pass for all six datasets and four arms before plotting. Its source-study,
reserved-role, paired-observation, operation-ledger, work-certificate and source
identity checks remain active. The builder neither promotes an incomplete
campaign to full coverage nor accepts a prewritten successful audit JSON.

The new tables show arm-minus-frozen-teacher effects, crossed seed-by-unit
intervals, within-dataset four-test Holm P values, executed work relative to qC,
the common ceiling and work divided by that ceiling. The last two normalizations
are distinct. The six frontier panels show qC, the frozen teacher and the four
executed arms on `D_compare`. They do not attach the teacher's `D_test` interval
to those points or turn configuration-grid ranges into confidence intervals.

The budget replay contract describes a **new experiment conditional on frozen
parent ensembles**. All emitted budget entries retain
`historical_table_reproduction: false`. A new conditional comparison can support
related columns/panels; it cannot by itself reconstruct historical Table S46b,
the original retained-share ranges, an ensemble sensitivity band or the other
main Figure 6 panels.

## Exact output map

Every table stem below is emitted as CSV, Markdown and LaTeX. Every figure stem
is emitted as PNG and SVG, with the indicated full-precision CSV as its numerical
source. Rendered tables use shorter human-readable precision; CSV numbers retain
the source floating-point values.

| Output ID | Paths relative to `--output-dir` | Manuscript relationship | Coverage |
| --- | --- | --- | --- |
| `factorial_arxiv_table` | `tables/S16.{csv,md,tex}` | All eight S16 cell rows and six measured quantities | `columns` |
| `factorial_pascal_table` | `tables/S16b.{csv,md,tex}` | All eight S16b cell rows and six measured quantities | `columns` |
| `factorial_arxiv_plot` | `data/factorial_arxiv.csv`; `figures/factorial_arxiv.{png,svg}` | Main Figure 4a cell means/intervals | `partial_panels` |
| `factorial_pascal_plot` | `data/factorial_pascal.csv`; `figures/factorial_pascal.{png,svg}` | Companion plot of S16b cells | `columns` |
| `factorial_point_contrasts` | `tables/factorial_point_contrasts.{csv,md,tex}` | S16/S16b point contrasts, without joint-covariance intervals | `columns` |
| `ensemble_law_fit` | `tables/S26b_fit.{csv,md,tex}` | S26b fitted coefficients and residual diagnostics | `columns` |
| `ensemble_grid_diagnostics` | `data/ensemble_grid.csv`; `figures/ensemble_fit.{png,svg}`; `figures/ensemble_residuals.{png,svg}` | Companion views of the actual S26b comparator grid | `columns` |
| `joint_s14_confirmation_table` | `tables/S14_joint.{csv,md,tex}`; `data/S14_joint_audit.json` | S14 zero-null/decision columns from the complete nine-test family | `columns` |
| `joint_s14_confirmation_forest` | `data/S14_joint_forest.csv`; `figures/S14_joint_forest.{png,svg}` | Main Figure 3a–b quantities; related S9 contrast view without rescaling | `partial_panels` |
| `budget_comparison_table` | `tables/S46b_comparison.{csv,md,tex}`; `data/budget_audit_summary.json` | S46b comparison columns for a new strict campaign | `columns` |
| `budget_comparison_forest_work` | `data/budget_comparison.csv`; `figures/budget_comparison.{png,svg}` | Companion effect/work view of the new S46b-related comparisons | `columns` |
| `budget_comparison_frontier` | `data/budget_frontier.csv`; `figures/budget_frontier.{png,svg}` | Related main Figure 6a / S3 `D_compare` point frontiers | `partial_panels` |

These twelve output groups cover columns or related panels of ten distinct
manuscript displays. Their existence does not turn the other displays, missing
columns, missing intervals or original execution records into reconstructed
results. No group currently declares `full_numerical_display`.

## Machine-readable identity and completion

`BUILD.json` uses schema `cross-morse-supplement-build-1`. It contains:

- `outputs`: stable IDs, relative files, manuscript `display_ids`, conservative
  `coverage`, `input_kind`, the source inventory digest, and SHA-256 of each
  generated file;
- `inputs`: the records-index/specification identity and complete per-file
  inventories for every successfully validated record set or campaign, including
  raw observations, original code, configuration and provenance;
- `implementation`: local builder/replay Python files and the pinned dependency
  contract, each with size and SHA-256; S14 also records its three manuscript
  verification-contract hashes;
- `skipped`: optional sources or record sets that were not supplied;
- `errors`: supplied inputs that failed validation or generation.

`status` is `PENDING` when no numerical output could be built, `PASS` when at
least one group was built with no errors, and `FAIL` when any supplied input
failed. The CLI exits nonzero for `FAIL`. `PASS` does not require every optional
input and does not mean that the manuscript is publication-ready. Top-level
`synthetic: false` means the builder refuses declared synthetic sources; it is
not an independent certification of the author's provenance declarations.

The original heterogeneous S14 test code is hashed but not rerun by this builder.
Record locators establish value traceability, not independent historical
execution or the validity of an inferential method. The strict budget replay
has the narrower conditional scope stated above. These limitations accompany
the generated tables, forests and machine-readable output groups.

Focused tests are in `tests/test_supplement_build.py`. They exercise complete
cell/grid reconstruction, the distinct factorial contrast conventions, exact CSV
numeric sources, the original joint family, strict budget validation, work
normalization, missing inputs, stale hashes, malformed identities, explicit
synthetic/advisory refusal and non-destructive output handling. Their temporary
fixtures are validator tests and are never registered as benchmark observations.
