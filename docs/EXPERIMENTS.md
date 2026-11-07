# Experiments and measurements

[Documentation](README.md) · [Method](METHOD.md) · [Availability](../AVAILABILITY.md)

The core lifecycle and additional comparison studies have separate outputs. This
guide describes their execution and export. The core lifecycle of all seven
datasets is regenerated on a Slurm cluster by `bash run_cross_morse.sh reproduce`
([HPC guide](HPC.md)); the studies below run from their own drivers on completed
lifecycle runs. [`release/experiment_code.json`](../release/experiment_code.json)
maps every experiment behind a reported table or figure to its code, command,
configurations, outputs and displays. The measured outputs of the reported
experiments are published with the release `paper-artifacts`
([publication guide](PUBLICATION.md)); `bash hpc/produce_paper_outputs_arrhenius.sh <suite>`
runs every driver below on the cluster as one job graph
([Producing the measured outputs](PRODUCTION.md)).

## Every experiment and its code

[`release/experiment_code.json`](../release/experiment_code.json) maps every experiment behind a reported table or figure to its code, its full command, its configurations and the artifacts it produces; `make statement-check` verifies the map. The runners outside the lifecycle read completed (sealed) lifecycle runs, never write them, and follow one pattern: `plan` records the identities of the parent runs, `run` executes the measurement with the released runtime (one job per parent with `--job K`; an interrupted run resumes), `export` writes the measurement-record set that `scripts/records.py` traces value by value. Loading local checkpoints needs `--trust-local-checkpoints`.

| Experiment | Displays | Command | Modules | Outputs |
|---|---|---|---|---|
| Lifecycle (E10, E15, E16, E19, E20) | — | `run_cross_morse.py`, `run_cross_morse.sh` | — | `run:*` |
| E3–E6 | — | `scripts/arm_campaign.py`, `scripts/run_budget_arms.py`, `hpc/run_budget_arms_arrhenius.sh` | `cm_experiments/runner.py`, `cm_experiments/campaign.py`, `cm_campaigns/arm_campaign.py` | `campaign:budget`, `budget_matched` |
| S14 joint confirmation | S14, Fig. 2, Fig. 3, Fig. S9 | `scripts/build_s14_campaign.py`, `scripts/audit_s14_confirmation.py` | `cm_campaigns/s14_builder.py`, `cm_experiments/confirmation_analysis.py`, `cm_experiments/ensemble_grid.py`, `cm_campaigns/confirmation.py` | `campaign:confirmation` |
| E12 | — | `scripts/run_upstream_baseline.py`, `hpc/run_upstream_baseline_arrhenius.sh`, `scripts/export_records.py` | `cm_experiments/upstream.py`, `cm_campaigns/baselines.py` | `baselines` |
| E17 | — | `hpc/capture_arrhenius_accounting.sh`, `scripts/export_records.py` | `cm_campaigns/timing.py`, `cm_campaigns/accounting.py` | `nominal_time`, `sensor_parameters`, `baselines` |
| E12, E17 internal rows | S15, S15b | `scripts/run_internal_baselines.py`, `scripts/export_records.py`, `hpc/capture_arrhenius_accounting.sh` | `cm_experiments/internal_baselines.py`, `cm_campaigns/baselines.py`, `cm_campaigns/timing.py`, `cm_campaigns/accounting.py`, `cm_experiments/timing_records.py`, `cm_experiments/ensemble_grid.py` | `baselines` |
| Observed environment | — | `environment/arrhenius/capture_environment.sh` | — | `environment:gpu` |
| E13 | S45, Fig. S18 | `scripts/run_experiment.py` | `cm_experiments/attribution.py`, `cm_experiments/ensemble_grid.py` | `inserted_site` |
| E14 | S26, S26b | `scripts/run_experiment.py` | `cm_experiments/ensemble_grid.py` | `ensemble_grid` |
| Cross-backbone substitutions | S25 | `scripts/run_experiment.py`, `run_cross_morse.py` | `cm_experiments/backbones.py`, `cm_experiments/ensemble_grid.py` | `cross_backbone` |
| Factorials | S16, S16b, Fig. 4 | `scripts/run_ablation.py`, `scripts/run_experiment.py` | `cm_experiments/factorial.py`, `cm_experiments/ablation_lifecycle.py`, `cm_experiments/ensemble_grid.py` | `factorials` |
| E19 sensitivity grids | S18 | `scripts/run_sweep.py`, `scripts/export_sweep_records.py` | `cm_experiments/sweeps.py` | `sensitivity_grids` |
| Horizon and beam sweep | S29, Fig. S5 | `scripts/run_sweep.py`, `scripts/export_sweep_records.py` | `cm_experiments/sweeps.py` | `horizon_beam_sweep` |
| E7, E8 module ablations | S17 | `scripts/run_ablation.py`, `scripts/run_sweep.py`, `scripts/export_sweep_records.py` | `cm_experiments/ablation_lifecycle.py`, `cm_experiments/sweeps.py`, `cm_experiments/ensemble_grid.py` | `module_ablations` |
| E11 | S27 | `scripts/run_sweep.py`, `scripts/export_sweep_records.py` | `cm_experiments/buffered_admission.py`, `cm_experiments/sweeps.py` | `buffer_sensitivity` |
| E2 | S30, S33, S47 | `scripts/run_experiment_b.py` | `cm_experiments/predictability.py`, `cm_experiments/schedules.py`, `cm_experiments/execution.py`, `cm_experiments/study_execution.py` | `predictability` |
| E9 | — | `scripts/run_experiment_b.py` | `cm_experiments/search_enumeration.py`, `cm_experiments/predictability.py` | `search_enumeration` |
| Negative controls | S19, Fig. S16 | `scripts/run_experiment_b.py` | `cm_experiments/negative_controls.py`, `cm_experiments/input_gate.py`, `cm_experiments/predictability.py` | `negative_controls` |
| Input gate | S23, S23b, S37, S47 | `scripts/run_experiment_b.py` | `cm_experiments/input_gate.py`, `cm_experiments/protocol.py`, `cm_experiments/predictability.py` | `input_gate` |
| E18 | Fig. S18, S43 | `scripts/run_experiment_b.py` | `cm_experiments/band_audit.py`, `cm_experiments/predictability.py` | `band_power_audit`, `band_bound_tightness` |
| E17 runtime scaling | Fig. S2 | `scripts/run_timing_records.py`, `hpc/capture_arrhenius_accounting.sh` | `cm_experiments/timing_records.py`, `cm_campaigns/timing.py`, `cm_campaigns/accounting.py`, `cm_experiments/ensemble_grid.py` | `runtime_scaling` |
| E17 timing anchors | S23b | `scripts/run_timing_records.py` | `cm_experiments/timing_records.py`, `cm_campaigns/timing.py`, `cm_campaigns/accounting.py` | `timing_anchors` |
| E14 official scores | S26 | `scripts/run_timing_records.py`, `scripts/run_experiment.py` | `cm_experiments/timing_records.py`, `cm_experiments/ensemble_grid.py` | `ensemble_grid_scores` |
| Figure 3 held-out mechanism contrasts | Fig. 3 | `scripts/run_confirmation_analysis.py` | `cm_experiments/confirmation_analysis.py`, `cm_experiments/ensemble_grid.py` | `mechanism_contrasts` |
| Admission-split oracle (C12) | S30, S43 | `scripts/run_confirmation_analysis.py` | `cm_experiments/admission_oracle.py`, `cm_experiments/confirmation_analysis.py`, `cm_experiments/ensemble_grid.py` | `oracle_dispersion` |
| Proposition 2 tube audit | S22, Fig. S17, Fig. 2 | `scripts/run_trajectory_audit.py` | `cm_experiments/trajectory_audit.py`, `cm_experiments/ensemble_grid.py` | `tube_audit` |
| Depth diagnostics | Fig. S14 | `scripts/run_trajectory_audit.py` | `cm_experiments/trajectory_audit.py`, `cm_experiments/ensemble_grid.py` | `depth_diagnostics` |
| E10 | S14b, Fig. S9, Fig. S18, S43 | `scripts/run_confirmation_analysis.py` | `cm_experiments/confirmation_analysis.py`, `cm_experiments/ensemble_grid.py`, `cm_campaigns/confirmation.py` | `confirmation_contrasts`, `confirmation_sensitivity` |
| Action-gap audit (Prop. 3) | S21, Fig. S17, S33, S34, S47, Fig. S7 | `scripts/run_decision_audit.py` | `cm_experiments/decision_audit.py`, `cm_experiments/ensemble_grid.py`, `cm_experiments/input_gate.py`, `cm_experiments/negative_controls.py`, `cm_experiments/budget.py` | `action_gap_audit` |
| Figure 3c-d decision traces | Fig. 3 | `scripts/run_decision_audit.py` | `cm_experiments/decision_audit.py`, `cm_experiments/ensemble_grid.py`, `cm_experiments/input_gate.py`, `cm_experiments/negative_controls.py` | `decision_traces` |
| E7, E8 reliability diagnostics | Fig. S16 | `scripts/run_decision_audit.py` | `cm_experiments/decision_audit.py`, `cm_experiments/ensemble_grid.py`, `cm_experiments/input_gate.py`, `cm_experiments/negative_controls.py` | `reliability_diagnostics` |
| E18 policy value | — | `scripts/run_ablation.py` | `cm_experiments/ablation_lifecycle.py`, `cm_experiments/ensemble_grid.py` | `spectral_policy_value` |
## Core benchmark campaign

Before evaluating outcome labels, freeze the source revision/fingerprint, model
inventory, configuration, splits, candidate family, thresholds, precision and
resource budgets. Use an official-data pilot to measure feasibility on the intended
hardware. Each full profile registers twelve optimization seeds; Roman-empire
additionally uses all ten official masks.

```bash
python -B run_cross_morse_arxiv.py --profile full --device cuda \
  --data-root ../cm-data --output-dir ../cm-runs/arxiv-full
python -B scripts/validate_run_manifest.py ../cm-runs/arxiv-full \
  --require-complete --json
```

Equivalent wrappers exist for all seven datasets. On a Slurm cluster,
`run_cross_morse.sh reproduce` executes this campaign for every dataset and
records the environment and allocation accounting of each job
([HPC guide](HPC.md)). Completed runs are packed and ingested with
[Run archives](RUN_ARCHIVES.md).

## E3–E6: comparison at a declared work budget

| Arm | Executed comparison |
|---|---|
| E3 | Initial-input schedule predictor; no intermediate feedback |
| E4 | Distilled schedule prediction |
| E5 | Replanning using predicted state without new observations |
| E6 | Budget-aware control with an explicit comparator-restart reserve |

Arm development uses permitted development roles. The teacher identity is fixed
through confirmation; the paired comparison uses `D_compare`, reserved during
initial preparation. Admission and test must remain untouched when the comparison
study is created. Each comparison records the selected configuration, actual
work, per-unit predictions and paired effects. Deploying a different arm requires
its own lock and admission.

Create a fresh parent through confirmation, then execute and verify one study:

```bash
make arm-parent ARM_RUN=../cm-runs/arxiv-arm-parent \
  ARM_DATASET=ogbn-arxiv ARM_PROFILE=pilot DEVICE=cuda
make arm-run ARM_RUN=../cm-runs/arxiv-arm-parent \
  ARM_OUTPUT=../cm-arms/arxiv DEVICE=cuda TRUST_CHECKPOINTS=1
make arm-check ARM_OUTPUT=../cm-arms/arxiv
```

The parent must satisfy the study's eligibility and identity checks; a pilot or
smoke run may legitimately retain `qC` and provide no confirmed feedback teacher.
`TRUST_CHECKPOINTS=1` authorizes loading the specified local checkpoint files.

Expected work and a hard per-input cap are separate quantities. The comparison
ratio and E6 hard-cap ratio can be set explicitly with `ARM_EXPECTED_RATIO` and
`ARM_HARD_CAP_RATIO`. Declare them before comparison. Wall time and energy are
not guaranteed by a multiply-add work bound.

### E6 executable scope

The standard study path uses bounded planning and acquisition without a regional
certificate callback, so its certified-skip count is zero. The lower-level API
accepts a callback with the required certificate premises. Record the executable
variant with the study; the formal S3 specification includes additional branches.

If the default conservative comparator-restart reserve exceeds the cap, E6
returns `INFEASIBLE` before control. Supply tighter predeclared bounds through
`ARM_COST_BOUNDS` when available; runtime checks enforce them. The family includes
comparator retention. Records identify the selected policy, execution status,
acquisitions, certified skips, fallback and actual work.

Each certified skip must use the score gap actually observed at that site (or a
smaller certified lower bound); an infinite gap is valid only for singleton
support. The execution ledger records one origin per action. Campaign replay
checks every original operation bound against both the paid work and the
before-operation affordability condition, including the frozen comparator bound,
the held reserve, the work unit and the terminal prediction. A legacy ledger
without its original bounds cannot substantiate the S3 work guarantee; observed
costs and a `prechecked` flag alone do not establish that guarantee.

### Complete campaign and export

```bash
make arm-campaign-template ARM_CAMPAIGN=../arm-campaign.json
make arm-campaign-plan ARM_CAMPAIGN=../arm-campaign.json
make arm-campaign-run ARM_CAMPAIGN=../arm-campaign.json \
  DEVICE=cuda TRUST_CHECKPOINTS=1
make arm-export ARM_CAMPAIGN=../arm-campaign.json ARM_EXPORT=../budget-matched
make arm-campaign-replay ARM_EXPORT=../budget-matched \
  DEVICE=cuda TRUST_CHECKPOINTS=1
```

Populate the template with real parent/study paths, source identities, run-archive
links and original accounting attachments before execution. The configured paper
campaign expects six dataset families, including all ten Roman masks and twelve
controller seeds. Each parent must pass the required teacher-eligibility checks.

These repeats estimate controller-optimization variability conditional on the
frozen parent ensemble. They are not independent backbone retrainings. The
exporter preserves this estimand and rejects duplicated studies under renamed
seed IDs. Use `arm-export-partial` only to inspect an unfinished campaign.

A complete export preserves parents, studies, predictions, paired observations
and execution attachments. `scripts/export_records.py budget-matched --help`
describes conversion to the measured `budget_matched` record set after a real
archive deposit. A DOI/checksum must identify that actual archive.

## External baselines

[`cm_experiments/upstream.py`](../cm_experiments/upstream.py) is a pinned-upstream
execution harness. Actual upstream source and dataset adapters must accompany the
measured reruns. It is not an implementation of every external method.

The study plan names the method/dataset, exact clean source and adapter commits,
adapter argument vector, 48 HPO configurations, twelve final-fit seeds and the
internal training-validation selection role. Roman also names its official
splits. Inspect the plan and trusted code, then execute individual attempts:

```bash
python -B scripts/run_upstream_baseline.py --plan ../baseline-plan.json \
  --output ../baseline-study --phase hpo --index 0 --trust-upstream-code
python -B scripts/run_upstream_baseline.py --plan ../baseline-plan.json \
  --output ../baseline-study --phase select
python -B scripts/run_upstream_baseline.py --plan ../baseline-plan.json \
  --output ../baseline-study --phase final --index 0 --split 0 --trust-upstream-code
```

All 48 attempts, including failed/pruned ones, must be present before selection.
The snippet illustrates one attempt; the remaining indices must also execute.
Final receipts identify checkpoints, prediction files, split identities and timing
traces. Export recomputes official scores from predictions rather than trusting a
receipt's score.

The study plans and adapter code travel inside the `baselines` record set. For every
study, `scripts/export_records.py baselines --checkouts DIR` (DIR holds the plan files;
each plan's `source_directory` and `adapter_directory` resolve against its own directory,
as `scripts/run_upstream_baseline.py` resolves them) writes the validated plan to
`raw/<dataset>/<method>/plan.json`, writes `raw/<dataset>/<method>/upstream.json`
(dataset, method, plan location, upstream repository URL of the source checkout's
`origin`, source and adapter commits, and the adapter archive with its SHA-256) and
archives the adapter checkout at `adapter_commit` with `git archive` to
`code/adapters/<dataset>/<method>/adapter-<commit>.tar`. The upstream third-party source
is referenced by URL and commit and is not copied. Each checkout's commit must equal the
plan's; a study without its checkouts makes the set partial (`--allow-partial`, never
into `records/`), and the record audit of a complete set requires the plan, the upstream
identity and the adapter archive of all fourteen comparisons.

The internal rows of Supplementary Tables S15 and S15b (per-seed scores and GPU-hours of
every internal route, E17 serial cost and wall time of q_C and of the deployed rule) are
measured on the seven sealed lifecycle runs by `scripts/run_internal_baselines.py`
(one job per run; Roman-empire: per split run) and added with `--internal`:

```bash
python -B scripts/run_internal_baselines.py plan --parent ../runs/arxiv --parent ../runs/roman_empire ... \
  --output ../baselines-internal
python -B scripts/run_internal_baselines.py run --output ../baselines-internal --job 0 --device cuda \
  --trust-local-checkpoints
python -B scripts/export_records.py baselines --internal ../baselines-internal --study ../studies/arxiv_AMP ... \
  --checkouts ../baselines --accounting ../acct/sacct_original.psv --hardware "..." --period "..."
```

Its definitions, recorded in the plan and the set's provenance: the method rows are those
of `method_rows` in `paper/build/displays/records_campaigns.py` (keys from
`cm_campaigns/baselines.py`); q_C and q_D are each run's own selected rules, and every other
route takes the configuration with the smallest counts-weighted mean D_select loss among the
route's candidates at the comparator's ensemble size that executed without failure (the
run's comparator rule restricted to the route, its tie order included); each configuration is
executed on D_test once per plant checkpoint as a one-member ensemble and scored with the
official metric (Roman-empire: per seed the mean over the splits); q_C and the deployed rule
are timed with the E17 TimingSession against the nominal execution on the full graph or on
the test graphs at edge-count quantiles 0, 0.5 and 1; GPU-hours follow Supplementary
Table S41 (backbone training of the configuration's checkpoints, plus the qualification
stages for feedback routes and the deployed rule). A run whose lifecycle did not run on CUDA,
or whose stage times are missing, is refused.

The Slurm wrapper is
[`hpc/run_upstream_baseline_arrhenius.sh`](../hpc/run_upstream_baseline_arrhenius.sh);
[`hpc/run_budget_arms_arrhenius.sh`](../hpc/run_budget_arms_arrhenius.sh) runs
the E3–E6 arms. Both read the cluster settings from the SITE SETTINGS block of
`run_cross_morse.sh` (`bash run_cross_morse.sh site-env` prints them), so one
edit configures every Slurm job of the repository.

## Timing and accounting

Use the same device, precision, batch protocol, warm-up, synchronization and
repetition procedure for internal and external comparisons. Preserve:

- Per-input operation/work counts and controller/comparator branch events.
- Synchronized wall-time samples, repetitions, dispersion and peak memory.
- HPO and final-fit allocations, failed attempts, training and qualification cost.
- Original scheduler job IDs and allocation accounting with one attribution rule.

[`cm_campaigns/timing.py`](../cm_campaigns/timing.py) supplies timing machinery;
[`hpc/capture_arrhenius_accounting.sh`](../hpc/capture_arrhenius_accounting.sh)
captures scheduler exports. Process duration is not allocation GPU-hours. Expected
branch cost, executing both assurance branches, observed maxima and modelled tail
scenarios must retain their separate definitions.

The additional factorial, negative-control, sensor, input-gate, cross-backbone,
ensemble and inserted-site results use the contracts in
[records/specs.json](../records/specs.json). After actual runs are exported:

```bash
make records-index
make records-check
make records-tables
make scientific-audit
```

The last gate requires complete applicable benchmark archives, measured records
and manuscript bindings.
