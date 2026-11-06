# Experiments and measurements

[Documentation](README.md) · [Method](METHOD.md) · [Availability](../AVAILABILITY.md)

The core lifecycle and additional comparison studies have separate outputs. This
guide describes their execution and export. Registered archives are indexed under
[`runs/`](../runs/index.json) and [`records/`](../records/index.json).

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

Equivalent wrappers exist for all seven datasets. Capture the actual environment
and allocation accounting, then pack and ingest completed runs using
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
receipt's score. The Slurm wrapper is
[`hpc/run_upstream_baseline_arrhenius.sh`](../hpc/run_upstream_baseline_arrhenius.sh).

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
