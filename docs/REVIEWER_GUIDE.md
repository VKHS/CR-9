# Reviewer guide

[Documentation](README.md) · [Reproduce the paper](HPC.md) · [Reviewer FAQ](REVIEWER_FAQ.md) · [Availability](../AVAILABILITY.md) · [Complete paper support](../paper/PAPER_SUPPORT.md) · [Claims ledger](../paper/CLAIMS.md) · [Reproducibility checklist](../REPRODUCIBILITY.md)

This repository is a code release. It distributes the implementation, the
configurations of all seven datasets, the Slurm reproduction launcher and a
registry of reported results and protocol constants. The launcher produces new
run artifacts; original and regenerated artifacts keep their own source identities.
The [artifact catalog](ARTIFACT_CATALOG.md) maps the availability statement to
actual files and producing commands. This guide shows how to check each part and maps
each review concern to its code, its commands and the records a reproduction
produces.

## 1. Check the software (no GPU)

From a checkout in the recorded CPU environment ([Getting started](GETTING_STARTED.md)):

```bash
make review REVIEW_OFFLINE=1 REPORT_ROOT=../cm-reports
```

The command checks the package, the registry and its arithmetic, the
analytical statements, the release integrity and the generated views, replays
the two bundled synthetic runs and builds their tables and figures. It trains
no model and downloads nothing. `../cm-reports/review/REVIEW_REPORT.md` and
`review.json` record every check with its output; each invocation keeps its own
outputs, so an earlier report cannot stand in for a later one.
`make docker-review REVIEW_OFFLINE=1 REPORT_ROOT=../cm-reports` runs the same
entry point in the CPU container.

| Verdict | Meaning |
|---|---|
| `SOFTWARE-ONLY` | Every software and example check passed; benchmark results come from a reproduction (section 2) |
| `SOFTWARE FAILURE` | At least one software check failed |
| `EVIDENCE COMPLETE` / `EVIDENCE INCOMPLETE` | Evidence mode, for run archives registered in a run index; this release registers none |

The individual checks are also available separately: `make verify`,
`make test`, `make theory`, `make si-check`, `make registry-check`,
`make model-inventory` and `make reproduce`.

## 2. Regenerate the results (GPU cluster)

```bash
$EDITOR run_cross_morse.sh             # SITE SETTINGS for the cluster
bash run_cross_morse.sh check          # settings, runtime fingerprint and plan
bash run_cross_morse.sh reproduce smoke   # the whole orchestration on synthetic data, minutes
bash run_cross_morse.sh reproduce      # the full study
```

The [HPC guide](HPC.md) documents the settings, the jobs, the compute the paper
reports per dataset, the recovery from time limits and GPU failures, and the
outputs. The suite's `REPRODUCTION_REPORT.md` places every dataset's deployed
policy and paired effect beside main Table 1, summarises the comparison of every
quantity `scripts/compare_with_manuscript.py` reads from a run (exact at the
printed precision, inside the printed 95% interval, or outside), embeds Tables 1
and 2 rebuilt from the regenerated runs, and records the GPU-hours used.

## 3. Inspect a regenerated run

| Question | Command | Reads |
|---|---|---|
| Does the run agree with the paper? | `make compare RUN_ROOT=/abs/run REPORT_ROOT=/abs/reports` | `config.json`, `select/`, `confirm/`, `admit/`, `policy_lock.json`, `final/results_summary.json` |
| Do its statistics recompute from its own observations? | `make replay RUN_ROOT=/abs/run REPORT_ROOT=/abs/reports` | sealed per-unit records; no PyTorch |
| Do its checkpoints produce its predictions? | `make reexecute RUN_ROOT=/abs/run REPORT_ROOT=/abs/reports` | sealed checkpoints and predictions (PyTorch) |
| Is the stage chain intact? | `python -B scripts/archive_run.py /abs/run --output /abs/run.tar.gz` | stage manifests, source and configuration identities |
| Which roles, seeds and environment did it use? | `python -B scripts/run_manifests.py runs --runs-root /abs/suite/runs --label mine --output-root /abs/manifests` | `prepared/`, seeds, `environment.json` |
| The Roman-empire split summary | `python -B scripts/summarize_roman_splits.py /abs/campaign --output /abs/roman.json` | `split_00` … `split_09` |
| The paper's run-derived tables | `python -B scripts/build_paper.py all --runs-root /abs/suite/runs --label mine --output-dir /abs/paper` | every completed run of a suite |

The reproduction report runs the comparison, the split summary, the table
builder and the manifest extraction for every completed run.

## Concern by concern

| Concern | Code and protocol | Commands | Records a reproduction produces |
|---|---|---|---|
| **R1.1 — Gains jointly assessed with expected cost and compute-matched alternatives** | Work ledger of every candidate in `cross_morse/pipeline.py` and `cross_morse/policies.py`; E3–E6 arms and Algorithm S3 in `cm_experiments/`; [Experiments](EXPERIMENTS.md) | `reproduce`; `hpc/run_budget_arms_arrhenius.sh`; `make records-check` | Rebuilt Tables 2, S23 and S41; `select/candidates.json` work ledger; the E3–E6 campaign with paired observations and actual work |
| **R1.2 — Scope of admission guarantees and single-graph criteria** | `cross_morse/statistics.py` (`empirical_bernstein`, `admission`); [Method](METHOD.md); Theorem 2 in `cm_theory/` | `reproduce`; `make compare`; `make theory` | `admit/decision.json` with the regime, units and bounds of every dataset; rebuilt Tables 1 and S13 |
| **R1.3 — Measured versus declared external-baseline costs** | Pinned upstream harness `cm_experiments/upstream.py`; E17 timing in `cm_campaigns/timing.py` | `hpc/run_upstream_baseline_arrhenius.sh`; `make records-check`; `make degeneracy-check` | Per-seed scores, timing traces and scheduler accounting of every retrained baseline; `records/baselines/` traced to raw files |
| **R1.4 / R2.2 — Availability claims supported by accessible artifacts** | [Statement compliance](../STATEMENT_COMPLIANCE.md) evaluates each item of the registered statement; [Availability](../AVAILABILITY.md) and the generated [manuscript paragraph](../paper/AVAILABILITY_STATEMENT.txt) state what the repository and its release `paper-artifacts` contain | `make statement-check`; `make paper-artifacts-status`; `make availability-check`; `make claims-check` | `make paper`: every published output verified by SHA-256, every numbered table and figure rebuilt with its source data and compared with the printed values |
| **R2.1 — Concrete frozen-backbone setting and inadequacy of an optimized fixed schedule** | Motivation and supported setting in [Method](METHOD.md); nested open-loop comparator q_C in `cross_morse/pipeline.py` | `reproduce`; `make compare`; E3–E6 driver | `select/decision.json` with the strongest open-loop comparator; the paired effect over q_C; E3–E6 comparisons on the reserved units |
| **R2.3 — Supported architectures and separate mathematical/empirical claims** | Declared operators in [Method](METHOD.md); [Analytical statements](THEORY.md); the frozen plants in `cross_morse/core.py` and `cross_morse/training.py` | `make model-inventory` (Table S5 counts); `make theory`; `make si-check`; `make compare` | `train/model_inventory.json` and sealed checkpoints of every run |

## Read the statuses precisely

- **Identical runtime:** the fingerprint of `run_cross_morse*.py` and
  `cross_morse/*.py` equals the released `0aba361b…`; `check` and `reproduce`
  verify it and every run seals the identity it ran with.
- **Exact / interval / outside:** a regenerated quantity equals the printed
  value at the printed precision, lies inside the printed 95% interval, or
  neither. A regenerated run is a new execution with the registered seeds;
  hardware, kernels and library versions can differ from the authors' runs.
- **Replayed:** `scripts/reproduce_results.py` recomputed the run's statistics
  from its sealed per-unit observations.
- **Traced:** `scripts/records.py check` re-read a summary value at its recorded
  precision from the declared raw file and locator.

`make decision-check` compares the transcribed confirmation contrasts with the
registered rule and policy fields. A reported evaluated arm and the final policy
are separate quantities; on Peptides the paper reports the radius arm as an
evaluated arm and the registered rule returns q_C. A regenerated run records its
own decision in `confirm/decision.json`.
[Joint S14 confirmation](S14_CONFIRMATION.md) specifies the paper's
nine-contrast family and its checker.
