# Data and code availability

## Availability statement

The CROSS-MORSE reference implementation, the seven dataset adapters with their configurations, the Slurm reproduction launcher, environment specifications, statistical analysis tools and reported-result references are available at [https://github.com/VKHS/CR-9](https://github.com/VKHS/CR-9). The release includes **2 complete synthetic example runs** with checkpoints, predictions and exploration logs for software verification. Official datasets are obtained from their providers by the adapters. Original benchmark execution archives can be supplied through the run/record registration workflow; this code release does not package those archives. `bash run_cross_morse.sh reproduce` executes the registered dataset lifecycles on a Slurm GPU cluster; each run seals the runtime fingerprint, configuration, data-role membership, checkpoints and decision records it was produced with. The suite report rebuilds supported run-derived outputs and compares their quantities with the reported references. [The artifact catalog](docs/ARTIFACT_CATALOG.md) distinguishes included files, provider data, new execution products and registered experimental archives.

## What the release contains

| Component | Location |
|---|---|
| Reference implementation and the sealed eight-stage lifecycle | `cross_morse/`, `run_cross_morse*.py` |
| Reproduction launcher for Slurm GPU clusters and the seven dataset wrappers | `run_cross_morse.sh`, `run_cross_morse_<dataset>.sh`; [HPC guide](docs/HPC.md) |
| Smoke, pilot and full-profile configurations; split rules, seeds and folds | `configs/`, `manifests/registered/`, `cross_morse/split_contract.py` |
| Experiments outside the lifecycle: E3–E6 arms, retrained baselines, E17 timing, scheduler accounting | `cm_experiments/`, `cm_campaigns/`, `hpc/`; [Experiments](docs/EXPERIMENTS.md) |
| Curated paper-reported values and registered protocol constants | `paper/reported_results.json`, `paper/registered_constants.json` |
| Builders of the run-derived tables and figures; comparison of a run with the paper | `paper/build/`, `scripts/build_paper.py`, `scripts/compare_with_manuscript.py` |
| Analytical statements and supplementary arithmetic | `cm_theory/`, `scripts/check_si_identities.py` |
| Environment specifications and containers | `environment/`, `Dockerfile` |
| Typed artifact catalog: included files, provider data, generated outputs and registered archives | [Artifact catalog](docs/ARTIFACT_CATALOG.md); `release/artifact_catalog.json` |
| Complete synthetic example runs | 2 under `examples/runs/` |

## Execution and reconstruction

Edit the SITE SETTINGS block of `run_cross_morse.sh` for the cluster (or select a profile with `CM_ENV_FILE`), then `bash run_cross_morse.sh check` verifies the settings and the runtime fingerprint and prints the job plan, and `bash run_cross_morse.sh reproduce` submits the lifecycle suite: one setup job, the eight-stage lifecycle of every dataset (Roman-empire on its ten official splits) and one report job. Runs that reach their time limit checkpoint and requeue themselves; a GPU or driver failure requeues the job on another node.

The report rebuilds the run-derived columns of main Tables 1–2 and Supplementary Tables S4, S12, S12c, S13–S15, S23 and S41; numerical plots related to main Figs. 2c and 4d and Supplementary Fig. S9; a partial internal-method map related to main Fig. 5a; an entry-loss depth diagnostic; and stage-cost bars related to Table S41 with `scripts/build_paper.py`, and `scripts/compare_with_manuscript.py` places every compared quantity of a full-profile run beside the printed value (exact at the printed precision, inside the printed 95% interval, or outside). The experiments outside the lifecycle and their record exporters are run as described in [Experiments](docs/EXPERIMENTS.md); `make records-check` traces a record set to its raw files. Registry plots use transcribed table values and analytical scenarios; they are not a replay of benchmark observations. Auxiliary tables require their corresponding validated measurement-record sets. Numerical plot builders do not reproduce every printed panel or its exact artwork; paper/PAPER_SUPPORT.md states the display-specific scope.

`make reconstruct REPORT_ROOT=/abs/reports` assembles the supported numerical outputs and a manifest of all numbered displays. `make reconstruction-example REPORT_ROOT=/abs/reports` uses the bundled synthetic examples. Each report distinguishes generated columns, partial or related plots, and check-only references. See [Reconstruction](docs/RECONSTRUCTION.md).

`make reproduce` and `make example-paper` replay the bundled synthetic runs without a GPU. The run index (`runs/index.json`) registers sealed archives packed with `scripts/archive_run.py`; the record index (`records/index.json`) registers measurement sets and their raw-file provenance. An original experiment and a new reproduction retain their actual source identities and an explicit relationship to the manuscript.

## Source identity

Runtime source identity: `0aba361b54f6561f407f097af257cf27e4bf7f85d1ff5f47a4bcdd8bcda3a4fb` (`make fingerprint`). `run_cross_morse.sh` checks it before a reproduction and every run records it with its environment. Paper PDFs are not distributed here; [paper/sources.json](paper/sources.json) retains their hashes and page references. Numerical comparison targets remain in [paper/reported_results.json](paper/reported_results.json).

See [HPC and reproduction](docs/HPC.md) and [Getting started](docs/GETTING_STARTED.md). Generated by `scripts/availability.py`; `make availability-check` verifies freshness and claim consistency.
