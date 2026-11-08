# Data and code availability

## Availability statement

The paper's registered data-availability statement:

> The complete experiment code, configurations, source data, split and seed manifests, environment specification, checkpoints, predictions, propensity-logged trajectories, statistical outputs and figure/table generators are available in the CROSS-MORSE GitHub repository at https://github.com/VKHS/CR-9. The repository provides the one-command reconstruction of every reported table and figure and follows established reproducibility guidance.

The table below identifies the available resources and the measured outputs that await publication. [Statement compliance](STATEMENT_COMPLIANCE.md) records the checks for each item (`make statement-check`).

| Statement item | Status | Where in the repository |
|---|---|---|
| complete experiment code | Awaiting measured outputs | 32 experiments mapped to their code in `release/experiment_code.json` |
| configurations | Awaiting measured outputs | `configs/` and the experiment configurations |
| source data | Awaiting measured outputs | `paper/source_data/` (written by the publication step) |
| split and seed manifests | Awaiting measured outputs | `manifests/registered/` (design); `manifests/published/` (executed) |
| environment specification | Awaiting measured outputs | `environment/`, `Dockerfile`, SITE SETTINGS of `run_cross_morse.sh` |
| checkpoints | Awaiting measured outputs | release `paper-artifacts` (`release/paper_artifacts.json`) |
| predictions | Awaiting measured outputs | release `paper-artifacts` (`release/paper_artifacts.json`) |
| propensity-logged trajectories | Awaiting measured outputs | release `paper-artifacts` (`release/paper_artifacts.json`) |
| statistical outputs | Awaiting measured outputs | release `paper-artifacts`: runs, measurement-record sets and campaigns |
| figure/table generators | Fulfilled | `paper/build/displays/`: a generator for each of the 90 numbered displays |
| one-command reconstruction of every reported table and figure | Awaiting measured outputs | `make paper` (scripts/reconstruct_paper.py) |
| follows established reproducibility guidance | Fulfilled | `REPRODUCIBILITY.md` |

## Measured outputs

The publication workflow distributes checkpoints, predictions, propensity-logged trajectories and statistical outputs through the `paper-artifacts` release in [GitHub Releases](https://github.com/VKHS/CR-9/releases); [`release/paper_artifacts.json`](release/paper_artifacts.json) lists every asset with the SHA-256 of each published part, archive and file. The status of each artifact appears below.

| Artifact | Status | Files | Size | Checkpoints | Predictions | Trajectories | Statistical outputs |
|---|---|---|---|---|---|---|---|
| `run:ogbn-arxiv` | awaiting measured outputs | — | — | — | — | — | — |
| `run:ogbn-proteins` | awaiting measured outputs | — | — | — | — | — | — |
| `run:ogbn-products` | awaiting measured outputs | — | — | — | — | — | — |
| `run:Roman-empire` | awaiting measured outputs | — | — | — | — | — | — |
| `run:PascalVOC-SP` | awaiting measured outputs | — | — | — | — | — | — |
| `run:COCO-SP` | awaiting measured outputs | — | — | — | — | — | — |
| `run:Peptides-func` | awaiting measured outputs | — | — | — | — | — | — |
| `records:action_gap_audit` | awaiting measured outputs | — | — | — | — | — | — |
| `records:band_bound_tightness` | awaiting measured outputs | — | — | — | — | — | — |
| `records:baselines` | awaiting measured outputs | — | — | — | — | — | — |
| `records:budget_matched` | awaiting measured outputs | — | — | — | — | — | — |
| `records:buffer_sensitivity` | awaiting measured outputs | — | — | — | — | — | — |
| `records:confirmation_contrasts` | awaiting measured outputs | — | — | — | — | — | — |
| `records:confirmation_sensitivity` | awaiting measured outputs | — | — | — | — | — | — |
| `records:cross_backbone` | awaiting measured outputs | — | — | — | — | — | — |
| `records:decision_traces` | awaiting measured outputs | — | — | — | — | — | — |
| `records:depth_diagnostics` | awaiting measured outputs | — | — | — | — | — | — |
| `records:ensemble_grid` | awaiting measured outputs | — | — | — | — | — | — |
| `records:ensemble_grid_scores` | awaiting measured outputs | — | — | — | — | — | — |
| `records:factorials` | awaiting measured outputs | — | — | — | — | — | — |
| `records:horizon_beam_sweep` | awaiting measured outputs | — | — | — | — | — | — |
| `records:input_gate` | awaiting measured outputs | — | — | — | — | — | — |
| `records:inserted_site` | awaiting measured outputs | — | — | — | — | — | — |
| `records:mechanism_contrasts` | awaiting measured outputs | — | — | — | — | — | — |
| `records:module_ablations` | awaiting measured outputs | — | — | — | — | — | — |
| `records:negative_controls` | awaiting measured outputs | — | — | — | — | — | — |
| `records:nominal_time` | awaiting measured outputs | — | — | — | — | — | — |
| `records:oracle_dispersion` | awaiting measured outputs | — | — | — | — | — | — |
| `records:reliability_diagnostics` | awaiting measured outputs | — | — | — | — | — | — |
| `records:runtime_scaling` | awaiting measured outputs | — | — | — | — | — | — |
| `records:sensitivity_grids` | awaiting measured outputs | — | — | — | — | — | — |
| `records:sensor_parameters` | awaiting measured outputs | — | — | — | — | — | — |
| `records:timing_anchors` | awaiting measured outputs | — | — | — | — | — | — |
| `records:tube_audit` | awaiting measured outputs | — | — | — | — | — | — |
| `campaign:budget` | awaiting measured outputs | — | — | — | — | — | — |
| `campaign:confirmation` | awaiting measured outputs | — | — | — | — | — | — |
| `environment:gpu` | awaiting measured outputs | — | — | — | — | — | — |

## One command

`make paper` downloads the published outputs (analysis tier; `PAPER_TIER=complete` adds lifecycle checkpoints and full prediction arrays), verifies every part, archive and file against the manifest, verifies each run's sealed eight-stage chain and each record set's trace to its raw files, rebuilds every numbered table and figure with its source data (`paper/build/displays/`) and compares every computed value with the value printed in the paper. It exits 0 only when every display is complete and every compared value agrees. See [Reconstruction](docs/RECONSTRUCTION.md).

## Publication of the measured outputs

`bash hpc/publish_paper_artifacts_arrhenius.sh` packs the reported runs, record sets and campaigns on the cluster, uploads them to the release, downloads them back for verification, and writes the commit-sized views: `manifests/published/` (executed role memberships and seeds), `environment/observed/` (execution environments) and `paper/source_data/` (the source data of every display). See [Publishing the measured outputs](docs/PUBLICATION.md).

## Independent regeneration

`bash run_cross_morse.sh reproduce` executes the registered lifecycles on any Slurm GPU cluster after its SITE SETTINGS block is edited; its report compares the new runs with the paper ([HPC guide](docs/HPC.md)). Official datasets are obtained from their providers by the adapters. The release also includes 2 complete synthetic example runs for software verification without a GPU.

## Source identity

Runtime source identity: `0aba361b54f6561f407f097af257cf27e4bf7f85d1ff5f47a4bcdd8bcda3a4fb` (`make fingerprint`). Every published run records the runtime identity it was produced with; verification resolves it to this tree, a bundled snapshot or the run's own frozen source. Paper PDFs are not distributed here; [paper/sources.json](paper/sources.json) retains their hashes and page references, and [paper/reported_results.json](paper/reported_results.json) holds the printed values used only for comparison.

Generated by `scripts/availability.py`; `make availability-check` verifies freshness and claim consistency.
