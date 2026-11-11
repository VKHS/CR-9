# Compliance with the data-availability statement

The paper's registered data-availability statement:

> The complete experiment code, configurations, source data, split and seed manifests, environment specification, checkpoints, predictions, propensity-logged trajectories, statistical outputs and figure/table generators are available in the CROSS-MORSE GitHub repository at https://github.com/VKHS/CROSS-MORSE. The repository provides the one-command reconstruction of every reported table and figure and follows established reproducibility guidance.

Experiment and reconstruction tooling is included. Full original-paper publication is not yet registered for the artifacts identified below in `release/paper_artifacts.json`. The separately published CPU execution resources are reported independently.

This offline check validates the release manifest and its committed views. `make statement-fulfilled` also downloads and verifies the complete artifacts and executes the reconstruction before evaluating this statement.

**Original-paper evidence: 2 of 12 requirements fulfilled.** All requirements are assessed; listing a requirement is not the same as verifying it.

The separate [CPU execution release](https://github.com/VKHS/CROSS-MORSE/releases/tag/demo-artifacts) has **31 of 37 artifact groups registered as published**. The original-paper status below does not mean this CPU release is empty. See [demo/RESULTS.md](demo/RESULTS.md) for its measured scope and comparison outcomes.

| Statement item | Status | Where | Remaining |
|---|---|---|---|
| complete experiment code | Awaiting measured outputs | 32 experiments mapped to their code in `release/experiment_code.json` | E12: adapter code of the fourteen external comparisons, archived at its pinned commit (published inside records:baselines and copied to experiments/baselines/) |
| configurations | Awaiting measured outputs | `configs/` and the experiment configurations | E12: study plans of the fourteen external comparisons (48 HPO configurations, 12 optimization seeds, pinned upstream source and adapter commits) (published inside records:baselines and copied to experiments/baselines/) |
| source data | Awaiting measured outputs | `paper/source_data/` (written by the publication step) | generated from the published outputs |
| split and seed manifests | Awaiting measured outputs | `manifests/registered/` (design); `manifests/published/` (executed) | executed memberships of 7 runs |
| environment specification | Awaiting measured outputs | `environment/`, `Dockerfile`, SITE SETTINGS of `run_cross_morse.sh` | observed environment of the reported runs: environment/observed/arxiv.json, environment/observed/proteins.json, environment/observed/products.json, environment/observed/roman_empire.json, environment/observed/pascalvoc_sp.json, environment/observed/coco_sp.json, environment/observed/peptides_func.json, environment:gpu |
| checkpoints | Awaiting measured outputs | release `paper-artifacts` (`release/paper_artifacts.json`) | checkpoints of 7 reported runs |
| predictions | Awaiting measured outputs | release `paper-artifacts` (`release/paper_artifacts.json`) | prediction arrays of 7 reported runs |
| propensity-logged trajectories | Awaiting measured outputs | release `paper-artifacts` (`release/paper_artifacts.json`) | exploration trajectories of 7 reported runs |
| statistical outputs | Awaiting measured outputs | release `paper-artifacts`: runs, measurement-record sets and campaigns | awaiting publication: campaign:budget, campaign:confirmation, records:action_gap_audit, records:band_bound_tightness, records:baselines, records:budget_matched, records:buffer_sensitivity, records:confirmation_contrasts, records:confirmation_sensitivity, records:cross_backbone, records:decision_traces, records:depth_diagnostics, records:ensemble_grid, records:ensemble_grid_scores, records:factorials, records:horizon_beam_sweep, records:input_gate, records:inserted_site, records:mechanism_contrasts, records:module_ablations, records:negative_controls, records:nominal_time, records:oracle_dispersion, records:reliability_diagnostics, records:runtime_scaling, records:sensitivity_grids, records:sensor_parameters, records:timing_anchors, records:tube_audit, run:COCO-SP, run:PascalVOC-SP, run:Peptides-func, run:Roman-empire, run:ogbn-arxiv, run:ogbn-products, run:ogbn-proteins |
| figure/table generators | Fulfilled | `paper/build/displays/`: a generator for each of the 90 numbered displays | — |
| one-command reconstruction of every reported table and figure | Awaiting measured outputs | `make paper` (scripts/reconstruct_paper.py) | 37 artifacts awaiting publication; `make paper` rebuilds the displays that read them from the release |
| follows established reproducibility guidance | Fulfilled | `REPRODUCIBILITY.md` | — |

Items awaiting measured outputs have their code, structure and checks in place. On the cluster, after `bash run_cross_morse.sh reproduce` has completed the seven lifecycle runs:

```bash
bash hpc/produce_paper_outputs_arrhenius.sh <suite>      # every other measured output, one Slurm job graph
bash hpc/publish_paper_artifacts_arrhenius.sh all <work>/paper-artifacts.map.json   # release paper-artifacts
make statement-fulfilled                                 # every item fulfilled
```

The first command is described in [docs/PRODUCTION.md](docs/PRODUCTION.md); its last job writes the map of all artifacts. The second uploads them to the release `paper-artifacts`, downloads and verifies them, rebuilds every table and figure with `make paper` and writes the manifests, observed environments and source data into the checkout (`CM_PUBLISH_COMMIT=1` commits and pushes them). The reconstruction item is fulfilled when every display is rebuilt and every compared value agrees with the printed one; `paper/source_data/compare/` lists each difference.

## CPU execution on public data

CPU execution of the released code on public benchmark data (demo/README.md). Every artifact was produced by the repository's own lifecycle and runners on hosted CPU runners: Roman-empire on its complete official graph with its ten official splits, the other six datasets on deterministic subsets of their official data, at the pilot settings. The paper's printed values are the measurements of its full-scale GPU runs. It is published as the release [`demo-artifacts`](https://github.com/VKHS/CROSS-MORSE/releases/tag/demo-artifacts) (31 of 37 artifacts) and listed in [release/demo_artifacts.json](release/demo_artifacts.json); `make demo-paper` runs every table and figure generator on it ([demo/README.md](demo/README.md); [demo/RESULTS.md](demo/RESULTS.md) lists the displays its artifacts support). The table below records the scope of this execution; the original-paper statuses above remain separate.

| Statement item | Executed | What the CPU execution supplies | Evidence |
|---|---|---|---|
| complete experiment code | yes | the lifecycle and every runner executed from this checkout (the E12 adapters excepted) | 7 artifact(s) |
| configurations | yes | the effective configuration sealed in every CPU run | 7 artifact(s) |
| source data | yes | the source data written by the CPU execution's reconstruction | 56 of 90 displays built |
| split and seed manifests | yes | the executed split and seed manifests of the CPU runs | 7 artifact(s) |
| environment specification | yes | the observed CPU environment of the runs | 8 artifact(s) |
| checkpoints | yes | the plant and controller checkpoints of the CPU runs | 7 artifact(s) |
| predictions | yes | the member and deployed predictions of the CPU runs | 8 artifact(s) |
| propensity-logged trajectories | yes | the propensity-logged exploration trajectories of the CPU runs | 7 artifact(s) |
| statistical outputs | yes | the selection, confirmation and admission statistics, record sets and campaigns | 30 artifact(s) |
| figure/table generators | yes | every table and figure generator run on the CPU execution's artifacts | 56 of 90 displays built |
| one-command reconstruction of every reported table and figure | partly | make demo-paper | 56 of 90 displays built |
| follows established reproducibility guidance | yes | the same checklist mapping | demo/published/paper/source_data |

## The availability text printed in the manuscript

The identified manuscript also prints a longer availability paragraph (main text, page 15; Supplementary Information, page 30) and two repository sentences in the main Methods. [`release/printed_availability.json`](release/printed_availability.json) holds the text and divides it into 39 claims; `scripts/printed_statement_check.py` evaluates each one: 29 fulfilled, 8 awaiting measured outputs, 2 missing.

### Main text, Data and code availability (page 15)

| Claim | Printed text | Status | Where | Remaining |
|---|---|---|---|---|
| M01 | frozen plants | Fulfilled | `cross_morse/core.py`: FrozenPlant, EdgeWeightedPlant, HeterophilyPlant, DesignatedSitePlant, build_plant; `cross_morse/training.py`: audit_training | — |
| M02 | transport adapter | Fulfilled | `cross_morse/core.py`: GraphOperator, apply_kernel, apply_transport, FrozenPlant.native_step, FrozenPlant.step, DesignatedSitePlant.step | — |
| M03 | action registry | Fulfilled | `cross_morse/core.py`: ACTIONS (10 registered actions A0–A9) | — |
| M04 | sensing | Fulfilled | `cross_morse/core.py`: MeasurementBank, chebyshev_window_coefficients | — |
| M05 | observer | Fulfilled | `cross_morse/learning.py`: ConditionedObserver, ControllerState | — |
| M06 | consequence models | Fulfilled | `cross_morse/learning.py`: ActionCritic, ResidualModel, Controller.stage_cost, Controller.terminal_value | — |
| M07 | the nine policy families | Fulfilled | `cross_morse/policies.py`: make_registry registers 9 route families (6 open loop, 3 feedback) | — |
| M08 | selection, confirmation and admission statistics | Fulfilled | `cross_morse/statistics.py`: nested_selection, confirmation, admission, empirical_bernstein, holm_adjust | — |
| M09 | the sealed eight-stage lifecycle | Fulfilled | STAGES in `cross_morse/pipeline.py`, `cross_morse/provenance.py`, `scripts/archive_run.py`; 2 bundled runs sealed at all 8 stages | — |
| M10 | the dataset adapters | Fulfilled | `cross_morse/datasets.py` (BENCHMARKS: 7 official dataset adapters) | — |
| M11 | one entry point and three configurations per dataset | Fulfilled | `run_cross_morse_<dataset>.py` and `configs/<dataset>_{smoke,pilot,full}.json` for 7 datasets | — |
| M12 | the Slurm launchers with the Arrhenius site profile | Fulfilled | `run_cross_morse.sh` (SITE SETTINGS preset to Arrhenius), seven dataset wrappers, `configs/arrhenius-site.env.example` | — |
| M13 | the environment specifications | Fulfilled | `environment/requirements-cpu.lock`, `environment/apptainer.def`, `Dockerfile`, `environment/create_cpu_env.sh`, `environment/check_environment.py`, `environment/arrhenius/capture_environment.sh`, `requirements.txt` | — |
| M14 | the test suite | Fulfilled | `tests/` (at least 50 test modules); `make test`, `make test-no-torch` | — |
| M15 | the scripts that build the run-derived tables and figures from completed run directories (make tables, make figures) | Fulfilled | `make tables` (scripts/build_paper.py), `make figures` (scripts/build_paper.py); `RUNS_ROOT=` names a directory of completed runs | — |
| M16 | the budget-matched arms E3–E6 | Fulfilled | E3–E6 in `release/experiment_code.json` | — |
| M17 | the pinned-upstream driver of the retrained external baselines | Fulfilled | `cm_experiments/upstream.py`, `scripts/run_upstream_baseline.py`, `hpc/run_upstream_baseline_arrhenius.sh` | — |
| M18 | the E17 timing protocol | Fulfilled | E17 in `release/experiment_code.json`; `cm_campaigns/timing.py`, `scripts/run_timing_records.py` | — |
| M19 | the scheduler-accounting capture | Fulfilled | `hpc/capture_arrhenius_accounting.sh`, `cm_campaigns/accounting.py`, `hpc/finalize_arm_accounting.sh` | — |
| M20 | the exporter that turns their outputs into traceable measurement records | Fulfilled | `scripts/export_records.py` writes 31 record contracts of `records/specs.json`; `scripts/records.py check` traces each value to its raw file | — |
| M21 | are available at https://github.com/VKHS/CROSS-MORSE | Fulfilled | https://github.com/VKHS/CROSS-MORSE in `CITATION.cff`, `pyproject.toml`, `codemeta.json` | — |
| M22 | The runtime source identity printed by make fingerprint is sealed into every run record. | Fulfilled | `make fingerprint`; `source_sha256` sealed in every stage record (bundled runs); publication re-verifies the seals of every reported run | — |
| M23 | All seven datasets are official public releases obtained from their providers through the repository’s adapters | Fulfilled | `cross_morse/datasets.py` (BENCHMARKS: 7 official dataset adapters); provider documentation and download resolution in `data/sources.json` (no provider data bundled) | — |
| M24 | the split rule, seeds, fold structure and margins of every configuration are registered in the repository | Fulfilled | `manifests/registered/` (14 official-data configurations): split rule and protocol, partition and training seeds, outer/inner/cross-fitting folds, selection and admission margins; smoke configurations use synthetic data | — |
| M25 | the role membership of D_fit, D_explore, D_select, D_confirm, D_compare, D_admit and D_test of every completed run is extracted into its manifest | Awaiting measured outputs | `scripts/run_manifests.py` extracts the seven role memberships with their hashes (`manifests/examples/`: 2 runs; `manifests/published/`: reported runs) | executed memberships of 7 reported runs (bash hpc/produce_paper_outputs_arrhenius.sh <suite>, then bash hpc/publish_paper_artifacts_arrhenius.sh all <work>/paper-artifacts.map.json) |
| M26 | The run archive of each dataset (prepared data and split indices, frozen checkpoints, propensity-logged exploration trajectories, candidate and final predictions, selection, confirmation and admission records, stage manifests and environment records) | Fulfilled | `scripts/archive_run.py` packs and verifies the sealed run directory; the 2 bundled runs contain every listed component | — |
| M27 | are deposited at the Zenodo DOIs listed in the repository’s runs/index.json | Awaiting measured outputs | `runs/index.json` (0 of 7 datasets registered with a Zenodo DOI; written by `scripts/ingest_run.py`) | 7 datasets without a registered full-run Zenodo DOI: deposit each run archive and register its DOI with the evidence-registration route (docs/EVIDENCE_REGISTRATION.md: make evidence-deposit, evidence-publish PUBLISH=1, evidence-register) |
| M28 | the measurement records of the experiments outside the eight-stage run (per-seed scores and timing traces of the retrained baselines with their scheduler accounting, the E3–E6 campaign with its paired observations, and the factorial, negative-control, ensemble-grid, attribution and cross-backbone records) are deposited at the Zenodo DOIs listed in the repository’s runs/index.json and records/index.json | Missing | `records/index.json` (0 of 7 record sets with a Zenodo DOI) | the record contract of `records/index.json` has no deposit-DOI field; the record sets are distributed as `records:*` assets of the release `paper-artifacts` (`release/paper_artifacts.json`). Either add record-set deposit registration to the record contract or state the release as the location of the measurement records in the printed paragraph |
| M29 | make paper rebuilds every run-derived table and figure of this article from the ingested archives | Awaiting measured outputs | `make paper` (`scripts/reconstruct_paper.py`): downloads and verifies the published archives, rebuilds every numbered display and compares each value with the printed one | 37 artifacts awaiting publication; `make paper` rebuilds the displays that read them from the release |
| M30 | make records-tables prints the record-derived supplementary tables, with every value traced to the raw file it is read from | Awaiting measured outputs | `make records-tables` (`scripts/records.py tables`): every value traced to its raw file; reads `records/` or the record sets that `make paper` verified under `ARTIFACTS_ROOT` | 27 measurement-record sets unpublished (bash hpc/produce_paper_outputs_arrhenius.sh <suite>, then bash hpc/publish_paper_artifacts_arrhenius.sh all <work>/paper-artifacts.map.json) |

### Main text, Methods: Reported results and statistical power (pages 14–15)

| Claim | Printed text | Status | Where | Remaining |
|---|---|---|---|---|
| R01 | The result registry links every table and figure value to its source record, dataset and split identity, configuration hash, checkpoint identity, seed and unit arrays and the generator that produces it. | Awaiting measured outputs | `paper/display_manifest.json` binds each of the 90 numbered displays to its generator and inputs; reconstruction writes every value with its source record to `paper/source_data/` | generated from the published outputs |
| R02 | Synthetic smoke runs check the software paths and are kept apart from the benchmark tables. | Fulfilled | bundled runs are labelled synthetic (`manifests/examples/`), their tables carry a synthetic banner (`paper/generated/examples/`), and publication accepts only full scientific runs | — |

### Supplementary Information, Data and code availability (page 30)

| Claim | Printed text | Status | Where | Remaining |
|---|---|---|---|---|
| S01 | The reference implementation, dataset adapters, entry points, configurations, Slurm launchers, environment specification, tests, run-archive and record contracts and the table and figure builders are available at https://github.com/VKHS/CROSS-MORSE. | Fulfilled | claims M01–M21; run-archive contract (`scripts/archive_run.py`, `release/schemas/`); record contracts (`records/specs.json`); a generator for every numbered display (`paper/build/displays/`) | — |
| S02 | Official datasets are obtained from their providers through the adapters. | Fulfilled | as M23: `cross_morse/datasets.py` (BENCHMARKS: 7 official dataset adapters); provider documentation and download resolution in `data/sources.json` (no provider data bundled) | — |
| S03 | The run archive of every dataset [...] are deposited at the DOIs listed in the repository’s runs/index.json | Awaiting measured outputs | `runs/index.json` (0 of 7 datasets registered with a DOI; written by `scripts/ingest_run.py`) | 7 datasets without a registered full-run DOI: deposit each run archive and register its DOI with the evidence-registration route (docs/EVIDENCE_REGISTRATION.md: make evidence-deposit, evidence-publish PUBLISH=1, evidence-register) |
| S04 | the measurement records of the experiments outside the eight-stage run are deposited at the DOIs listed in the repository’s runs/index.json and records/index.json | Missing | `records/index.json` (0 of 27 record sets with a DOI) | the record contract of `records/index.json` has no deposit-DOI field; the record sets are distributed as `records:*` assets of the release `paper-artifacts` (`release/paper_artifacts.json`). Either add record-set deposit registration to the record contract or state the release as the location of the measurement records in the printed paragraph |
| S05 | make paper rebuilds every run-derived table and figure of this Supplementary Information from the ingested archives | Awaiting measured outputs | as M29: `make paper` (`scripts/reconstruct_paper.py`): downloads and verifies the published archives, rebuilds every numbered display and compares each value with the printed one | 37 artifacts awaiting publication; `make paper` rebuilds the displays that read them from the release |
| S06 | make records-tables prints the record-derived tables, each value traced to the raw file it is read from | Awaiting measured outputs | as M30: `make records-tables` (`scripts/records.py tables`): every value traced to its raw file; reads `records/` or the record sets that `make paper` verified under `ARTIFACTS_ROOT` | 27 measurement-record sets unpublished (bash hpc/produce_paper_outputs_arrhenius.sh <suite>, then bash hpc/publish_paper_artifacts_arrhenius.sh all <work>/paper-artifacts.map.json) |
| S07 | The synthetic example runs of the repository exercise the execution paths and are kept out of the benchmark tables. | Fulfilled | as R02: bundled runs are labelled synthetic (`manifests/examples/`), their tables carry a synthetic banner (`paper/generated/examples/`), and publication accepts only full scientific runs | — |

Claims marked missing describe a location that the repository's publication route does not produce: the printed paragraph names the location, or the route adds it.

Generated by `scripts/statement_check.py --write`; `make statement-check` verifies it is current.
