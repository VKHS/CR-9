# Compliance with the data-availability statement

The paper's registered data-availability statement:

> The complete experiment code, configurations, source data, split and seed manifests, environment specification, checkpoints, predictions, propensity-logged trajectories, statistical outputs and figure/table generators are available in the CROSS-MORSE GitHub repository at https://github.com/VKHS/CR-9. The repository provides the one-command reconstruction of every reported table and figure and follows established reproducibility guidance.

**2 of 12 items fulfilled**; 10 awaiting the publication of the measured outputs; 0 missing. Published artifacts: 0 of 37 (`release/paper_artifacts.json`).

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
| statistical outputs | Awaiting measured outputs | release `paper-artifacts`: runs, measurement-record sets and campaigns | 36 of 36 artifacts: campaign:budget, campaign:confirmation, records:action_gap_audit, records:band_bound_tightness, records:baselines, records:budget_matched, records:buffer_sensitivity, records:confirmation_contrasts, records:confirmation_sensitivity, records:cross_backbone, records:decision_traces, records:depth_diagnostics, records:ensemble_grid, records:ensemble_grid_scores, records:factorials, records:horizon_beam_sweep, records:input_gate, records:inserted_site, records:mechanism_contrasts, records:module_ablations, records:negative_controls, records:nominal_time, records:oracle_dispersion, records:reliability_diagnostics, records:runtime_scaling, records:sensitivity_grids, records:sensor_parameters, records:timing_anchors, records:tube_audit, run:COCO-SP, run:PascalVOC-SP, run:Peptides-func, run:Roman-empire, run:ogbn-arxiv, run:ogbn-products, run:ogbn-proteins |
| figure/table generators | Fulfilled | `paper/build/displays/`: a generator for each of the 90 numbered displays | — |
| one-command reconstruction of every reported table and figure | Awaiting measured outputs | `make paper` (scripts/reconstruct_paper.py) | 37 artifacts unpublished; displays needing them cannot be rebuilt yet |
| follows established reproducibility guidance | Fulfilled | `REPRODUCIBILITY.md` | — |

Items awaiting measured outputs have their code, structure and checks in place. On the cluster, after `bash run_cross_morse.sh reproduce` has completed the seven lifecycle runs:

```bash
bash hpc/produce_paper_outputs_arrhenius.sh <suite>      # every other measured output, one Slurm job graph
bash hpc/publish_paper_artifacts_arrhenius.sh all <work>/paper-artifacts.map.json   # release paper-artifacts
make statement-fulfilled                                 # every item fulfilled
```

The first command is described in [docs/PRODUCTION.md](docs/PRODUCTION.md); its last job writes the map of all artifacts. The second uploads them to the release `paper-artifacts`, downloads and verifies them, rebuilds every table and figure with `make paper` and writes the manifests, observed environments and source data into the checkout (`CM_PUBLISH_COMMIT=1` commits and pushes them). The reconstruction item is fulfilled when every display is rebuilt and every compared value agrees with the printed one; `paper/source_data/compare/` lists each difference.

Generated by `scripts/statement_check.py --write`; `make statement-check` verifies it is current.
