# Measurement records outside the eight-stage run

[Repository home](../README.md) · [Run archives](../docs/RUN_ARCHIVES.md) · [Availability](../AVAILABILITY.md) ·
[Implementation notes](../paper/IMPLEMENTATION_NOTES.md)

The eight-stage runner supplies the run-derived columns of main Tables 1–2 and
Supplementary Tables S4, S12, S12c, S13–S15, S23 and S41 through
[run archives](../docs/RUN_ARCHIVES.md). External baselines, input-gate timings,
sensor traces and campaign accounting enter through the measurement sets below.
Each set carries the raw logs its summary values are read from.
`scripts/records.py` traces every summary value to its raw file and builds the
corresponding record table. [AVAILABILITY.md](../AVAILABILITY.md) states the
supported reconstruction scope and current coverage.

## Record sets

Defined in [`specs.json`](specs.json); the directory name of a set is its key.

| Set | Displays | Experiments | Keys | Quantities | Producing command |
|---|---|---|---|---|---|
| `action_gap_audit` | S21, Fig. S17 | — | dataset | `median_gap_nats`, `p5_gap_nats`, `near_zero_threshold_nats`, `near_zero_fraction`, `switch_fraction`, `recovery_index_stability`, `beam_sequence_stability`, `decision_sites` | scripts/run_decision_audit.py |
| `band_bound_tightness` | Fig. S18 | E18 | action, spectral_scale | `displacement_ratio` | scripts/run_experiment_b.py |
| `band_power_audit` | Fig. S18c, S8, S22 | E18 | dataset, baseline, action, spectrum | `tightness_ratio`, `tightness_ratio_max`, `window_bound_ratio`, `retained_statistic_ratio`, `omitted_band_share`, `approximation_error_share`, `degree_factor`, `update_gain_p95`, `logged_statistic_ratio`, `audited_states`, `exact_spectrum_states`, `ranking_spearman`, `replay_max_abs_deviation`, `update_lipschitz_max`, `shared_weight_spectral_norm`, `activation_lipschitz` | scripts/run_experiment_b.py |
| `baselines` | S15, S15b | E12, E17 | dataset, method, seed | `serial_cost`, `wall_time`, `gpu_hours`, `score`, `hpo_trials`, `params_millions` | scripts/export_records.py; scripts/run_upstream_baseline.py |
| `budget_matched` | S46, S46b | E3, E4, E5, E6 | dataset, arm, configuration | `effect_vs_qc_nats`, `effect_vs_feedback_nats`, `effect_vs_feedback_ci_low_nats`, `effect_vs_feedback_ci_high_nats`, `p_holm`, `work_ratio`, `ceiling` | scripts/arm_campaign.py; scripts/run_budget_arms.py |
| `buffer_sensitivity` | S27 | E11 | dataset, buffer_hops | `clipped_effect_nats`, `half_width_nats`, `upper_nats`, `admitted`, `resampling_units` | scripts/export_sweep_records.py; scripts/run_sweep.py |
| `confirmation_contrasts` | S14b | E10 | contrast | `effect_nats`, `interval_low_nats`, `interval_high_nats`, `p_value_zero`, `unit_count` | scripts/run_confirmation_analysis.py |
| `confirmation_sensitivity` | Fig. S18 | E10 | contrast | `effect_nats`, `ci_low_nats`, `ci_high_nats`, `units_old`, `graph_variance_share`, `holm_p`, `label_survival_probability` | scripts/run_confirmation_analysis.py |
| `cross_backbone` | S25 | — | regime, model | `effect_nats`, `route_stable_folds`, `outer_folds` | scripts/run_experiment.py |
| `decision_traces` | Fig. 3c-d | — | dataset | `same_depth_action_entropy`, `fraction_schedule_nominal`, `fraction_state_conditioned`, `fraction_exclusive_sequence`, `fraction_recovery`, `fraction_input_fallback` | scripts/run_decision_audit.py |
| `depth_diagnostics` | Fig. S14 | — | dataset, depth, phase | `source_alignment_change`, `policy_loss_change`, `first_intervention_depth_q25`, `first_intervention_depth_median`, `first_intervention_depth_q75`, `supported_deterioration_depth`, `transport_fraction` | scripts/run_trajectory_audit.py |
| `ensemble_grid` | S26b | E14 | dataset, ensemble | `loss_nats` | scripts/run_experiment.py |
| `ensemble_grid_scores` | S26 | E14 | dataset, ensemble | `official_score` | scripts/run_experiment.py; scripts/run_timing_records.py |
| `factorials` | S16, S16b | — | dataset, cell | `ld_integrated_loss`, `interval_low`, `interval_high`, `official_metric`, `activity`, `relative_time` | scripts/run_experiment.py |
| `horizon_beam_sweep` | S29, Fig. S5 | — | dataset, horizon, beam_width | `selection_effect_nats`, `expected_serial_cost`, `oracle_headroom_bound_nats`, `realized_fraction` | scripts/export_sweep_records.py; scripts/run_sweep.py |
| `input_gate` | S23 | — | dataset | `fallback_probability`, `deployed_wall_overhead_percent`, `gpu_memory_gib`, `cpu_memory_gib` | scripts/run_experiment_b.py |
| `inserted_site` | S45 | E13 | dataset, arm | `effect_nats`, `standard_error` | scripts/run_experiment.py |
| `mechanism_contrasts` | Fig. 3a-b | — | dataset, contrast | `effect_nats`, `interval_low_nats`, `interval_high_nats` | scripts/run_confirmation_analysis.py |
| `module_ablations` | S17 | E7, E8 | module, arm, dataset | `loss_change_nats`, `warning_lead_change_depths`, `false_safe_rate_change`, `risk_rmse_change`, `action_entropy_change`, `innovation_rmse`, `clean_score_change`, `worst_harm_change_nats`, `retained_effect_fraction`, `cost_fraction` | scripts/export_sweep_records.py; scripts/run_sweep.py |
| `negative_controls` | S19 | — | control | `residual_activity`, `input_fallback`, `worst_pre_fallback_harm` | scripts/run_experiment_b.py |
| `nominal_time` | S23b | E17 | dataset | `t_nom_seconds` | scripts/export_records.py |
| `oracle_dispersion` | S30, Fig. S13 | — | dataset | `median_oracle_depth`, `oracle_depth_q25`, `oracle_depth_q75`, `fraction_differs_from_qc`, `inputs` | scripts/run_confirmation_analysis.py |
| `predictability` | S30, S33, S47 | E2 | dataset | `phi_seq`, `phi_gain`, `feedback_gain_nats`, `predictor_gain_nats`, `compared_sequences`, `evaluation_units` | scripts/run_experiment_b.py |
| `reliability_diagnostics` | Fig. S16 | E7, E8 | diagnostic, level | `coverage`, `coverage_ci_low`, `coverage_ci_high`, `risk_rmse`, `nis_coverage`, `false_safe_rate`, `false_safe_ci_low`, `false_safe_ci_high`, `harm_miss_rate`, `harm_miss_ci_low`, `harm_miss_ci_high`, `input_fallback`, `residual_activity` | scripts/run_decision_audit.py |
| `runtime_scaling` | Fig. S2 | E17 | dataset, instance | `processed_edges_millions`, `wall_time_seconds`, `peak_memory_gib` | scripts/run_timing_records.py |
| `search_enumeration` | Supplementary Note S20, E9 record | E9 | dataset, planner, horizon | `two_e_H_nats`, `zeta_H_nats`, `realized_regret_nats`, `bound_coverage`, `beam_optimal_fraction`, `two_e_H_zero_uncertainty_nats`, `mean_family_size`, `planning_instances` | scripts/run_experiment_b.py |
| `sensitivity_grids` | S18 | E19 | parameter, setting, dataset | `qualified_fraction`, `strata_without_transport`, `eligible_count`, `selected_complexity`, `route_changed`, `route_simpler`, `clipped_fraction`, `clipped_mean_nats`, `admission_upper_nats`, `raw_maximum_harm_nats`, `admitted` | scripts/export_sweep_records.py; scripts/run_sweep.py |
| `sensor_parameters` | S6c | E17 | dataset | `sketch_dim`, `observer_period`, `residual_fraction`, `sensing_period` | scripts/export_records.py |
| `timing_anchors` | S23b | E17 | dataset | `throughput_anchor_seconds`, `training_anchor_low_seconds`, `training_anchor_high_seconds` | scripts/run_timing_records.py |
| `tube_audit` | S22, Fig. S17, Fig. 2a | — | dataset | `beta_median`, `beta_p95`, `m_eps_op`, `reference_term`, `tube_coverage`, `mode_stable_coverage`, `final_bound_nats`, `observed_error_median_nats`, `observed_error_p95_nats` | scripts/run_trajectory_audit.py |

## Layout of one set

```text
records/<set>/
├── summary.csv        one measured value per row
├── raw/               the logs the values are read from (training logs, scheduler accounting, timing traces, metric JSON)
├── code/              the scripts, configurations and commit identities that produced them
├── provenance.json    experiment, produced_by, hardware, period, protocol
└── MANIFEST.json      SHA-256 of every file (written by `make records-index`)
```

`summary.csv` has the set's key columns followed by
`quantity,value,unit,source_file,locator`. The shape of three rows of the `baselines` set (angle
brackets mark what the deposited row contains):

```text
dataset,method,seed,quantity,value,unit,source_file,locator
<dataset>,<method>,,gpu_hours,<value>,GPU-h,raw/<scheduler export>.csv,csv:JobName=<job>;GPUHours
<dataset>,<method>,<seed>,score,<value>,%,raw/<metrics>.json,json:/test/score
<dataset>,<method>,,wall_time,<value>,nominal,raw/<timing trace>.log,regex:wall_time_nominal=([0-9.]+)
```

A locator says where in `source_file` the value is:

| Locator | Meaning |
|---|---|
| `json:/a/b/0` | RFC 6901 pointer into a JSON file |
| `csv:column=value;target` | the first row whose `column` equals `value`; the cell in column `target` |
| `regex:pattern` | the first match of a regular expression with one capture group in a text file |

The value in `summary.csv` must equal the raw value at the precision it is printed with: a raw
`4.663` may be printed as `4.66`, not as `4.70` or `4.6600`.

## Writing a set from the experiment outputs

`scripts/export_records.py` writes a set from the outputs the experiment code of this repository produces, so that
no value is typed by hand:

| Set | Command | Reads |
|---|---|---|
| `budget_matched` | `export_records.py budget-matched --campaign <export>` | `analysis.json` of a complete campaign export (`make arm-export`) |
| `baselines` | `export_records.py baselines --study ... --timing ... --accounting ...` | pinned-upstream studies (`scripts/run_upstream_baseline.py`), E17 timing rows, original sacct exports |
| `sensor_parameters` | `export_records.py from-runs --set sensor_parameters --run ...` | each sealed run's `config.json` and the deployed rule's executed actions |
| `nominal_time` | `export_records.py from-runs --set nominal_time --run ...` | each sealed run's nominal-member execution record |

Original outputs are copied into `raw/` unchanged. A computed quantity (a score from saved predictions,
GPU-hours from the scheduler export, a ratio from timing rows, a resolvent share from action counts) is
written into a small JSON file under `raw/` with the SHA-256 of its inputs. Small inputs are copied next to it;
bulky ones (per-fit test predictions, per-unit campaign observations) stay in the archived studies or campaign,
whose DOI and checksum enter `provenance.json`, and are named by SHA-256 and archive path
(`--copy-predictions`, `--copy-observations` copy them when they are small enough for the repository).
`code/SOURCE.json` records the command, the runtime source identity and the repository commit. The
exporter stops instead of filling a gap: a registered seed without its final fit, an allocation missing
from the sacct export, a timing row outside a completed GPU allocation or an incomplete campaign
ends the export. The other six sets are deposited by hand as below, with the producing code under `code/`.

## Synthetic example sets

`examples/records/` holds `sensor_parameters` and `nominal_time` written by the exporter from the two
bundled synthetic example runs. Their `provenance.json` carries `"synthetic": true`, their tables are titled
*SYNTHETIC EXAMPLE* (`paper/generated/examples/records/`), and `make example-records-check` re-traces them in
CI. `scripts/records.py` refuses a synthetic set under `records/`, and the exporter refuses to write one there.

## Depositing a set

```bash
mkdir -p records/baselines/raw records/baselines/code
cp <logs, scheduler exports, metric files> records/baselines/raw/
cp <launch scripts, configurations, commit list> records/baselines/code/
$EDITOR records/baselines/summary.csv records/baselines/provenance.json
make records-index        # hashes the set, traces every value, writes records/index.json and AVAILABILITY.md
make records-check        # what CI runs on every push
make records-tables       # paper/generated/official/records/<set>.{tex,md,csv}
```

Remove allocation identifiers and personal paths from logs before depositing; the release audit
rejects NAISS project identifiers anywhere in the tree.

## What the check establishes

| Check | Establishes | Outside its scope |
|---|---|---|
| `MANIFEST.json` | the deposited files are the files that were indexed | when or where they were produced |
| locator trace | every printed value appears in a deposited raw file | that the raw file comes from the stated experiment; `provenance.json` and `code/` document that |
| printed table | the supplementary table is a function of the deposited summary | statistical claims beyond the summarised values |
