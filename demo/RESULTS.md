# CPU demonstration: results

> **Demonstration.** CPU demonstration of the released code at reduced scale (demo/README.md). Every artifact was produced by the repository's own lifecycle and runners on public benchmark data on hosted CPU runners. The values are not the paper's measurements and are not expected to equal the printed values.

Release: [demo-artifacts](https://github.com/VKHS/CR-9/releases/tag/demo-artifacts) — 31 of 37 artifacts published (38.55 GiB, 28,913 files).

## Rebuilt from the release with `make demo-paper`

- Display generators: 56 of 90 built, 35 of them from the release and 21 from the repository's own sources (protocol and analytical displays); 34 wait for an input (below). Every display built from the release is labelled as the demonstration (synthetic software example).
- Supplementary builds from the release's record sets and campaign: 10 output(s), status PASS.
- Printed values: 1263 equal, 1202 different, 248 missing. Differences are expected: these are reduced-scale CPU runs, not the runs the paper reports.

Displays waiting for an input, with what they lack:

- `main:Figure 2`: campaign:confirmation
- `main:Figure 3`: campaign:confirmation
- `main:Figure 4`: records:baselines
- `main:Figure 5`: records:baselines
- `supplementary:Figure S2`: runtime_scaling: peak_memory_gib {'dataset': 'ogbn-arxiv', 'instance': 'full'} is not in the record set
- `supplementary:Figure S4`: records:baselines
- `supplementary:Figure S6`: ogbn-arxiv: the run deploys no feedback rule with a positive gain over q_C
- `supplementary:Figure S7`: records:action_gap_audit
- `supplementary:Figure S8`: PascalVOC-SP: a zero-width admission or test interval leaves the two-way shares undefined
- `supplementary:Figure S9`: campaign:confirmation; records:confirmation_sensitivity
- `supplementary:Figure S10`: ogbn-arxiv: the run deploys no feedback rule with a positive gain over q_C
- `supplementary:Figure S12`: ogbn-arxiv: the run deploys no feedback rule with a positive gain over q_C
- `supplementary:Figure S16`: reliability_diagnostics: no coverage rows
- `supplementary:Figure S17`: records:action_gap_audit
- `supplementary:Figure S18`: records:confirmation_sensitivity
- `supplementary:Figure S20`: ogbn-arxiv: the run deploys no feedback rule with a positive gain over q_C
- `supplementary:Table S6c`: sensor_parameters: sketch_dim {'dataset': 'ogbn-arxiv'} is not in the record set
- `supplementary:Table S14`: campaign:confirmation
- `supplementary:Table S14b`: records:confirmation_contrasts
- `supplementary:Table S15`: records:baselines
- `supplementary:Table S15b`: records:baselines
- `supplementary:Table S17`: records:module_ablations
- `supplementary:Table S21`: records:action_gap_audit
- `supplementary:Table S23`: records:baselines
- `supplementary:Table S23b`: records:baselines
- `supplementary:Table S28`: ogbn-arxiv: no qualified support cell
- `supplementary:Table S33`: records:action_gap_audit
- `supplementary:Table S34`: records:action_gap_audit
- `supplementary:Table S37`: input_gate: fallback_probability {'dataset': 'PascalVOC-SP'} is not in the record set
- `supplementary:Table S41`: studies/ogbn-arxiv/seed-0/split-0: no captured Slurm job ID
- `supplementary:Table S42`: ogbn-arxiv: the run deploys no feedback rule with a positive gain over q_C
- `supplementary:Table S43`: records:baselines; records:confirmation_sensitivity
- `supplementary:Table S44`: ogbn-arxiv: the run deploys no feedback rule with a positive gain over q_C
- `supplementary:Table S47`: records:action_gap_audit


## The seven lifecycle runs

| Dataset | Data | Profile | Seconds per stage (prepare, train, explore, fit, select, confirm, admit, evaluate) | Size |
|---|---|---|---|---|
| ogbn-arxiv | 21,168 of 169,343 nodes (1 part of a 8-way METIS partition) | smoke | 0 / 218 / 347 / 3 / 216 / 1 / 1 / 1 | 356 MiB |
| ogbn-proteins | 4,266 of 132,534 nodes (one part per official role of a 96-way METIS partition) | smoke | 1 / 111 / 237 / 4 / 175 / 0 / 9 / 9 | 260 MiB |
| ogbn-products | 18,575 of 2,449,029 nodes (1 part of a 128-way METIS partition) | smoke | 1 / 166 / 735 / 5 / 454 / 1 / 1 / 1 | 373 MiB |
| Roman-empire split_00 | complete official graph | pilot | 2 / 162 / 219 / 6 / 138 / 1 / 1 / 1 | 210 MiB |
| Roman-empire split_01 | complete official graph | pilot | 1 / 162 / 203 / 5 / 140 / 1 / 1 / 1 | 210 MiB |
| Roman-empire split_02 | complete official graph | pilot | 1 / 258 / 223 / 8 / 163 / 1 / 1 / 1 | 210 MiB |
| Roman-empire split_03 | complete official graph | pilot | 2 / 161 / 147 / 4 / 101 / 0 / 0 / 1 | 210 MiB |
| Roman-empire split_04 | complete official graph | pilot | 2 / 278 / 261 / 8 / 176 / 1 / 1 / 1 | 210 MiB |
| Roman-empire split_05 | complete official graph | pilot | 1 / 256 / 226 / 7 / 159 / 1 / 1 / 1 | 210 MiB |
| Roman-empire split_06 | complete official graph | pilot | 1 / 254 / 226 / 8 / 139 / 1 / 1 / 1 | 210 MiB |
| Roman-empire split_07 | complete official graph | pilot | 1 / 171 / 143 / 4 / 87 / 0 / 0 / 1 | 210 MiB |
| Roman-empire split_08 | complete official graph | pilot | 1 / 173 / 148 / 3 / 108 / 0 / 0 / 1 | 210 MiB |
| Roman-empire split_09 | complete official graph | pilot | 1 / 124 / 214 / 7 / 147 / 1 / 1 / 1 | 210 MiB |
| PascalVOC-SP | first 600 of 11,355 official graphs (287,918 nodes) | smoke | 1 / 3241 / 7 / 6 / 300 / 6 / 9 / 10 | 3,153 MiB |
| COCO-SP | first 600 of 123,286 official graphs (285,742 nodes) | smoke | 2 / 1020 / 8 / 7 / 399 / 17 / 21 / 22 | 12,033 MiB |
| Peptides-func | first 1,500 of 15,535 official graphs (223,414 nodes) | smoke | 3 / 490 / 4 / 8 / 375 / 1 / 2 / 3 | 208 MiB |

Every run executed the eight stages of the lifecycle from this checkout on a hosted 4-CPU runner (demo-runs.yml). Configurations: demo/configure.py (the pilot settings, fp32 plant).

## The production graph

The repository's own production graph (scripts/paper_outputs.py, planned with `--allow-fixture --device cpu`) executed every node it could plan on CPU (demo/execute.py; workflow runs [42674584772](https://github.com/VKHS/CR-9/actions/runs/42674584772), [42738711578](https://github.com/VKHS/CR-9/actions/runs/42738711578), [42746503398](https://github.com/VKHS/CR-9/actions/runs/42746503398), [42746423814](https://github.com/VKHS/CR-9/actions/runs/42746423814), [42753821512](https://github.com/VKHS/CR-9/actions/runs/42753821512), [42800949895](https://github.com/VKHS/CR-9/actions/runs/42800949895), [42817409486](https://github.com/VKHS/CR-9/actions/runs/42817409486), [42820132747](https://github.com/VKHS/CR-9/actions/runs/42820132747), [42843295081](https://github.com/VKHS/CR-9/actions/runs/42843295081), [42860910909](https://github.com/VKHS/CR-9/actions/runs/42860910909)). Its declared choices are recorded in the plan: {"arm_seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "band_audit": {"baseline": "shared", "dataset": "ogbn-arxiv"}, "budget_parents": "fresh executions prepare->confirm of each sealed run's own configuration; the sealed runs carry admission and test decisions, which the E3-E6 protocol (cm_experiments/runner.py) and the complete campaign export refuse", "from_runs_roman_member": "split_00 (the first split run, as the E17 timing records time it)", "horizon_sweep_configuration": {"ensemble": 1, "profile": "full"}, "negative_controls_dataset": "PascalVOC-SP", "provisional": [], "s14_excluded": ["Roman-empire"]}.

| Node | Level | Tasks complete | State |
|---|---|---|---|
| band_audit.run | 0 | 1/1 | complete |
| budget.parent.arxiv | 0 | 1/1 | complete |
| budget.parent.coco_sp | 0 | 1/1 | complete |
| budget.parent.pascalvoc_sp | 0 | 1/1 | complete |
| budget.parent.peptides_func | 0 | 1/1 | complete |
| budget.parent.proteins | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_00 | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_01 | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_02 | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_03 | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_04 | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_05 | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_06 | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_07 | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_08 | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_09 | 0 | 1/1 | complete |
| buffer_sensitivity.run | 0 | 2/2 | complete |
| cross_backbone.run | 0 | 12/12 | complete |
| depth_diagnostics.run | 0 | 16/16 | complete |
| ensemble_grid.run | 0 | 15/15 | complete |
| environment.capture | 0 | 1/1 | complete |
| factorials.lifecycles | 0 | 12/12 | complete |
| input_gate.run | 0 | 15/15 | complete |
| inserted_site.run | 0 | 3/3 | complete |
| mechanism_contrasts.run | 0 | 16/16 | complete |
| oracle_dispersion.run | 0 | 352/352 | complete |
| runtime_scaling.run | 0 | 29/29 | complete |
| s14_campaign.run | 0 | 6/6 | complete |
| sensitivity_grids.variants | 0 | 160/160 | complete |
| tube_audit.run | 0 | 14/14 | complete |
| budget.studies.arxiv | 1 | 0/12 | incomplete (FAILED, NOT RUN) |
| budget.studies.coco_sp | 1 | 0/12 | incomplete (FAILED, NOT RUN) |
| budget.studies.pascalvoc_sp | 1 | 0/12 | incomplete (FAILED, NOT RUN) |
| budget.studies.peptides_func | 1 | 0/12 | incomplete (FAILED, NOT RUN) |
| budget.studies.proteins | 1 | 0/12 | incomplete (FAILED, NOT RUN) |
| budget.studies.roman_empire_split_00 | 1 | 0/12 | incomplete (FAILED, NOT RUN) |
| budget.studies.roman_empire_split_01 | 1 | 0/12 | incomplete (FAILED, NOT RUN) |
| budget.studies.roman_empire_split_02 | 1 | 0/12 | incomplete (FAILED, NOT RUN) |
| budget.studies.roman_empire_split_03 | 1 | 0/12 | incomplete (FAILED, NOT RUN) |
| budget.studies.roman_empire_split_04 | 1 | 0/12 | incomplete (FAILED, NOT RUN) |
| budget.studies.roman_empire_split_05 | 1 | 0/12 | incomplete (FAILED, NOT RUN) |
| budget.studies.roman_empire_split_06 | 1 | 0/12 | incomplete (FAILED, NOT RUN) |
| budget.studies.roman_empire_split_07 | 1 | 0/12 | incomplete (FAILED, NOT RUN) |
| budget.studies.roman_empire_split_08 | 1 | 0/12 | incomplete (FAILED, NOT RUN) |
| budget.studies.roman_empire_split_09 | 1 | 0/12 | incomplete (FAILED, NOT RUN) |
| decision_audit.plan | 1 | 1/1 | complete |
| factorials.cells | 1 | 2/2 | complete |
| negative_controls.run | 1 | 1/1 | complete |
| s14_campaign.assemble | 1 | 0/1 | incomplete (FAILED) |
| timing_anchors.run | 1 | 6/6 | complete |
| confirmation.plan | 2 | 0/1 | incomplete (NOT RUN) |
| decision_audit.run | 2 | 16/16 | complete |
| module_ablations.plan | 2 | 1/1 | complete |
| confirmation.run | 3 | 0/6 | incomplete (NOT RUN) |
| module_ablations.lifecycles | 3 | 192/192 | complete |
| module_ablations.measure | 4 | 15/16 | incomplete (NOT RUN) |

A second graph, planned for `horizon_beam_sweep` alone in its own work directory (demo-sweep.yml; workflow run [42742259867](https://github.com/VKHS/CR-9/actions/runs/42742259867)), produced the artifacts that demo/merge_maps.py added to the main graph's map. Its declared choices are recorded in the plan: {"arm_seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "band_audit": {"baseline": "shared", "dataset": "ogbn-arxiv"}, "budget_parents": "fresh executions prepare->confirm of each sealed run's own configuration; the sealed runs carry admission and test decisions, which the E3-E6 protocol (cm_experiments/runner.py) and the complete campaign export refuse", "from_runs_roman_member": "split_00 (the first split run, as the E17 timing records time it)", "horizon_sweep_configuration": {"beam": 16, "ensemble": 1, "profile": "full"}, "negative_controls_dataset": "PascalVOC-SP", "provisional": []}.

| Node | Level | Tasks complete | State |
|---|---|---|---|
| horizon_beam_sweep.variants | 0 | 7/7 | complete |

A second graph, planned for `records:runtime_scaling`, `records:timing_anchors` alone in its own work directory (demo-sweep.yml; workflow run [42758399098](https://github.com/VKHS/CR-9/actions/runs/42758399098)), produced the artifacts that demo/merge_maps.py added to the main graph's map. Its declared choices are recorded in the plan: {"arm_seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "band_audit": {"baseline": "shared", "dataset": "ogbn-arxiv"}, "budget_parents": "fresh executions prepare->confirm of each sealed run's own configuration; the sealed runs carry admission and test decisions, which the E3-E6 protocol (cm_experiments/runner.py) and the complete campaign export refuse", "from_runs_roman_member": "split_00 (the first split run, as the E17 timing records time it)", "negative_controls_dataset": "PascalVOC-SP", "provisional": []}.

| Node | Level | Tasks complete | State |
|---|---|---|---|
| runtime_scaling.run | 0 | 29/29 | complete |
| timing_anchors.run | 1 | 6/6 | complete |

A second graph, planned for `records:decision_traces`, `records:reliability_diagnostics`, `records:mechanism_contrasts` alone in its own work directory (demo-sweep.yml; workflow run [42790196009](https://github.com/VKHS/CR-9/actions/runs/42790196009)), produced the artifacts that demo/merge_maps.py added to the main graph's map. Its declared choices are recorded in the plan: {"arm_seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "band_audit": {"baseline": "shared", "dataset": "ogbn-arxiv"}, "budget_parents": "fresh executions prepare->confirm of each sealed run's own configuration; the sealed runs carry admission and test decisions, which the E3-E6 protocol (cm_experiments/runner.py) and the complete campaign export refuse", "from_runs_roman_member": "split_00 (the first split run, as the E17 timing records time it)", "negative_controls_dataset": "PascalVOC-SP", "provisional": []}.

| Node | Level | Tasks complete | State |
|---|---|---|---|
| input_gate.run | 0 | 15/15 | complete |
| mechanism_contrasts.run | 0 | 16/16 | complete |
| decision_audit.plan | 1 | 1/1 | complete |
| decision_audit.run | 2 | 16/16 | complete |

A second graph, planned for `campaign:budget`, `records:budget_matched` alone in its own work directory (demo-sweep.yml; workflow run [42811732665](https://github.com/VKHS/CR-9/actions/runs/42811732665)), produced the artifacts that demo/merge_maps.py added to the main graph's map. Its declared choices are recorded in the plan: {"arm_seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "band_audit": {"baseline": "shared", "dataset": "ogbn-arxiv"}, "budget_parents": "fresh executions prepare->confirm of each sealed run's own configuration; the sealed runs carry admission and test decisions, which the E3-E6 protocol (cm_experiments/runner.py) and the complete campaign export refuse", "from_runs_roman_member": "split_00 (the first split run, as the E17 timing records time it)", "negative_controls_dataset": "PascalVOC-SP", "provisional": []}.

| Node | Level | Tasks complete | State |
|---|---|---|---|
| budget.parent.arxiv | 0 | 1/1 | complete |
| budget.parent.coco_sp | 0 | 1/1 | complete |
| budget.parent.pascalvoc_sp | 0 | 1/1 | complete |
| budget.parent.peptides_func | 0 | 1/1 | complete |
| budget.parent.proteins | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_00 | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_01 | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_02 | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_03 | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_04 | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_05 | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_06 | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_07 | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_08 | 0 | 1/1 | complete |
| budget.parent.roman_empire_split_09 | 0 | 1/1 | complete |
| budget.studies.arxiv | 1 | 12/12 | complete |
| budget.studies.coco_sp | 1 | 12/12 | complete |
| budget.studies.pascalvoc_sp | 1 | 12/12 | complete |
| budget.studies.peptides_func | 1 | 12/12 | complete |
| budget.studies.proteins | 1 | 12/12 | complete |
| budget.studies.roman_empire_split_00 | 1 | 12/12 | complete |
| budget.studies.roman_empire_split_01 | 1 | 12/12 | complete |
| budget.studies.roman_empire_split_02 | 1 | 12/12 | complete |
| budget.studies.roman_empire_split_03 | 1 | 12/12 | complete |
| budget.studies.roman_empire_split_04 | 1 | 12/12 | complete |
| budget.studies.roman_empire_split_05 | 1 | 12/12 | complete |
| budget.studies.roman_empire_split_06 | 1 | 12/12 | complete |
| budget.studies.roman_empire_split_07 | 1 | 12/12 | complete |
| budget.studies.roman_empire_split_08 | 1 | 12/12 | complete |
| budget.studies.roman_empire_split_09 | 1 | 12/12 | complete |

## Not in the demonstration

Each artifact the demonstration does not hold, with its reason:

- `campaign:confirmation`: The S14 family is planned without Roman-empire (declared choice, demo/README.md): assemble_s14 refuses a partial family, and check_campaign admits only a non-synthetic campaign of all seven datasets.
- `records:action_gap_audit`: The audit scores the decisions of a closed-loop deployed rule. Every demonstration run deployed an open-loop rule, so the audit has no decision to score and the exporter writes no set.
- `records:baselines`: The fourteen external comparisons of E12 run the pinned upstream studies and adapters that the author supplies; the demonstration does not run them.
- `records:confirmation_contrasts`: Reads campaign:confirmation, which the demonstration does not assemble.
- `records:confirmation_sensitivity`: Reads campaign:confirmation, which the demonstration does not assemble.
- `records:module_ablations`: Fifteen of the sixteen measurement jobs of the S17 module ablations completed. The remaining job reads the sealed variant lifecycle runs of its parent, each opened and verified through its confirm stage with its prediction arrays; restoring them filled a hosted runner's disk twice (No space left on device). The exporter writes only the complete set.

Generated by `demo/report.py` in the publishing workflow.
