# CPU demonstration: results

> **Demonstration.** CPU demonstration of the released code at reduced scale (demo/README.md). Every artifact was produced by the repository's own lifecycle and runners on public benchmark data on hosted CPU runners. The values are not the paper's measurements and are not expected to equal the printed values.

Release: [demo-artifacts](https://github.com/VKHS/CR-9/releases/tag/demo-artifacts) — 31 of 37 artifacts published (38.55 GiB, 28,913 files).

## Rebuilt from the release with `make demo-paper`

- Displays: 0 of 0 complete in content and comparison coverage; the others name the artifact or printed-value contract they lack.
- Printed values: 0 equal, 0 different, 0 missing. Differences are expected: these are reduced-scale CPU runs, not the runs the paper reports.


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

The repository's own production graph (scripts/paper_outputs.py, planned with `--allow-fixture --device cpu`) executed every node it could plan on CPU (demo/execute.py; workflow runs [42674584772](https://github.com/VKHS/CR-9/actions/runs/42674584772), [42738711578](https://github.com/VKHS/CR-9/actions/runs/42738711578), [42746503398](https://github.com/VKHS/CR-9/actions/runs/42746503398), [42746423814](https://github.com/VKHS/CR-9/actions/runs/42746423814), [42753821512](https://github.com/VKHS/CR-9/actions/runs/42753821512), [42800949895](https://github.com/VKHS/CR-9/actions/runs/42800949895), [42817409486](https://github.com/VKHS/CR-9/actions/runs/42817409486), [42820132747](https://github.com/VKHS/CR-9/actions/runs/42820132747), [42843295081](https://github.com/VKHS/CR-9/actions/runs/42843295081)). Its declared choices are recorded in the plan: {"arm_seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "band_audit": {"baseline": "shared", "dataset": "ogbn-arxiv"}, "budget_parents": "fresh executions prepare->confirm of each sealed run's own configuration; the sealed runs carry admission and test decisions, which the E3-E6 protocol (cm_experiments/runner.py) and the complete campaign export refuse", "from_runs_roman_member": "split_00 (the first split run, as the E17 timing records time it)", "horizon_sweep_configuration": {"ensemble": 1, "profile": "full"}, "negative_controls_dataset": "PascalVOC-SP", "provisional": [], "s14_excluded": ["Roman-empire"]}.

| Node | Level | Tasks complete | State |
|---|---|---|---|
| band_audit.run | 0 | 0/1 | incomplete (NOT RUN) |
| budget.parent.arxiv | 0 | 0/1 | incomplete (NOT RUN) |
| budget.parent.coco_sp | 0 | 0/1 | incomplete (NOT RUN) |
| budget.parent.pascalvoc_sp | 0 | 0/1 | incomplete (NOT RUN) |
| budget.parent.peptides_func | 0 | 0/1 | incomplete (NOT RUN) |
| budget.parent.proteins | 0 | 0/1 | incomplete (NOT RUN) |
| budget.parent.roman_empire_split_00 | 0 | 0/1 | incomplete (NOT RUN) |
| budget.parent.roman_empire_split_01 | 0 | 0/1 | incomplete (NOT RUN) |
| budget.parent.roman_empire_split_02 | 0 | 0/1 | incomplete (NOT RUN) |
| budget.parent.roman_empire_split_03 | 0 | 0/1 | incomplete (NOT RUN) |
| budget.parent.roman_empire_split_04 | 0 | 0/1 | incomplete (NOT RUN) |
| budget.parent.roman_empire_split_05 | 0 | 0/1 | incomplete (NOT RUN) |
| budget.parent.roman_empire_split_06 | 0 | 0/1 | incomplete (NOT RUN) |
| budget.parent.roman_empire_split_07 | 0 | 0/1 | incomplete (NOT RUN) |
| budget.parent.roman_empire_split_08 | 0 | 0/1 | incomplete (NOT RUN) |
| budget.parent.roman_empire_split_09 | 0 | 0/1 | incomplete (NOT RUN) |
| buffer_sensitivity.run | 0 | 0/2 | incomplete (NOT RUN) |
| cross_backbone.run | 0 | 0/12 | incomplete (NOT RUN) |
| depth_diagnostics.run | 0 | 0/16 | incomplete (NOT RUN) |
| ensemble_grid.run | 0 | 0/15 | incomplete (NOT RUN) |
| environment.capture | 0 | 0/1 | incomplete (NOT RUN) |
| factorials.lifecycles | 0 | 0/12 | incomplete (NOT RUN) |
| input_gate.run | 0 | 0/15 | incomplete (NOT RUN) |
| inserted_site.run | 0 | 0/3 | incomplete (NOT RUN) |
| mechanism_contrasts.run | 0 | 0/16 | incomplete (NOT RUN) |
| oracle_dispersion.run | 0 | 0/352 | incomplete (NOT RUN) |
| runtime_scaling.run | 0 | 0/29 | incomplete (NOT RUN) |
| s14_campaign.run | 0 | 0/6 | incomplete (NOT RUN) |
| sensitivity_grids.variants | 0 | 0/160 | incomplete (NOT RUN) |
| tube_audit.run | 0 | 0/14 | incomplete (NOT RUN) |
| budget.studies.arxiv | 1 | 0/12 | incomplete (NOT RUN) |
| budget.studies.coco_sp | 1 | 0/12 | incomplete (NOT RUN) |
| budget.studies.pascalvoc_sp | 1 | 0/12 | incomplete (NOT RUN) |
| budget.studies.peptides_func | 1 | 0/12 | incomplete (NOT RUN) |
| budget.studies.proteins | 1 | 0/12 | incomplete (NOT RUN) |
| budget.studies.roman_empire_split_00 | 1 | 0/12 | incomplete (NOT RUN) |
| budget.studies.roman_empire_split_01 | 1 | 0/12 | incomplete (NOT RUN) |
| budget.studies.roman_empire_split_02 | 1 | 0/12 | incomplete (NOT RUN) |
| budget.studies.roman_empire_split_03 | 1 | 0/12 | incomplete (NOT RUN) |
| budget.studies.roman_empire_split_04 | 1 | 0/12 | incomplete (NOT RUN) |
| budget.studies.roman_empire_split_05 | 1 | 0/12 | incomplete (NOT RUN) |
| budget.studies.roman_empire_split_06 | 1 | 0/12 | incomplete (NOT RUN) |
| budget.studies.roman_empire_split_07 | 1 | 0/12 | incomplete (NOT RUN) |
| budget.studies.roman_empire_split_08 | 1 | 0/12 | incomplete (NOT RUN) |
| budget.studies.roman_empire_split_09 | 1 | 0/12 | incomplete (NOT RUN) |
| decision_audit.plan | 1 | 0/1 | incomplete (NOT RUN) |
| factorials.cells | 1 | 0/2 | incomplete (NOT RUN) |
| negative_controls.run | 1 | 0/1 | incomplete (NOT RUN) |
| s14_campaign.assemble | 1 | 0/1 | incomplete (NOT RUN) |
| timing_anchors.run | 1 | 0/6 | incomplete (NOT RUN) |
| confirmation.plan | 2 | 0/1 | incomplete (NOT RUN) |
| decision_audit.run | 2 | 0/16 | incomplete (NOT RUN) |
| module_ablations.plan | 2 | 0/1 | incomplete (NOT RUN) |
| confirmation.run | 3 | 0/6 | incomplete (NOT RUN) |
| module_ablations.lifecycles | 3 | 0/192 | incomplete (NOT RUN) |
| module_ablations.measure | 4 | 0/16 | incomplete (NOT RUN) |

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
- `records:module_ablations`: nodes not complete: module_ablations.measure

Generated by `demo/report.py` in the publishing workflow.
