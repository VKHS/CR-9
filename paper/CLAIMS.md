# Claim-to-evidence map

Generated from `paper/sources.json`, `records/specs.json` and the run/record indexes. `make claims-check` checks freshness. This map covers the registered manuscript groups, including those without a dedicated display builder.

## Registered coverage

- Official full-profile archives registered: **0 of 7**.
- Auxiliary measurement-record sets registered: **0 of 10**.
- Runtime source identity: `0aba361b54f6561f407f097af257cf27e4bf7f85d1ff5f47a4bcdd8bcda3a4fb`.

Registration flags and DOI references do not establish fresh byte integrity, public deposit availability, or agreement with the manuscript. `make review` performs those checks; its scientific audit covers enumerated core results and declared auxiliary records, not every statement in the paper.

| Dataset | Index registration | Registered DOI reference |
|---|---|---|
| ogbn-arxiv | not registered | — |
| ogbn-proteins | not registered | — |
| ogbn-products | not registered | — |
| Roman-empire | not registered | — |
| PascalVOC-SP | not registered | — |
| COCO-SP | not registered | — |
| Peptides-func | not registered | — |

| Record set | Index registration |
|---|---|
| `baselines` | not registered |
| `budget_matched` | not registered |
| `cross_backbone` | not registered |
| `ensemble_grid` | not registered |
| `factorials` | not registered |
| `input_gate` | not registered |
| `inserted_site` | not registered |
| `negative_controls` | not registered |
| `nominal_time` | not registered |
| `sensor_parameters` | not registered |

## Manuscript groups and builders

Run builders reconstruct the named run-derived columns; record builders emit files named after the record set (for example `baselines.tex`), with the corresponding manuscript table labels in their titles. Analytical calculations use registered formulas and inputs. Registry plots remain transcriptions or analytical scenarios rather than replays of benchmark observations.

| Manuscript item | Pages | Evidence kind | Builder or source | Required evidence | Status |
|---|---|---|---|---|---|
| Table 1 (main) | 9, 10 | measurement | run-derived columns: eight-stage archive (`make paper`) | official run archives for the reported datasets | 0/7 official archives registered; 0 DOI references |
| Table 2 (main) | 10, 11 | measurement + derived | run-derived columns: eight-stage archive (`make paper`) | official run archives for the reported datasets | 0/7 official archives registered; 0 DOI references |
| Abstract and results (main) | 1, 2, 9, 10, 11, 12, 13 | measurement + derived | source observations and the comparison target in `paper/reported_results.json` | original observations identified by the manuscript provenance | no dedicated display builder is declared for this registry group |
| Table S4 (SI) | 2 | measurement | run-derived columns: eight-stage archive (`make paper`) | official run archives for the reported datasets | 0/7 official archives registered; 0 DOI references |
| Table S5d (SI) | 3 | measurement + derived | `scripts/model_inventory.py`; sealed `train/model_inventory.json` for benchmark identity | archived checkpoints and inventories for the original benchmark models | released configurations are countable; historical identity requires run evidence |
| Table S6c (SI) | 5 | measurement + derived | record set `sensor_parameters` (E17; `make records-tables`) | `records/sensor_parameters/` with raw sources | `sensor_parameters` not registered; trace check required |
| Table S7 (SI) | 5 | measurement | source observations and the comparison target in `paper/reported_results.json` | original observations identified by the manuscript provenance | no dedicated display builder is declared for this registry group |
| Table S12 (SI) | 8, 9 | measurement | run-derived columns: eight-stage archive (`make paper`) | official run archives for the reported datasets | 0/7 official archives registered; 0 DOI references |
| Table S12b (SI) | 9 | measurement | source observations and the comparison target in `paper/reported_results.json` | original observations identified by the manuscript provenance | no dedicated display builder is declared for this registry group |
| Table S12c (SI) | 9 | derived arithmetic | run-derived columns: eight-stage archive (`make paper`) | official run archives for the reported datasets | 0/7 official archives registered; 0 DOI references |
| Table S13 (SI) | 10 | measurement + derived | run-derived columns: eight-stage archive (`make paper`) | official run archives for the reported datasets | 0/7 official archives registered; 0 DOI references |
| Table S14 (SI) | 10 | measurement | run-derived columns: eight-stage archive (`make paper`) | official run archives for the reported datasets | 0/7 official archives registered; 0 DOI references |
| Table S14b (`supplementary_table_S14b_separated_data_sensitivity`) (SI) | 10, 11 | analytical sensitivity | registered formulas and inputs (`make registry-check` covers enumerated identities) | the source observations for any measured inputs | arithmetic validation does not establish the measured inputs |
| Table S14b (`supplementary_table_S14b_unchanged_contrasts`) (SI) | 10, 11 | analytical sensitivity | registered formulas and inputs (`make registry-check` covers enumerated identities) | the source observations for any measured inputs | arithmetic validation does not establish the measured inputs |
| Table S15 (SI) | 11, 12 | measurement + derived | run-derived columns: eight-stage archive (`make paper`); record set `baselines` (E12, E17; `make records-tables`) | official run archives for the reported datasets; `records/baselines/` with raw sources | 0/7 official archives registered; 0 DOI references; `baselines` not registered; trace check required |
| Table S15b (SI) | 12 | measurement | record set `baselines` (E12, E17; `make records-tables`) | `records/baselines/` with raw sources | `baselines` not registered; trace check required |
| Table S16 and Figure S17d (SI) | 12, 13, 36 | measurement + derived | record set `factorials` (declared experiment records; `make records-tables`) | `records/factorials/` with raw sources | `factorials` not registered; trace check required |
| Table S16b (SI) | 12, 13 | measurement + derived | record set `factorials` (declared experiment records; `make records-tables`) | `records/factorials/` with raw sources | `factorials` not registered; trace check required |
| Table S20 (SI) | 15 | measurement + derived | source observations and the comparison target in `paper/reported_results.json` | original observations identified by the manuscript provenance | no dedicated display builder is declared for this registry group |
| Table S21 (SI) | 15, 16 | measurement | source observations and the comparison target in `paper/reported_results.json` | original observations identified by the manuscript provenance | no dedicated display builder is declared for this registry group |
| Table S22 (SI) | 16 | measurement + derived | source observations and the comparison target in `paper/reported_results.json` | original observations identified by the manuscript provenance | no dedicated display builder is declared for this registry group |
| Table S23 (SI) | 16 | measurement + derived | run-derived columns: eight-stage archive (`make paper`); record set `input_gate` (declared experiment records; `make records-tables`) | official run archives for the reported datasets; `records/input_gate/` with raw sources | 0/7 official archives registered; 0 DOI references; `input_gate` not registered; trace check required |
| Table S23b (SI) | 16 | measurement + derived | record set `nominal_time` (E17; `make records-tables`) | `records/nominal_time/` with raw sources | `nominal_time` not registered; trace check required |
| Table S26b (SI) | 18 | measurement + derived | record set `ensemble_grid` (E14; `make records-tables`) | `records/ensemble_grid/` with raw sources | `ensemble_grid` not registered; trace check required |
| Table S30 (SI) | 19 | measurement + derived | source observations and the comparison target in `paper/reported_results.json` | original observations identified by the manuscript provenance | no dedicated display builder is declared for this registry group |
| Table S34 (SI) | 21 | analytical sensitivity | registered formulas and inputs (`make registry-check` covers enumerated identities) | the source observations for any measured inputs | arithmetic validation does not establish the measured inputs |
| Table S37 (SI) | 23 | measurement | source observations and the comparison target in `paper/reported_results.json` | original observations identified by the manuscript provenance | no dedicated display builder is declared for this registry group |
| Table S41 (SI) | 24, 25 | measurement + derived | run-derived columns: eight-stage archive (`make paper`) | official run archives for the reported datasets | 0/7 official archives registered; 0 DOI references |
| Table S42 (SI) | 25 | analytical sensitivity | registered formulas and inputs (`make registry-check` covers enumerated identities) | the source observations for any measured inputs | arithmetic validation does not establish the measured inputs |
| Table S44 (SI) | 26 | analytical sensitivity | registered formulas and inputs (`make registry-check` covers enumerated identities) | the source observations for any measured inputs | arithmetic validation does not establish the measured inputs |
| Table S45 (SI) | 26 | measurement + derived | record set `inserted_site` (E13; `make records-tables`) | `records/inserted_site/` with raw sources | `inserted_site` not registered; trace check required |
| Tables S46 and S46b (SI) | 26, 27 | measurement | record set `budget_matched` (E3, E4, E5, E6; `make records-tables`) | `records/budget_matched/` with raw sources | `budget_matched` not registered; trace check required |
| Table S46b (SI) | 27 | measurement | record set `budget_matched` (E3, E4, E5, E6; `make records-tables`) | `records/budget_matched/` with raw sources | `budget_matched` not registered; trace check required |
| Table S47 (SI) | 27 | analytical sensitivity | registered formulas and inputs (`make registry-check` covers enumerated identities) | the source observations for any measured inputs | arithmetic validation does not establish the measured inputs |
