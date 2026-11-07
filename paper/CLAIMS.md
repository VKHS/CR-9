# Claim-to-evidence map

Generated from `paper/sources.json`, `paper/display_manifest.json`, the display generators and `release/paper_artifacts.json`. `make claims-check` checks freshness.

## Outputs in this release

- The measured outputs of the reported experiments are assets of the [release](https://github.com/VKHS/CR-9/releases/tag/paper-artifacts): 0 of 8 required artifacts published.
- `make paper` downloads and verifies them, rebuilds every numbered display with `paper/build/displays/` and compares every computed value with the printed one.
- Availability statement: 2 of 12 items fulfilled ([compliance](../STATEMENT_COMPLIANCE.md)).
- Runtime source identity: `0aba361b54f6561f407f097af257cf27e4bf7f85d1ff5f47a4bcdd8bcda3a4fb`.

## Manuscript groups, generators and inputs

| Manuscript item | Pages | Evidence kind | Builder or source | Required evidence | Status |
|---|---|---|---|---|---|
| Table 1 (main) | 9, 10 | measurement | Table 1: no generator; Figure S15: no generator; Figure S19: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table 2 (main) | 10, 11 | measurement + derived | Table 2: no generator | repository sources | generated from the repository's configuration and analytical code |
| Abstract and results (main) | 1, 2, 9, 10, 11, 12, 13 | measurement + derived |  | repository sources | no numbered display reads this group |
| Table S4 (SI) | 2 | measurement | Figure S14: no generator; Figure S15: no generator; Table S4: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S5d (SI) | 3 | measurement + derived | Table S5d: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S6c (SI) | 5 | measurement + derived | Figure S6: no generator; Figure S10: no generator; Table S6c: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S7 (SI) | 5 | measurement | Table S7: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S12 (SI) | 8, 9 | measurement | Figure 2: no generator; Table S12: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S12b (SI) | 9 | measurement | Table S12b: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S12c (SI) | 9 | derived arithmetic | Figure 2: no generator; Table S12c: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S13 (SI) | 10 | measurement + derived | Figure S8: no generator; Figure S17: no generator; Figure S19: no generator; Table S13: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S14 (SI) | 10 | measurement | Figure 3: no generator; Figure S9: no generator; Figure S18: no generator; Table S14: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S14b (`supplementary_table_S14b_separated_data_sensitivity`) (SI) | 10, 11 | analytical sensitivity | Figure 3: no generator; Figure S9: no generator; Figure S18: no generator; Table S14b: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S14b (`supplementary_table_S14b_unchanged_contrasts`) (SI) | 10, 11 | analytical sensitivity | Figure 3: no generator; Figure S9: no generator; Figure S18: no generator; Table S14b: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S15 (SI) | 11, 12 | measurement + derived | Figure 5: no generator; Figure S4: no generator; Table S15: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S15b (SI) | 12 | measurement | Figure S2: no generator; Figure S4: no generator; Table S15b: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S16 and Figure S17d (SI) | 12, 13, 36 | measurement + derived | Figure 4: no generator; Figure S17: no generator; Table S16: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S16b (SI) | 12, 13 | measurement + derived | Figure 4: no generator; Table S16b: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S20 (SI) | 15 | measurement + derived | Figure 2: no generator; Figure S17: no generator; Table S20: `paper/build/displays/run_audits.py` (S20) | `run:ogbn-arxiv`; `run:ogbn-proteins`; `run:ogbn-products`; `run:Roman-empire`; `run:PascalVOC-SP`; `run:COCO-SP`; `run:Peptides-func` | awaiting publication: `run:ogbn-arxiv`, `run:ogbn-proteins`, `run:ogbn-products`, `run:Roman-empire`, `run:PascalVOC-SP`, `run:COCO-SP`, `run:Peptides-func` |
| Table S21 (SI) | 15, 16 | measurement | Figure 2: no generator; Figure S7: no generator; Table S21: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S22 (SI) | 16 | measurement + derived | Figure 2: no generator; Table S22: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S23 (SI) | 16 | measurement + derived | Figure 5: no generator; Figure S1: no generator; Figure S10: no generator; Table S23: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S23b (SI) | 16 | measurement + derived | Figure 5: no generator; Figure S2: no generator; Table S23b: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S26b (SI) | 18 | measurement + derived | Figure 4: no generator; Figure S20: no generator; Table S26b: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S30 (SI) | 19 | measurement + derived | Figure S13: no generator; Table S30: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S34 (SI) | 21 | analytical sensitivity | Figure S7: no generator; Table S34: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S37 (SI) | 23 | measurement | Figure S16: no generator; Table S37: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S41 (SI) | 24, 25 | measurement + derived | Figure 4: no generator; Table S41: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S42 (SI) | 25 | analytical sensitivity | Figure S12: no generator; Figure S20: no generator; Table S42: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S44 (SI) | 26 | analytical sensitivity | Figure 5: no generator; Table S44: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S45 (SI) | 26 | measurement + derived | Figure S18: no generator; Table S45: no generator | repository sources | generated from the repository's configuration and analytical code |
| Tables S46 and S46b (SI) | 26, 27 | measurement | Figure 6: no generator; Figure S3: no generator; Table S46: no generator; Table S46b: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S46b (SI) | 27 | measurement | Figure 6: no generator; Figure S3: no generator; Table S46b: no generator | repository sources | generated from the repository's configuration and analytical code |
| Table S47 (SI) | 27 | analytical sensitivity | Figure 6: no generator; Figure S20: no generator; Table S47: no generator | repository sources | generated from the repository's configuration and analytical code |
