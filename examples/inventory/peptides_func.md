# Complete-run artifact inventory

**Peptides-func — executed_synthetic_software_example**

Verified all eight stages and 294 sealed files. The directory contains 311 inventoried files (26,480,767 bytes).

This report checks artifact integrity and actual split membership; checkpoint re-execution (`scripts/evaluate_checkpoints.py`) and statistical replay (`scripts/reproduce_results.py`) are separate checks.

## Identities

| Identity | SHA-256 |
|---|---|
| source_sha256 | `b5005fc53d31dfcb430c3b8f82a1fcbb7f97c2e8cb41eefbcb4a1f87e7a4bb5e` |
| config_sha256 | `0c0875b4f7e8cd925798cea3096169894bc619e8af63dff40217e1076210784e` |
| unit_sha256 | `2129ea766109b668a30aacd01f41c824b599c7048a77824e5f1739e32b3f25ad` |

## Contents

| Category | Files | Bytes |
|---|---:|---:|
| ancillary records | 11 | 44,948 |
| candidate and member predictions | 160 | 293,886 |
| configuration and environment | 3 | 9,460 |
| decision and statistical records | 13 | 86,125 |
| final predictions and statistics | 8 | 17,593 |
| model and recovery checkpoints | 50 | 25,328,438 |
| offline trajectories | 24 | 171,707 |
| prepared data and splits | 7 | 97,764 |
| stage manifests | 8 | 254,267 |
| training and freeze records | 27 | 176,579 |

## Data and seeds

Master/split seed: `314159`. Training seeds: `[1, 2]`. These are this run's recorded seeds.

Prediction axis: **graph**; split axis: **graph**; statistical units: **graphs**.

| Role | Label rows | Split units | Nonempty statistical units |
|---|---:|---:|---:|
| fit | 28 | 28 | 28 |
| fit_train | 26 | 26 | 26 |
| fit_val | 2 | 2 | 2 |
| explore | 10 | 10 | 10 |
| select | 5 | 5 | 5 |
| confirm | 2 | 2 | 2 |
| compare | 3 | 3 | 3 |
| admit | 16 | 16 | 16 |
| test | 16 | 16 | 16 |

## Models, trajectories, and decisions

The run contains 4 best checkpoints, 12 complete offline trajectories and 96 recorded transitions. Every checkpoint file is present and matches its seal. Recorded backbone/head hashes agree across stages; logical tensor hashes are not independently recomputed by this stdlib tool.

Admission passed: `False`. Exact qC fallback: `True`. Statistical scope: paired graph bootstrap conditional on the fixed trained ensemble. Seed scope: one fixed prefix ensemble; retraining-seed uncertainty is not estimated.

The JSON companion contains every relative path, size, SHA-256, producing stage, numeric-array shape, trajectory identity, and the captured environment. `unsealed_ancillary` records are hashed for this inventory but are not stage-sealed scientific outputs.
