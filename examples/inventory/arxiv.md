# Complete-run artifact inventory

**ogbn-arxiv — executed_synthetic_software_example**

Verified all eight stages and 292 sealed files. The directory contains 309 inventoried files (27,524,623 bytes).

This report checks artifact integrity and actual split membership; checkpoint re-execution (`scripts/evaluate_checkpoints.py`) and statistical replay (`scripts/reproduce_results.py`) are separate checks.

## Identities

| Identity | SHA-256 |
|---|---|
| source_sha256 | `b5005fc53d31dfcb430c3b8f82a1fcbb7f97c2e8cb41eefbcb4a1f87e7a4bb5e` |
| config_sha256 | `ffdcdec3ad6544029cad37b68a4e0a4bf18cdae4edced64ab985247777ebb9f4` |
| unit_sha256 | `102f5f005748224879d337ba8c8fb33cdbf830615e628f06578fecb639262641` |

## Contents

| Category | Files | Bytes |
|---|---:|---:|
| ancillary records | 11 | 13,192 |
| candidate and member predictions | 160 | 1,452,941 |
| configuration and environment | 3 | 9,379 |
| decision and statistical records | 12 | 70,011 |
| final predictions and statistics | 8 | 25,344 |
| model and recovery checkpoints | 50 | 25,330,338 |
| offline trajectories | 24 | 159,592 |
| prepared data and splits | 6 | 42,498 |
| stage manifests | 8 | 252,611 |
| training and freeze records | 27 | 168,717 |

## Data and seeds

Master/split seed: `314159`. Training seeds: `[1, 2]`. These are this run's recorded seeds.

Prediction axis: **node**; split axis: **node**; statistical units: **balanced_synthetic_only**.

| Role | Label rows | Split units | Nonempty statistical units |
|---|---:|---:|---:|
| fit | 57 | 57 | 8 |
| fit_train | 52 | 52 | 8 |
| fit_val | 5 | 5 | 4 |
| explore | 19 | 19 | 7 |
| select | 10 | 10 | 5 |
| confirm | 5 | 5 | 3 |
| compare | 5 | 5 | 4 |
| admit | 32 | 32 | 8 |
| test | 32 | 32 | 8 |

## Models, trajectories, and decisions

The run contains 4 best checkpoints, 12 complete offline trajectories and 72 recorded transitions. Every checkpoint file is present and matches its seal. Recorded backbone/head hashes agree across stages; logical tensor hashes are not independently recomputed by this stdlib tool.

Admission passed: `False`. Exact qC fallback: `True`. Statistical scope: descriptive dependent-core inference. Seed scope: one fixed prefix ensemble; retraining-seed uncertainty is not estimated.

The JSON companion contains every relative path, size, SHA-256, producing stage, numeric-array shape, trajectory identity, and the captured environment. `unsealed_ancillary` records are hashed for this inventory but are not stage-sealed scientific outputs.
