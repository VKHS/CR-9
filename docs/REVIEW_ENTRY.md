# Reviewer entry point

[Repository](../README.md) · [Method](METHOD.md) · [Reconstruction](RECONSTRUCTION.md) · [Reproducibility checklist](../REPRODUCIBILITY.md)

CROSS-MORSE provides inference-time control for frozen graph predictors,
configurations, experiment drivers and artifact-based reconstruction tools.
Each execution's records identify its data, seeds, checkpoints, policy and
environment. Package names do not substitute for those records.

## Published release and manuscript artifact status

The existing [GitHub Release](https://github.com/VKHS/CR-9/releases/tag/demo-artifacts) is titled **Paper artifacts**, with the actual tag `demo-artifacts`. It contains **40 uploaded assets totaling approximately 10.85 GB** (checked 11 November 2026); the [execution-package manifest](../release/demo_artifacts.json) records **31 of 37 artifact groups as published**. They are real released resources, with their execution scale and data selections documented in the [package guide](../demo/README.md). The separate [full-manuscript manifest](../release/paper_artifacts.json) lists 37 original-paper artifact groups not yet registered as published in that channel. They are separate publication scopes; the second status does not mean the first release is empty.

## Available checks

| Purpose | Command | Scope |
|---|---|---|
| Software and bundled-example review | `make review REVIEW_OFFLINE=1 REPORT_ROOT=../cm-reports` | The checks and numerical replays listed in the report. |
| Published execution-package inventory | `make demo-status` | Assets identified by the existing manifest. |
| Published execution-package reconstruction | `make demo-paper` | Available-input integrity and rendering; inspect the report for absent inputs, incomplete displays and numerical comparisons. |
| Strict manuscript reconstruction | `make paper PAPER_TIER=complete` | Complete-tier retrieval, complete display content and manuscript comparisons. A nonzero exit is not a pass. |
| Full availability-statement verification | `make statement-fulfilled` | The repository's artifact and reconstruction requirements. |

Install the [documented environment](GETTING_STARTED.md) first. Existing command
and release identifiers are retained for compatibility with published URLs and
byte-verified manifests. Their spelling does not determine scientific scope.

## Trace a reported value

The [display inventory](../paper/display_manifest.json) identifies each generator
and its inputs. A reconstruction workspace includes input identities, full-precision
source data, comparison CSVs and output hashes. Check both display-content
completeness and comparison coverage; generating a file alone is not sufficient.

The [execution-package guide](../demo/README.md) lists exact dataset selections
and production settings. Its [execution report](../demo/RESULTS.md) and
[source-data index](../demo/published/paper/source_data/INDEX.json) retain the
recorded outcomes and coverage. Independent executions retain their own identities.

## Documentation and data

Wording corrections do not modify measurements or certify an unexecuted check.
Experiment outputs, analytical calculations and software-validation fixtures retain
their distinct meanings. Planning material describes an intended publication layout,
not an execution record. For manuscript correspondence, use the measured records
and strict reconstruction report rather than a planning summary.
