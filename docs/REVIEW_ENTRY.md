# Reviewer entry point

[Repository](../README.md) · [Method](METHOD.md) · [Reconstruction](RECONSTRUCTION.md) · [Reproducibility checklist](../REPRODUCIBILITY.md)

CROSS-MORSE provides inference-time control for frozen graph predictors, with the
configurations, experiment drivers and artifact-based reconstruction tools of the
paper. Each execution's records identify its data, seeds, checkpoints, policy and
environment.

## Releases

The release [`demo-artifacts`](https://github.com/VKHS/CROSS-MORSE/releases/tag/demo-artifacts)
holds the complete execution of the pipeline on public data on CPU: **40 assets, 10.85 GB**,
carrying 31 of the 37 artifact groups that [release/demo_artifacts.json](../release/demo_artifacts.json)
lists. The [CPU execution guide](../demo/README.md) gives its dataset selections and
settings, and [demo/RESULTS.md](../demo/RESULTS.md) what `make demo-paper` rebuilds from it.

[release/paper_artifacts.json](../release/paper_artifacts.json) lists the 37 artifact
groups of the paper's measured outputs, which `hpc/publish_paper_artifacts_arrhenius.sh`
uploads to the release `paper-artifacts`; [STATEMENT_COMPLIANCE.md](../STATEMENT_COMPLIANCE.md)
reports their status item by item.

## Checks

| Purpose | Command | Scope |
|---|---|---|
| Software and bundled examples | `make review REVIEW_OFFLINE=1 REPORT_ROOT=../cm-reports` | The checks and numerical replays listed in the report. |
| CPU execution inventory | `make demo-status` | The assets of the release `demo-artifacts`. |
| CPU execution reconstruction | `make demo-paper` | Download and verify the assets, rebuild the tables and figures, and compare the values with the printed ones. |
| Paper reconstruction | `make paper PAPER_TIER=complete` | The paper's measured outputs: complete-tier retrieval, every display and the comparison with the printed values. |
| Availability statement | `make statement-fulfilled` | Every item of the data-availability statement. |

Install the [documented environment](GETTING_STARTED.md) first.

## Trace a reported value

The [display inventory](../paper/display_manifest.json) identifies each generator
and its inputs. A reconstruction workspace includes input identities, full-precision
source data, comparison CSVs and output hashes, and its report states each display's
content completeness and comparison coverage.

The [CPU execution guide](../demo/README.md) lists the dataset selections and
settings of the CPU runs; its [results](../demo/RESULTS.md) and
[source-data index](../demo/published/paper/source_data/INDEX.json) record their
outcomes and coverage.
