# Reconstruct every table and figure

[Documentation](README.md) · [Statement compliance](../STATEMENT_COMPLIANCE.md) · [Publishing the measured outputs](PUBLICATION.md) · [Display inventory](../paper/PAPER_SUPPORT.md) · [Artifact catalog](ARTIFACT_CATALOG.md)

## One command

```bash
make paper                                   # analysis tier: no checkpoints or full prediction arrays
make paper PAPER_TIER=complete               # every published file
make paper ARTIFACTS_ROOT=/abs/artifacts PAPER_ROOT=/abs/paper
make paper-offline                           # reuse artifacts already downloaded
```

`make paper` (`scripts/reconstruct_paper.py`):

1. reads [`release/paper_artifacts.json`](../release/paper_artifacts.json), the manifest
   of the measured outputs published with the release `paper-artifacts`;
2. downloads every published artifact and checks the SHA-256 of each part, of the
   reassembled archive and of every file inside it;
3. verifies each run's sealed eight-stage chain (the analysis tier checks every present
   file and every declared omission against the stage manifests) and each record set's
   trace from summary values to raw files;
4. runs the display generators in `paper/build/displays/`, one per numbered table and
   figure, and the run-derived and supplementary builders;
5. compares every computed value with the value printed in the paper at the printed
   precision.

| Exit status | Meaning |
|---|---|
| `0` | every display generated in full from the published outputs and every compared value agrees |
| `1` | outputs still await publication, a display is incomplete or a value differs; the report names which |
| `2` | a download, an input verification or a generator failed |

The summary is printed and saved beside the workspace as `<workspace>.status.json`.

## Display generators

Each numbered display has a generator registered with
`@display("supplementary:Table S20", stem="S20", needs=(...), scope=...)` in
`paper/build/displays/`:

| Module | Displays |
|---|---|
| `main_paper.py` | main Tables 1–2 and Figures 2–6 |
| `run_audits.py` | selection, support, confirmation, admission and qualification displays read from the lifecycle runs |
| `records_campaigns.py` | displays read from measurement-record sets and campaign exports |
| `experiments.py` | ablations, sensitivity grids, buffer and horizon/beam sweeps, outcome definitions |
| `protocol_tables.py` | protocol tables regenerated from the configuration, registered constants and record contracts |
| `analytical.py` | analytical displays computed from the stated formulas and registered inputs |

`needs` names the inputs: all seven lifecycle runs, one run, a measurement-record set, a
campaign, or only the repository's sources. A generator reads its inputs and the
repository's configuration and code; it never reads the printed values. It declares
which printed value each computed cell should reproduce, and the comparison step looks
the printed value up in [`paper/reported_results.json`](../paper/reported_results.json)
afterwards. A generator whose inputs are not yet published is reported as awaiting
those inputs; nothing is substituted for them.

## Workspace contents

| File or directory | Purpose |
|---|---|
| `RECONSTRUCTION_REPORT.md` | every display with its generator status, its comparison counts and any awaited input |
| `reconstruction.json` | input and software identities, outputs and their relationship to the 90 displays |
| `FILES.json` | SHA-256 and byte size of every exported file |
| `displays/tables/` | complete tables (CSV, Markdown, LaTeX) |
| `displays/figures/` | complete figures (PNG, SVG) |
| `displays/data/<stem>/` | full-precision source data of every table and figure panel |
| `displays/compare/<stem>.csv` | every printed value beside the computed one, with its status |
| `runs/`, `records/`, `supplement/` | the run-derived, record-set and supplementary builder outputs |

`python -B scripts/reconstruct.py --verify /abs/paper` detects missing, extra or
changed files. The publication step copies `displays/{tables,data,compare}` into
[`paper/source_data/`](../paper/source_data) so that the source data of every display
is also part of the Git tree.

## Other inputs

`scripts/reconstruct.py` builds the same workspace from local inputs:

```bash
python -B scripts/reconstruct.py --runs-root /abs/runs --records-index /abs/records/index.json \
  --confirmation-campaign /abs/s14 --budget-campaign /abs/e3_e6 --campaign NAME=/abs/dir \
  --output-dir /abs/workspace --label my-runs
make reconstruction-example REPORT_ROOT=../cm-reports      # the bundled synthetic examples
```

`--runs-root` accepts a completed run, a directory of completed runs (for example the
`runs/` of a `run_cross_morse.sh reproduce` suite) or a Roman-empire campaign with its
ten official splits. Synthetic example inputs are labelled as such in every output and
never count as a reconstruction of the paper.
