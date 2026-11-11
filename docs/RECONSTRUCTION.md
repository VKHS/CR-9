# Reconstruct every table and figure

[Documentation](README.md) · [Statement compliance](../STATEMENT_COMPLIANCE.md) · [Publishing the measured outputs](PUBLICATION.md) · [Display inventory](../paper/PAPER_SUPPORT.md) · [Artifact catalog](ARTIFACT_CATALOG.md)

## Existing released outputs versus full-manuscript artifacts

The existing [10.85 GB release](https://github.com/VKHS/CR-9/releases/tag/demo-artifacts) has 40 uploaded assets and 31 of 37 groups published in its execution-package manifest. That is distinct from the full-manuscript `paper-artifacts` channel required by the reconstruction command below. The published execution-package verification does not by itself certify every numerical result printed in the manuscript.

## One command

```bash
make paper                                   # omits heavy lifecycle checkpoint/prediction files
make paper PAPER_TIER=complete               # every published file
make paper ARTIFACTS_ROOT=/abs/artifacts PAPER_ROOT=/abs/paper
make paper-offline                           # reuse artifacts already downloaded
make paper-offline PAPER_TIER=complete       # require complete cached lifecycle artifacts
```

The smaller analysis tier applies to lifecycle run archives. Record sets and
campaigns retain their complete evidence, including any baseline-study archives
or controller checkpoints they contain.

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
5. compares declared computed values with independently reviewed printed targets,
   and checks that no required registered target or display content was omitted.

| Exit status | Meaning |
|---|---|
| `0` | every display has complete content and comparison coverage from the published outputs; all required values agree at printed precision |
| `1` | a required artifact is unregistered, unpublished, absent locally or cached at an insufficient tier; content or a comparison definition is missing; a value cannot be compared or differs |
| `2` | a download, an input verification or a generator failed |

The summary is printed and saved beside the workspace as `<workspace>.status.json`.
Offline reuse checks each artifact against the current release manifest, verification
stamp and file inventory. It does not trust an old download directory solely because
the files are present.

An analysis-tier cache cannot satisfy a complete-tier request. Canonical required
artifacts remain mandatory even if their manifest row is absent or its `required`
flag is edited. This includes the observed environment, which may not be read by
an individual figure generator. The status JSON reports `artifact_materialization`
with the requested tier, verified tiers and named missing inputs. An incomplete
local cache still produces the available reconstruction and a diagnostic report;
changed or conflicting bytes fail verification.

To reconstruct from original local experiment folders before upload, use
`make paper-preflight PUBLISH_MAP=/abs/map.json PAPER_PREFLIGHT_ROOT=/abs/new-work`.
The [publication guide](PUBLICATION.md) describes the verified input workspace and
its local-only readiness verdict.

### The commands named in the manuscript on the published outputs

After `make paper` (or `make fetch-artifacts`), the verified artifacts are under `ARTIFACTS_ROOT`:
`runs/<dataset>/` (Roman-empire: `split_00` to `split_09`), `records/<set>/` with its
`records/index.json`, `campaigns/` and `environment/`. The other commands that the manuscript
names read them directly:

```bash
make records-tables                                  # record-derived supplementary tables, every value traced to its raw file
make tables  RUNS_ROOT=../cm-artifacts/runs          # run-derived tables from completed run directories
make figures RUNS_ROOT=../cm-artifacts/runs          # run-derived figures
make tables  RUNS_ROOT=<suite>/runs                  # the same builders on a reproduction suite
```

`make records-tables` reads `records/` when it registers record sets and otherwise the record
sets that `make paper` downloaded and verified (`RECORDS_ROOT=` selects another directory).
`make fingerprint` prints the runtime source identity that every stage seal of a run carries.

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
and the PDF-bound [`paper/printed_display_values.json`](../paper/printed_display_values.json)
afterwards. Runtime-filled protocol templates are not comparison references. The
reviewed sidecar identifies the document SHA-256, display and source pages; typed
numeric targets preserve printed trailing-zero precision. A generator whose inputs are not yet published is reported as awaiting
those inputs; nothing is substituted for them.

### Content and comparison coverage

`built` means a generator rendered files. It does not, by itself, mean that the whole
display reproduces the paper. Each display separately records `content_complete`,
`missing_content`, `comparison_complete` and `comparison_coverage`. Full numerical
coverage requires complete content and the required independent comparison targets.
A quantitative figure needs plotted source data and a reviewed display-specific
comparison contract; a few checks against a related table do not certify all its panels.

Table S28b is a fully specified analytical calculation. It builds without lifecycle
archives from the count, variance and target-effect grids printed on supplementary
page 19. Table S28 retains its observed-effect dependency, while its listed reference
counts and ESS define an analytical scenario. Figure S17g uses that declared ESS
scenario; its actual-support companion data remain separate. Successful checks of
panel g do not certify all eight panels of Figure S17.

Comparison CSVs distinguish `exact`, `differs`, `not_comparable`, `no_reference` and
`missing`. Required registered values remain in this accounting even when a generator
omits its comparison call. A printed dash represented by an explicit reference null
can agree with a not-applicable computation; an absent field is different. Required
blank or nonfinite table cells, absent factorial intervals, missing S14 margin tests
and unprovided configuration ranges remain incomplete. Intentional table dashes use
`not_applicable(reason)` so their meaning travels with the output metadata.

The machine-readable summary includes `values_no_reference`, `values_missing`,
`comparison_coverage_complete` and `display_content_incomplete`, in addition to the
matching and differing counts. Every incomplete comparison state prevents a successful
`make paper` exit. A computed disagreement is retained for investigation; it is not
changed to the printed number.

### Paired statistical and campaign inputs

Tables S16/S16b reconstruct factorial contrast intervals from a
recognized `cm-factorial-2` raw record, its hashed `unit_dil.npz`, paired per-unit
losses and the original shared bootstrap stream. They verify all eight means and
marginal intervals, replay the declared seed and replicate count, and compute each
contrast from the same draws. Whole-graph Pascal means use equal graph weights;
arxiv means retain the scored-node weights of the buffered clusters. A summary-only
record supports point contrasts, but cannot determine the joint contrast covariance.
It therefore produces no substituted independent-cell interval.

Figure S17d reproduces the analytical illustration explicitly labelled
"intervals: independence" on supplementary page 36. Its effect-coded coefficients
have magnitude 1/4, so its illustrated half-width is
`sqrt(sum(cell_half_width**2)) / 4`, using the eight measured marginal 95% intervals
and the printed independence assumption. The plot, `panel_d.csv` and the
`panel_d_interval_method` entry in `evidence_scope` identify that assumption; `panel_d_cells.csv`
preserves its source means and interval endpoints. This approximation does not
replace paired inference in Tables S16/S16b. When a recognized joint record is
available, it is also replayed and its actual intervals are exported separately as
`panel_d_paired_joint.csv`, with the source hashes, seed and replicate count in
the factorial provenance. An invalid supplied joint record remains an error; an
absent one is disclosed and is not fabricated from marginal intervals.

The sigma grid `[0.005, 0.007, 0.010]` comes from the S28b caption on supplementary
page 19; panel S17g plots its endpoint curves, as printed on page 36. Registration
of that grid and comparisons for panel d do not establish agreement for the full
eight-panel Figure S17. That requires independently reviewed comparison targets
for all its quantitative content.

Main Figure 6 uses the same strict E3–E6 campaign replay as Tables S46/S46b. It selects
the `selected` configuration explicitly, accepts the registered producer arm labels,
and checks its values against the 24 replayed campaign contrasts. Other configurations
remain available for configuration ranges; they cannot be mistaken for the selected
estimate.

S14-related displays preserve the campaign's production kind, admission findings and
`post_hoc` flag in `evidence_scope`. A new joint family computed after admission can
render actual statistics, but cannot establish that an earlier admission was governed
by a prospective joint lock. Missing original margin-test results are not inferred
from the zero-null P values.

## Workspace contents

| File or directory | Purpose |
|---|---|
| `RECONSTRUCTION_REPORT.md` | every display with its generator status, its comparison counts and any awaited input |
| `reconstruction.json` | input and software identities, outputs and their relationship to the 90 displays |
| `FILES.json` | SHA-256 and byte size of every exported file |
| `displays/tables/` | rendered tables (CSV, Markdown, LaTeX); completeness is recorded per display |
| `displays/figures/` | rendered figures (PNG, SVG); completeness is recorded per display |
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
