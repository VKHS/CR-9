# Reconstruct supported paper outputs

[Documentation](README.md) · [Artifact catalog](ARTIFACT_CATALOG.md) · [Display inventory](../paper/PAPER_SUPPORT.md) · [Supplementary builders](SUPPLEMENT_BUILD.md)

`make reconstruct` builds one traceable workspace from sealed run artifacts,
registered measurement records and optional confirmation or budget campaigns.
The workspace includes a report for every numbered table and figure in the
registered paper and supplement. Each output states whether it supplies table
columns, a partial panel, a full numerical display or a supporting artifact.
Those scopes describe what was actually generated.

The cluster launcher and this command serve different stages of the workflow.
`bash run_cross_morse.sh reproduce` executes the registered dataset lifecycles.
`make reconstruct` reads completed inputs and builds their supported numerical
outputs without training, loading checkpoints into a model or downloading data.
The [artifact catalog](ARTIFACT_CATALOG.md) identifies the producing command and
availability of each input category.

## Verify the workflow with bundled examples

In the recorded CPU environment:

```bash
make reconstruction-example REPORT_ROOT=../cm-reports
python -B scripts/reconstruct.py --verify ../cm-reports/reconstruction-examples
```

This uses the two sealed synthetic examples and their example measurement
records. Tables, plots and the top-level report carry their synthetic scope.
These values demonstrate software execution and remain distinct from the
paper's benchmark observations.

## Build from registered experiment artifacts

First materialize registered archives using the [run archive workflow](RUN_ARCHIVES.md)
or point the builder at already completed run directories. Both paths verify
the sealed input bytes before constructing outputs.

```bash
make reconstruct \
  RUNS_INDEX=/absolute/runs/index.json \
  RECORDS_INDEX=/absolute/records/index.json \
  RECONSTRUCTION_ROOT=/absolute/paper-build
```

For completed runs that have not been placed in an archive index:

```bash
python -B scripts/reconstruct.py \
  --runs-root /absolute/completed-runs \
  --records-index /absolute/records/index.json \
  --output-dir /absolute/paper-build \
  --label regenerated-full-profile
```

`--runs-root` accepts an individual completed run, a directory of completed runs,
or a Roman-empire campaign. A Roman campaign must contain each of its ten official
split IDs exactly once. Multiple unrelated runs for the same dataset are rejected
because they would overwrite the same evidence file; declare their intended
campaign aggregation first.

The run index identifies the registered artifacts, while actual local bytes
establish whether the builder can read them. A `verified` registration flag does
not replace byte verification. Downloadable artifacts whose files have not been
materialized are listed as input requirements in the report.

## Include supplementary analyses

```bash
make reconstruct \
  RUNS_INDEX=/absolute/runs/index.json \
  RECORDS_INDEX=/absolute/records/index.json \
  CONFIRMATION_CAMPAIGN=/absolute/original-confirmation-campaign \
  BUDGET_CAMPAIGN=/absolute/budget-campaign \
  RECONSTRUCTION_ROOT=/absolute/paper-build
```

The factorial and ensemble builders read the corresponding traced record sets.
The S14 builder reads the validated original joint confirmation family; it does
not combine unrelated per-run tests into a historical nine-test result. The
budget builder verifies the campaign contract and retains its declared relation
to the historical table. See [Supplementary builders](SUPPLEMENT_BUILD.md) for
the exact generated columns, panels, CSV sources and statistical scope.

## Workspace contents

| File or directory | Purpose |
|---|---|
| `RECONSTRUCTION_REPORT.md` | Linked generated outputs and coverage of all 90 numbered displays |
| `reconstruction.json` | Input and software identities, output-to-display relationships and build status |
| `FILES.json` | SHA-256 and byte size of every exported file |
| `runs/evidence/` | Evidence extracted from sealed runs, retaining runtime, configuration and unit identities |
| `runs/tables/` | Run-derived tables in CSV, Markdown and LaTeX |
| `runs/figures/` | Related numerical plots in PNG and SVG |
| `records/` | Traced measurement-set tables |
| `supplement/` | Validated supplementary analyses, including the plotted numeric source data |

The builder checks the complete eight-stage chain and its sealed bytes without
deserializing checkpoints. Record-set checks verify both the manifest and each
summary value's locator in its raw source. The report records the builder source
hashes and the interpreter and numerical-library versions used for the build.

Builds use a fresh staging directory. A failed builder leaves an existing valid
workspace intact. A successful rebuild replaces the complete managed workspace,
so obsolete files from an earlier build cannot appear to be current outputs.
Directories that are not managed workspaces, changed workspaces and input
directories are preserved rather than overwritten.

## Verify or require completeness

```bash
make reconstruction-check RECONSTRUCTION_ROOT=/absolute/paper-build
python -B scripts/reconstruct.py --verify /absolute/paper-build --require-all-displays
```

The first command verifies the exported bytes and file set. The second also
requires full numerical coverage of every display from non-synthetic inputs.
The equivalent build-time option is `make reconstruct REQUIRE_ALL_DISPLAYS=1`.

| Exit status | Meaning |
|---|---|
| `0` | Requested supported outputs built, or exported workspace bytes verified; any requested completeness condition also passed |
| `1` | Inputs need materialization, no numerical outputs were available, or the requested full-display coverage condition was not met |
| `2` | Invalid input, failed verification, malformed output or a builder error |

`SUPPORTED_OUTPUTS_BUILT` records successful generation within the stated scope.
The separate `all_displays_numerically_reconstructed` field remains false when
any display is represented only by a partial output, protocol check or input
requirement. `historical_experiment_identity_verified` is not set by this builder:
the [scientific evidence audit](VALIDATION.md#scientific-evidence-gate) checks
original result bindings, campaign identities and agreement with the paper.

The reported-results registry supplies comparison targets. The builder never
fills an unmeasured cell with a printed number, converts an analytical scenario
into an observation, or infers historical identity from a successful new run.
