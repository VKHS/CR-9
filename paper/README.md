# Paper references and result reconstruction

The canonical repository is [VKHS/CR-9](https://github.com/VKHS/CR-9).
Paper PDFs are not distributed here. [sources.json](sources.json) identifies the
reference documents by SHA-256, size and page count, and maps tables to pages.
[reported_results.json](reported_results.json) contains numerical comparison
targets; [registered_constants.json](registered_constants.json) records the
protocol. [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) explains executable
choices and the run artifacts that identify them.
The [paper-support contract](../docs/PAPER_SUPPORT.md) maps the twenty experiment
requirements to inspectable records and the associated manuscript decisions.

The registry preserves printed values. Its status fields distinguish an
evaluated arm from the outcome allowed by the registered lifecycle. In
particular, Peptides records Holm `P = 0.070`, registered level `0.05`, and
`lifecycle_resolution: return_qC`. Final-policy reports use the comparator
identity; radius-arm measurements retain their own evaluated-arm targets and
require separate evidence bindings. The
[seven-dataset parameter inventory](../docs/DATASETS.md#model-parameter-inventory)
records the explicit architecture settings, exact full-profile counts and their
agreement with Table S5 at its printed precision.
`make model-inventory` constructs and counts the released models;
`make compare-runs` checks official archived checkpoint identities and actual
parameter counts against the manuscript targets.

[CLAIMS.md](CLAIMS.md) is the generated claim-to-evidence ledger: every table group
of `sources.json` with its producing experiment, its builder and how this release
regenerates it (`make claims`, checked by `make claims-check`).
[AVAILABILITY_STATEMENT.txt](AVAILABILITY_STATEMENT.txt) is the journal availability
paragraph generated from the same state (`make availability-statement`).
`run_cross_morse.sh reproduce` regenerates the run-derived results and compares
them with `reported_results.json` ([HPC guide](../docs/HPC.md)).

## Check references

```bash
make registry-check
```

This checks metadata, executable protocol constants, reference formulas and
arithmetic within transcribed tables. S6c residuals and admissible sketch ratios,
for example, use the reported cost multiplier. Arithmetic checks use those
registered inputs; measurement replay uses the originating execution records.

```bash
make si-check     # derived supplementary entries recomputed from their printed inputs
make theory       # analytical statements checked numerically (docs/THEORY.md)
```

`scripts/check_si_identities.py` recomputes the enumerated derived entries
from their printed inputs (Holm family, margin and two-way rescaling of S14/S14b,
the S16/S16b factorials, the S12/S12c selection readings, S23b, S26b, S34, S42,
S44-S47 and Note S19/S24 statements) over the rounding box of its inputs. An entry
that only an alternative reading reproduces is reported as an advisory note naming
that reading; such notes describe the transcribed text, not the measurements. PDF bytes are verified only when the documents
are supplied. To verify copies you already have:

```bash
python scripts/check_manuscript_sources.py \
  --manuscript-pdf /path/to/main.pdf \
  --supplementary-pdf /path/to/supplementary.pdf
```

When the final PDFs change, use the [source-registration workflow](../docs/EVIDENCE_REGISTRATION.md#7-review-and-register-the-final-pdfs).
It reports document identities and searches declared pages for registry numbers.
Those searches do not identify a number's row, unit or experimental meaning, and
they can miss percentages, rounded values, scientific notation and derived
quantities. Review the affected groups in context before registering new source
identities. Registration retains that review with the PDF and registry hashes;
it does not edit numerical targets or supply missing experimental evidence.

For an optional compact Markdown/CSV export, run
`make registry-views REPORT_ROOT=../cm-reports`. Reading views are generated on
request; the JSON files remain the single reference source.

## Build results from observations

The modules under `build/` read completed run directories. Synthetic outputs
carry an explicit label; official builds require validated archives and records.

```bash
make example-paper     # two bundled synthetic examples
make paper             # registered official archives and measurement records
make compare-runs      # compare complete official coverage to reference values
make scientific-audit  # check scientific evidence, model identity and bindings
```

[AVAILABILITY.md](../AVAILABILITY.md) lists current coverage and supported
displays. [Run archives](../docs/RUN_ARCHIVES.md) explains how to ingest a run,
register a measurement set and bind an observation to a manuscript result.
Output reports identify any observation that the selected input set does not supply.
## Current manuscript support

[RESULTS.md](RESULTS.md) is the readable seven-dataset results view. It draws
from the reported measurements in this directory and preserves the registered
confirmation outcome separately from an evaluated arm. This reading view uses
registered manuscript values; reconstructed outputs are generated separately
from the corresponding execution artifacts.

[PAPER_SUPPORT.md](PAPER_SUPPORT.md) inventories all 90 numbered tables and
figures in the current main text and supplement. The source-bound
[display manifest](display_manifest.json) records page locations and the
reproduction path for each display. `make paper-report-check` checks its
structure, referenced paths, source identities and generated views.

[MANUSCRIPT_REVIEW.md](MANUSCRIPT_REVIEW.md) records the contextual review of the
provided PDFs. The associated review binds the exact document bytes, registry
files and table-page map. This checks correspondence of the supplied sources;
`run_cross_morse.sh reproduce` regenerates the observations and its report
compares them with these registries ([HPC guide](../docs/HPC.md)).
