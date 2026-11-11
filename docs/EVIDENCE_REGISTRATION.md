# Evidence registration

[Documentation](README.md) · [Run archives](RUN_ARCHIVES.md) · [Reviewer guide](REVIEWER_GUIDE.md) · [Availability](../AVAILABILITY.md)

Use this workflow to publish completed runs into a run index and make their
relationship to manuscript results inspectable. The runs are produced by
`run_cross_morse.sh reproduce` ([HPC guide](HPC.md)); the workflow publishes
regenerated or original runs. One private plan names the runs, deposit location,
record exporters and E3–E6 campaign. The commands reuse the archive, record and
scientific-audit contracts; the scientific audit decides whether the benchmark
evidence is complete.

`hpc/discover_evidence_arrhenius.sh` locates complete runs under an export tree
on the cluster and `hpc/pack_evidence_arrhenius.sh` preflights and packs them as
a Slurm array; both read the site settings of `run_cross_morse.sh`.

## 1. Locate completed runs and write the plan

If completed Arrhenius runs or extracted exports are already available under one
directory, start with discovery:

```bash
make evidence-discover EVIDENCE_ROOT=/abs/original-exports \
  DISCOVERY_OUTPUT=/abs/cm-evidence/discovery.json \
  EVIDENCE_PLAN=/abs/cm-evidence/plan.json \
  DEPOSITS_DIR=/abs/cm-evidence/deposits
```

The equivalent direct command is:

```bash
python -B scripts/evidence_pipeline.py discover \
  --root /abs/original-exports --output /abs/cm-evidence/discovery.json \
  --plan-output /abs/cm-evidence/plan.json --deposits-dir /abs/cm-evidence/deposits
```

Discovery scans only that explicit root in deterministic path order. It reuses
the complete-run inventory to check all eight seals, checkpoint and other
artifact bytes, source resolution, split membership and numeric archive
structure. It neither loads checkpoint objects nor trains, downloads, deposits,
registers, or changes the original runs. Hashing large complete exports can take
substantial I/O time. The source snapshot that produced each run must be
available to the existing source resolver; historical identities are preserved.

| Discovery outcome | Meaning and next action |
|---|---|
| `UNIQUE_CANDIDATE` | One complete official full-profile candidate passed the inventory. Its path can appear in the proposed plan; review it and run full preflight. This is not a scientific readiness verdict. |
| `AMBIGUOUS` | Multiple eligible paths exist, even if their byte identities agree. The plan entry stays unset. Inspect the paths and original experiment records; discovery never chooses by timestamp, filename or loss. |
| `MISSING` | No eligible candidate was found. Inspect rejected candidates or locate the original completed output. |
| `UNRESOLVED_SCAN` | A directory could not be read or a scan bound was reached. Every plan path stays unset because another candidate may remain unseen. Narrow the root or increase the explicit bound, then rescan. |
| `REJECTED` candidate | The report gives the exact scope, completeness, source or inventory failure. Synthetic and pilot runs, missing checkpoints, altered seals and incomplete campaigns cannot become official candidates. |

Roman-empire is accepted only as one parent campaign containing `split_00`
through `split_09`, with each split's own official index and a common producing
source and non-split configuration. Individual split runs cannot fill its plan
entry. Symlink directories are reported and never followed. The default scan
bounds are relative depth 6 and 10,000 directories; set
`DISCOVERY_MAX_DEPTH` / `DISCOVERY_MAX_DIRECTORIES`, or the CLI's `--max-depth` /
`--max-directories`, when needed. Recognized runs and campaigns are scan leaves,
so their data directories are checked by the inventory rather than searched for
more candidate roots.

The report is JSON with a readable Markdown file beside it. The optional plan
uses the existing schema and can be read by preflight. It records candidate
source/configuration/unit identities and sealed-content hashes for inspection,
while leaving `license`, `relationship_to_manuscript`, `code_doi`,
`budget_campaign`, and all ten record-set mappings unset. Finding bytes cannot
establish those declarations. Preflight rechecks the actual inputs; discovery
does not authorize publication or substitute for a fresh preflight.

Keep reports, plans and deposits outside both the checkout and the scanned
source directory. Files are created with private permissions and never
overwritten. Choose new report/plan names for another scan; omit `EVIDENCE_PLAN`
(or both `--plan-output` and `--deposits-dir` in the direct command) to produce a
report without a plan. Discovery exits 0 only
when the scan is complete, at least one unique candidate exists, and no dataset
is ambiguous. It exits 1 for an empty, ambiguous or incomplete scan while
retaining the diagnostic report; invalid inputs or unsafe output paths return 2.
Missing datasets still require evidence even when discovery exits 0.

If the paths are already known, or no exported runs are available yet, write an
empty plan instead:

```bash
make evidence-plan EVIDENCE_PLAN=/abs/cm-evidence/plan.json
```

Keep the plan outside the checkout because it contains local experiment paths.
Fill the generated template with the actual inputs:

| Field | Meaning |
|---|---|
| `deposits_dir` | Absolute directory outside the checkout for archives, receipts, reports and `evidence-state.json` |
| `datasets.<name>.run_root` | Completed full-profile run; for Roman-empire, the campaign containing `split_00` through `split_09` |
| `relationship_to_manuscript` | The author's declaration: `exact_result_producing_run`, `refactor_with_prediction_equivalence` or `post_result_revision`; a dataset may override the common declaration |
| `license` | Zenodo license identifier covering the archived material, including applicable dataset rights |
| `code_doi` | Optional published version DOI of the corresponding code snapshot |
| `budget_campaign` | Extracted complete E3–E6 campaign required by the scientific audit |
| `record_sets.<set>` | Either `exporter`, an argument list for an existing exporter, or `mapping`, a path to an original-output mapping |

`EVIDENCE_DATASET=Peptides-func`, for example, limits a command to one dataset.
An unset run path remains unregistered. `make evidence-status EVIDENCE_PLAN=...`
shows the recorded progress and next outstanding steps.

## 2. Preflight the original runs

In an environment with the recorded dependencies, including PyTorch:

```bash
make evidence-preflight EVIDENCE_PLAN=/abs/cm-evidence/plan.json FULL_AUDIT=always
```

On Arrhenius, the array launcher performs preflight and packing, without training
or uploading. Pass the allocation and environment paths explicitly:

```bash
sbatch --account=<allocation> --partition=<partition> \
  --export=ALL,CM_SOURCE_ROOT=/abs/CROSS-MORSE,CM_ENV_FILE=/abs/arrhenius.env,CM_EVIDENCE_PLAN=/abs/cm-evidence/plan.json \
  hpc/pack_evidence_arrhenius.sh
```

Preflight checks the eight-stage chain, sealed artifacts, official-data scope,
60/20/10/5/5 data roles, reserved comparison units, role isolation, actual S5
inventory, numerical replay, registered selection, support reconstruction and
manuscript result bindings. The report identifies the original runtime and the
checker. Historical runs retain their source identity; replay does not relabel
them as experiments performed with the current checkout.

| Verdict | Meaning and next action |
|---|---|
| `READY` | Required run checks and manuscript binding checks passed. Continue with packing. |
| `INTEGRITY_ONLY` | Required integrity checks passed, but manuscript agreement or its declaration is incomplete. Inspect the differences and identify the correct evidence or manuscript revision. |
| `PENDING_FULL_AUDIT` | Required support reconstruction was not completed. Repeat preflight with the required dependencies. |
| `BLOCKED` | A required run check failed. Inspect the per-dataset report before proceeding. |

The plan-driven preflight returns success for `READY` and `INTEGRITY_ONLY`, so
either can be packed for inspection. `BLOCKED`, `PENDING_FULL_AUDIT` and an empty
selection return nonzero while preserving reports. The standalone
`scripts/evidence_preflight.py` returns zero only for `READY`. An interrupted or
skipped required check cannot appear as passed. Synthetic examples are refused
as official benchmark evidence.

## 3. Pack and rehearse the deposit

```bash
make evidence-pack EVIDENCE_PLAN=/abs/cm-evidence/plan.json
read -r -s -p 'Sandbox Zenodo token: ' ZENODO_SANDBOX_TOKEN
export ZENODO_SANDBOX_TOKEN
make evidence-deposit EVIDENCE_PLAN=/abs/cm-evidence/plan.json SANDBOX=1
```

The archive must match the preflight's sealed contents and relevant checker and
registry inputs. The pipeline records its SHA-256 and size and checks them again
before subsequent operations. Replacing predictions or an archive requires a new
validation; unchanged dataset/configuration names cannot reuse an old verdict.
State updates use locks and atomic replacement so interrupted commands and array
tasks do not silently overwrite one another's progress.

Sandbox receipts and credentials are separate from production. A sandbox draft
or publication cannot enter the official archive index. Receipts are retained so
an interrupted request can resume the same deposit.

## 4. Publish and register

```bash
read -r -s -p 'Production Zenodo token: ' ZENODO_TOKEN
export ZENODO_TOKEN
make evidence-deposit EVIDENCE_PLAN=/abs/cm-evidence/plan.json
# Inspect the production drafts and their metadata before this explicit publication step.
make evidence-publish EVIDENCE_PLAN=/abs/cm-evidence/plan.json PUBLISH=1
make evidence-register EVIDENCE_PLAN=/abs/cm-evidence/plan.json
```

Publication requires an explicit switch and a current eligible preflight for the
exact archive. `BLOCKED` and `PENDING_FULL_AUDIT` cannot be promoted to publication
by an override. Publishing an `INTEGRITY_ONLY` archive requires
`ALLOW_INTEGRITY_ONLY=1`; it remains evidence with unresolved manuscript agreement
and cannot make the review verdict complete. Reserved DOIs remain distinct from
published version DOIs verified through the public record.

Registration verifies and extracts each archive, writes its identity to
`runs/index.json`, generates `manifests/official/`, and refreshes availability
views. The large payloads stay outside Git. On a fresh clone, `make verify` checks
that each absent deposited run has a complete matching index/manifest descriptor;
this is offline metadata validation. Any present payload is verified in full.
`make fetch-deposits` and `make review` separately check public access, archive
bytes and the materialized evidence.

## 5. Bind observations to manuscript results

```bash
make evidence-bind EVIDENCE_PLAN=/abs/cm-evidence/plan.json
```

Single-run bindings use the existing comparator and binding checker. The Roman
binding uses all ten official splits and the audit's paired split aggregate.
A binding is installed only when it covers the required targets, has the declared
source relationship, and agrees at printed precision. Failed comparisons remain
available in the external report directory for inspection.

An evaluated controller and the policy allowed by confirmation are separate
quantities. For Peptides, the recorded Holm `P = 0.070` exceeds the registered
`0.05` level; the registered lifecycle returns `q_C`. A radius-arm result needs its
own sealed observations and must be labelled accordingly. Resolving a different
manuscript claim requires the relevant original confirmation record or a paper
correction; registering a file does not change the decision rule.

## 6. Build measurement-record sets

Use the existing `budget-matched`, `baselines` and `from-runs` exporters when their
input formats match the original outputs. Otherwise create a mapping:

```bash
make record-mapping-template SET=factorials OUTPUT=/abs/cm-evidence/factorials.mapping.json
# Fill provenance, original file paths, row keys, units and locators.
make record-from-mapping MAPPING=/abs/cm-evidence/factorials.mapping.json
make evidence-records EVIDENCE_PLAN=/abs/cm-evidence/plan.json
```

A mapping copies the named original files and locates each summary value in those
copies. Declared sums, means, ratios and differences retain their input hashes and
are recomputed by record verification. The exported mapping is retained with the
set. Existing sets are preserved unless replacement is requested, and a failed
replacement restores the previous set and index.

This checks traceability and declared arithmetic. It does not independently prove
how the original files were measured. Preserve the instrument, seed, protocol,
hardware and timing/accounting records needed to assess that origin. Values or
units cannot be taken from manuscript targets to fill missing observations.

The required coverage is checked against the record contract. After export,
`make records-check` verifies the sets and `make records-tables` rebuilds their
tables. `make degeneracy-check` highlights repeated baseline patterns for
inspection; a repeated value alone neither invalidates a measurement nor
justifies changing it. Reviewer 1's concern is answered by the raw per-rerun
records and a correct measured-versus-declared description.

## 7. Review and register the final PDFs

This author-side step needs Poppler (`pdfinfo` and `pdftotext`) or the Python
`pypdf` package. Missing text extraction is reported as unavailable, not passed.

```bash
make manuscript-sources \
  MANUSCRIPT_PDF=/abs/main.pdf SUPPLEMENTARY_PDF=/abs/si.pdf \
  REVIEW_TEMPLATE=/abs/cm-evidence/manuscript-review.json
# Review every listed group in context; complete reviewed_by, review_status and notes.
make register-manuscript-sources \
  MANUSCRIPT_PDF=/abs/main.pdf SUPPLEMENTARY_PDF=/abs/si.pdf \
  REVIEW_NOTES=/abs/cm-evidence/manuscript-review.json
```

The first command identifies the PDF bytes and searches the declared pages for
numeric tokens. The occurrence fraction is a search diagnostic, not a percentage
of correct table entries. It does not establish row, unit, sign interpretation or
experimental meaning. For example, AP `0.0058` and `0.58` percentage points can
represent the same effect; a `3.0%` variance share can be stored as `0.03`, and an
exact model count can be derived while the table prints a rounded count.
If the supplied PDFs differ from the current registered identities, inspection
returns nonzero after writing the review template; inspect that difference before
the separate registration step.

The review template is bound to the exact PDFs, registry and page map. Record a
contextual note for each group after checking values, units, derived quantities
and source locations. Correct the registry or page mapping only where that review
establishes a change, then regenerate the template if its inputs changed.
Registration rechecks those inputs and retains the scoped contextual source review in
`paper/transcription_check.json`. A blanket acceptance flag cannot substitute for
that review. PDFs remain outside the repository, and registration does not edit
the manuscript or its numerical targets.

Complete source review before the final production preflight when possible. If
PDF registration or numerical/page-map corrections happen after depositing,
repeat preflight and binding against the updated references before the final
review. The pipeline deliberately refuses an old approval after those inputs
change; unchanged archive bytes and their deposit receipts can still be reused.

## 8. Review the evidence and commit the registration

Refresh generated views and checksums, then run the reviewer workflow with the
actual E3–E6 and [joint S14 confirmation](S14_CONFIRMATION.md) campaigns and
matching PDFs:

```bash
make availability claims availability-statement paper-report manifest
make verify
make review REQUIRE_EVIDENCE=1 REPORT_ROOT=/abs/cm-evidence/review-reports \
  BUDGET_CAMPAIGN=/abs/e3-e6-campaign \
  CONFIRMATION_CAMPAIGN=/abs/s14-confirmation \
  MANUSCRIPT_PDF=/abs/main.pdf SUPPLEMENTARY_PDF=/abs/si.pdf
make readiness \
  BUDGET_CAMPAIGN=/abs/e3-e6-campaign \
  CONFIRMATION_CAMPAIGN=/abs/s14-confirmation \
  MANUSCRIPT_PDF=/abs/main.pdf SUPPLEMENTARY_PDF=/abs/si.pdf \
  REVIEW_REPORT=/abs/cm-evidence/review-reports/review/review.json
```

Readiness reports missing, failed and advisory items. It checks the current
inputs and the review's artifacts instead of accepting an old verdict string.
It can be run earlier to identify the next missing requirement. The final review
must precede final readiness because the review is itself one of those inputs.

Inspect `git diff` before committing the index, official manifests, validated
bindings, record sets, source review and refreshed generated files. Keep private
plans, tokens, receipts, cluster paths, PDF files and large run payloads outside
the commit. A complete verdict describes the implemented evidence checks; it is
not a prediction of a journal's acceptance decision.

Generated registration files include the status paragraph in `README.md`,
`AVAILABILITY.md`, `paper/CLAIMS.md`, `paper/AVAILABILITY_STATEMENT.txt`,
`paper/PAPER_SUPPORT.md`, `paper/RESULTS.md`, `paper/RESULTS.csv`,
`docs/badges/`, the derived registered-dataset list in
`release/release_scope.json`, `RELEASE_MANIFEST.json` and `SHA256SUMS.txt`.
Review and commit those changes together with their underlying indexes so a fresh
clone has current views. Updating the registered-dataset list does not enable an
affirmative claim of original-result reproduction.
