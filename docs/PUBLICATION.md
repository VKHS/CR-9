# Publishing the measured outputs

[Documentation](README.md) · [Availability](../AVAILABILITY.md) · [Statement compliance](../STATEMENT_COMPLIANCE.md) · [Producing the outputs](PRODUCTION.md) · [Reconstruction](RECONSTRUCTION.md) · [HPC guide](HPC.md)

The paper's data-availability statement places the checkpoints, predictions,
propensity-logged trajectories, statistical outputs, split and seed manifests and
observed environment of the reported experiments in this repository. The publication workflow attaches verified artifacts to the **`paper-artifacts`** release in
[VKHS/CR-9 Releases](https://github.com/VKHS/CR-9/releases), listed
with their SHA-256 identities in [`release/paper_artifacts.json`](../release/paper_artifacts.json).
GitHub accepts release assets up to 2 GiB each and limits neither the total size of a
release nor its download bandwidth; larger archives are split into numbered parts.

## What is published

`python -B scripts/paper_artifacts.py plan` derives the list from the display
generators: every lifecycle run (ogbn-arxiv, ogbn-proteins, ogbn-products,
Roman-empire with its ten splits, PascalVOC-SP, COCO-SP, Peptides-func), every
measurement-record set and campaign that a table or figure reads, and the observed
GPU environment. Each run is published in two tiers:

| Tier | Contents | Used by |
|---|---|---|
| `analysis` | every sealed file except checkpoints and full prediction arrays: configuration, split arrays, exploration trajectories with propensities, fit audits, selection, confirmation and admission records, paired test units, final summaries, stage manifests | `make paper` (default) |
| `complete` | every sealed file, including all checkpoints (`*.pt`), member and candidate predictions and final probability arrays | `make paper PAPER_TIER=complete`; re-execution audits |

The analysis tier applies to lifecycle run archives. Record sets and campaigns
have a complete tier, so analysis-mode reconstruction still downloads their
baseline-study archives and any controller checkpoints they contain.

The analysis tier carries the complete inventory of its run. Verification checks
every present file against the run's sealed stage manifests and every omitted file
against its declared SHA-256 and size, so the two tiers describe the same run.

## Check original local folders before upload

The checked [input-map template](../release/templates/paper-artifacts.map.json)
lists the canonical required artifacts. Copy it outside the checkout and fill
the paths and `_relationships` with the original experiment folders and their
documented relationship to the paper. A prepared template identifies inputs to
provide; it does not represent measured evidence.

```bash
make paper-preflight \
  PUBLISH_MAP=/abs/paper-artifacts.map.json \
  PAPER_PREFLIGHT_ROOT=/abs/new-preflight
```

The local preflight verifies supplied lifecycle stage seals, all required Roman
splits, raw-to-summary record traces, campaign contracts and the observed GPU
environment with the same source verifiers used by publication. It then constructs
a canonical input workspace, indexes the verified record sets and runs the real
display reconstruction. Missing canonical artifacts are reported by name, including
required inputs not consumed by an individual display. Invalid supplied artifacts
remain errors even if other artifacts are missing.

The workspace must be new and must not overlap a supplied input directory or the
repository checkout. Input and workspace paths must not traverse symbolic links.
By default it uses hard links on the same filesystem to avoid duplicating large
checkpoints. Those links share the source bytes; they are not independent immutable
copies. The preflight does not change permissions or modify the originals. For
separate filesystems or a physically separate copy, use:

```bash
make paper-preflight \
  PUBLISH_MAP=/abs/paper-artifacts.map.json \
  PAPER_PREFLIGHT_ROOT=/abs/new-copied-preflight \
  PAPER_PREFLIGHT_COPY=1
```

The equivalent Python option is `--copy-mode copy`. Source inventories, the input
map, reconstruction code and materialized inputs are checked again after the build
to detect changes during verification. An original run's own `source/` snapshot
can resolve its historical runtime only when its files match the recorded runtime
fingerprint; a mismatching or symlinked snapshot is rejected. Existing bundled
snapshots remain available for the historical examples.

`REPORT.md` provides the readable result, `report.json` records the identities,
relationships and named gaps, `inventories/` records verified input files, and
`reconstruction/` holds the generated outputs and comparisons.

| Exit | Local verdict | Meaning |
|---|---|---|
| `0` | `READY_FOR_PUBLICATION` | Required local inputs verify and every required display comparison agrees. |
| `1` | `INCOMPLETE` | Required inputs, display content or comparison definitions are absent, or computed values differ. |
| `2` | `INVALID` | Input bytes, materialization, source verification or a generator failed. |

This command performs no upload, submits no jobs and does not change the release
inventory. `public_availability_verified` is always false in its report. A locally
ready collection becomes publicly available only after upload and successful
retrieval and verification. `make statement-fulfilled` remains the final public
availability check.

## Publish from Arrhenius

On a login node, in a checkout of this repository, after `bash run_cross_morse.sh reproduce`
has completed the seven lifecycle runs:

```bash
bash hpc/produce_paper_outputs_arrhenius.sh "$CM_BASE/outputs/CROSS_MORSE_REPRODUCTION_0001" \
  --baselines /abs/pinned-baseline-study-plans
python3 -B scripts/paper_outputs.py status --work $CM_BASE/paper-outputs/CROSS_MORSE_REPRODUCTION_0001
export GITHUB_TOKEN=...                                           # fine-grained token, contents: write on VKHS/CR-9
bash hpc/publish_paper_artifacts_arrhenius.sh all $CM_BASE/paper-outputs/CROSS_MORSE_REPRODUCTION_0001/paper-artifacts.map.json
make statement-fulfilled                                          # verify the full availability claim
```

[Producing the outputs](PRODUCTION.md) describes the job graph that the first command
submits; its final job writes the named artifact map and records each verification outcome.
`all` runs `pack` and `publish` below in one step. Directories produced outside that
workflow are published with a completed input map. The checked
[publication-map template](../release/templates/paper-artifacts.map.json) already
lists every required artifact and its relationship field. Copy it to your work
directory, fill the absolute paths and explicit relationships, then run `pack`
and `publish`. Empty fields are preparation inputs, not experimental records; the
validator requires actual artifacts before publication. Regenerate the template
with the command below if the artifact inventory changes.

```bash
bash hpc/publish_paper_artifacts_arrhenius.sh template           # writes paper-artifacts.map.json
$EDITOR $CM_BASE/paper-artifacts/paper-artifacts.map.json         # absolute paths and _relationships
bash hpc/publish_paper_artifacts_arrhenius.sh pack $CM_BASE/paper-artifacts/paper-artifacts.map.json
bash hpc/publish_paper_artifacts_arrhenius.sh publish
```

- `pack` runs as a Slurm job with the SITE SETTINGS of `run_cross_morse.sh`
  (`CM_PUBLISH_LOCAL=1` packs in the current shell). Each directory is checked with
  the repository's own verifiers before it is archived: the sealed eight-stage chain
  of every run, all ten Roman-empire splits, the summary-to-raw trace of every
  record set, both campaign evidence contracts and the captured GPU environment. Only
  full scientific runs are accepted. Archives are deterministic and content-addressed:
  packing identical inputs produces identical asset names and bytes.
  If an original run depends on an external frozen runtime beside its folder,
  packing includes that verified runtime's Python files under the run's `source/`
  in both tiers and records their hashes. These are the files covered by the
  runtime fingerprint, not a claim to include an entire unrelated checkout.
  Heavy run files are streamed from their existing location; original folders
  are not rewritten. Changed source bytes or conflicting source destinations
  stop packing before the new assets and entry are installed.
- `publish` creates the release if needed, uploads every part that is not already
  present with the same SHA-256. Existing assets are never deleted or replaced; a
  conflicting asset is an error. The manifest is updated only after all uploads verify.
  It then retrieves the complete tier, rechecks every file, reconstructs the supported
  displays with `scripts/reconstruct_paper.py`, and writes the small committed views: `manifests/published/` (role memberships and seeds of every
  run), `environment/observed/` (the execution environment of every run) and
  `paper/source_data/` (every generated CSV/JSON source file, the reconstruction report,
  its file inventory and a recomputed comparison status). Every view is linked to the
  current release manifest and artifact identities; source data also identifies its
  builders and registered comparison inputs. `make statement` refreshes
  `STATEMENT_COMPLIANCE.md`.
- Commit those files (or set `CM_PUBLISH_COMMIT=1` to commit and push them from a clean
  checkout). Verification or generator errors stop publication before derived views
  are committed. Numerical differences or incomplete display coverage remain explicit
  in the report and do not become a claim that the paper was reproduced exactly.

The map is a JSON object from artifact identifier to directory:

```json
{
  "run:ogbn-arxiv": "/abs/suite/runs/01_arxiv",
  "run:Roman-empire": "/abs/suite/runs/04_roman_empire",
  "records:budget_matched": "/abs/records/budget_matched",
  "campaign:budget": "/abs/campaigns/e3_e6",
  "environment:gpu": "/abs/environment-capture",
  "_relationships": {
    "run:ogbn-arxiv": "post_result_revision",
    "run:Roman-empire": "post_result_revision",
    "records:budget_matched": "post_result_revision",
    "campaign:budget": "post_result_revision",
    "environment:gpu": "post_result_revision"
  }
}
```

The excerpt above shows selected artifacts; the wrapper requires the complete map
created by `template` or by production collection.

Runs produced by `bash run_cross_morse.sh reproduce` already have this layout. The
environment capture is the directory that
`environment/arrhenius/capture_environment.sh "$CM_VENV_DIR/bin/python" /abs/environment-capture`
writes inside a GPU allocation with the runs' interpreter (`requirements-gpu.lock`,
`observed_environment.json`, `nvidia-smi.txt`, `modules.txt`).
Every mapped artifact requires an explicit relationship in `_relationships` or via
`--relationship` on the pack command (`CM_ARTIFACT_RELATIONSHIP` in the wrapper).
`exact_result_producing_run` identifies supplied original archives;
`post_result_revision` identifies newly executed experiments;
`refactor_with_prediction_equivalence` is the declared equivalence case. No original
relationship is assumed. A repository-built S14 campaign whose lock follows the
lifecycle admissions is recorded as post hoc and cannot be classified as an original
result-producing campaign.

## Retrieve and verify

```bash
make paper                                  # download (analysis tier), verify, rebuild, compare
make paper PAPER_TIER=complete              # also checkpoints and prediction arrays
make fetch-artifacts ARTIFACTS_ROOT=/abs/dir
make paper-artifacts-status                 # published and awaiting artifacts
python -B scripts/paper_artifacts.py verify /abs/dir/runs/arxiv
```

Every downloaded part, reassembled archive and unpacked file is checked against the
manifest, including when reusing cached files offline. A changed byte stops the
command with exit status 2. Unpacking and derived-view replacement are transactional:
verification failures preserve the previous managed directories.
