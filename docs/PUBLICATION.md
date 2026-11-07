# Publishing the measured outputs

[Documentation](README.md) · [Availability](../AVAILABILITY.md) · [Statement compliance](../STATEMENT_COMPLIANCE.md) · [Reconstruction](RECONSTRUCTION.md) · [HPC guide](HPC.md)

The paper's data-availability statement places the checkpoints, predictions,
propensity-logged trajectories, statistical outputs, split and seed manifests and
observed environment of the reported experiments in this repository. They are
attached to the GitHub release **`paper-artifacts`** of
[VKHS/CR-9](https://github.com/VKHS/CR-9/releases/tag/paper-artifacts) and listed
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

The analysis tier carries the complete inventory of its run. Verification checks
every present file against the run's sealed stage manifests and every omitted file
against its declared SHA-256 and size, so the two tiers describe the same run.

## Publish from Arrhenius

On a login node, in a checkout of this repository:

```bash
bash hpc/publish_paper_artifacts_arrhenius.sh template           # writes paper-artifacts.map.json
$EDITOR $CM_BASE/paper-artifacts/paper-artifacts.map.json         # absolute directory of each artifact
bash hpc/publish_paper_artifacts_arrhenius.sh pack $CM_BASE/paper-artifacts/paper-artifacts.map.json
export GITHUB_TOKEN=...                                           # fine-grained token, contents: write on VKHS/CR-9
bash hpc/publish_paper_artifacts_arrhenius.sh publish
```

- `pack` runs as a Slurm job with the SITE SETTINGS of `run_cross_morse.sh`
  (`CM_PUBLISH_LOCAL=1` packs in the current shell). Each directory is checked with
  the repository's own verifiers before it is archived: the sealed eight-stage chain
  of every run, all ten Roman-empire splits, the summary-to-raw trace of every
  record set. Only full scientific runs are accepted. Archives are deterministic:
  packing the same directory twice gives identical bytes.
- `publish` creates the release if needed, uploads every part that is not already
  present with the same SHA-256, and records the assets in
  `release/paper_artifacts.json`. It then downloads everything back, verifies it,
  rebuilds every table and figure with `scripts/reconstruct_paper.py`, and writes the
  commit-sized views: `manifests/published/` (role memberships and seeds of every
  run), `environment/observed/` (the execution environment of every run) and
  `paper/source_data/` (the numerical source data of every display with its
  comparison to the printed values). `make statement` refreshes
  `STATEMENT_COMPLIANCE.md`.
- Commit those files (or set `CM_PUBLISH_COMMIT=1` to commit and push them).

The map is a JSON object from artifact identifier to directory:

```json
{
  "run:ogbn-arxiv": "/abs/suite/runs/01_arxiv",
  "run:Roman-empire": "/abs/suite/runs/04_roman_empire",
  "records:budget_matched": "/abs/records/budget_matched",
  "campaign:budget": "/abs/campaigns/e3_e6",
  "environment:gpu": "/abs/environment-capture"
}
```

Runs produced by `bash run_cross_morse.sh reproduce` already have this layout. The
environment capture is the directory that
`environment/arrhenius/capture_environment.sh "$CM_VENV_DIR/bin/python" /abs/environment-capture`
writes inside a GPU allocation with the runs' interpreter (`requirements-gpu.lock`,
`observed_environment.json`, `nvidia-smi.txt`, `modules.txt`).
`--relationship` records whether a run is the result-producing run of the paper
(`exact_result_producing_run`, the default), a later reproduction
(`post_result_revision`) or the declared equivalence case.

## Retrieve and verify

```bash
make paper                                  # download (analysis tier), verify, rebuild, compare
make paper PAPER_TIER=complete              # also checkpoints and prediction arrays
make fetch-artifacts ARTIFACTS_ROOT=/abs/dir
make paper-artifacts-status                 # published and awaiting artifacts
python -B scripts/paper_artifacts.py verify /abs/dir/runs/arxiv
```

Every downloaded part, reassembled archive and unpacked file is checked against the
manifest. A changed byte stops the command with exit status 2.
