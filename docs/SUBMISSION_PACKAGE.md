# Connect the submission package to the availability statement

[Repository](../README.md) · [Reconstruction](RECONSTRUCTION.md) · [Publication](PUBLICATION.md) · [Reproducibility](../REPRODUCIBILITY.md)

The submission tools connect existing experimental outputs to the canonical
artifact inventory and the strict reconstruction command. They do not train a
model, change a measurement, replace manuscript reference values, remove old
records, or change repository visibility.

## 1. Inspect the connections without changing the checkout

```bash
make submission-inventory SUBMISSION_ROOT=/absolute/new-inventory
```

This writes `REPORT.md` and `report.json` outside the checkout. Each required
artifact is mapped to its archive descriptors, identity and affected displays;
the report also locates descriptors in other manifests without silently assigning
them to the manuscript. The twelve availability clauses are evaluated using the
existing repository checker. Missing registration does not mean that the authors'
original experiment files do not exist.

This is a local inventory check, not a network-access test or reconstruction
certificate. Exit 0 means the canonical descriptors are connected, 1 means
connection work remains, and 2 indicates an execution/input error.

## 2. Connect the existing result folders

Create an input-map template using the existing publication command:

```bash
python -B scripts/paper_artifacts.py map-template --output /absolute/paper-inputs.json
```

Fill in absolute paths to the existing seven dataset runs (including the Roman
split campaign), auxiliary record sets, campaigns and captured environment.
Preserve each input's actual relationship to the manuscript in `_relationships`.
The original execution is `exact_result_producing_run`; a newly executed revision
is `post_result_revision`. Do not change archived data identities or profiles to
make an input satisfy a different execution contract.

First check the existing data, without any upload:

```bash
make submission-finalize PUBLISH_MAP=/absolute/paper-inputs.json \
  SUBMISSION_ROOT=/absolute/new-local-check
```

The command uses a detached Git worktree and independent copies for its preflight.
It checks original source identities and reconstructs before permitting publication.
Input folders and the calling branch remain unchanged. The default independent
copies require additional disk space. The script's optional `--copy-mode hardlink`
shares input bytes and should be used only when the underlying inputs are immutable.

Explicitly authorize publication only after the local check passes:

```bash
# Authenticate with gh auth login, or supply GH_TOKEN/GITHUB_TOKEN through your shell.
# Never put credentials in a command argument or a committed file.
make submission-finalize PUBLISH_MAP=/absolute/paper-inputs.json \
  SUBMISSION_ROOT=/absolute/new-publication PUBLISH=1
```

This packages the existing outputs, publishes immutable release assets, downloads
and verifies the complete tier, runs strict reconstruction, derives the executed
manifests/environment/source-data/baseline views, refreshes generated availability
pages, and verifies the resulting source package. No training is performed.

The result is `submission-registration.patch` and a staged registration in the
isolated checkout. Review and commit it through your normal branch workflow.
There is no automatic push of registration changes or change of repository
visibility. A failed step stops the process and keeps its diagnostic log. Uploads
completed before a later failure remain preserved; the tool never deletes or
replaces release assets. Keep the isolated checkout and input files until the
registration has been reviewed.

## 3. Verify the registered release in one command

After the generated registrations have been committed:

```bash
make submission-verify SUBMISSION_ROOT=/absolute/new-verification
```

This requests the complete artifact tier, invokes the existing strict
reconstruction, checks all registered displays and comparisons, and validates the
committed views. A success requires every required artifact, complete display
content, complete comparison coverage and zero differing/missing/unreferenced
values. An empty report, zero-of-zero comparison, partial execution channel or
analysis-only cache cannot satisfy this gate.

The existing `make paper` and `make paper PAPER_TIER=complete` commands retain
their meanings and acceptance rules. The new entry point adds an explicit
submission-level report; it does not relax those rules.

## Access and storage

A reviewer needs access to both the repository and the selected release assets.
While the repository is private, authenticate with an authorized account using
`gh auth login` or a read-capable `GH_TOKEN`/`GITHUB_TOKEN`. The transport resolves
GitHub browser-download links through the release-assets API, follows signed asset
redirects without forwarding GitHub credentials, and retains archive and file
checksum verification. Public access is not certified by a successful authenticated
check. Arrange authorized reviewer access or publish an accessible release before
making an unrestricted availability claim.

The analysis tier saves space by omitting declared heavy lifecycle files. The
complete tier retains those files; selecting a smaller download must not change
the experimental population or the numerical result being reconstructed.

Original measurements, manuscript reference transcriptions, planning calculations
and software-validation fixtures keep their distinct identities. Correct a
transcription against the final manuscript and originating records, then regenerate
the reports; do not replace computed values merely to obtain a passing comparison.
