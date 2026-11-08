# Validation

[Documentation](README.md) · [Availability](../AVAILABILITY.md)

The machine-readable [validation record](VALIDATION.json) records the executed
checks of this release with their environment, scope and the runtime
fingerprint they validated: the full test suite with PyTorch, the analytical
statements, the supplementary arithmetic, the model inventory and fresh
synthetic smoke runs of the runtime, and, under
`reproduction_launcher_validation`, the checks of the reproduction launcher
(the test suite without PyTorch and a simulated Slurm cluster covering
submission, recovery and the report). Both name the released runtime
`0aba361b…`. A source fingerprint identifies runtime code; a Git commit
additionally identifies documentation, examples and packaging. Historical
fixture validation does not establish that the current runtime produced those
observations.

The record identifies the checks executed for this revision and distinguishes
current-source validation from replay of historical examples. Deposit-client
tests use a stateful mocked transport; a passing test does not imply that a live
Zenodo upload was performed. GitHub Actions results belong to the displayed
commit.

## Re-run the software checks

From the recorded CPU environment at the checkout root:

```bash
make paper-report-check PYTHON=python
make verify PYTHON=python
make test PYTHON=python
make reproduce PYTHON=python REPORT_ROOT=../cm-validation/examples
make example-paper PYTHON=python PAPER_EXAMPLES_ROOT=../cm-validation/paper
make review PYTHON=python REVIEW_OFFLINE=1 REPORT_ROOT=../cm-validation
```

`make verify` checks source metadata, registered arithmetic, release integrity,
the two declared synthetic examples and generated artifacts. The test
suite covers algorithms, statistical rules, role separation, provenance and
workflow contracts.

Two of the `make verify` steps concern the manuscript's mathematics and need only
NumPy and the standard library. `make theory` runs a seeded battery that checks
every definition, lemma, proposition, theorem and corollary on random instances
of its hypotheses, including the executable admission bound of
`cross_morse.statistics` and the budget-aware rule of `cm_experiments.budget`,
and confirms that [Analytical statements](THEORY.md) is current. `make si-check`
recomputes the derived entries of the supplementary tables from their printed
inputs: values that a different reading reproduces are reported as advisory
notes, and a value that no documented reading reproduces fails the check.
`make theory-full` and `make si-report` write the complete tables to `REPORT_ROOT`. A fresh smoke run separately checks execution of the current
runtime; see [Getting started](GETTING_STARTED.md).

Registration tests cover stale preflight reports, changed archive bytes, absent
deposited payloads, manifest/index disagreement, interrupted record replacement,
derived-value tampering, PDF review freshness and stale reviewer reports. They
exercise both refusal cases and valid fixture-based paths; fixtures do not count
as official experimental deposits. A real cluster preflight and public deposit
verification are separate operations in the [registration guide](EVIDENCE_REGISTRATION.md).

## Scientific evidence gate

```bash
make scientific-audit PYTHON=python REPORT_ROOT=../cm-validation
```

This checks registered benchmark archives, measurement sets, model identity and
result bindings, and returns nonzero while required records are missing or
invalid; this release registers none ([AVAILABILITY.md](../AVAILABILITY.md)). A
regenerated suite is checked by its own report (`run_cross_morse.sh reproduce`)
and by `make compare RUN_ROOT=...` for a single run.

The repository's GitHub Actions workflow validates the published checkout. Check
its result for the commit you use.

## Evidence completion validation — 8 November 2026

The `evidence_completion_validation` section of [VALIDATION.json](VALIDATION.json)
records the upgrade reviewed against `f34aa5a`, preserving the subsequent author
change in `578837f`. Earlier dated sections below remain historical records.

| Check | Executed result and scope |
|---|---|
| Integrated CPU suite | 1,698 tests: 1,696 passed, 2 skipped; 582.769 seconds |
| Integrated suite without PyTorch | 1,405 tests: 1,354 passed, 51 skipped; 301.977 seconds |
| Final focused suite after the source-reference corrections | 195 passed, no skips; 39.318 seconds |
| Source and release verification | `make verify verify-manifest` passed; 1,229 source files in the release inventory |
| CPU environment | Python 3.12.14, PyTorch 2.10.0+cpu; all 53 recorded package versions match |
| Offline reviewer workflow | 17 steps passed; verdict `SOFTWARE-ONLY` |
| Supplied paper and supplement | Both PDF byte identities verified; 34 registry groups checked |
| Offline paper reconstruction | 20 repository-source displays rendered, no generator errors; the command correctly returns incomplete while measured artifacts await publication |

The integrated suites and offline reviewer ran before the final bounded S15/S19,
S17d and textual source-reference corrections. The final focused suite covers
those corrections and their reconstruction, integrity and availability paths.
The machine-readable record identifies both software snapshots and the files
changed between them. GitHub's fresh-clone and CPU-container workflows identify
the exact proposed commit they validate; a local record does not pre-attest
their result.

The CPU suite's two skips concern a synthetic smoke execution that did not
confirm a state contrast and a private cluster profile excluded from the public
package. The no-PyTorch suite skips cases that require PyTorch. Synthetic
fixtures, mocked release transports and simulated Slurm state validate software
behavior; they do not supply original benchmark measurements.

The available offline comparisons contain 437 exact matches, 15 literal
differences and four missing computed values. Eighteen displays have complete
content and comparison coverage; coverage alone does not mean every value
agrees. All ninety numbered displays remain registered. The differences include
wording, schema representation and scientific protocol questions, as classified
in [the availability review](AVAILABILITY_REVIEW.md). They are not fifteen
independent numerical defects. The complete availability statement remains
unverified until the measured artifacts and all required comparisons pass.

The upgrade preserves the core runtime, configurations, sealed examples and
source snapshots. It preserves all 2,349 existing numeric leaves in the results
registry and every existing numerical constant. Eleven text cells are corrected
against the PDFs, and three PDF-described protocol groups are added to the
constants registry. Their review is AI-assisted, not an author certification.

Arrhenius hostname resolution failed before authentication. No cluster files
were accessed and no benchmark jobs were submitted. Recovering the original
result-producing archives remains the preferred next step; newly executed
experiments must retain their new-run provenance.

## Registered statement validation — 7 November 2026

The `registered_statement_validation` section of [VALIDATION.json](VALIDATION.json)
records the checks of the availability-statement work based on commit `f513af0`:

| Check | Executed result |
|---|---|
| Full suite, CPU PyTorch job of the hosted workflow | 1,618 tests: 1,616 passed, 2 skipped |
| Suite in an interpreter without PyTorch | 1,325 tests: 1,274 passed, 51 PyTorch-dependent tests skipped |
| Repository verification | Passed, including the statement check, the artifact plan and generated views |
| `make paper-offline` with nothing published | 20 displays built from repository sources, 70 awaiting the measured outputs, no generator errors |
| Statement items | 2 fulfilled, 10 awaiting the measured outputs, none missing |

The runner tests execute the factorial cells, the history-MLP observer, the S14
builder and the internal rows of Tables S15 and S15b on synthetic smoke lifecycles;
the production pipeline is exercised against a simulated Slurm cluster. No benchmark
observation was produced.

## Availability and reconstruction validation — 7 November 2026

The `availability_reconstruction_validation` section of [VALIDATION.json](VALIDATION.json)
records the integrated checks on the upgrade based on commit `5c1399e`:

| Check | Executed result |
|---|---|
| Full suite in the recorded CPU environment | 1,132 tests: 1,131 passed, one private-profile test skipped |
| Suite in an interpreter without PyTorch | 839 tests: 823 passed, 16 PyTorch-dependent tests skipped |
| Repository verification | Passed, including generated views, artifact catalog, source integrity and sealed examples |
| Offline reviewer workflow | Passed, 17 steps, verdict `SOFTWARE-ONLY` |
| Launcher and reproduction regressions | 35 passed, including a fresh current-source synthetic CPU lifecycle and actual report build |
| Supplementary builders | 14 focused tests passed; five representative plots visually checked |

The reconstruction checks cover input tampering, indexed-archive identity,
Roman split membership, synthetic labeling, duplicate datasets, failed-build
preservation, output tampering and the strict all-display gate. The launcher
tests cover frozen-source consistency, report failures, saved settings, active
jobs, unfinished Roman splits and GPU-health requeue using a simulated scheduler.

The twelve supplementary output groups retain their ten related displays'
column or partial-panel scope. Their tests use temporary validator fixtures.
The fresh CPU integration also uses synthetic data. Original benchmark artifacts
and live cluster execution remain separate evidence inputs. Runtime source and
the manuscript numerical registries retain their existing identities; the
validation record hashes the changed analysis and orchestration tools.
