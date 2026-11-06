# Validation

[Documentation](README.md) · [Availability](../AVAILABILITY.md)

The current manuscript support revision records its executed checks in
[VALIDATION.json](VALIDATION.json). The original development records remain
available in [Git history](https://github.com/VKHS/CR-9/tree/89bbd136625d581027a84755cef0715cf2231a5f).

The machine-readable [validation record](VALIDATION.json) records the executed
checks, environment and scope. A source fingerprint identifies runtime code;
a Git commit additionally identifies documentation, examples and packaging.
Historical fixture validation does not establish that the current runtime
produced those observations.

The record identifies the checks executed for this revision and distinguishes
current-source validation from replay of historical examples. Deposit-client
tests use a stateful mocked transport; a passing test does not imply that a live
Zenodo upload was performed. Historical development records remain available
through the linked Git revision, and GitHub Actions results belong to the
displayed commit.

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

This checks official benchmark archives, measurement sets, model identity and
result bindings. It returns nonzero when required records are missing or invalid;
[AVAILABILITY.md](../AVAILABILITY.md) lists the registered deposits.

The repository's GitHub Actions workflow validates the published checkout. Check
its result for the commit you use.
