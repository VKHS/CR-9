# Reviewer guide

[Documentation](README.md) · [Availability](../AVAILABILITY.md) · [Complete paper support](../paper/PAPER_SUPPORT.md) · [Claims ledger](../paper/CLAIMS.md) · [Reproducibility checklist](../REPRODUCIBILITY.md)

This guide maps the review concerns to commands and inspectable records. The
current coverage is reported from the evidence indexes; expected artifact paths
below do not imply that missing measurements have been deposited.
The [paper-support contract](PAPER_SUPPORT.md) extends this map to E1–E20 and
separates executable checks, required original observations and unresolved
manuscript wording. The archive and record-set counts alone do not measure that
broader scientific coverage.

## Start with the software

From a checkout in the recorded CPU environment:

```bash
make review REVIEW_OFFLINE=1 REPORT_ROOT=../cm-reports
```

For installation, see [Getting started](GETTING_STARTED.md). This command checks
the package, replays the two synthetic examples and builds their supported
displays. It trains no model and requests no benchmark data or deposit download.
`make docker-review REVIEW_OFFLINE=1 REPORT_ROOT=../cm-reports` provides the same
entry point when Docker is available. The container recipes record Python
package versions; their base image and operating-system packages can change.

The stable entry points are `../cm-reports/review/REVIEW_REPORT.md` and
`review.json`. Each invocation records its own check outputs and logs so that a
previous successful audit cannot supply the verdict for a later failed run.
The report also identifies its inputs and check outputs. The author-side
`make readiness` checks their current hashes before accepting a complete report;
an old verdict string alone is insufficient.

## Examine official evidence

Omit `REVIEW_OFFLINE=1` to check registered public Zenodo run records, download
archives as needed, verify bytes, ingest them alongside the selected index, rebuild
supported displays, compare results and run the scientific audit. The registered
index is preserved. Download size and verification time depend on the archives;
this is not a fixed-duration reproduction of the experiments.

```bash
make review REPORT_ROOT=../cm-reports
```

A complete check also needs the E3–E6 campaign, the original joint S14
confirmation campaign, all auxiliary record sets, manuscript result bindings
and the matching final PDFs. For example:

```bash
make review REPORT_ROOT=../cm-reports \
  BUDGET_CAMPAIGN=/path/to/e3-e6-campaign \
  CONFIRMATION_CAMPAIGN=/path/to/s14-confirmation \
  MANUSCRIPT_PDF=/path/to/main.pdf \
  SUPPLEMENTARY_PDF=/path/to/supplementary.pdf \
  REQUIRE_EVIDENCE=1
```

`RUNS_INDEX` and `RECORDS_ROOT` can select existing evidence outside the checkout.
The same selected evidence must be used by fetch, builds, comparisons and the
audit. `REQUIRE_EVIDENCE=1` makes incomplete evidence a nonzero exit; the default
allows a report of missing evidence to be inspected without treating it as a
software failure.

| Verdict | Meaning |
|---|---|
| `EVIDENCE COMPLETE` | All checks required by this workflow passed, including the scientific audit, registered deposit verification, comparisons, decision consistency and supplied PDF identities |
| `EVIDENCE INCOMPLETE` | Software checks passed, but evidence is missing, invalid, inconsistent, unavailable or not fully checked; the report identifies the affected steps |
| `SOFTWARE-ONLY` | Offline software and example checks passed; official benchmark evidence was not established |
| `SOFTWARE FAILURE` | At least one required software check failed, including in offline mode |

These verdicts describe the enumerated automated checks. They do not verify all
mathematical proofs, establish every observation's historical origin or certify
all statements in the manuscript. The claims ledger identifies builder coverage
for the 34 registered table groups; it does not imply that every group has a
dedicated reconstruction script. The [complete display inventory](../paper/PAPER_SUPPORT.md)
also covers every numbered figure and protocol table, with explicit partial
builder and original-analysis requirements.

The [joint confirmation guide](S14_CONFIRMATION.md) describes the original nine
unrounded tests, pairing, fixed policies and official-run bindings. Its checker
reconstructs the joint Holm decisions while preserving the heterogeneous raw-test
methods and their source files. Saved reviews also bind every nested campaign
input; changing a raw file invalidates a previously complete verdict.

## Concern by concern

Paths in the following table are relative to the checkout or the corresponding
report workspace. `make paper` uses `paper/generated/official/` by default;
`make review` records its generated-output paths in its own report.

| Concern | Commands | Evidence to inspect |
|---|---|---|
| **R1.1 — Gains jointly assessed with expected cost and compute-matched alternatives** | `make paper`, `make records-check`, `make scientific-audit` | Run-derived `tables/T2.*` and `tables/S23.*`; `records/ensemble_grid.*` and `records/budget_matched.*` outputs; raw work/timing observations and the complete E3–E6 campaign. Formula checks alone do not establish measured cost |
| **R1.2 — Scope of admission guarantees and single-graph criteria** | `make paper`, `make decision-check`, `make scientific-audit` | `tables/T1.*`, each run's `admit/decision.json`, independent-unit definition and locked policy. Read the distinction between descriptive single-graph results and conditional independent-graph guarantees in [Method](METHOD.md) |
| **R1.3 — Measured versus declared external-baseline costs** | `make records-check`, `make records-tables`, `make degeneracy-check` | `records/baselines/summary.csv`, its raw-file locators, E17 timing traces and scheduler exports; generated `records/baselines.*`. Pattern diagnostics highlight repeated ratios or SDs for inspection; raw provenance determines what was measured |
| **R1.4 / R2.2 — Availability claims supported by accessible artifacts** | `make fetch-deposits`, `make review`, `make availability-statement-check` | `runs/index.json`, download verification report, [availability](../AVAILABILITY.md), [claims ledger](../paper/CLAIMS.md) and the generated [manuscript paragraph](../paper/AVAILABILITY_STATEMENT.txt). A DOI string or draft reservation does not establish public access |
| **R2.1 — Concrete frozen-backbone setting and inadequacy of an optimized fixed schedule** | `make records-tables`, campaign replay, `make scientific-audit` | Motivation and supported deployment setting in [Method](METHOD.md); `records/budget_matched/` with paired E3–E6 comparisons, reserved comparison units and actual work. Implementation alone does not demonstrate a measured advantage |
| **R2.3 — Supported architectures and separate mathematical/empirical claims** | `make model-inventory`, `make compare-runs`, `make records-check`, `make theory`, `make si-check` | Declared operators in [Method](METHOD.md), per-run `train/model_inventory.json`, nominal-identity records and `records/cross_backbone/`. S5 count agreement checks configuration consistency; sealed checkpoints identify the models actually used. [Analytical statements](THEORY.md) separates each mathematical statement, its reference implementation and its numerical check from the empirical records |

## Read the statuses precisely

- **Registered:** an index declares an archive or record set and its identity.
  Ingestion verifies archives before creating entries; the reviewer rechecks the
  deposited bytes rather than trusting the flag alone.
- **Fetched and verified:** the public record's version DOI and matching file were
  checked, and the archive's size and SHA-256 agree with the index.
- **Rebuilt:** a supported output was generated from ingested observations.
  `paper/reported_results.json` remains a comparison target, not a source of
  missing observations.
- **Traced:** `scripts/records.py check` re-read a summary value at its recorded
  precision from the declared raw file and locator.
  Sets exported from a mapping also retain that mapping and recheck declared
  calculations against the copied inputs. This verifies the recorded calculation,
  while the original experiment records establish how its inputs were measured.
- **Bound:** the declared manuscript-to-run relationship passed the implemented
  result and identity checks. Numerical agreement alone is not proof of origin.

`make decision-check` compares the transcribed confirmation contrasts with the
registered rule and policy fields. A reported evaluated arm and the final policy
are separate quantities. Missing or contradictory inputs remain visible;
`make review` cannot promote them to complete evidence.

`make degeneracy-check-registry` applies the pattern diagnostics to the
transcribed baseline ledger. Findings are prompts to inspect measurement records,
not instructions to change genuine measured values. Registration freshness is
checked with `make claims-check`, `make availability-check` and
`make availability-statement-check`.
