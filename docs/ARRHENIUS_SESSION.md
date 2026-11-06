# Register completed Arrhenius experiments

[Documentation](README.md) · [Evidence registration](EVIDENCE_REGISTRATION.md) · [Paper support](../paper/PAPER_SUPPORT.md)

The authors describe the paper and supplement as a completed measured study.
This session locates the corresponding original outputs and makes them available
for independent reconstruction. Keep the producing source revision, checkpoint
bytes and original decision records attached to each run.

## 1. Locate original outputs

Use an existing full-profile export root. Keep the evidence directory outside
both that root and the source checkout. The discovery tool reads the original
files and writes a new report and proposed registration plan.

```bash
umask 077
mkdir -p /abs/cm-evidence
make evidence-discover \
  EVIDENCE_ROOT=/abs/original-exports \
  DISCOVERY_OUTPUT=/abs/cm-evidence/discovery.json \
  EVIDENCE_PLAN=/abs/cm-evidence/plan.json \
  DEPOSITS_DIR=/abs/cm-evidence/deposits
```

For a storage-intensive scan on Arrhenius, configure the existing discovery
wrapper with absolute paths in the submission environment:

```bash
export CM_SOURCE_ROOT=/abs/CR-9
export CM_ENV_FILE=/abs/private/arrhenius.env
export CM_EVIDENCE_ROOT=/abs/original-exports
export CM_EVIDENCE_DIR=/abs/cm-evidence
mkdir -p "$CM_EVIDENCE_DIR/logs"
sbatch --account=<allocation> --partition=<partition> \
  --chdir="$CM_EVIDENCE_DIR" \
  --output="$CM_EVIDENCE_DIR/logs/discovery-%j.out" \
  --error="$CM_EVIDENCE_DIR/logs/discovery-%j.err" --export=ALL \
  "$CM_SOURCE_ROOT/hpc/discover_evidence_arrhenius.sh"
```

The trusted environment profile selects the Python interpreter and site modules.
The wrapper preserves discovery exit status and creates private, nonoverwriting
reports. A candidate is eligible only after its complete stage chain and source
identities pass the inventory. Pilot and synthetic runs remain separate. Roman
requires the parent campaign with all ten official split directories.

## 2. Resolve each discovered candidate

| Discovery status | Action |
|---|---|
| `UNIQUE_CANDIDATE` | Match its checkpoint, source, configuration, seeds and policy to the original experiment record, then preflight it |
| `AMBIGUOUS` | Identify the original result-producing experiment from its provenance; do not select a run by its score |
| `MISSING` | Locate the corresponding original run or export; retain the missing status until its files are available |
| `UNRESOLVED_SCAN` | Inspect unreadable paths or explicit scan bounds, choose a suitable root or limit, and create a new report |
| `REJECTED` | Inspect the stated missing or inconsistent artifact; preserve the original files and report the issue |

A new reproduction is a distinct experiment. Use the plan's
`relationship_to_manuscript` field to identify an exact original run, a refactor
with established prediction equivalence, or a later revision. The run's source
snapshot and fingerprint remain those of its producer.

## 3. Collect the complete evidence package

| Required material | Where it belongs |
|---|---|
| Seven full-profile archives, including ten Roman splits | Dataset entries in the private plan, then `runs/index.json` |
| Checkpoints, complete selection ledgers, role membership, confirmation/admission records, predictions and measured work | Original sealed run archives |
| Ten auxiliary sets: baselines, factorials, negative controls, input gate, nominal timing, sensor settings, cross-backbone, ensemble grid, inserted-site and budget-matched measurements | Original raw files imported with the existing record exporters or traceable mappings |
| E3–E6 comparison campaign, paid-operation logs and common budgets | `budget_campaign` in the plan and `BUDGET_CAMPAIGN` in the reviewer commands |
| Joint nine-test S14 confirmation family and exact policy decisions before admission | The confirmation campaign consumed by `make confirmation-audit` |
| Original GH200 timing repetitions and scheduler accounting | Raw sources behind timing and resource records; retain precision, batch, warmup, synchronization and denominators |
| Final main and supplementary PDFs | Local copies matching `paper/sources.json` for byte verification |

The [complete display inventory](../paper/PAPER_SUPPORT.md) also identifies
analyses without a dedicated figure builder. Preserve their original generating
scripts and source arrays. Protocol tables and analytical calculations remain
linked to their actual definitions and source inputs.

## 4. Preflight, pack and register

Complete the private plan with actual paths, the manuscript relationship,
license and raw-record mappings. Then run:

```bash
make evidence-preflight EVIDENCE_PLAN=/abs/cm-evidence/plan.json FULL_AUDIT=always
make evidence-pack EVIDENCE_PLAN=/abs/cm-evidence/plan.json
make evidence-status EVIDENCE_PLAN=/abs/cm-evidence/plan.json
```

Follow [the registration guide](EVIDENCE_REGISTRATION.md#4-publish-and-register)
for deposit, publication and registration. Its explicit publication switch and
public-byte verification remain in force. Keep credentials in the environment
and local paths in the private plan. Registration refreshes the public indexes;
it does not infer that every manuscript conclusion is established.

The joint S14 campaign is checked separately from per-run local confirmation:

```bash
make confirmation-audit CONFIRMATION_CAMPAIGN=/abs/cm-evidence/confirmation
```

The heterogeneous original raw-test outputs retain their analysis provenance;
the checker replays their joint Holm family and frozen decision rule. It does
not replace block, permutation or crossed bootstrap tests with a different test.

## 5. Rebuild and inspect the public evidence

After the archive, auxiliary records and manuscript bindings are registered:

```bash
make paper
make compare-runs REPORT_ROOT=/abs/cm-evidence/review
make scientific-audit \
  BUDGET_CAMPAIGN=/abs/cm-evidence/budget-campaign \
  CONFIRMATION_CAMPAIGN=/abs/cm-evidence/confirmation \
  MANUSCRIPT_PDF=/abs/paper.pdf \
  SUPPLEMENTARY_PDF=/abs/supplement.pdf \
  REPORT_ROOT=/abs/cm-evidence/review
make review REQUIRE_EVIDENCE=1 \
  BUDGET_CAMPAIGN=/abs/cm-evidence/budget-campaign \
  CONFIRMATION_CAMPAIGN=/abs/cm-evidence/confirmation \
  MANUSCRIPT_PDF=/abs/paper.pdf \
  SUPPLEMENTARY_PDF=/abs/supplement.pdf \
  REPORT_ROOT=/abs/cm-evidence/review
```

Review the exact current inputs and generated comparisons. Retain unfavorable
or contradictory outcomes and resolve them against the original records. In
particular, the source review identifies Peptides confirmation, the complete
Roman/Pascal/COCO selection ledgers, model identities and the executed E6 variant
as requiring explicit evidence. Use `make readiness` with the same campaign,
PDF and review-report paths for the final completeness check.
