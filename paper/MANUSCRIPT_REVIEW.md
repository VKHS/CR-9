# Measured-study manuscript review

Reviewed on 6 November 2026 against the author's supplied 16-page main paper
and 38-page supplementary information. The author confirms that the experiments
were implemented and measured. The empirical numerical entries are therefore
handled as **reported measurements**, with their original units, signs and
uncertainty intervals preserved.

This was an **AI-assisted contextual review requested by the author**. It
compared manuscript statements, table entries, definitions and source locations
with the repository's reference registry. It did not inspect the original
benchmark observations, certify experiment execution or imply that an author
personally reviewed every registry cell during this update.

The exact PDF identities are recorded in [sources.json](sources.json).
[transcription_check.json](transcription_check.json) binds the named reviewer's
contextual notes for all 34 existing result groups to those PDF bytes, both
numerical registry files and the table-page map. The entire text of both PDFs
was read; important results tables and figures were also inspected as rendered
pages. No numerical comparison target was changed by source registration.

## Main results checked

The following values reproduce main Table 1. Loss effects are paired against
the listed dataset's nested open-loop comparator. A negative effect is a loss
reduction. Metric changes are percentage points except the explicitly stated
native AP entry.

| Dataset | Reported evaluated route | Official-metric change | Paired loss effect, nats | 95% test interval, nats |
|---|---|---|---|---|
| ogbn-arxiv | radius, E=4 | +0.07 accuracy pp | -0.00450 CE | [-0.0075, -0.0015] |
| ogbn-proteins | radius, E=4 | +0.11 ROC-AUC pp | -0.00380 BCE | [-0.0068, -0.0008] |
| ogbn-products | comparator returned, E=2 | 0, exact comparator identity | 0 | not applicable |
| Roman-empire | MPC-RH, E=4 | +0.26 accuracy pp | -0.00572 CE | [-0.0089, -0.0026] |
| PascalVOC-SP | MPC-RH, E=8 | +0.60 macro-F1 pp | -0.00842 CE | [-0.0128, -0.0041] |
| COCO-SP | MPC-RH, E=8 | +0.65 macro-F1 pp | -0.00905 CE | [-0.0127, -0.0054] |
| Peptides-func | radius evaluated arm, E=4 | +0.0058 native AP (=0.58 AP pp) | -0.00303 BCE | [-0.0047, -0.0013] |

Main Table 2 reports expected serial costs of 4.62, 4.68, 2.1, 5.22, 10.30,
10.60 and 4.73 nominal executions in the same dataset order. The corresponding
comparator costs are 4, 4, 2.1, 4, 8, 8 and 4. Printed expected overheads are
15%, 17%, 0%, 30%, 29%, 33% and 18%. These are branch-weighted cost summaries;
their underlying ledger values and printed rounding govern derived ratios.

## Source distinctions that remain meaningful

| Source quantity | Correct interpretation | Reproduction requirement |
|---|---|---|
| Test scores, paired effects, timing observations and executed action counts | Author-reported measurements | Original observations, units, seeds, role membership, configurations and checkpoint identities |
| Expected branch cost, factorial contrasts, admission bounds and exchange rates | Calculations from observed or registered inputs | Recompute the stated formula, including units and rounding |
| Tail-cost quantile C_0.99, ensemble-spend scenarios and prospective power | Model-based analyses explicitly identified in the paper | Preserve model assumptions and distinguish them from observed cost maxima and empirical outcomes |
| Registered thresholds, action constants and data-role definitions | Protocol specifications | Check executable configurations and immutable run manifests |
| Exact released-model parameter counts | Implementation-derived inventory | Instantiate the released architecture and compare the exact count with the paper's rounded precision |

The paper's statistical estimates, expected costs, fitted laws, sensitivity
analyses and prospective power calculations retain their mathematical meanings.
Calling the study completed does not turn an analytical scenario into a recorded
observation. For example, S5d prints 3.1 million arxiv parameters; the exact
3,116,632 count in the registry comes from the executable model inventory.

## Printed decisions requiring their original evidence

**Selection on Roman, Pascal and COCO.** Main page 9, Table 1 and supplementary
Table S12c distinguish two records: applying the registered upper-limit rule
with `epsilon_eq=0.002` to the displayed representative rows selects radius,
whereas the reported complete-ledger execution selected MPC-RH. Reproducing the
historical decision requires the original complete candidate ledger, nesting,
eligible set and deterministic selector trace. A route label or a threshold
changed after evaluation cannot reconstruct that history.

**Peptides confirmation.** Main page 9 and supplementary Table S14 report
Holm-adjusted `P=0.070` at the registered level `0.05`. Under the printed frozen
confirmation map the lifecycle returns the comparator. The radius-arm values
above remain the evaluated-arm results printed in the paper; its admission
arithmetic alone does not establish prior confirmation. The registry therefore
retains `registered_confirmation_pass=false`, `evaluation_status=evaluated_arm`
and `lifecycle_resolution=return_qC`. A different historical deployment requires
its original corrected confirmation record and corresponding policy identity.

**Single-graph admission.** The four single-graph datasets use descriptive
buffered-cluster or repeated-split criteria. The independent-graph admission
theorem is scoped to the multigraph datasets and their untouched admission
graphs. The source review preserves that distinction.

**Rounded test outputs.** Supplementary Table S14 prints some raw and adjusted
P values at different precision. For example, the displayed Pascal raw value
`0.0002` and adjusted value `0.0013` require the unrounded test outputs for an
exact Holm replay. Recalculating from rounded displays is an arithmetic check,
not a replacement for the original nine-contrast record.

## Repository evidence and the availability paragraph

At this review, `runs/index.json` contains no registered official run archives,
and `records/index.json` contains no deposited measurement sets. This describes
the contents available in this checkout; it does not say that the author has
not performed the experiments.

The main paper's page-15 availability paragraph describes seven deposited run
archives and the external experiment records with Zenodo DOIs. That deposit
statement needs the actual archive registrations and DOI receipts to be
verifiable from the repository. Source registration, reported numerical tables
and successful software tests do not supply those missing original files.

The existing [run archive workflow](../docs/RUN_ARCHIVES.md),
[evidence registration workflow](../docs/EVIDENCE_REGISTRATION.md) and
[paper support contract](../docs/PAPER_SUPPORT.md) define the original files
and replay steps. The generated [claim ledger](CLAIMS.md) and
[availability statement](AVAILABILITY_STATEMENT.txt) report the current indexed
coverage. Official result builds continue to require their source observations.

## Verify these sources

```bash
python scripts/check_manuscript_sources.py \
  --manuscript-pdf /absolute/path/to/main.pdf \
  --supplementary-pdf /absolute/path/to/supplementary.pdf
make registry-check
make si-check
```

The first check compares exact PDF bytes; the latter checks inspect registered
constants and the enumerated arithmetic identities. The numeric-token search
saved in the review is an advisory location check. Its occurrence fraction is
not a percentage of correctly measured or correctly transcribed results.
