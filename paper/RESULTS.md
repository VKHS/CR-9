# Results reported in the paper

[Complete display support](PAPER_SUPPORT.md) · [Contextual manuscript review](MANUSCRIPT_REVIEW.md) · [Machine-readable results](RESULTS.csv) · [Artifact availability](../AVAILABILITY.md)

The authors identify the supplied main paper and supplementary information as a completed measured study. The values below preserve their reported numerical values. They are generated from [reported_results.json](reported_results.json), with the original document identities recorded below. No benchmark observations are created by this report.

## Main Table 1: reported evaluated outcomes

Loss is the paired CE or mean multilabel BCE difference against the named dataset's open-loop comparator; negative values favor the evaluated rule. Metric changes and their intervals are percentage points. Peptides +0.58 AP percentage points is +0.0058 in native AP units. Brackets reproduce reported paired 95% intervals.

| Dataset | Reported rule | Loss | Loss difference, nats | 95% interval, nats | Metric gain, pp | 95% interval, pp |
|---|---|---|---|---|---|---|
| ogbn-arxiv | radius, E=4 | CE | -0.00450 | [-0.00750, -0.00150] | +0.07 accuracy | [0.01, 0.13] |
| ogbn-proteins | radius, E=4 | BCE | -0.00380 | [-0.00680, -0.00080] | +0.11 ROC-AUC | [0.03, 0.19] |
| ogbn-products | qC returned, E=2 | CE | +0.00000 | not applicable | +0.00 accuracy | not applicable |
| Roman-empire | MPC-RH, E=4 | CE | -0.00572 | [-0.00890, -0.00260] | +0.26 accuracy | [0.10, 0.42] |
| PascalVOC-SP | MPC-RH, E=8 | CE | -0.00842 | [-0.01280, -0.00410] | +0.60 macro F1 | [0.38, 0.82] |
| COCO-SP | MPC-RH, E=8 | CE | -0.00905 | [-0.01270, -0.00540] | +0.65 macro F1 | [0.42, 0.88] |
| Peptides-func | radius, E=4 (evaluated arm) | BCE | -0.00303 | [-0.00470, -0.00130] | +0.58 AP | [0.09, 1.07] |

**Peptides:** the radius values are the reported evaluated-arm measurements. The printed Holm value 0.070 does not pass the registered 0.05 threshold, so the registered confirmation rule returns q_C; a different final policy requires its own corrected confirmation record. The evaluated-arm values remain visible here and separately named in the CSV. A regenerated run records its own confirmation decision in `confirm/decision.json`.

## Main Table 2: reported work and analytical cost quantities

The unit is one nominal full-depth execution. Expected cost is branch-weighted work; the printed overhead percentages retain the paper's rounding. Assurance cost and the modelled 0.99 quantile have different meanings. The latter is a cost-model calculation, not a measured maximum or an empirical timing percentile.

| Dataset | Comparator cost | Expected cost | Printed overhead | Assurance cost | Modelled C_0.99 |
|---|---|---|---|---|---|
| ogbn-arxiv | 4 | 4.62 | 15% | 8.7 | 4.8 |
| ogbn-proteins | 4 | 4.68 | 17% | 8.8 | 6.9 |
| ogbn-products | 2.1 | 2.1 | 0% | 2.1 | not applicable |
| Roman-empire | 4 | 5.22 | 30% | 9.4 | 5.4 |
| PascalVOC-SP | 8 | 10.3 | 29% | 18.5 | 11 |
| COCO-SP | 8 | 10.6 | 33% | 18.8 | 11.3 |
| Peptides-func (evaluated radius arm) | 4 | 4.73 | 18% | 8.9 | 5 |

## Registered confirmation interpretation

This separate view applies the frozen confirmation interpretation already recorded in the repository. When it returns qC, zero paired effect follows from comparator identity; it is not a replacement measurement. Each published run carries the complete decision trail of its deployment: selection ledger, confirmation and admission records and the policy lock.

| Dataset | Registered final rule | Loss difference, nats | Interpretation |
|---|---|---|---|
| ogbn-arxiv | radius | -0.00450 | Reported policy outcome; a regenerated run records its own decision trail |
| ogbn-proteins | radius | -0.00380 | Reported policy outcome; a regenerated run records its own decision trail |
| ogbn-products | qC returned | +0.00000 | Comparator identity; no new measurement |
| Roman-empire | MPC-RH | -0.00572 | Reported arm; a regenerated run records the full selector ledger |
| PascalVOC-SP | MPC-RH | -0.00842 | Reported arm; a regenerated run records the full selector ledger |
| COCO-SP | MPC-RH | -0.00905 | Reported arm; a regenerated run records the full selector ledger |
| Peptides-func | qC returned | +0.00000 | Comparator identity; no new measurement |

On Roman, Pascal and COCO the complete eligible candidate ledger decides between the reported MPC-RH choices and the radius preference of the representative rows in S12/S12c; a regenerated run records that ledger in `select/candidates.json` and `select/decision.json`. S14 reports one joint nine-test family; local per-run tests have a different scope. See [the source review](MANUSCRIPT_REVIEW.md) and [implementation notes](IMPLEMENTATION_NOTES.md).

## Reproducing these results

`make paper` downloads the measured outputs published with the repository's release, verifies every byte, rebuilds Tables 1–2 and every other numbered display with `paper/build/displays/` and places each value above beside the rebuilt one (`displays/compare/`). `bash run_cross_morse.sh reproduce` regenerates the runs independently on another Slurm GPU cluster ([HPC guide](../docs/HPC.md)). See the [reconstruction guide](../docs/RECONSTRUCTION.md) and [publication guide](../docs/PUBLICATION.md).

## Source identities

| Document | Pages | Bytes | SHA-256 |
|---|---|---|---|
| main | 16 | 5,498,468 | `7c4b5145035d5873057a510a84077ee54ea9dbb2af49e2c7cb7306305a9a15e2` |
| supplementary | 38 | 12,252,216 | `120ce46ec1b4e4a14000c90d0f7ef64bad4c6b3d7a3800739b8153703675552e` |

Regenerate with `make paper-report`; check with `make paper-report-check`. The companion CSV contains both the registered final-rule view and separately named reported-arm effects and costs.
