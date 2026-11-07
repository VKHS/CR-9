# Complete paper support inventory

[Reported results](RESULTS.md) · [Manuscript review](MANUSCRIPT_REVIEW.md) · [Numerical claims ledger](CLAIMS.md) · [Artifact availability](../AVAILABILITY.md)

This source-bound inventory covers **90 numbered displays**: two main tables, six main figures, 62 supplementary tables and 20 supplementary figures. The authors identify the supplied PDFs as the completed measured study. The inventory names the available implementation and reconstruction path for every display, including those outside the 34 curated numerical registry groups.

**Every numbered display has a generator** in `paper/build/displays/`: measured displays are rebuilt from the outputs published with the repository's release (`release/paper_artifacts.json`), protocol and analytical displays from the repository's configuration, registered constants and analytical code. `make paper` runs them all, writes each display's source data and compares every computed value with the printed one ([reconstruction guide](../docs/RECONSTRUCTION.md), [statement compliance](../STATEMENT_COMPLIANCE.md)).

## What the checking paths cover

A command in the tables below is a relevant check or builder. It does not imply that the command recreates the entire printed display. Run builders reconstruct their documented columns; record builders emit set-level summaries; figure builders provide related summaries and selected panel data. The original analysis and artwork remain required where no full-display builder is declared. Mathematical and protocol checks do not establish the empirical origin of an input.

| Checking or reconstruction scope | Displays |
|---|---|
| Run-derived columns | 10 |
| Record-set tables | 13 |
| Protocol checks | 16 |
| Theory/arithmetic checks | 18 |
| Model inventory checks | 2 |
| Related plots / partial panels | 7 |
| Method documentation | 1 |
| Original analysis required | 23 |
| Full-display generator | 0 |

[display_manifest.json](display_manifest.json) retains each display's complete implementation list, related numerical groups, record sets, required evidence and qualifications. Its reviewed PDF and registry identities are checked before either report is rendered.

## Main paper

| Display | PDF pages | Content | Scope | Implementation / analysis | Command |
|---|---|---|---|---|---|
| **Table 1** — Policy and paired test outcomes | 10 | Reported measurements | Run-derived columns | [tables.py](../paper/build/tables.py); [check_decision_consistency.py](../scripts/check_decision_consistency.py) | `make paper`; `make compare-runs` |
| **Table 2** — Predictive gains and serial-work accounting | 11 | Measurements and/or derived quantities | Run-derived columns | [tables.py](../paper/build/tables.py) | `make paper`; `make compare-runs` |
| **Figure 1** — Frozen predictor and policy lifecycle | 2 | Method illustration | Method documentation | [draw_diagrams.py](../scripts/draw_diagrams.py) | `make diagrams-check`; `make theory` |
| **Figure 2** — Measured evidence and analytical statements | 9 | Measurements and/or derived quantities | Related plots / partial panels | [control.py](../cm_theory/control.py); [selection.py](../cm_theory/selection.py); [stability.py](../cm_theory/stability.py); [policies.py](../cross_morse/policies.py); [figures.py](../paper/build/figures.py) | `make paper` |
| **Figure 3** — Feedback and replanning identification | 10 | Measurements and/or derived quantities | Related plots / partial panels | [statistics.py](../cross_morse/statistics.py); [check_si_identities.py](../scripts/check_si_identities.py); [supplement.py](../paper/build/supplement.py); [build_supplement.py](../scripts/build_supplement.py); [SUPPLEMENT_BUILD.md](../docs/SUPPLEMENT_BUILD.md) | `make scientific-audit`; `make reconstruct` |
| **Figure 4** — Ablations, ensemble response and resource accounting | 11 | Measurements and/or derived quantities | Related plots / partial panels | [accounting.py](../cm_campaigns/accounting.py); [economics.py](../cm_theory/economics.py); [factorial.py](../cm_theory/factorial.py); [specs.json](../records/specs.json); [figures.py](../paper/build/figures.py); [supplement.py](../paper/build/supplement.py); [build_supplement.py](../scripts/build_supplement.py); [SUPPLEMENT_BUILD.md](../docs/SUPPLEMENT_BUILD.md) | `make paper`; `make reconstruct` |
| **Figure 5** — Deployment effect and cost | 12 | Measurements and/or derived quantities | Related plots / partial panels | [timing.py](../cm_campaigns/timing.py); [upstream.py](../cm_experiments/upstream.py); [economics.py](../cm_theory/economics.py); [check_paper_constants.py](../scripts/check_paper_constants.py); [figures.py](../paper/build/figures.py) | `make paper` |
| **Figure 6** — Budget-matched comparison | 13 | Measurements and/or derived quantities | Related plots / partial panels | [arm_campaign.py](../cm_campaigns/arm_campaign.py); [execution.py](../cm_experiments/execution.py); [economics.py](../cm_theory/economics.py); [inference.py](../cm_theory/inference.py); [supplement.py](../paper/build/supplement.py); [build_supplement.py](../scripts/build_supplement.py); [SUPPLEMENT_BUILD.md](../docs/SUPPLEMENT_BUILD.md) | `make scientific-audit`; `make reconstruct` |

## Supplementary tables

| Display | PDF pages | Content | Scope | Implementation / analysis | Command |
|---|---|---|---|---|---|
| **Table S1** — Dataset statistical units and admission criteria | 1, 2 | Protocol specification | Protocol checks | [statistics.py](../cross_morse/statistics.py); [split_contract.py](../cross_morse/split_contract.py) | `make registry-check` |
| **Table S2** — Data roles, folds, seeds and decisions | 2 | Protocol specification | Protocol checks | [split_contract.py](../cross_morse/split_contract.py); [INDEX.json](../manifests/registered/INDEX.json) | `make registry-check` |
| **Table S3** — Target-regime qualification contract | 2 | Protocol specification | Protocol checks | [pipeline.py](../cross_morse/pipeline.py) | `make registry-check` |
| **Table S4** — Target-qualification observations | 2 | Reported measurements | Run-derived columns | [pipeline.py](../cross_morse/pipeline.py) | `make paper`; `make compare-runs` |
| **Table S5** — Frozen-model inventory and training cost | 3 | Measurements and/or derived quantities | Model inventory checks | [core.py](../cross_morse/core.py); [model_inventory.py](../scripts/model_inventory.py) | `make model-inventory`; `make compare-runs` |
| **Table S5b** — Native and inserted transport sites | 3, 4 | Protocol specification | Protocol checks | [core.py](../cross_morse/core.py) | `make registry-check` |
| **Table S5c** — Nominal identity and site accounting | 4 | Measurements and/or derived quantities | Original analysis required | [core.py](../cross_morse/core.py); [test_theory_runtime_parity.py](../tests/test_theory_runtime_parity.py) | `make scientific-audit` |
| **Table S5d** — Executable model inventory and reconciliation | 3 | Measurements and/or derived quantities | Model inventory checks | [model_inventory.py](../scripts/model_inventory.py) | `make model-inventory`; `make compare-runs` |
| **Table S6** — Ten-action registry and per-step cost | 4 | Mathematical or analytical calculation | Theory/arithmetic checks | [core.py](../cross_morse/core.py); [operators.py](../cm_theory/operators.py) | `make theory`; `make si-check` |
| **Table S6b** — Action range, reach and product identities | 5 | Mathematical or analytical calculation | Theory/arithmetic checks | [operators.py](../cm_theory/operators.py) | `make theory`; `make si-check` |
| **Table S6c** — Sensing-work identity and executed settings | 5 | Measurements and/or derived quantities | Record-set tables | [core.py](../cross_morse/core.py); [check_paper_constants.py](../scripts/check_paper_constants.py) | `make records-check`; `make records-tables` |
| **Table S7** — Per-cell support audit | 5 | Reported measurements | Original analysis required | [scientific_evidence_audit.py](../scripts/scientific_evidence_audit.py) | `make scientific-audit` |
| **Table S8** — Online features and observer diagnostics | 5, 6 | Protocol specification | Protocol checks | [core.py](../cross_morse/core.py); [learning.py](../cross_morse/learning.py) | `make registry-check`; `make scientific-audit` |
| **Table S9** — Observer, consequence-model and search configuration | 6 | Protocol specification | Protocol checks | [learning.py](../cross_morse/learning.py); [policies.py](../cross_morse/policies.py) | `make registry-check` |
| **Table S10** — Executable policy definitions | 6 | Protocol specification | Protocol checks | [policies.py](../cross_morse/policies.py) | `make registry-check` |
| **Table S11** — Frozen confirmation map | 8 | Protocol specification | Protocol checks | [statistics.py](../cross_morse/statistics.py) | `make registry-check` |
| **Table S12** — Representative candidate ledger | 8, 9 | Reported measurements | Run-derived columns | [statistics.py](../cross_morse/statistics.py) | `make paper`; `make compare-runs` |
| **Table S12b** — Nested open-loop family comparison | 9 | Reported measurements | Original analysis required | [statistics.py](../cross_morse/statistics.py) | `make scientific-audit` |
| **Table S12c** — Representative and complete-ledger selector audit | 9 | Measurements and/or derived quantities | Run-derived columns | [selection.py](../paper/build/selection.py) | `make paper`; `make compare-runs` |
| **Table S13** — Independent-graph admission and clipping | 10 | Measurements and/or derived quantities | Run-derived columns | [statistics.py](../cross_morse/statistics.py) | `make paper`; `make compare-runs` |
| **Table S14** — Nine-test mechanism confirmation family | 10 | Reported measurements | Run-derived columns | [statistics.py](../cross_morse/statistics.py); [confirmation.py](../cm_campaigns/confirmation.py); [audit_s14_confirmation.py](../scripts/audit_s14_confirmation.py); [supplement.py](../paper/build/supplement.py); [build_supplement.py](../scripts/build_supplement.py); [SUPPLEMENT_BUILD.md](../docs/SUPPLEMENT_BUILD.md) | `make paper`; `make compare-runs`; `make confirmation-audit`; `make reconstruct` |
| **Table S14b** — Confirmation-size sensitivity | 10, 11 | Mathematical or analytical calculation | Theory/arithmetic checks | [check_si_identities.py](../scripts/check_si_identities.py) | `make theory`; `make si-check` |
| **Table S15** — Method scores and resource ranges | 11, 12 | Measurements and/or derived quantities | Run-derived columns | [upstream.py](../cm_experiments/upstream.py) | `make paper`; `make compare-runs` |
| **Table S15b** — Per-rerun resource accounting | 12 | Reported measurements | Record-set tables | [accounting.py](../cm_campaigns/accounting.py); [timing.py](../cm_campaigns/timing.py) | `make records-check`; `make records-tables` |
| **Table S16** — arxiv component factorial | 12 | Measurements and/or derived quantities | Record-set tables | [factorial.py](../cm_theory/factorial.py); [supplement.py](../paper/build/supplement.py); [build_supplement.py](../scripts/build_supplement.py); [SUPPLEMENT_BUILD.md](../docs/SUPPLEMENT_BUILD.md) | `make records-check`; `make records-tables`; `make reconstruct` |
| **Table S16b** — Pascal component factorial | 12, 13 | Measurements and/or derived quantities | Record-set tables | [factorial.py](../cm_theory/factorial.py); [supplement.py](../paper/build/supplement.py); [build_supplement.py](../scripts/build_supplement.py); [SUPPLEMENT_BUILD.md](../docs/SUPPLEMENT_BUILD.md) | `make records-check`; `make records-tables`; `make reconstruct` |
| **Table S17** — Module ablations and observed signatures | 13 | Measurements and/or derived quantities | Original analysis required | [core.py](../cross_morse/core.py); [learning.py](../cross_morse/learning.py); [policies.py](../cross_morse/policies.py) | `make scientific-audit` |
| **Table S18** — Registered sensitivity grids and observations | 13 | Measurements and/or derived quantities | Original analysis required | [registered_constants.json](../paper/registered_constants.json) | `make scientific-audit` |
| **Table S19** — Negative controls | 13 | Measurements and/or derived quantities | Record-set tables | [specs.json](../records/specs.json) | `make records-check`; `make records-tables` |
| **Table S20** — Selection radii and theorem instantiation | 15 | Measurements and/or derived quantities | Original analysis required | [selection.py](../cm_theory/selection.py) | `make scientific-audit` |
| **Table S21** — Action gaps and switching | 15, 16 | Reported measurements | Original analysis required | [policies.py](../cross_morse/policies.py) | `make scientific-audit` |
| **Table S22** — Trajectory tube audit | 16 | Measurements and/or derived quantities | Original analysis required | [stability.py](../cm_theory/stability.py) | `make scientific-audit` |
| **Table S23** — Expected and assurance cost | 16 | Measurements and/or derived quantities | Run-derived columns | [economics.py](../cm_theory/economics.py) | `make paper`; `make compare-runs` |
| **Table S23b** — Nominal execution work, timing and amortization | 16 | Measurements and/or derived quantities | Record-set tables | [timing.py](../cm_campaigns/timing.py); [check_paper_constants.py](../scripts/check_paper_constants.py) | `make records-check`; `make records-tables` |
| **Table S24** — Claim-to-artifact registry | 16, 17 | Protocol specification | Protocol checks | [arxiv_full.json](../paper/bindings/arxiv_full.json); [specs.json](../records/specs.json) | `make registry-check` |
| **Table S25** — Cross-backbone substitutions | 17 | Measurements and/or derived quantities | Record-set tables | [core.py](../cross_morse/core.py) | `make records-check`; `make records-tables` |
| **Table S26** — Integer ensemble grid and equal-cost interpolation | 17, 18 | Measurements and/or derived quantities | Record-set tables | [specs.json](../records/specs.json) | `make records-check`; `make records-tables` |
| **Table S26b** — Fitted ensemble law and break-even | 18 | Measurements and/or derived quantities | Record-set tables | [economics.py](../cm_theory/economics.py); [supplement.py](../paper/build/supplement.py); [build_supplement.py](../scripts/build_supplement.py); [SUPPLEMENT_BUILD.md](../docs/SUPPLEMENT_BUILD.md) | `make records-check`; `make records-tables`; `make reconstruct` |
| **Table S27** — Buffer sensitivity on dependent graphs | 18 | Measurements and/or derived quantities | Original analysis required | [statistics.py](../cross_morse/statistics.py) | `make scientific-audit` |
| **Table S28** — Distribution-free one-step regret bound | 18, 19 | Mathematical or analytical calculation | Theory/arithmetic checks | [control.py](../cm_theory/control.py) | `make theory`; `make si-check` |
| **Table S28b** — Variance-aware one-step radius | 19 | Mathematical or analytical calculation | Theory/arithmetic checks | [control.py](../cm_theory/control.py) | `make theory`; `make si-check` |
| **Table S29** — Planning horizon and beam sweep | 19 | Measurements and/or derived quantities | Original analysis required | [policies.py](../cross_morse/policies.py) | `make scientific-audit` |
| **Table S30** — Oracle dispersion and retained-share grid | 19 | Measurements and/or derived quantities | Original analysis required | [economics.py](../cm_theory/economics.py) | `make scientific-audit` |
| **Table S31** — Reproduction map | 29, 30 | Protocol specification | Protocol checks | [build_paper.py](../scripts/build_paper.py); [records.py](../scripts/records.py) | `make registry-check` |
| **Table S32** — Budget-matched comparator definitions | 21 | Protocol specification | Protocol checks | [execution.py](../cm_experiments/execution.py) | `make registry-check` |
| **Table S33** — Retained-share model for comparison arms | 21 | Mathematical or analytical calculation | Theory/arithmetic checks | [economics.py](../cm_theory/economics.py) | `make theory`; `make si-check` |
| **Table S34** — Selective-sensing and branch-cost model | 21 | Mathematical or analytical calculation | Theory/arithmetic checks | [budget.py](../cm_experiments/budget.py) | `make theory`; `make si-check` |
| **Table S35** — Reporting conventions and record linkage | 21, 22 | Protocol specification | Protocol checks | [reported_results.json](../paper/reported_results.json) | `make registry-check` |
| **Table S36** — E1–E20 experiment and falsification definitions | 22, 23 | Protocol specification | Protocol checks | [PAPER_SUPPORT.md](../docs/PAPER_SUPPORT.md) | `make registry-check` |
| **Table S37** — Fallback, false-safe and harm-miss rates | 23 | Reported measurements | Record-set tables | [statistics.py](../cross_morse/statistics.py) | `make records-check`; `make records-tables` |
| **Table S38** — Results-record schema | 23, 24 | Protocol specification | Protocol checks | [specs.json](../records/specs.json); [provenance.py](../cross_morse/provenance.py) | `make registry-check` |
| **Table S39** — Separated data-role contract | 24 | Protocol specification | Protocol checks | [split_contract.py](../cross_morse/split_contract.py); [role_evidence.py](../cross_morse/role_evidence.py) | `make registry-check` |
| **Table S40** — Comparison-arm information restrictions | 24 | Protocol specification | Protocol checks | [execution.py](../cm_experiments/execution.py) | `make registry-check` |
| **Table S41** — Qualification cost components | 24, 25 | Measurements and/or derived quantities | Run-derived columns | [accounting.py](../cm_campaigns/accounting.py) | `make paper`; `make compare-runs` |
| **Table S42** — Budget-matched test power | 25 | Mathematical or analytical calculation | Theory/arithmetic checks | [inference.py](../cm_theory/inference.py) | `make theory`; `make si-check` |
| **Table S43** — Experiment outcomes, estimators and thresholds | 25, 26 | Measurements and/or derived quantities | Original analysis required | [reported_results.json](../paper/reported_results.json) | `make scientific-audit` |
| **Table S44** — Per-input tail-cost model | 26 | Mathematical or analytical calculation | Theory/arithmetic checks | [economics.py](../cm_theory/economics.py) | `make theory`; `make si-check` |
| **Table S45** — Inserted-site attribution | 26 | Measurements and/or derived quantities | Record-set tables | [core.py](../cross_morse/core.py) | `make records-check`; `make records-tables` |
| **Table S46** — Budget-matched frontier | 26 | Reported measurements | Record-set tables | [execution.py](../cm_experiments/execution.py); [arm_campaign.py](../cm_campaigns/arm_campaign.py) | `make records-check`; `make records-tables` |
| **Table S46b** — D_compare arm outcomes | 27 | Reported measurements | Record-set tables | [execution.py](../cm_experiments/execution.py); [arm_campaign.py](../cm_campaigns/arm_campaign.py); [supplement.py](../paper/build/supplement.py); [build_supplement.py](../scripts/build_supplement.py); [SUPPLEMENT_BUILD.md](../docs/SUPPLEMENT_BUILD.md) | `make records-check`; `make records-tables`; `make reconstruct` |
| **Table S47** — Decision boundaries and power | 27 | Mathematical or analytical calculation | Theory/arithmetic checks | [inference.py](../cm_theory/inference.py); [economics.py](../cm_theory/economics.py) | `make theory`; `make si-check` |
| **Table S48** — Policy outcomes and admission criteria | 31 | Measurements and/or derived quantities | Original analysis required | [tables.py](../paper/build/tables.py) | `make scientific-audit` |

## Supplementary figures

| Display | PDF pages | Content | Scope | Implementation / analysis | Command |
|---|---|---|---|---|---|
| **Figure S1** — Integrated decision surface | 31 | Measurements and/or derived quantities | Original analysis required | [economics.py](../cm_theory/economics.py); [tables.py](../paper/build/tables.py) | `make scientific-audit` |
| **Figure S2** — Runtime and memory scaling | 31 | Measurements and/or derived quantities | Original analysis required | [accounting.py](../cm_campaigns/accounting.py); [timing.py](../cm_campaigns/timing.py); [check_paper_constants.py](../scripts/check_paper_constants.py) | `make scientific-audit` |
| **Figure S3** — Budget-matched per-dataset frontiers | 31, 32 | Measurements and/or derived quantities | Related plots / partial panels | [arm_campaign.py](../cm_campaigns/arm_campaign.py); [execution.py](../cm_experiments/execution.py); [supplement.py](../paper/build/supplement.py); [build_supplement.py](../scripts/build_supplement.py); [SUPPLEMENT_BUILD.md](../docs/SUPPLEMENT_BUILD.md) | `make scientific-audit`; `make reconstruct` |
| **Figure S4** — Method score and cost maps | 32 | Measurements and/or derived quantities | Original analysis required | [accounting.py](../cm_campaigns/accounting.py); [timing.py](../cm_campaigns/timing.py); [upstream.py](../cm_experiments/upstream.py) | `make scientific-audit` |
| **Figure S5** — Horizon and beam sensitivity | 32 | Measurements and/or derived quantities | Original analysis required | [policies.py](../cross_morse/policies.py) | `make scientific-audit` |
| **Figure S6** — Action-cost and sensing-work model | 32 | Measurements and/or derived quantities | Theory/arithmetic checks | [operators.py](../cm_theory/operators.py); [core.py](../cross_morse/core.py); [check_paper_constants.py](../scripts/check_paper_constants.py) | `make theory`; `make si-check` |
| **Figure S7** — Acquisition gate | 32 | Measurements and/or derived quantities | Theory/arithmetic checks | [budget.py](../cm_experiments/budget.py); [policies.py](../cross_morse/policies.py) | `make theory`; `make si-check` |
| **Figure S8** — Clipping moments and uncertainty units | 32 | Measurements and/or derived quantities | Theory/arithmetic checks | [statistics.py](../cross_morse/statistics.py) | `make theory`; `make si-check` |
| **Figure S9** — Mechanism contrasts and size sensitivity | 33 | Measurements and/or derived quantities | Related plots / partial panels | [statistics.py](../cross_morse/statistics.py); [check_si_identities.py](../scripts/check_si_identities.py); [figures.py](../paper/build/figures.py); [supplement.py](../paper/build/supplement.py); [build_supplement.py](../scripts/build_supplement.py); [SUPPLEMENT_BUILD.md](../docs/SUPPLEMENT_BUILD.md) | `make paper`; `make reconstruct` |
| **Figure S10** — Sensing-cost identities | 33 | Measurements and/or derived quantities | Theory/arithmetic checks | [economics.py](../cm_theory/economics.py); [core.py](../cross_morse/core.py); [check_paper_constants.py](../scripts/check_paper_constants.py) | `make theory`; `make si-check` |
| **Figure S11** — Terminal-site budget ledger | 33 | Mathematical or analytical calculation | Theory/arithmetic checks | [budget.py](../cm_experiments/budget.py); [budget.py](../cm_theory/budget.py) | `make theory`; `make si-check` |
| **Figure S12** — Comparison-test power | 34 | Mathematical or analytical calculation | Theory/arithmetic checks | [inference.py](../cm_theory/inference.py) | `make theory`; `make si-check` |
| **Figure S13** — Application settings and schedule adequacy | 34 | Measurements and/or derived quantities | Original analysis required | [economics.py](../cm_theory/economics.py) | `make scientific-audit` |
| **Figure S14** — Depth diagnostics | 34 | Measurements and/or derived quantities | Original analysis required | [pipeline.py](../cross_morse/pipeline.py); [figures.py](../paper/build/figures.py) | `make paper` |
| **Figure S15** — Spectral fingerprint, action range and headroom | 35 | Measurements and/or derived quantities | Theory/arithmetic checks | [operators.py](../cm_theory/operators.py); [core.py](../cross_morse/core.py); [tables.py](../paper/build/tables.py) | `make theory`; `make si-check`; `make scientific-audit` |
| **Figure S16** — Reliability, harm and fallback | 35 | Measurements and/or derived quantities | Original analysis required | [core.py](../cross_morse/core.py); [learning.py](../cross_morse/learning.py); [statistics.py](../cross_morse/statistics.py); [specs.json](../records/specs.json) | `make scientific-audit` |
| **Figure S17** — Analytical and statistical audits | 36 | Measurements and/or derived quantities | Original analysis required | [control.py](../cm_theory/control.py); [factorial.py](../cm_theory/factorial.py); [selection.py](../cm_theory/selection.py); [statistics.py](../cross_morse/statistics.py) | `make scientific-audit` |
| **Figure S18** — Inserted-site attribution and separated protocols | 37 | Measurements and/or derived quantities | Original analysis required | [core.py](../cross_morse/core.py); [statistics.py](../cross_morse/statistics.py); [check_si_identities.py](../scripts/check_si_identities.py) | `make scientific-audit` |
| **Figure S19** — Guarantee scope and supported models | 37 | Measurements and/or derived quantities | Original analysis required | [core.py](../cross_morse/core.py); [split_contract.py](../cross_morse/split_contract.py); [statistics.py](../cross_morse/statistics.py); [tables.py](../paper/build/tables.py) | `make diagrams-check`; `make theory`; `make scientific-audit` |
| **Figure S20** — Break-even, resampling and seed-design power | 38 | Mathematical or analytical calculation | Theory/arithmetic checks | [economics.py](../cm_theory/economics.py); [inference.py](../cm_theory/inference.py) | `make theory`; `make si-check` |

## Decisions and analysis qualifications

- **Figure 2 (main):** F3 draws related selection geometry using the full eligible candidate ledger for panel 2c; the complete printed figure requires its original panel data and design.
- **Figure 3 (main):** The supplementary builder validates the complete original joint S14 campaign and plots the state/RH effects and original intervals for quantities in panels 3a–b. It preserves original heterogeneous test P values. Action entropy and action-origin panels 3c–d require their original traces; the forest is not the complete printed figure.
- **Figure 4 (main):** F1 reconstructs effect versus overhead for panel 4d; other factorial and ensemble panels require their original analyses. The supplementary builder also produces panel-4a-related arxiv eight-cell factorial means and original intervals from the complete verified factorial record set. Comparator ensemble fit diagnostics do not replace panel 4b feedback-by-E effects/work; other existing partial-panel limits remain.
- **Figure 5 (main):** F2 is a partial internal-method map using absolute official scores and serial costs. Panel 5a additionally uses external baselines and gain/cost normalization; the complete panel and other cost-model panels require their original data, assumptions and generator.
- **Figure 6 (main):** A strictly replayed 24-arm budget campaign can produce related panel-6a D_compare point frontiers and effect/work diagnostics. This is a new conditional experiment, not historical D_test reconstruction; original teacher intervals, configuration ranges, decision boundaries, power and tail panels remain required.
- **Table 1 (main):** Peptides radius measurements are an evaluated arm; the printed Holm value 0.070 does not pass 0.05, so the registered lifecycle returns qC on those inputs.
- **Figure S3 (SI):** The supplementary builder can plot related executed-work D_compare point frontiers from a strict 24-arm campaign. The original D_test teacher intervals, arm configuration ranges and ensemble sensitivity band are absent, so this is partial quantitative support rather than the complete printed figure.
- **Figure S9 (SI):** F6 reconstructs available local mechanism contrasts; joint-family and sensitivity panels require the complete S14 sources. The supplementary builder adds a validated original joint-S14 forest. It does not reconstruct the original-versus-rescaled J_old/J_new sensitivity intervals or replace the complete Supplementary Figure S9 design.
- **Figure S14 (SI):** F4 plots recorded nominal validation loss against depth at entry. It is a related diagnostic, not the selected-policy paired loss effect in panel S14b or an exact builder for any printed S14 panel.
- **Figure S15 (SI):** Panels a-c are exact registered action calculations. Panel d uses the measured oracle upper confidence bound and deployed policy effect; its headroom denominator is a confidence bound, not the oracle point estimate.
- **Figure S19 (SI):** Panel b reports dataset-specific admission outcomes, including the distinction between descriptive single-graph criteria and independent-graph admission. The figure is not solely a method illustration.
- **Table S8 (SI):** S8 defines 25/26 sensor coordinates and separate observer diagnostics; it does not tabulate measured innovation, NIS or calibration results.
- **Table S12 (SI):** Representative printed rows do not reconstruct the full-ledger Roman/Pascal/COCO choices; retain the eligible family, nested comparator and every feasibility mask.
- **Table S12c (SI):** Representative printed rows do not reconstruct the full-ledger Roman/Pascal/COCO choices; retain the eligible family, nested comparator and every feasibility mask. Representative-row rule readings are derived from displayed bounds; the executed and separated-protocol choice columns report historical complete-ledger selections.
- **Table S14 (SI):** The run builder emits local-family contrasts. The separate joint S14 campaign must preserve all nine original unrounded tests and decisions fixed before admission. The supplementary builder emits the original joint nine-test family: signed effects, original intervals/raw P, recomputed pooled Holm P, statistical units and strict point-margin decisions. Tests against the practical margin and equivalence tests remain separate. Native official-run linkage is established by scientific-audit, not by plotting alone.
- **Table S14b (SI):** Sample-size sensitivity rescales uncertainty under the declared variance model; it is separate from original confirmation observations.
- **Table S15 (SI):** The run builder covers internal methods; the fourteen external dataset/method pairs require the baselines record set.
- **Table S16 (SI):** The supplementary builder requires all eight arxiv cells and all six measured quantities, preserves original cell intervals and time units, and derives both effect-coded and baseline-anchored point contrasts. It does not infer interaction intervals from marginal cell intervals.
- **Table S16b (SI):** The supplementary builder requires all eight Pascal cells and all six measured quantities, preserves original cell intervals/time denominator, and derives both point-contrast conventions. Joint-covariance main-effect/interaction intervals require the original resampling covariance and are not manufactured.
- **Table S23b (SI):** Keep observed timing, arithmetic nominal-work approximations, adopted timing anchors and modelled tails separately identified.
- **Table S26b (SI):** The supplementary builder fits the actual seven-point E={1,2,4,8,12,16,20} comparator grid and emits coefficients, points and residual diagnostics. Loss-only records do not supply the printed D_inf band, kappa, break-even cost inputs, feedback-by-E response or additional-checkpoint GPU-hours.
- **Table S44 (SI):** Keep observed timing, arithmetic nominal-work approximations, adopted timing anchors and modelled tails separately identified.
- **Table S46 (SI):** Record the executed E6 variant and paid-operation ledger; the S46b retained-share range is not a confidence interval.
- **Table S46b (SI):** Record the executed E6 variant and paid-operation ledger; the S46b retained-share range is not a confidence interval. The supplementary builder requires strict replay of all 24 D_compare arm contrasts and emits paired effects/intervals, within-dataset Holm P and executed work/ceiling. The campaign remains a new conditional experiment with historical_table_reproduction=false; retained-share configuration ranges are not reconstructed.
- **Table S48 (SI):** Peptides radius measurements are an evaluated arm; the printed Holm value 0.070 does not pass 0.05, so the registered lifecycle returns qC on those inputs.

The [E1–E20 support contract](../docs/PAPER_SUPPORT.md) supplies the remaining experiment-level requirements, including source identity, pairing, supported actions, separated roles, exact evaluator, timing denominator and the distinction between the formal and executed algorithms.

## Reviewed source identities

| Document | Pages | Bytes | SHA-256 |
|---|---|---|---|
| main | 16 | 5,498,468 | `7c4b5145035d5873057a510a84077ee54ea9dbb2af49e2c7cb7306305a9a15e2` |
| supplementary | 38 | 12,252,216 | `120ce46ec1b4e4a14000c90d0f7ef64bad4c6b3d7a3800739b8153703675552e` |

The PDFs are identified as external source documents. Their bytes are verified when matching copies are supplied. The contextual review is bound in [transcription_check.json](transcription_check.json).

Regenerate with `make paper-report`; check freshness with `make paper-report-check`. Changing a PDF or numerical registry requires explicit review and update of the inventory before regeneration.
