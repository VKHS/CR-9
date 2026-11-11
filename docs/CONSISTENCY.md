# Scientific consistency

[Statement compliance](../STATEMENT_COMPLIANCE.md) · [Figure coverage](FIGURE_REFERENCE_COVERAGE.md)

Questions on which the printed methods or printed values and the released code (or two printed statements) do not yet agree, each with the original record that decides it and the rule that decides it. Reference values, measurements and their provenance are never changed to force agreement. scripts/consistency_audit.py evaluates the machine-checkable rules on completed runs (make consistency-audit RUNS_ROOT=<runs>) and renders docs/CONSISTENCY.md.

| Item | Question | Status | Decided by |
|---|---|---|---|
| K1 | Studentized maximum of the selection bootstrap (Algorithm S4) | Code aligned; records decide the printed values | select/decision.json, select/candidates.json, select/unit_scores.npz and select/metric_observations.npz of every original run |
| K2 | Peptides-func confirmation: Holm P = 0.070 above 0.05 | Awaiting original records | confirm/decision.json of the Peptides-func run and, where used, the S14 joint confirmation campaign (campaign:confirmation) or a separate corrected confirmation record registered before admission |
| K3 | Selection on the complete candidate ledger of Roman-empire, PascalVOC-SP and COCO-SP | Awaiting original records | select/decision.json and select/candidates.json of the original runs (all ten Roman-empire splits) |
| K4 | Seed hierarchy of Table S2 | Design difference; records decide | fit/<profile>/ and select/ of the original runs (controller fits per profile, any per-seed controller packages) and the E3-E6 campaign records |
| K5 | Settings of the budget-matched comparison (Tables S46b and S47, main Figure 6) | Needs an author declaration | the declared settings of the S47 decoding-cost scenario (decoding cost, ceiling, ensemble spend) and the campaign:budget records |
| K6 | Failure responses of the target-regime gates (Table S3) | Awaiting original records | train/entry_initial.json and select/entry_complete.json of every original run |
| K7 | Table S20 \|Q\|: the feasible family, not the registry | Resolved | nothing further: the definition is fixed by Algorithm S4 |
| K8 | Literal cells of the protocol displays | Needs an author declaration | the author, cell by cell: the manuscript cell is corrected, or the runtime or record schema is changed before the records are exported |
| K9 | Manuscript corrections found while registering the figure contracts | Needs an author declaration | the author's correction of the manuscript |

On the cluster, after the seven lifecycles have run:

```bash
make consistency-audit RUNS_ROOT=<suite>/runs     # K1, K2, K3, K4 and K6 on the original runs
```

Each check reports `consistent`, `needs correction` (the rule names what changes), `pending` (no such run) or `structure only` (smoke and synthetic runs, which exercise the check but decide nothing about printed values).

## K1. Studentized maximum of the selection bootstrap (Algorithm S4)

**Displays:** supplementary:Table S20, supplementary:Table S12, supplementary:Table S12c, main:Table 1. **Status:** Code aligned; records decide the printed values.

- **Printed** (supplementary, p. 8): Algorithm S4 requires the feasible family Q_feas, evaluates every q in Q_feas against the nested comparator in each resample and records the studentized maximum over q; its line 10 reports |Q|, |Q_s|, r_max and the slack of Table S20.
- **Printed** (main, p. 5): Eq. (18) defines Q_feas by dynamic support, the official-metric bound and the resource bounds; Eq. (19) forms Q_s from the simultaneous upper limits.
- **Code before:** nested_selection took the studentized maximum over every executed registered configuration, a superset of Q_feas whose radius is at least as wide.
- **Code now:** nested_selection takes the maximum over Q_feas (select/decision.json: max_t_family = feasible, feasible_family_size, max_t_family_size); records sealed before carry no max_t_family and replay with every executed configuration (cross_morse/review.py, paper/build/displays/run_audits.py::screened_radius, cm_experiments/backbones.py).
- **Decided by:** select/decision.json, select/candidates.json, select/unit_scores.npz and select/metric_observations.npz of every original run
- **Rule:** Replay each original selection with both families. The family that reproduces the recorded max_radius, U and joint_feasible is the family the run used. Q_feas: the printed r_max, |Q_s| and slack follow Algorithm S4. Every executed configuration: either the Algorithm S4 text is corrected to the family the reported runs used, or the selection is repeated with Q_feas from the select stage onward and the affected printed values are replaced.
- **Check:** `selection_family` in `scripts/consistency_audit.py`

## K2. Peptides-func confirmation: Holm P = 0.070 above 0.05

**Displays:** supplementary:Table S14, supplementary:Table S14b, main:Table 1, supplementary:Figure S1, supplementary:Table S1. **Status:** Awaiting original records.

- **Printed** (supplementary, p. 10): The Peptides radius-minus-one-step contrast has P = 0.035 and Holm P = 0.070 > 0.05; the registered frozen confirmation map would return q_C unless the selected radius outcome is supported by a separate corrected confirmation record.
- **Printed** (main, p. 10): Table 1 reports radius, E = 4, as the deployed Peptides-func rule with a boundary admission.
- **Code before:** The frozen confirmation map (cross_morse/statistics.py::confirmation) retains a provisional route only when its contrast exceeds the margin and is Holm-significant against zero; otherwise it downgrades to the eligible predecessor or q_C.
- **Code now:** Unchanged: the map is applied as registered; nothing overrides a failed Holm test.
- **Decided by:** confirm/decision.json of the Peptides-func run and, where used, the S14 joint confirmation campaign (campaign:confirmation) or a separate corrected confirmation record registered before admission
- **Rule:** Radius confirmed with Holm P <= 0.05 in the original confirmation record: Table 1 stands and the printed 0.070 is the analytical sensitivity of Table S14b. The record returns q_C: radius may be reported only with the separate corrected confirmation record the caption names; without it, the Peptides rows of Table 1, Figure S1 and Table S1 report q_C. Radius retained while its Holm P exceeds 0.05: the record contradicts the registered map and the run is not reported.
- **Check:** `peptides_confirmation` in `scripts/consistency_audit.py`

## K3. Selection on the complete candidate ledger of Roman-empire, PascalVOC-SP and COCO-SP

**Displays:** supplementary:Table S12c, supplementary:Table S12, main:Table 1. **Status:** Awaiting original records.

- **Printed** (supplementary, p. 9): On the representative rows the registered rule reads radius (radius lies 0.00110, 0.00085 and 0.00101 above U_min, inside the band), whereas the executed selection on the complete candidate ledger chose MPC-RH; Table 1 deploys MPC-RH on the three datasets.
- **Code before:** The Table S12c generator replays main Eq. (19) on the representative rows and on every configuration of select/decision.json (paper/build/selection.py::registered_rule_on_all_configurations) and compares both readings with the printed table.
- **Code now:** Unchanged.
- **Decided by:** select/decision.json and select/candidates.json of the original runs (all ten Roman-empire splits)
- **Rule:** The registered rule on the complete ledger must return MPC-RH on each of the three datasets, as the executed selection did. If it returns radius, the executed selection was not the registered rule's outcome: either the selection or the printed selection and Table 1 rows are corrected.
- **Check:** `complete_ledger_selection` in `scripts/consistency_audit.py`

## K4. Seed hierarchy of Table S2

**Displays:** supplementary:Table S2. **Status:** Design difference; records decide.

- **Printed** (supplementary, p. 2): D_select: twelve plant seeds repeated within fold and twelve controller seeds; D_explore: twelve controller seeds; D_compare: paired frozen plant seeds.
- **Code before:** Selection evaluates precomputed frozen-ensemble losses without a plant refit inside each fold; the controller is fitted once per Lean/Full profile (with cross-fitting folds) and selection has no twelve-controller-seed loop; E3-E6 repeat controller optimization conditional on one paired parent ensemble.
- **Code now:** Unchanged.
- **Decided by:** fit/<profile>/ and select/ of the original runs (controller fits per profile, any per-seed controller packages) and the E3-E6 campaign records
- **Rule:** Original records with twelve controller fits per profile (or plant refits inside selection folds): the released runtime adds that loop before the runs are reproduced. Records with one fit per profile: the controller-seed and plant-seed columns of Table S2 are corrected to the executed hierarchy.
- **Check:** `seed_hierarchy` in `scripts/consistency_audit.py`

## K5. Settings of the budget-matched comparison (Tables S46b and S47, main Figure 6)

**Displays:** supplementary:Table S46b, supplementary:Table S47, main:Figure 6. **Status:** Needs an author declaration.

- **Printed** (supplementary, p. 27): The E3 execution of Table S46b consumes its whole work ceiling, while the ensemble-spend crossover of Table S47 uses a separate decoding-cost scenario.
- **Printed** (main, p. 13): Figure 6 draws the budget frontier, retained shares and the ensemble-spend comparison together.
- **Code before:** The E3-E6 arms run at the common ceiling (cm_campaigns); the settings of the S47 decoding-cost scenario are not registered.
- **Code now:** Unchanged.
- **Decided by:** the declared settings of the S47 decoding-cost scenario (decoding cost, ceiling, ensemble spend) and the campaign:budget records
- **Rule:** Register the scenario's settings before Figure 6 and Table S47 are compared; a plotted point or rounded value compatible with several settings does not identify them.

## K6. Failure responses of the target-regime gates (Table S3)

**Displays:** supplementary:Table S3. **Status:** Awaiting original records.

- **Printed** (supplementary, p. 2): Nominal graph value failure: stop or improve the plant; registry support failure: prune, collect and re-explore; depth qualification failure: do not credit the controller with plant rescue.
- **Code before:** A failed entry contract makes every feedback configuration infeasible, so the lifecycle returns q_C; a support failure does not prune, collect or re-explore; a depth failure applies an operational exclusion rather than a reporting instruction.
- **Code now:** Unchanged.
- **Decided by:** train/entry_initial.json and select/entry_complete.json of every original run
- **Rule:** Every gate passed in the original runs: the failure responses never executed and no reported value depends on them. A gate failed: the run is handled with the printed failure action (and the runtime's response aligned with it) before its values are reported.
- **Check:** `entry_gates` in `scripts/consistency_audit.py`

## K7. Table S20 |Q|: the feasible family, not the registry

**Displays:** supplementary:Table S20, supplementary:Figure S17. **Status:** Resolved.

- **Printed** (supplementary, p. 8, 15): Algorithm S4 reports |Q| of the family it runs over, Q_feas; Table S20 prints 180, 180, 144, 216, 240, 240 and 180.
- **Code before:** The Table S20 generator reported the registry size (270, 265, 260, 270, 265, 265 and 260), the upper bound of |Q|.
- **Code now:** The generator reports |Q_feas| (select/decision.json joint_feasible), verified against its Algorithm S4 replay, with the registry size beside it in the source data.
- **Decided by:** nothing further: the definition is fixed by Algorithm S4
- **Rule:** The realized |Q_feas| of each run is compared with the printed |Q| by make paper.

## K8. Literal cells of the protocol displays

**Displays:** supplementary:Table S1, supplementary:Table S11, supplementary:Table S38. **Status:** Needs an author declaration.

- **Printed** (supplementary, p. 1, 8, 23, 24): Table S1 annotates the Peptides-func interval role 'inferential; boundary'; Table S11 prints Lean as the fail output of the Full profile and 'strongest open-loop predecessor or q_C' for the state-conditioned route; Table S38 lists stage, unit and protocol as keys of the results record.
- **Code before:** The offline build reported fifteen literal differences. Five were wording cells of Tables S1, S3 and S8 whose printed wording describes the executed behavior: the partitioner named twice for ogbn-proteins, the per-split cores and the 'descriptive (repeated splits)' role of Roman-empire, the observer-period annotation of the innovation row and the gate response written as 'feedback candidates made infeasible; strongest open-loop q_C retained'.
- **Code now:** The generators print those five cells in the printed wording and keep the detail in the table notes and source data (paper/build/displays/protocol_tables.py), so Table S8 and the residual-headroom row of Table S3 reproduce exactly. Six cells remain, each a statement about behavior or schema rather than wording: the Peptides-func boundary annotation is an admission outcome, not a protocol constant; for a failed Full confirmation the runtime returns Lean only when Lean belongs to Q_s (q_C otherwise), and for a failed state confirmation it always returns q_C; the records carry no stage or unit column and store the protocol as a value. The S3 gate responses and the S2 seed columns are items K6 and K4.
- **Decided by:** the author, cell by cell: the manuscript cell is corrected, or the runtime or record schema is changed before the records are exported
- **Rule:** Each remaining difference is listed in paper/source_data/compare/ after make paper. Changing the runtime (S11) or the record schema (S38) after the records are sealed is not allowed: the change precedes the export, or the manuscript cell is corrected.

## K9. Manuscript corrections found while registering the figure contracts

**Displays:** supplementary:Figure S6, supplementary:Figure S7, supplementary:Figure S8, supplementary:Figure S13, supplementary:Figure S20, supplementary:Table S21. **Status:** Needs an author declaration.

- **Printed** (supplementary, p. 32): Figure S6c and Note S19 declare the sensing share s = 0.06 for the reduced ogbn-proteins sensor (k = 32, P_s = 4), whose own Note S19 cost 8r(k/d)/P_s is 0.073; with s >= 0.073 the T = 8 bound would be f_res <= 0.060, below the executed 0.061 of Table S6c.
- **Printed** (supplementary, p. 32): Figure S6b prints each bar pair's labels over the other bar (e.g. 36 above the 50% bar), and labels COCO at 50% with 15 where Note S19 states T <= 14 (14.84 rounded against its integer part).
- **Printed** (supplementary, p. 32, 15): The Figure S7b legend prints the ogbn-proteins and COCO-SP fifth-percentile gaps as 1.0 mnats where Table S21 prints 0.001 nats.
- **Printed** (supplementary, p. 32, 34): The captions of Figure S8 and the title of Figure S13c refer to Table 2 for the paired test intervals, which main Table 1 prints.
- **Printed** (supplementary, p. 38): On the printed page of Figure S20a a white overlay covers the E = 4 legend entry and the Roman-empire filled marker and label, which the embedded figure image contains.
- **Code before:** Not applicable: these are differences inside the manuscript.
- **Code now:** The figure contracts follow the printed values and record each difference in their group notes (paper/printed_display_values.json).
- **Decided by:** the author's correction of the manuscript
- **Rule:** Correct each item in the manuscript source and regenerate the PDFs; then register the new PDF identities (paper/sources.json) and re-register the affected contract values from the corrected pages, so that the comparison targets stay the printed values.

Generated from `paper/consistency_ledger.json` by `scripts/consistency_audit.py --write`; `make consistency-check` verifies it is current.

