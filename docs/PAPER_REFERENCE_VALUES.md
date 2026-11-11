# Paper references and reconstruction evidence

[Documentation](README.md) · [Reconstruction](RECONSTRUCTION.md) · [Statement compliance](../STATEMENT_COMPLIANCE.md)

Reconstruction computes a result from an execution artifact or a declared
protocol, then compares it with an independently registered paper value. A
rendered table or plot is complete only when its required cells and panels have
source data and its required comparisons have reference values. A missing value,
an unknown reference and a numerical disagreement remain distinct outcomes.

## Printed values

The main paper and supplementary PDF identities are recorded in
[`paper/sources.json`](../paper/sources.json). The numerical registry is
[`paper/reported_results.json`](../paper/reported_results.json). The reviewed display
references are in [`paper/printed_display_values.json`](../paper/printed_display_values.json),
which covers the comparisons in Tables S1, S2, S3, S5b, S6, S6b, S8, S9, S10, S11, S15,
S19, S24, S28, S28b, S29, S31, S32, S35, S36, S38, S39 and S40, and Figures S1, S5, S8,
S10, S11 and S19. It records the PDF checksums, display identity and source pages for
each group; its review note describes the contextual transcription and normalization.

A typed numeric entry preserves the paper's decimal precision. For example,
`{"$value": 0.020, "$decimals": 3}` retains three decimal places even though JSON
otherwise loses the trailing zero. An explicit `null` represents a printed dash;
an absent reference path means that a comparison target has not been transcribed.
Text normalization handles whitespace and mathematical minus signs. It does not
rewrite a conflicting protocol to match the implementation.

Source-defined analytical coordinates can also have explicitly declared comparison
precision. Their group metadata identifies the formulas, input sources, units and
rounding allowance. Such coordinates are derived targets, not measurements printed
to extra decimal places. In particular, Figure S10's six-decimal sensing-model
coordinates are independently evaluated from its declared formulas and verified
benchmark statistics; its observed-cost comparisons retain the precision of S23.

Only the comparison layer reads these additional targets. Experiment runners and
display computations use their recorded observations, configurations and source
definitions. The loader rejects a changed PDF identity, a table/page mismatch or
an ambiguous duplicate reference group.

Tables S31 and S35 preserve the paper's descriptive record names and add a
repository-location column. Every location is checked against the released code
and record contracts, and the source-data file lists the resolved links. This
keeps the paper's descriptions readable while making the actual files discoverable.

Three source-defined groups are registered in
`paper/registered_constants.json`: S15's declared accounting ranges, S19's
construction/conclusion descriptions, and S28b's `sigma1` grid. S15 exports its
declared ranges as labelled companion data while retaining actual costs in the
measured ledger. Table S19's numerical targets are comparison targets; actual activity,
fallback and harm come from the verified control records. The S28b grid contains
three values; Figure S17g plots its two endpoints. A single panel or parameter
grid does not establish complete comparison coverage for that eight-panel figure.

Two independent comparison groups for S28/S28b hold **110 targets**. Supplementary pages 18–19 explicitly distinguish the listed
reference ESS from S7's observed support minima. The analytical input registry
therefore retains the stated counts, ESS, count grid and target-effect scales;
the generators export observed minima separately. S28's final column uses
the actual lifecycle effect. S28b uses its four expressly printed target scales,
including 0.0090 nats, rather than rounding an observed effect into a new scenario.

The paper displays the Hoeffding count at equality rounded to an integer. This can
be one below the smallest integer satisfying the bound: for 0.0030 nats those
values are 53,257 and 53,258 respectively. The generator exports the real equality,
the printed rounded convention and the sufficient integer separately. Bernstein
requirements are checked against the bound at the reported integer and its
predecessor. These formula evaluations establish analytical agreement.

## Figure and sweep contracts

Six groups hold the contracts for Figures S1, S5, S8, S10 and S19 and Table S29
(**430 targets**); with the S28/S28b groups they hold **540 comparison targets**,
counting text and categorical cells as well as numeric coordinates.

| Display | Targets | Coverage and inputs |
|---|---:|---|
| Figure S1 | 135 | Seven datasets across endpoint/loss means and intervals, units, mechanism categories, fallback roles and coordinates, expected-work marker encoding and annotations. Original runs and `input_gate` records are its inputs; the products global return is not an observed fallback probability. |
| Table S29 | 36 | Every horizon/beam configuration, selection effect, expected cost, headroom fraction and registered setting from SI p. 19. The generator reads verified `horizon_beam_sweep` measurements. |
| Figure S5 | 57 | Both complete sweeps, every numerical headroom label, the shared horizon/beam setting and the horizon overlay. The actual oracle-headroom denominator must come from the sweep records, not inversion of a rounded reference fraction. |
| Figure S8 | 12 | Three clipped-fraction bars, both uncertainty sources for each dataset, and all three ratio annotations. Actual run moments and intervals determine the output; a one-decimal ratio cannot recover hidden unrounded intervals or variance components. |
| Figure S10 | 52 | All six dataset bar pairs and residuals/annotations, every point of both eight-width curves, their domain and reference boundaries. Original run costs are measured inputs; sensing shares and sketch precision are source-defined analytical identities. |
| Figure S19 | 138 | Every statement/class/condition cell, the six-class legend, source scope distinctions, seven admission rows and all eleven model-scope bullets. Protocol descriptions are separate from run-derived unit counts, coverage and admission outcomes. |

The [figure-reference coverage](FIGURE_REFERENCE_COVERAGE.md) lists the fourteen of the
25 quantitative figures with dedicated full-figure contracts, their exact source pages
and the inputs each comparison reads.

All required panel targets are enumerated independently of what a generator emits.
Omitting a comparison leaves a coverage failure, and perturbing an observed input
or an analytical coordinate is detected against the independent targets;
synthetic-fixture tests establish this software behavior.

Figure S8 provides a precision example: the rounded Peptides interval endpoints
in main Table 1 imply a ratio around 3.73, while Figure S8 prints 3.8. Those rounded
endpoints do not determine the original unrounded interval. Reconstruction keeps
the actual recorded interval and compares its computed ratio with the figure's
one-decimal annotation.

## Runtime and source distinctions

S3's selected-strata threshold is read from the runtime implementation. Runs record
the nominal A0 support fraction over every registered Full-profile stratum and apply
the 1.00 requirement before selection when entry enforcement is enabled; each run's
records carry the source identity it ran with.

The analytical inputs `o_dec` and `phi_OL` are defined symbolically in the paper, and
rounded table outputs do not identify them uniquely; the affected analytical scenario
takes them as explicit inputs, independently of the E3–E6 campaign measurements.

The textual references in S14b, S20 and S42 hold the literal printed wording;
explanatory qualifications are kept as underscore-prefixed repository metadata,
outside the comparison targets. The source-review record binds the registries to the
supplied PDF bytes and identifies the review as AI-assisted.

## A successful reconstruction

`make paper` checks more than the existence of generated files:

- Downloaded and reused artifacts must match the current release manifest.
- Required table cells and quantitative figure source data must be present.
- Required reference cells are enumerated independently of the comparisons a
  generator happens to emit, so dropping a comparison cannot make a display pass.
- Missing, incomparable or unreferenced values prevent successful completion.
- Values that differ at the printed precision are reported with both values and
  their source paths; the paper's targets are not adjusted to force agreement.

The generated `displays/compare/` files provide the per-cell comparisons.
`reconstruction.json` records display content, comparison coverage, inputs,
software identities and output hashes. The status JSON summarizes this report.

## Published source data

Publication derives small repository views from the verified artifacts:
`paper/source_data/`, `manifests/published/`, `environment/observed/` and the E12
study-code view. Their indexes bind the copied files to the release manifest and
the reconstruction tools. The source-data index also retains the reconstruction
report and its file inventory.

`make statement-check` checks these committed views offline. It verifies actual
study contracts, executed memberships, observed environments and the report
behind the completion summary. Empty files, duplicate studies, changed source
data and stale reconstruction tools cannot establish fulfillment.

The E12 view includes each complete study's file inventory and release-archive
identity. It checks all 48 attempted search configurations, final seeds and
official splits, and the checkpoint, prediction, split and timing identities.
The large study archives are release assets; the complete-tier gate
verifies those bytes and their original attempt seals before reconstruction.

`make statement-fulfilled` first downloads and verifies the complete artifact tier
and executes reconstruction, then applies the statement check. This is the final
publication gate. Synthetic fixtures exercise the software paths.
