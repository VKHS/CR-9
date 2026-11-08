# Paper references and reconstruction evidence

[Documentation](README.md) · [Reconstruction](RECONSTRUCTION.md) · [Statement compliance](../STATEMENT_COMPLIANCE.md)

Reconstruction computes a result from an execution artifact or a declared
protocol, then compares it with an independently registered paper value. A
rendered table or plot is complete only when its required cells and panels have
source data and its required comparisons have reference values. A missing value,
an unknown reference and a numerical disagreement remain distinct outcomes.

## Printed values

The main paper and supplementary PDF identities are recorded in
[`paper/sources.json`](../paper/sources.json). The existing numerical registry is
[`paper/reported_results.json`](../paper/reported_results.json). Additional reviewed
display references are in
[`paper/printed_display_values.json`](../paper/printed_display_values.json).

The additional reference file covers the previously unregistered comparisons in
Tables S1, S2, S3, S5b, S6, S6b, S8, S9, S10, S11, S15, S19, S24, S28, S28b, S29, S31,
S32, S35, S36, S38, S39 and S40, and Figures S1, S5, S8, S10, S11 and S19. It
records the PDF checksums, display identity and source pages for each group. Its
review note describes the contextual transcription and normalization; it is not
an execution record or an author certification.

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

Three source-defined groups from the prior work report are now registered in
`paper/registered_constants.json`: S15's declared accounting ranges, S19's
construction/conclusion descriptions, and S28b's `sigma1` grid. S15 exports its
declared ranges as labelled companion data while retaining actual costs in the
measured ledger. Table S19's numerical targets remain comparison-only; actual activity,
fallback and harm come from the verified control records. The S28b grid contains
three values; Figure S17g plots its two endpoints. A single panel or parameter
grid does not establish complete comparison coverage for that eight-panel figure.

The S28/S28b review adds two independent comparison groups containing **110
targets**. Supplementary pages 18–19 explicitly distinguish the listed
reference ESS from S7's observed support minima. The analytical input registry
therefore retains the stated counts, ESS, count grid and target-effect scales;
the generators export observed minima separately. S28's final column still uses
the actual lifecycle effect. S28b uses its four expressly printed target scales,
including 0.0090 nats, rather than rounding an observed effect into a new scenario.

The paper displays the Hoeffding count at equality rounded to an integer. This can
be one below the smallest integer satisfying the bound: for 0.0030 nats those
values are 53,257 and 53,258 respectively. The generator exports the real equality,
the printed rounded convention and the sufficient integer separately. Bernstein
requirements are checked against the bound at the reported integer and its
predecessor. These formula evaluations establish analytical agreement; they do
not create new experiment observations.

## Completed figure and sweep contracts

The sidecar now contains **29 independent reference groups**. Six further groups
complete the contracts for Figures S1, S5, S8, S10 and S19 and Table S29, adding
**430 targets**. Together with the S28/S28b additions, this upgrade adds **540
comparison targets**. These include text and categorical cells as well as numeric
coordinates; they are not a count of newly executed experiments.

| Display | New targets | Coverage and evidence boundary |
|---|---:|---|
| Figure S1 | 135 | Seven datasets across endpoint/loss means and intervals, units, mechanism categories, fallback roles and coordinates, expected-work marker encoding and annotations. Original runs and `input_gate` records remain required; the products global return is not an observed fallback probability. |
| Table S29 | 36 | Every horizon/beam configuration, selection effect, expected cost, headroom fraction and registered setting from SI p. 19. The generator still requires verified `horizon_beam_sweep` measurements. |
| Figure S5 | 57 | Both complete sweeps, every numerical headroom label, the shared horizon/beam setting and the horizon overlay. The actual oracle-headroom denominator must come from the sweep records, not inversion of a rounded reference fraction. |
| Figure S8 | 12 | Three clipped-fraction bars, both uncertainty sources for each dataset, and all three ratio annotations. Actual run moments and intervals determine the output; a one-decimal ratio cannot recover hidden unrounded intervals or variance components. |
| Figure S10 | 52 | All six dataset bar pairs and residuals/annotations, every point of both eight-width curves, their domain and reference boundaries. Original run costs remain measured inputs; sensing shares and sketch precision are source-defined analytical identities. |
| Figure S19 | 138 | Every statement/class/condition cell, the six-class legend, source scope distinctions, seven admission rows and all eleven model-scope bullets. Protocol descriptions are separate from run-derived unit counts, coverage and admission outcomes. |

Including the prior analytical Figure S11 contract, **six of the 25 quantitative
figures have dedicated complete contracts; 19 remain pending**. Main Figure 1 is
an illustration. The [figure-reference coverage plan](FIGURE_REFERENCE_COVERAGE.md)
records the exact source pages, original-data prerequisites and remaining work.
This is contract completion, not a claim that the original experiment collection
has been supplied or reproduced. The new contracts preserve the measured-input
requirements of Figures S1, S5, S8, S10 and S19.

All required panel targets are enumerated independently of what a generator emits.
Omitting a comparison leaves a coverage failure, and perturbing an observed input
or an analytical coordinate is detected against the independent targets. A
successful synthetic-fixture test establishes this software behavior; it does
not become scientific evidence for the paper.

Figure S8 provides a precision example: the rounded Peptides interval endpoints
in main Table 1 imply a ratio around 3.73, while Figure S8 prints 3.8. Those rounded
endpoints do not determine the original unrounded interval. Reconstruction keeps
the actual recorded interval and compares its computed ratio with the figure's
one-decimal annotation. It does not alter the interval to force agreement.

## Runtime and remaining source distinctions

S3's selected-strata threshold is now read from the runtime implementation. New
runs record the nominal A0 support fraction over every registered Full-profile
stratum and apply the 1.00 requirement before selection when entry enforcement is
enabled. Historical run records retain their original source identity and do not
acquire retrospective evidence that this new gate was enforced.

The exact numerical analytical inputs `o_dec` and `phi_OL` are defined symbolically
but are not printed in the supplied PDFs. Rounded table outputs do not identify
them uniquely. They remain explicit input requirements for the affected analytical
scenario, independently of actual E3–E6 campaign measurements.

Four groups of textual references in S14b, S20 and S42 now retain the literal
printed wording. Eleven text cells changed; all 2,349 existing numerical leaves
in the results registry are preserved. Earlier explanatory qualifications remain
as underscore-prefixed repository metadata, outside the comparison targets. The
source-review record binds the revised registries to the same supplied PDF bytes
and identifies the review as AI-assisted.

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
The large study archives remain in the release assets; the complete-tier gate
verifies those bytes and their original attempt seals before reconstruction.

`make statement-fulfilled` first downloads and verifies the complete artifact tier
and executes reconstruction, then applies the statement check. This is the final
publication gate. Synthetic fixtures exercise the software paths and retain their
software-only scope.
