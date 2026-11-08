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
protocol references are in
[`paper/printed_display_values.json`](../paper/printed_display_values.json).

The additional reference file covers the previously unregistered comparisons in
Tables S1, S2, S3, S5b, S6, S6b, S8, S9, S10, S11, S15, S19, S24, S31, S32,
S35, S36, S38, S39 and S40, and the analytical budget illustration in Figure S11. It
records the PDF checksums, display identity and source pages for each group. Its
review note describes the contextual transcription and normalization; it is not
an execution record or an author certification.

A typed numeric entry preserves the paper's decimal precision. For example,
`{"$value": 0.020, "$decimals": 3}` retains three decimal places even though JSON
otherwise loses the trailing zero. An explicit `null` represents a printed dash;
an absent reference path means that a comparison target has not been transcribed.
Text normalization handles whitespace and mathematical minus signs. It does not
rewrite a conflicting protocol to match the implementation.

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
measured ledger. S19's numerical targets remain comparison-only; actual activity,
fallback and harm come from the verified control records. The S28b grid contains
three values; Figure S17g plots its two endpoints. A single panel or parameter
grid does not establish complete comparison coverage for that eight-panel figure.

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
