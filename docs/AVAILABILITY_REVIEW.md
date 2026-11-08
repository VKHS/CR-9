# Availability and reconstruction review

[Documentation](README.md) · [Validation](VALIDATION.md) · [Statement compliance](../STATEMENT_COMPLIANCE.md) · [Publication](PUBLICATION.md)

## Scope and conclusion

This review examines the author-supplied **CR-9 work report and remaining steps**,
the main paper and supplementary file identified in
[`paper/sources.json`](../paper/sources.json), and the successive repository
upgrades beginning at `f34aa5a8f1d9c3d379fc0e6a288aa4373c1b980b`. The current
completion work was evaluated against published main
`ba44e9a5d13cce6749514e46937dcc4cbfef5f53` on 8 November 2026. The author-line change
in `578837f9c165a3d1bbe4cc2d9fce3783ae5d5f03` is preserved.

The repository now carries a substantially stronger production, export,
publication and reconstruction implementation. The full availability statement
still requires the measured experiment artifacts and a successful comparison
with every reported display. Code, artifact descriptors and synthetic tests do
not establish that those measurements have been published. The registered
paragraph is preserved as the target; `make statement-fulfilled` is the check
that must succeed before it is treated as fulfilled.

The review did not execute benchmark training or access the Arrhenius filesystem.
The attempted connection failed at hostname resolution, before authentication.
The original checkpoints, predictions, trajectories and historical experiment
records were consequently not recovered during this revision. Existing sealed
examples remain explicitly synthetic.

## Progress in the current revision

| Area | Previous behavior | Current behavior |
|---|---|---|
| Selected-strata qualification | S3's separate 1.00 criterion was not computed and enforced. | New runs audit A0 qualification in every registered Full-profile stratum and apply the gate before selection when the entry contract is enabled. Missing nominal cells remain in the denominator and fail. |
| Historical runtime identity | The active source matched the preceding implementation. | The new runtime has its own fingerprint; the exact preceding 29-file runtime is preserved for historical replay. Old runs are not relabeled as having executed the new gate. |
| S28/S28b and S17g | Observed support minima and rounded effects could replace the source-declared analytical coordinates. | Declared analytical counts, ESS and target scales are evaluated directly, with observed support and effects exported separately. S28b reconstructs without experiment archives. |
| Figure 5a and Figure S4 cost ranges | Observed scalar costs could replace the declared method ranges shown in the source figures. | Render the declared endpoints and preserve measured costs as companions. Three additional S4 method points remain explicit source requirements. |
| Independent figure contracts | Figure S11 was the only quantitative figure with a dedicated complete reference contract. | Add full contracts for Figures S1, S5, S8, S10 and S19, plus every cell of Table S29. Changed or missing empirical inputs still produce differences or incomplete results. |
| Offline artifact completeness | A cached analysis tier or a missing environment artifact could escape the final completeness decision. | Requested tiers and every canonical artifact are checked independently of which inputs a display consumes. Missing manifest rows cannot remove a requirement. |
| Original-folder preparation | Verification and reconstruction were coupled to packaging or a published manifest. | `make paper-preflight` verifies an original-folder map and reconstructs before upload, with a local-only verdict and pre/post input checks. |
| Source snapshot discovery | Some consumers accepted a run's own snapshot while reconstruction required a bundled one. | Exact-source reconstruction can resolve a run's own verified source snapshot without adding it to this repository. |

The source-only reconstruction now renders 21 displays, compared with 20 at the
reviewed base. Twenty have complete content and required comparison coverage,
compared with eighteen previously. Available comparisons increased from 437 to
482 exact agreements; the unresolved computed cells decreased from four to three.
The fifteen literal differences remain visible. These are software, protocol and
analytical checks, not the recovery or reproduction of benchmark measurements.

The independent reference sidecar now contains 29 groups. This revision adds
540 required reference leaves: 110 for S28/S28b and 430 for Figures S1, S5, S8,
S10 and S19 and Table S29. S28b's 72 leaves are covered by 44 comparison rows
and agree without lifecycle artifacts. The five new figure contracts cover every
specified panel, coordinate family, interval, boundary and numerical annotation.
Positive tests use explicitly synthetic fixtures; the contracts do not supply
missing measurements. Figure S17's related-table checks remain partial.

### Figure-level evidence still needed

The inventory contains 26 figures. One is classified as an illustration; six of
the remaining 25 now have dedicated complete reference contracts: S1, S5, S8,
S10, S11 and S19. Nineteen still require complete figure-specific coverage of
their panels, coordinates and declared analytical assumptions. Some targets exist
in the table registries and can be mapped explicitly; other panels need original
plot data or scenario clarification. Depositing run archives alone cannot finish
these definitions. The [figure reference coverage map](FIGURE_REFERENCE_COVERAGE.md)
identifies completed contracts and each remaining figure's source pages and
bounded requirements. A complete contract defines what must agree; reproduction
additionally requires the original inputs to pass those comparisons.

The current Figure S4 on supplementary page 32 contains three retrained external
methods in each of its arxiv, Roman and Pascal panels. Its extra GRAMA/arxiv,
AMP/Roman and AMP/Pascal points are not among the fourteen external dataset/method
pairs tabulated in S15/S15b. Their precise scores and generating study records
must be supplied separately; visual positions on the printed plot do not provide
an exact numerical reference. The figure retains an incomplete result until those
points and its complete comparison contract are available.

## Corrections implemented

| Area | Finding | Implemented correction |
|---|---|---|
| Factorial uncertainty | Marginal cell intervals cannot determine the covariance of paired contrasts. | Replay the actual paired per-unit observations and shared bootstrap stream, checking all eight original cell summaries. Summary-only records provide point contrasts without an invented contrast interval. |
| E11 dependence blocks | Grouping around independent roots could leave a dependence edge between two purported resampling blocks. | Use connected components of the thresholded dependence graph, with a new versioned estimator. A single component supplies no bootstrap interval or positive admission. Old semantics cannot silently resume under the new definition. |
| Factorial and ablation evidence | Some exports referenced prediction or fitted-controller identities without carrying the files. | Retain the actual prediction arrays and fitted controller packages, with hashes and original lifecycle identity. |
| Internal baselines | Per-seed predictions were reduced to hashes; Roman member coverage was insufficiently constrained. | Save and verify each method/seed prediction array; require all ten distinct official Roman splits and reject mixed synthetic/benchmark campaigns. |
| External E12 studies | Plans and adapter archives did not provide the complete original studies. | Export deterministic complete study archives and file inventories, including all 48 original search attempts, final fits, checkpoints, predictions, splits, timing and source snapshots. Production's E12 export requests both `--copy-studies` and `--copy-predictions`. |
| Display completeness | A written file could be treated as complete despite absent cells, missing panels or unregistered comparison values. | Track required content and independently enumerated comparison targets. Missing, uncomparable or unreferenced values prevent success. Deliberate dashes carry explicit not-applicable reasons. |
| Release transport | Cached stamps could outlive changed files; conflicting assets and swallowed failures weakened publication. | Reverify cached bytes, use immutable content-derived asset names, and stop publication after transfer, verification or generator failures. |
| Provenance and resume | New outputs could inherit an original-run label; changed producer code or active jobs could be reused ambiguously. | Require an explicit manuscript relationship, pin auxiliary producer sources and contracts, verify task seals and check both live and historical Slurm state before resubmission. |
| Committed evidence views | A completion count or empty index could appear sufficient. | Bind actual source files, comparison CSVs, membership arrays, observed environments and E12 inventories to the current release, exact fetched inputs, builder sources and reconstruction report. Install all views transactionally. |

The publication wrapper permits an honest comparison report containing
disagreements to be published, but does not turn that report into successful
statement fulfillment. Verification or generator failures stop before derived
views are committed. Complete-tier verification remains necessary because the
small committed views deliberately leave large checkpoint and study archives in
release assets.

## Paper comparisons and remaining scientific decisions

Twenty-nine additional comparison groups are independently transcribed from the
identified supplementary PDF in
[`paper/printed_display_values.json`](../paper/printed_display_values.json).
They cover previously unreferenced protocol tables, the action registry and
executed-action functions, the claim/restriction tables, the S28/S28b analytical
scenario, the horizon/beam sweep and six quantitative figure contracts. Generators
do not read those targets as their computation inputs. The review is AI-assisted and is not an author certification.

Three of the work report's four unregistered source groups are recoverable from
the supplied supplement and are now registered: S15's declared reference
accounting ranges (page 12), S19's construction/conclusion descriptions (page 13),
and S28b's three-value `sigma1` grid (page 19, with endpoints used in Figure S17g).
The S15 ranges are labelled companion reference data and never replace executed
costs. S19 measurements still come from the actual control records.

The remaining group, `o_dec` and `phi_OL`, is symbolically defined without exact
numerical settings. Inferring `phi_OL=0.20` from rounded S33 outputs would not be a
unique transcription; `o_dec` cannot be isolated from the printed total-work
fractions. Their original scenario configurations or an explicit author setting
are needed. The report also refers to a separate **CR-9: decisions left for the
author** document. That document was not located among the accessible files;
the review does not claim to have settled its broader definition choices and
internal manuscript inconsistencies.

The four textual transcription groups identified by the work report are also
corrected: S14b sensitivity and unchanged-contrast interpretations, the S20
Peptides unit description, and S42's graph-comparison unit descriptions. Eleven
text cells now retain the literal PDF wording; earlier qualifications remain in
separate repository metadata. All 2,349 pre-existing numerical leaves in the
results registry are unchanged, and the source-review binding is refreshed
without an author-certification claim.

An offline build without measured deposits renders twenty-one displays from
repository sources. Its 500 comparisons contain 482 agreements, fifteen literal
differences and three missing computed values; none lacks a registered reference.
Those numbers describe the available protocol and analytical comparisons, not
reproduction of the paper's benchmark results. The command correctly exits with
an incomplete result. All ninety numbered displays remain registered.

The fifteen differences require different interpretations:

| Group | Interpretation |
|---|---|
| Six wording or annotation differences in S1, S3 and S8 | Added degree/split detail, descriptive repeated-split terminology, the Peptides boundary annotation, the observer-period annotation and the residual-headroom response. These are not six independent numerical or runtime defects. Literal comparisons are retained. |
| Two S11 fallback descriptions | Full-to-Lean is qualified by the caption's eligible-simpler-model rule. State failure always selects `q_C`, which is permitted but narrower than the row's predecessor-or-`q_C` wording. Neither should be described as an unauthorized-policy result. |
| Three S3 failure responses | Nominal qualification retains `q_C` instead of stopping/improving the plant; registry-support failure does not perform the prescribed prune/collect/re-explore response; depth failure applies a conservative operational exclusion rather than only a reporting instruction. |
| One S2 seed-column difference | Twelve controller-optimization replicates appear in the comparison controller-seed column where the PDF has a dash. The stage and seed terminology differs. |
| Three S38 representation differences | Stage and statistical unit are not row keys in the mapped record schema; evaluation protocol is stored as a value rather than part of the observation key. |

The three missing comparison cells concern the seed hierarchy:

- **S2 selection plant seeds:** selection uses precomputed frozen-ensemble losses;
  it does not record a separate plant refit or seed repetition inside each fold.
- **S2 selection controller seeds:** the released fitting stage fits once per
  Lean/Full profile, and selection has no twelve-controller-seed loop. Twelve
  plant checkpoint seeds do not establish twelve controller replicates.
- **S2 comparison plant seeds:** E3–E6 repeat controller optimization conditional
  on a fixed, paired parent ensemble. The displayed field does not demonstrate
  the printed plant-seed hierarchy.

The previously absent **S3 selected-strata fraction** is implemented prospectively
using Main Eq. (5), Table S3 and S35 convention C28. Its denominator is the Full
controller's registered model-fit support strata; its numerator is those with
qualified A0 cells. The existing support audit supplies the propensity, ESS and
maximum-weight qualification. The new helper does not relax those thresholds or
look at selection labels. Tests establish exclusion before label access even when
the feedback candidate would otherwise have a clear paired improvement.

These items require the original experiment records or an explicit scientific
choice about the implementation and manuscript. Reference targets were not
rewritten to force equality.

### Measurement scope retained

Figure S17d explicitly labels its printed error bars as an independence
illustration. Its reconstruction retains that analytical assumption and formula,
while exporting any verified paired-bootstrap intervals as separate companion
data. Tables S16 and S16b use joint resampling. The two interval constructions are
not interchangeable; reproducing the labelled illustration does not establish
paired inferential uncertainty or certify the whole eight-panel figure.

Observer-stop factorial timing ends at the stop exception. The replay that
produces the returned prediction is outside that timer, and the corresponding
work count excludes stopping-rule evaluations and the stopping-site measurement.
The new `measurement_scope` record makes this explicit. These measurements cannot
be presented as complete inference latency without a changed measurement
procedure and actual reruns.

S14 campaigns retain their original or post hoc status and their admission
findings. A retrospectively computed joint family does not establish that the
historical admission was governed by a prospective joint lock. E11's connected
components still concern dependent units from a single graph; they do not create
an independent-graph population guarantee.

## What completes the availability statement

| Requirement | Repository support | Required evidence |
|---|---|---|
| Experiment code and configurations | Lifecycle and auxiliary executors, explicit experiment map, E12 adapters and study contracts | The fourteen actual external study plans and pinned adapter/source archives used by the reported comparisons |
| Source data and statistical outputs | Verified record exporters, campaign audits, all numbered display generators and PDF-bound comparisons | Actual raw records, campaign inputs and reconstructed CSV/JSON outputs matching the paper |
| Splits, seeds and environment | Registered contracts, executed-membership export and observed-environment validation | The original run membership arrays, actual seed records and observed execution environments |
| Checkpoints, predictions and trajectories | Complete-tier archives; repaired external/internal/factorial exports | The original trained files and propensity-logged trajectories, including the auxiliary studies |
| One-command reconstruction | `make paper`, with complete content, comparison and integrity checks; local `make paper-preflight` before upload | All required artifacts publicly retrievable, complete figure reference contracts and every required comparison successful |

The preferred next input is the existing result-producing archive collection.
It should be discovered and verified before starting expensive replacement runs.
Any new runs retain `post_result_revision` provenance. The
[production guide](PRODUCTION.md) and [publication guide](PUBLICATION.md) describe
the exact artifact map, relationship declaration, job graph and release commands.

After measured artifacts are published, the final gate is:

```bash
make statement-fulfilled
```

This retrieves and verifies complete artifacts, executes reconstruction and
checks the committed source-data and manifest views. `make statement-check` is
the lighter offline check of those views. Neither a green software test nor
editing a status file substitutes for this gate.

The checks preserve and verify each declared manuscript relationship. They do
not independently authenticate historical attribution: identifying an archive
as the actual result-producing experiment still requires the author's records.

## Validation and preserved inputs

[VALIDATION.md](VALIDATION.md) and [VALIDATION.json](VALIDATION.json) record the
executed integration checks, their environments and their scope. Regression
cases include altered cached files, missing original model/prediction bytes,
duplicate studies/splits, changed producer code, omitted comparisons, false
completion flags, joint bootstrap covariance and publication failure handling.

The current core runtime adds `cross_morse/qualification.py` and five integration
lines in `pipeline.py`. Its identity is recorded in `docs/VALIDATION.json` and
`release/release_scope.json`. All 29 files of the preceding runtime are preserved
byte-for-byte under `examples/source_snapshots/0aba361b…/`, with their origin
commit and file hashes. All 23 configuration files, 620 sealed-example files and
the earlier `b5005fc…` source snapshot retain their bytes from the reviewed base.

All 2,349 pre-existing numeric leaves in the results registry, all 582 in the
constants registry and all 483 in the prior printed-reference sidecar are
unchanged in this revision. Added values are explicitly source-declared
analytical coordinates or independent comparison targets. The prior literal
text corrections retain their explanatory qualifications as separate metadata.
Historical validation events retain the source identities they actually tested.
Public status headlines now use resource descriptions and named artifact states;
the machine-readable checks retain the actual evidence status.
