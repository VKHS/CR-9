# Availability and reconstruction review

[Documentation](README.md) · [Validation](VALIDATION.md) · [Statement compliance](../STATEMENT_COMPLIANCE.md) · [Publication](PUBLICATION.md)

## Scope and conclusion

This review examines the author-supplied **CR-9 work report and remaining steps**,
the main paper and supplementary file identified in
[`paper/sources.json`](../paper/sources.json), and the repository beginning at
`f34aa5a8f1d9c3d379fc0e6a288aa4373c1b980b`. The subsequent author-line change in
`578837f9c165a3d1bbe4cc2d9fce3783ae5d5f03` is preserved. The changes described below
are in the revision introducing this review.

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

Twenty-one additional comparison groups are independently transcribed from the
identified supplementary PDF in
[`paper/printed_display_values.json`](../paper/printed_display_values.json).
They cover previously unreferenced protocol tables, the action registry and
executed-action functions, the claim/restriction tables, and the analytical
terminal-budget illustration. Generators do not read those targets as their
computation inputs. The review is AI-assisted and is not an author certification.

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

An offline build without measured deposits renders twenty displays from
repository sources. Its 456 comparisons contain 437 agreements, fifteen literal
differences and four missing computed values; none lacks a registered reference.
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

The four missing comparison cells are substantive evidence questions:

- **S2 selection plant seeds:** selection uses precomputed frozen-ensemble losses;
  it does not record a separate plant refit or seed repetition inside each fold.
- **S2 selection controller seeds:** the released fitting stage fits once per
  Lean/Full profile, and selection has no twelve-controller-seed loop. Twelve
  plant checkpoint seeds do not establish twelve controller replicates.
- **S2 comparison plant seeds:** E3–E6 repeat controller optimization conditional
  on a fixed, paired parent ensemble. The displayed field does not demonstrate
  the printed plant-seed hierarchy.
- **S3 selected-strata fraction:** aggregate registry support is checked, but a
  separately computed and enforced selected-strata fraction of 1.00 is absent.
  This does not mean that online support masking is absent.

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
| One-command reconstruction | `make paper`, with complete content, comparison and integrity checks | All required artifacts publicly retrievable and every required comparison successful |

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

The core `cross_morse/` runtime, configurations, sealed examples and source
snapshots remain unchanged relative to the reviewed base. Every pre-existing
numerical value in the paper registries is preserved. The constants registry
adds the three PDF-described protocol groups above; literal text corrections in
the results registry retain explanatory qualifications as separate metadata.
Auxiliary executors and reconstruction tools changed as described above and have
their own recorded source identities.
Public status headlines now use resource descriptions and named artifact states;
the machine-readable checks retain the actual evidence status.
