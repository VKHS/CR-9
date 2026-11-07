# Changelog

## 2026-11-07 — Availability catalog and traced reconstruction

- Added a typed artifact catalog that maps code, configurations, dataset sources,
  split and seed manifests, environments, checkpoints, predictions, propensity
  logs, statistical outputs and generators to their files or producing commands.
  Bundled examples, new executions and original experiment artifacts retain
  separate identities. Availability prose now matches those distinctions.
- Added `make reconstruct`: sealed input-byte verification, run and auxiliary
  builders in one staged workspace, a report covering all 90 manuscript displays,
  source and output hashes, preserved workspaces on failure, and an optional
  strict full-display completeness check. `make reconstruction-example` verifies
  the complete workflow on the bundled examples; CI retains its output bundle.
- Added supplementary numerical builders for the original S14 confirmation
  family, factorial cell records, ensemble-law fits and certified budget
  campaigns. Generated plots include their numeric CSV sources; partial panels
  and analytical fits retain their stated scope.
- Corrected launcher resume and report behavior: frozen source verification,
  active-job awareness, unfinished Roman split selection, restored suite
  settings, and explicit failure/incomplete statuses for report construction.
  GPU availability follows the infrastructure recovery path.
- Kept the public overview free of deficit counters and removed universal
  reconstruction wording where the workflow supports only specified outputs.
  The frozen prediction runtime and manuscript numerical registries are unchanged.

## 2026-11-06 — Reproduction launcher and code-release scope

- `run_cross_morse.sh` is now the reproduction launcher of the paper. A SITE
  SETTINGS block at the top holds everything cluster-specific (account,
  partition, GPU request, storage, modules, PyTorch wheel, wall times); the
  defaults are the authors' Arrhenius setting with the allocation left empty.
  `check` verifies the settings, the runtime fingerprint and the complete job
  plan with the paper's GPU-hours without submitting; `reproduce [full|pilot|smoke]`
  freezes the source and configurations into a numbered suite and submits the
  environment setup, one run for each of six datasets, the ten Roman-empire
  splits as one array and a report job; `status`, `report` and
  `reproduce resume SUITE` follow and recover a suite.
- Resilience: checkpointed requeue at the time limit (USR1 180 s before the
  limit, exit 75), requeue on another node after a GPU or driver failure (exit
  76), GPU health checks with retries, failure classes and exit codes, stable
  live logs, a run registry and job table per suite, and a finalizer that
  records scheduler accounting.
- `scripts/reproduction.py` writes the suite identity and the report:
  `REPRODUCTION_REPORT.md` places each dataset's deployed policy and paired
  effect beside main Table 1, summarizes the comparison of every quantity
  `scripts/compare_with_manuscript.py` reads (exact, interval, outside), embeds
  Tables 1 and 2 rebuilt from the regenerated runs, records the GPU-hours used
  and packs a results ZIP.
- `release/release_scope.json` declares a code release whose results are
  regenerated; the availability page, the claim ledger, the manuscript
  availability paragraph, the paper reports and the badge are generated from
  that scope. The registration workflow is unchanged for releases that publish
  archives.
- `hpc/` drivers read the launcher's site settings (`run_cross_morse.sh site-env`).
- Tests: `tests/test_reproduction.py` and `tests/test_reproduction_launcher.py`
  (a simulated Slurm cluster: suite submission, setup, single runs, the Roman
  array, the report, checkpointed and infrastructure requeues, GPU-health and
  runtime failures, missing artifacts, report deferral and resumption).
- The runtime (`run_cross_morse*.py`, `cross_morse/`) is unchanged: source
  fingerprint `0aba361b54f6561f407f097af257cf27e4bf7f85d1ff5f47a4bcdd8bcda3a4fb`.

## 2026-11-06 — Current manuscript support

- Replaced the obsolete revision workspace with the current paper results, complete display inventory and source-bound contextual review.
- Added joint S14 confirmation evidence replay and tightened E6 certificate and budget replay checks.
- Kept the reported numerical targets and original runtime source identity. Registered artifact coverage is reported separately from the authors’ completed measurements.

## 2026-11-04 — Paper-support review and authentic-run discovery

- Hardened the analytical checks: mandatory runtime checks fail when unavailable;
  non-finite comparisons cannot pass; the lifecycle simulation must exercise
  both deployment and comparator return; invalid statement selections fail.
  The spectral reference uses a derivative/remainder bound instead of treating
  a sampled maximum as a certificate. Degenerate variance, sample-size and
  empty-score cases are explicit.
- Preserved the runtime's feedback-only selector domain and exposed the
  all-feasible reading as a labelled reference diagnostic. Added actual runtime
  parity for the observer, inserted sites, selector terminal rule and budget
  episode details. Runtime source and historical run identities are unchanged.
- Supplementary checks now enforce complete dataset/route/contrast families,
  source decimal precision and native-AP/percentage-point conversion. Informational
  inverse fits do not count as successful source checks; alternative readings
  remain advisories.
- Added bounded discovery of completed official exports to the existing evidence
  pipeline and `make evidence-discover`. It proposes only unique complete
  candidates, leaves author/deposit metadata unset and reports ambiguity or an
  incomplete scan. It does not register experiments or execute training.
- Aligned the decision checker with the runtime's strict numeric Holm boundary,
  while preserving the meaning of a printed strict P-value bound.
- Added [Paper support](docs/PAPER_SUPPORT.md), with E1–E20 acceptance criteria,
  selection/confirmation semantics, measured-versus-modelled distinctions and
  precise manuscript issues that require original evidence or author resolution.
  Evidence counts remain generated from the actual indexes.

## 2026-11-04 — Analytical statements and supplementary arithmetic

- `cm_theory/` implements every definition, lemma, proposition, theorem and corollary of
  the main text, the Supplementary Lemma 3 (work bound of Algorithm S3), the entry contract,
  the support identity, the cost identities, the Note S19 cost model and the Note S24
  variance hierarchy as functions of explicit inputs (NumPy and the standard library).
  `cm_theory/statements.py` records each statement with its source, reference
  implementation and runtime counterpart.
- `scripts/verify_theory.py` (`make theory`, part of `make verify`; `make theory-full`)
  checks each statement on seeded random instances of its hypotheses: periodic and
  weighted graphs, bounded loss distributions, finite decision processes and random budgets.
  Theorem 2 is checked through `cross_morse.statistics.empirical_bernstein` and Lemma 3
  through `cm_experiments.budget.run_budget_episode` together with an independent
  restatement of Algorithm S3. `docs/THEORY.md` is generated from the same table and
  checked for currency.
- `scripts/check_si_identities.py` (`make si-check`, part of `make verify`; `make si-report`)
  recomputes the derived entries of Supplementary Tables S2-S47 and the related in-text
  values from their printed inputs over the rounding box of those inputs, complementing
  `scripts/check_paper_constants.py`. Values that only an alternative reading reproduces
  are reported as advisory notes with that reading.
- `cm_theory.inference` adds two one-sided equivalence tests, minimum detectable effects
  and power at a Holm step, and the balanced two-way graph-seed variance decomposition;
  the Student-t distribution is computed with the standard library.
- Tests: `tests/test_theory_battery.py`, `tests/test_theory_formulas.py` (including the
  Table S28/S28b columns) and `tests/test_si_identities.py` run without PyTorch;
  `tests/test_theory_runtime_parity.py` compares the dense references with the sparse
  PyTorch operators when PyTorch is installed. No runtime file changes, so the source
  fingerprint of existing runs is unchanged.

## 2026-11-03 — Evidence registration from the cluster runs

- `scripts/evidence_preflight.py` applies the scientific audit's validators to completed
  official runs where they were produced, before packing or upload: stage chain, scope,
  data-role protocol, role isolation, Table S5 inventory, numeric replay, selector, the
  full audit with support reconstruction, and the manuscript comparison with a
  provisional binding. Verdicts READY, INTEGRITY_ONLY, PENDING_FULL_AUDIT or BLOCKED.
- `scripts/evidence_pipeline.py` (`make evidence-*`) runs preflight, pack, deposit,
  publish, register, bind and record steps from one plan file with a locked state file;
  publication requires explicit consent and an eligible current verdict bound to
  the archive bytes, sealed contents and checking inputs. Blocked or uncompleted
  full audits cannot be overridden into publication. Mismatching bindings cannot
  be force-installed.
  `hpc/pack_evidence_arrhenius.sh` preflights and packs the seven datasets as a Slurm array.
- `scripts/emit_campaign_binding.py` writes the Roman-empire ten-split binding that the
  audit's campaign checker requires (single-run emitters refuse campaigns).
- `export_records.py mapping-template` and `from-mapping` (`scripts/record_mapping.py`)
  build any record set from original outputs: values are located in the copied raw files
  or derived with input hashes. Record checks recompute those derivations;
  replacement preserves the prior set if installation or indexing fails.
- `scripts/register_manuscript_sources.py` reports PDF identities and numeric-token
  searches as review aids. Registering sources requires a contextual review of
  the declared groups, bound to the exact PDFs, registry and page map. Occurrence
  percentages do not certify transcription accuracy. Existing source identities
  and numerical targets are retained in this upgrade.
- `scripts/readiness.py` (`make readiness`) checks outstanding requirements and
  review freshness against the selected evidence and check artifacts. Repeated
  baseline patterns remain advisory; missing observations and decision
  inconsistencies remain visible.
- `make verify` on a clone without the runs reports each run registered at a production
  Zenodo DOI as held at its deposit and checks its committed manifest against
  `runs/index.json` (`--allow-deposited-absent` of `ingest_run.py --check` and
  `run_manifests.py`); present runs are verified in full, and registering on a machine
  without the earlier runs keeps their manifests.
- Availability generation synchronizes the derived registered-dataset list;
  separate declarations about original observations and reproduction retain
  their existing values. External-index ingestion preserves canonical views.
- Retained one practical registration guide and the existing Make interface;
  delivery-only patch, commit and token-prompt wrappers are not required in the
  published repository. Runtime code, full-profile model settings and the
  benchmark evidence indexes are unchanged.
- Container CI also runs when registration scripts, builders, tests or Make
  workflows change, exercising the package without a Git checkout or network.

## 2026-11-03 — Reviewer workflow and evidence deposit tools

- Added `make review`: software verification and synthetic replay in offline
  mode; registered-deposit verification, supported builds, comparisons and the
  scientific audit in evidence mode. Each invocation keeps separate outputs and
  reports missing or failed checks without promoting them to complete evidence.
- Added Zenodo archive/code deposit and reviewer fetch commands. Deposits reuse
  service-specific receipts, require explicit publication, bind file identities
  and distinguish reserved DOIs from verified public records. Archive licensing
  is explicit; reviewer downloads preserve authoritative indexes and documents.
- Added a generated table-to-evidence ledger, registration badge and manuscript
  availability paragraph. Registration, public access, reconstruction and
  scientific agreement remain distinct checks.
- Added diagnostics for repeated baseline resource patterns and confirmation-rule
  consistency. Pattern findings invite inspection of raw measurements; they do
  not establish measurement invalidity or authorize a change to observed values.
- Added a reviewer guide, compact reproducibility checklist and CodeMeta metadata.
  CPU container recipes use the recorded Python package versions. CI validates
  source, examples and full CPU execution, uploads bounded review artifacts and
  rechecks weekly; action references are pinned to verified commits.
- Retained the model implementation, S5 counts, numerical targets, original
  evidence indexes and runtime source identity. No official measurement archive
  or record set is manufactured by these tools.

## 2026-11-02 — Explicit plant widths and reference identities

- The seven full profiles now declare encoder, block-MLP and head settings whose
  parameter totals agree with Table S5 at its printed precision. The dataset
  guide records the exact counts and the scope of this agreement.
- Optional width keys preserve the previous tensor layouts when omitted.
  `scripts/model_inventory.py --default-widths` inventories that compatibility
  path, including the 1,138,216-parameter arxiv default recorded in Table S5d.
- Run manifests and `train/model_inventory.json` record the declared widths.
  GPS and linear-GINE full profiles use a two-layer prediction head, included
  in the parameter count and optimized by the feature-only reference `q0`.
- Reference checks cover S5 parameter totals, S5b/S5c site accounting, S6c
  sensing formulas and S23b nominal propagation work. Calculations that use
  reported cost measurements remain checks of the transcribed ledger.
- Transcription: Table S23 arxiv r = 0.013 as printed.

## 2026-11-01 — Policy resolution and evidence import

- Final-policy exports and comparisons now apply the registered confirmation
  rule. Peptides resolves to the comparator; its reported radius-arm results
  retain separate targets and require their own sealed observations.
- Corrected current graph-run record export to read the sealed test-role
  execution trace, with legacy archive support and role/ensemble checks.
- Aligned S19 and S25 evidence coverage with their actual control and
  regime/model rows, including required quantities and raw-file verification.
- Fresh model inventories record the actual class, parameters, buffers and
  optimizer-active counts. The dataset guide shows one exact count column;
  original S5 values remain in manuscript provenance.
- Added regression coverage and fresh-run record exports to CI. Official
  deposit status remains generated from the evidence indexes.

## 2026-11-01 — Seven-dataset model parameter inventory

- Added an all-seven inventory linking the rounded S5 benchmark counts to exact
  released-model counts, input/output dimensions and counting scope.
- Extended the existing inventory command with dataset-aware dimensions, task
  semantics, separate buffer counts and a full-profile report.
- Added consistency checks for S5/S15 parameter references, adapter dimensions
  and the generated documentation table.

## 2026-11-01 — Manuscript alignment and archive comparisons

- Updated source identities to the finalized 16-page manuscript and 38-page
  supplement. Table S5d now records the reported model-inventory distinction.
- Model comparisons check sealed checkpoint identities and named-tensor counts
  against S5, including every split in a Roman campaign.
- Confirmation comparisons use the current S14 measurements on `D_confirm`;
  S14b remains a separate analytical sensitivity.
- Generated captions read Peptides values and lifecycle status from the registry.
  Lifecycle diagrams and archive guides use the separated data-role protocol.
- Clarified which table columns come from runs and which require auxiliary
  measurement sets. Current archive coverage remains in `AVAILABILITY.md`.

## 2026-10-30 — Focused references and portable verification

- Removed bundled manuscript PDFs and duplicate review narratives and reading
  views. Retained source hashes, page references, numerical targets and concise
  implementation choices.
- Made registry reading views optional. Source checks report explicitly whether
  they checked metadata or verified a supplied PDF's bytes.
- Consolidated availability into one generated statement linked from the README.
  Official archive, measurement-record and result-binding requirements remain.
- Corrected the data-role guide to match the five-role 60/20/10/5/5 protocol.
- Allowed machine-precision floating-point variation in regenerated example
  CSV/JSON reports. Artifact hashes, decisions, counts and material numerical
  differences retain strict checks.

## 2026-10-30 — Published project snapshot

- Seven dataset adapters, 21 execution profiles and two complete synthetic
  examples, with environment specifications, tests and result builders.
- Practical guides and canonical links use `https://github.com/VKHS/CR-9`.
- Official benchmark archives and auxiliary measurement sets are not deposited
  in this snapshot; current coverage is recorded in [AVAILABILITY.md](AVAILABILITY.md).
