# Changelog

## 2026-11-09 — CPU demonstration of the complete pipeline

- A separately published demonstration ([demo/README.md](demo/README.md)) runs the whole pipeline with the
  repository's own code on GitHub-hosted CPU runners: the seven lifecycle runs (Roman-empire on its complete
  official graph, all ten splits; the other datasets on deterministic subsets of their official data built by
  `demo/data.py` with the runtime's own adapters), the production graph's record sets, campaigns and observed
  environment (`scripts/paper_outputs.py plan --allow-fixture --device cpu`, executed without Slurm by
  `demo/execute.py`), and every table and figure rebuilt from the release `demo-artifacts` with
  `make demo-paper`. [demo/RESULTS.md](demo/RESULTS.md) records what it contains and what it could not produce.
- The demonstration has its own channel: `release/demo_artifacts.json`, the release `demo-artifacts`, the
  relationship `reduced_scale_demonstration` and views under `demo/published/`. The paper's manifest refuses
  demonstration artifacts, `reconstruct_paper.py` refuses to mix the two manifests, and no status in
  `STATEMENT_COMPLIANCE.md` depends on the demonstration; its separate section reports what the demonstration
  shows for each item.
- The production planner gains `--device cpu`, `--s14-exclude`, `--horizon-sweep-profile/-ensemble` and
  `--horizon-sweep-beam` (fixture plans; recorded among the plan's declared choices). The Table S29 sweep can
  hold a beam width its parent was not planned with (`run_sweep.py plan --beam`): every point outside the
  parent's planner registry is then a variant lifecycle and no replicate of the parent's configuration is
  planned. `scripts/paper_artifacts.py` gains `pack/publish/fetch/status/derive --demo`; `make demo-paper`,
  `demo-fetch` and `demo-status` use them.
- Execution without Slurm on the CPU device: each E3-E6 study runs through `scripts/run_budget_study.py`
  (the study command of `hpc/run_budget_arms_arrhenius.sh`, with the same job-evidence directory and a
  `scheduler.json` recording that no allocation and no accounting exist); the accounting step skips such
  studies, a fixture plan exports the partial campaign, and the budget-matched record set is marked synthetic
  when a study is a software fixture. The E17 exports of a CPU plan skip the sacct capture and write the
  declared partial sets (`--allow-partial`).
- Workflows: `demo-runs.yml` (the runs), `demo-graph.yml` (the plan and its first dependency level, sixteen
  runners), `demo-continue.yml` (the later levels and the exports; each job restores from earlier archives
  only the outputs its tasks depend on, `demo/restore.py`, with the producer code pinned to the planned
  commit), `demo-grid.yml` (the PascalVOC-SP E14 job, whose eighteen additional plants train on several runners
  with the job's own training call, `demo/grid_seeds.py`), `demo-sweep.yml` (Table S29 as a graph of its own),
  `demo-publish.yml` (merge the maps, pack each artifact, publish, rebuild, record: `demo/merge_maps.py`,
  `demo/pack_each.py`, `demo/report.py`) and `demo-verify.yml` (`make demo-paper` from a fresh clone).
  `demo/execute.py` shares only unfinished tasks among runners and leaves named nodes to another workflow
  (`--skip-node`). Tests: `tests/test_demo.py`, `tests/test_sweeps.py`.

## 2026-11-09 — Protocol tables in the printed wording

- Five cells of Supplementary Tables S1, S3 and S8 whose printed wording describes what the code executes are
  generated in that wording, with the detail kept in the table notes and source data: the ogbn-proteins repeated
  units (`32 cores; 2-hop buffer`; the primary unit already names the degree-stratified partition), the
  Roman-empire repeated units (`10 splits`; each split run's 32 METIS cores with a 2-hop buffer are in the note)
  and interval role (`repeated-split empirical`), the period of the innovation row (`every update`; observer
  period r = 2 in the note) and the response of the entry gates (`retain strongest open loop`, the printed
  response of the residual-headroom gate). The offline build without measured outputs now reproduces 487
  printed values exactly (482 before), with 10 literal differences (15 before) and 3 missing seed-hierarchy
  fields; Table S8 is exact. Item K8 of the consistency ledger lists the six cells that remain, each a
  statement about behavior or schema: the Peptides-func boundary annotation of S1, the Full and state fallback
  descriptions of S11, and the stage, unit and protocol roles of S38. No reference value changes, and the
  runtime is unchanged.

## 2026-11-08 — Algorithm S4 in the runtime, the scientific-consistency ledger and figure contracts

- The selection's studentized maximum follows Supplementary Algorithm S4: `nested_selection` takes it over
  the feasible family Q_feas of main Eq. (18) (support, resource and official-metric bounds) instead of every
  executed registered configuration, a superset whose radius is at least as wide. `select/decision.json`
  records the family (`max_t_family`), `feasible_family_size` and `max_t_family_size`; r_max is the largest
  radius over Q_feas. Records sealed by earlier runtimes carry no family and replay exactly with every executed
  configuration (`cross_morse/review.py`, `paper/build/displays/run_audits.py::screened_radius`,
  `cm_experiments/backbones.py`); a test reproduces the complete selection record of each preserved runtime on
  identical inputs. The Table S20 finalist radius is evaluated on the same resamples through
  `diagnostic_max_t` instead of replacing a module function. The runtime fingerprint is now
  `9f7393f7f337d3bd8747e85e2aaf3481db33895e02f1f3f5a7881e6e8a3e6c5c`; the previous runtime `d9687652…` is
  preserved under `examples/source_snapshots/`, and `docs/VALIDATION.json` records the new validation event.
- The scientific-consistency ledger (`paper/consistency_ledger.json`, `docs/CONSISTENCY.md`) lists every open
  question between the printed methods or values and the code: the studentized-maximum family (K1), the
  Peptides-func confirmation at Holm P = 0.070 (K2), the complete-ledger selection on Roman-empire,
  PascalVOC-SP and COCO-SP (K3), the Table S2 seed hierarchy (K4), the budget-comparison settings (K5), the
  Table S3 failure responses (K6), Table S20 |Q| (K7, resolved), the literal protocol-table differences (K8)
  and the manuscript corrections found in the figure contracts (K9), each with the original record and the rule that decides it. `make consistency-audit RUNS_ROOT=<runs>`
  (`scripts/consistency_audit.py`) evaluates K1, K2, K3, K4 and K6 on completed runs; on the bundled runs it
  finds that their selections used every executed configuration, as sealed. No reference value is changed.
- Eight more quantitative figures carry complete, independently reviewed printed-value contracts: main Figure 5
  (268 targets) and supplementary Figures S6 (78), S7 (60), S9 (97), S12 (128), S13 (83), S15 (211) and S20
  (310), 1,235 targets in all, from the printed tables, captions and formulas evaluated in scratch code that
  imports no repository module. Each generator compares every target, and omission and perturbation tests guard
  each panel (`tests/test_figure_contracts_*.py`). Fourteen of the 25 quantitative figures now have contracts; the
  other eleven need original plotted data or a scenario declaration (`docs/FIGURE_REFERENCE_COVERAGE.md`).
  Figure 5 declares the `timing_anchors` record set it reads; the S9 margin ticks are drawn at the printed
  positive margins (products ±0.002) with the signed thresholds kept in the source data; S13 and S7 use the printed
  labels. Registering the contracts surfaced manuscript corrections (the Figure S6 sensing share, the S6b labels,
  the S7b/S21 p5 units, the Table 2 references of S8 and S13c, an overlay hiding part of Figure S20a), now ledger
  item K9.
- `make independent-verification VERIFY_WORK=<new directory>` (`scripts/independent_verification.sh`) is the
  acceptance test as a reviewer runs it: a fresh clone of the published repository at one revision, the complete
  tier retrieved there, `make paper`, `make statement-fulfilled` and `make final-destination-converge`, never
  reusing a local checkout or artifact cache. The final destination's close-out sequence runs it, after
  `make consistency-audit`.
- The destination's compliance page lists the consistency items, each decided at the destination by the
  original records or the author's declaration.

## 2026-11-08 — The destination tree at twelve of twelve, and Table S20 |Q| as the feasible family

- Table S20 reports |Q| as the paper defines it: the size of the feasible family Q_feas of main
  Eq. (18), over which Supplementary Algorithm S4 runs and whose size its line 10 reports. The
  generator (`paper/build/displays/run_audits.py::feasible_family_size`) counts the registered
  configurations that satisfy dynamic support, the resource bounds and the official-metric bound
  (`select/decision.json` `joint_feasible`), verifies that count against its Algorithm S4 replay, and
  lists the registered family size beside it in the source data; Figure S17c uses the same count.
  It previously reported the registry size (`candidate_count`), which bounds |Q| from above and
  could never equal the printed values. The design forecast now treats |Q| as a measured quantity
  with the registry as its bound: the printed 180, 180, 144, 216, 240, 240 and 180 lie within the
  registered 270, 265, 260, 270, 265, 265 and 260; no design-determined printed value conflicts
  with the released code.
- `destination/` holds every file that publication writes into the checkout, projected now at the
  path it will occupy (`scripts/destination.py`, written by `make final-destination`): the release
  manifest with all 37 required artifacts published, the seven executed split and seed manifests
  (runtime fingerprint, the configuration identity of every run and Roman split, seeds, role sizes
  from the registered floors and the providers' official split sizes, the pre-registered graph-role
  and official-split hashes of the LRGB benchmarks), the observed environments of the registered
  GH200 site, the E12 study view of the fourteen external comparisons and the source data of a
  complete reconstruction (90 displays, the 3747 printed values registered now, and the 19 figure
  contracts still to be registered with the input that closes each). The twelve registered items,
  evaluated on that tree with the checks of `scripts/statement_check.py`, are all fulfilled
  (`destination/STATEMENT_COMPLIANCE.md`); `STATEMENT_COMPLIANCE.md` keeps reporting the checkout.
- Every projected file carries a `projection` block that classifies its values (exact, registered,
  size, count, minimum, pending). `scripts/paper_artifacts.py` and `scripts/statement_evidence.py`
  refuse any file carrying it. `make final-destination-converge` compares every realized file of the
  checkout with its projection and exits non-zero when an exact value differs;
  `make final-destination-runs` additionally checks completed runs against the registered
  configurations, the released runtime, the role floors and the printed |Q|.
- The runtime (`run_cross_morse*.py`, `cross_morse/`) is unchanged by this entry: source fingerprint
  `d96876520ad80478dddcc71f8fdc1c1f277b7734da87681f047f4192f037c651`.

## 2026-11-08 — The printed availability text, the final destination and the commands on published outputs

- The availability text printed in the manuscript is registered and checked claim by claim.
  `release/printed_availability.json` holds the main-text paragraph (page 15), the two
  repository sentences of the Methods and the Supplementary paragraph (page 30), bound to
  the PDF identities of `paper/sources.json`, as 39 claims. `scripts/printed_statement_check.py`
  (`make printed-statement-check`) evaluates them with the vocabulary of the registered
  statement: symbols located with the Python AST, Make targets, the sealed bundled runs,
  the record contracts, `runs/index.json`, `records/index.json` and the release manifest.
  `STATEMENT_COMPLIANCE.md` lists them below the registered items; 29 are fulfilled by the
  checkout, 8 await the measured outputs, and the two sentences that place the measurement
  records at deposit DOIs in `records/index.json` are not supported by the record contract.
  `make availability-fulfilled` requires both texts.
- The final destination (`scripts/final_destination.py`, `make final-destination`,
  `docs/FINAL_DESTINATION.md`, `release/final_destination.json`) draws the repository after
  publication from what it already fixes:
  - a design forecast of the printed values that the released code determines before any
    measurement: 64 agree with the paper, and the printed |Q| of Table S20 lies within the
    registered candidate family of every dataset (next entry);
    `make final-destination-runs RUNS_ROOT=<runs>` checks completed runs against it before
    publication, and the registry formula reproduces both bundled runs route by route;
  - the data roles of PascalVOC-SP, COCO-SP and Peptides-func, pre-registered with the
    runtime's own role functions lifted from `cross_morse/data.py` without PyTorch; the same
    functions reproduce every role hash the bundled runs recorded at preparation;
  - the projected release: files, checkpoints, prediction and trajectory files, tier sizes
    and release parts of every run archive, counted with `content_counts`, `is_heavy` and the
    part size of `scripts/paper_artifacts.py` (about 29 GB in the analysis tier and 1.9 TB in
    the complete tier, dominated by the full-graph prediction arrays of COCO-SP and
    ogbn-products, whose complete tiers exceed one Zenodo record);
  - compliance at publication and the close-out sequence, including the two routes for the
    printed sentences on deposit DOIs.
  Projected values are marked, never enter a comparison and are replaced by the published
  values; publication refreshes the page.
- `make tables` and `make figures` read completed run directories with `RUNS_ROOT=` (a
  reproduction suite or the runs that `make paper` verified); `make records-tables` reads
  `RECORDS_ROOT`, and when `records/` registers no set, the record sets that `make paper`
  downloaded and verified.
- `records/index.json` declares all 31 record contracts of `records/specs.json`.
- The runtime (`run_cross_morse*.py`, `cross_morse/`) is unchanged by this entry: source
  fingerprint `d96876520ad80478dddcc71f8fdc1c1f277b7734da87681f047f4192f037c651`.

## 2026-11-08 — Local publication preflight and source-backed completion

- Enforce nominal A0 support in every registered Full-profile stratum before
  selection when the entry contract is enabled. Record the support fraction,
  missing cells and enforcement state in the sealed entry audit. Preserve the
  preceding runtime snapshot for exact-source historical replay.
- Reconstruct S28/S28b and Figure S17g from the paper's explicitly declared
  analytical counts, ESS and target-effect grids. Export actual run support and
  effect-dependent quantities separately. S28b now builds without archives.
  Add two PDF-bound reference groups while preserving all prior numerical values.
- Distinguish printed rounded Hoeffding equality counts from the smallest
  sufficient integer; export both and verify the Bernstein integer boundary.
- Plot the declared S15 cost ranges in main Figure 5a and Figure S4, retaining
  executed scalar costs separately. Record the three additional S4 method points
  that require original source data and study-scope clarification; do not infer
  their exact values from the plotted image.
- Add independent full-display contracts for Figures S1, S5, S8, S10 and S19 and
  Table S29: 430 additional required reference leaves, source-document bindings
  and input-perturbation/omission regressions. Correct S1's Peptides BCE label
  and distinguish a global comparator-return identity point from an interval.
  Preserve S8's rounded ratio annotations without inventing hidden precision.
  Keep declared analytical coordinates separate from measured outcomes.
- Require every canonical artifact and the requested cached tier during offline
  reconstruction, including environment evidence unused by a display. Emit named
  missing-input diagnostics without converting an incomplete cache into success.
- Add `make paper-preflight` to verify original local folders and reconstruct
  before upload. Recheck source and materialized input identities after the build;
  record local readiness separately from public availability. Original runs may
  supply their own exact, verified runtime snapshot.
- Carry an external run runtime snapshot in both release tiers so an archive
  verifies on another checkout. Stage only the small fingerprinted source files,
  stream heavy original files and recheck all source bytes before installing assets.
- Route the public `make replay` entry point through the same exact-source resolver
  as reconstruction, so historical runs use their recorded runtime implementation.
- Update the availability review and validation history. Current runtime checks
  remain separate from historical runs and from measured benchmark evidence.

## 2026-11-07 — Evidence-complete exports and independently checked reconstruction

- Retain actual factorial and internal-baseline predictions, fitted ablation
  controller packages, and complete external-baseline study archives. E12 exports
  verify every original search attempt and final checkpoint, prediction, split
  manifest and timing file, including all Roman official splits.
- Reconstruct factorial contrast intervals from paired per-unit observations and
  the original joint bootstrap. Replace E11's overlapping-dependence grouping with
  versioned connected components; one component cannot support an uncertainty
  interval or positive admission.
- Bind twenty-one additional comparison groups to the supplied supplementary PDF.
  Missing table cells, absent figure source data and omitted comparison targets
  prevent successful reconstruction. Printed disagreements remain visible.
- Register the printed S15 reference ranges, S19 control descriptions and S28b
  sigma grid. Keep the labelled Figure S17d independence illustration separate
  from the paired intervals of S16/S16b. Correct four groups of textual reference
  transcriptions, retaining explanations as metadata and preserving all prior
  numerical targets.
- Verify cached files, preserve immutable release assets, require explicit
  manuscript relationships, bind production plans to their auxiliary sources,
  and prevent duplicate Slurm submissions during resume.
- Publish source-data, membership, environment and study views atomically, with
  their release, input, source and report identities. The final statement gate
  fetches the complete artifact tier and executes reconstruction.
- Replace fraction-style public status headlines with a resource overview and
  named artifact status. The registered availability paragraph remains the target
  checked by the repository; measured benchmark outputs still require publication.
- Preserve the core runtime, configurations, prior numerical targets and sealed
  examples. Software fixtures and synthetic CPU checks do not establish benchmark
  reproduction. See [the availability review](docs/AVAILABILITY_REVIEW.md) and
  [validation record](docs/VALIDATION.md) for scope and executed checks.

## 2026-11-07 — Every reported experiment executable; one command produces the measured outputs

- One command produces every measured output of the availability statement on the
  cluster from the seven completed lifecycle runs:
  `bash hpc/produce_paper_outputs_arrhenius.sh <suite>` (`scripts/paper_outputs.py`
  `plan | submit | status | collect`, `make paper-outputs`) verifies the sealed runs,
  executes every runner's own plan step, submits one Slurm job graph with the
  dependencies between the runners and ends with the map of all artifacts that
  `bash hpc/publish_paper_artifacts_arrhenius.sh all <map>` publishes.
  `make statement-fulfilled` is the final gate. [Producing the measured
  outputs](docs/PRODUCTION.md) describes the job graph.
- Factorials (Tables S16, S16b): every cell of both 2^3 designs is executable and
  distinct. The observer factor carries a read-only stopping rule (the fitted
  terminal-value model stops an execution when the remaining nominal depths are
  predicted to raise the loss), risk-controlled execution is its own operation, and
  `factorial.json` records each cell's definition, stopping depths and action counts.
- E8 history-MLP observer (Table S36): a one-hidden-layer predictor over the last 2P
  latent vectors with a constant-gain update, executed as the module-ablation arm
  `observer:history_mlp` and measured in the decision audit.
- E11 buffer sensitivity (Table S27): `cm_experiments/buffered_admission.py` regroups
  the sealed per-core paired losses into the resampling units of each buffer (walk
  dependence, blocks around a maximum independent set) and applies the runtime's own
  admission statistic; at the registered buffer it reproduces the sealed decision.
- S14 joint confirmation built from the sealed runs: `scripts/build_s14_campaign.py
  plan | run | assemble` (`cm_campaigns/s14_builder.py`) executes the nine tests with
  the runtime's confirm-stage code; `check_campaign` audits such a campaign with the
  admission links reported as post hoc findings, and Figures 2–3 build from any
  validated campaign while the comparison reports differences from the printed S14.
- Internal rows of Tables S15 and S15b (E12, E17): `scripts/run_internal_baselines.py
  plan | run` (`cm_experiments/internal_baselines.py`) scores every internal route's
  configuration on D_test once per plant checkpoint, times q_C and the deployed rule
  with the E17 session inside completed GPU allocations, and takes the GPU-hours from
  the runs' stage times as Table S41; `scripts/export_records.py baselines --internal`
  adds these rows. `cm_campaigns/baselines.py` holds the method keys the S15 generator
  reads and the completeness audit of the set.
- E12: the export of the `baselines` record set packs every external comparison's
  study plan and its adapter code at the pinned commit (`--checkouts`); publication
  copies them to `experiments/baselines/`, and `make statement-check` lists the
  experiment-code and configuration items as awaiting that publication.
- Tables S31 and S35 link every row to the record set that its runner produces.
- The runtime (`run_cross_morse*.py`, `cross_morse/`) is unchanged: source
  fingerprint `0aba361b54f6561f407f097af257cf27e4bf7f85d1ff5f47a4bcdd8bcda3a4fb`.

## 2026-11-07 — Registered availability statement: published outputs, display generators, one command

- `release/availability_statement.json` holds the paper's registered data- and
  code-availability statement verbatim and divides it into its twelve items.
  `make statement-check` (`scripts/statement_check.py`) evaluates each item from
  the repository's contents and writes `STATEMENT_COMPLIANCE.md`; an item is
  fulfilled, awaiting the publication of the measured outputs, or missing.
- The measured outputs of the reported experiments (checkpoints, predictions,
  propensity-logged trajectories, statistical outputs, executed split and seed
  memberships, observed environments) are published as assets of this
  repository's release `paper-artifacts` and listed in
  `release/paper_artifacts.json` with the SHA-256 of every part, archive and file.
  `scripts/paper_artifacts.py` plans, packs (deterministic archives, parts below
  the 2 GiB asset limit), publishes, downloads and verifies them in an analysis
  tier and a complete tier; `hpc/publish_paper_artifacts_arrhenius.sh` runs the
  publication on the cluster and writes `manifests/published/`,
  `environment/observed/` and `paper/source_data/`.
- `make paper` (`scripts/reconstruct_paper.py`) is the one-command reconstruction:
  download, verification of every byte, of each run's sealed eight-stage chain and
  of each record set's trace, one generator per numbered table and figure
  (`paper/build/displays/`), full-precision source data of every display and a
  per-cell comparison with the printed values. Exit status 0 only when every
  display is complete and every value agrees.
- One generator per numbered display (`paper/build/displays/`): main Tables 1–2 and
  Figures 1–6, and every supplementary table and figure — 60 measured displays read
  from the published runs, record sets and campaigns, 16 protocol tables generated
  from the configurations, code and registered protocol (`paper/protocol/`), 13
  analytical displays computed from the stated formulas, and one illustration. Each
  writes the complete table (CSV, Markdown, LaTeX) or figure (PNG, SVG) with the
  full-precision source data of every panel and declares the printed value each cell
  reproduces; a generator never reads the printed values. `paper/display_manifest.json`
  names each display's generator and the record sets it reads.
- Experiment code for every measurement-record set a display reads
  (`cm_experiments/`, `scripts/run_experiment.py`, `run_experiment_b.py`,
  `run_sweep.py`, `run_timing_records.py`, `run_confirmation_analysis.py`,
  `run_trajectory_audit.py`, `run_decision_audit.py`): E2 predictability, E9 exact
  enumeration, E10 confirmation-size sensitivity, E13 inserted-site attribution, E14
  extended ensemble grid and its official scores, E18 spectral displacement audit
  and bound tightness, E19 sensitivity grids, the horizon/beam sweep, cross-backbone
  substitutions, factorials, negative controls, the serving-time input gate, the
  held-out mechanism contrasts, the admission-split oracle, the Proposition 2 tube
  audit, depth diagnostics, the action-gap, decision-trace and reliability audits,
  runtime scaling and timing anchors. Module ablations (E7, E8), the fitted
  factorial cells and the E18 policy value run as variant lifecycles
  (`scripts/run_ablation.py`): explore to confirm re-run from the sealed parent's
  prepare/train outputs with the variant installed in that process only and recorded
  in the configuration hash. Every runner reads sealed lifecycle runs without
  writing them, records the identities of its parents and exports a record set that
  `scripts/records.py` traces value by value. Thirty-one record contracts are
  registered in `records/specs.json`; `scripts/export_records.py from-mapping`
  accepts every one of them. Parts of a reported experiment that the released
  runtime cannot execute are declared in `release/experiment_code.json`
  (`not_executable`) and reported by `make statement-check`.
- Table S41's E3–E6 comparison row reads the Slurm allocations of the comparison
  campaign's arm jobs.
- `release/experiment_code.json` maps every experiment behind a reported table or
  figure to its code, command, configurations and published artifacts.
- Availability views (`AVAILABILITY.md`, `paper/AVAILABILITY_STATEMENT.txt`,
  `paper/CLAIMS.md`, the paper-support reports, the badge and the artifact
  catalog) follow the repository-release scope. `make verify` includes
  `paper-artifacts-check` and `statement-check`.
- The runtime (`run_cross_morse*.py`, `cross_morse/`) is unchanged: source
  fingerprint `0aba361b54f6561f407f097af257cf27e4bf7f85d1ff5f47a4bcdd8bcda3a4fb`.

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
