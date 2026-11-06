# Reviewer questions about this repository

[Documentation](README.md) · [Reviewer guide](REVIEWER_GUIDE.md) · [Claims ledger](../paper/CLAIMS.md) · [Availability](../AVAILABILITY.md) · [Paper support](../paper/PAPER_SUPPORT.md)

Each answer names the files that hold the fact and the command that checks it.
`make faq-check` confirms that every referenced file and make target exists;
it is a reference-integrity check, not an automated factual review of the answers.
Status lines describe the repository on the day of the commit; the generated
[availability page](../AVAILABILITY.md) is the live count.

## A. Provenance and identity

**A1. Is the code in this repository the code that produced the paper's numbers?**
The release fingerprint identifies executable bytes; it does not by itself
prove that those bytes produced the printed results. The current runtime hash
is `0aba361b54f6561f407f097af257cf27e4bf7f85d1ff5f47a4bcdd8bcda3a4fb`,
recorded in `docs/VALIDATION.json`. Original-run provenance still requires the
sealed archives and their source snapshots. `AVAILABILITY.md` reports the
registered coverage; passing software checks cannot fill missing archives.
Where: `scripts/source_fingerprint.py`, `scripts/_source_snapshots.py`.
Check: `make fingerprint`

**A2. Which manuscript version does the registry transcribe?**
`paper/sources.json` identifies the main text and supplement by SHA-256, size
and page count, and maps each table to its pages. The PDFs are not in the
repository. `paper/reported_results.json` holds the printed values,
`paper/registered_constants.json` the protocol constants, and each registry
group records its source pages. A new PDF is registered through a named contextual
review of every group (`make manuscript-sources`, then
`make register-manuscript-sources`), never by overwriting numbers.
Where: `paper/README.md`, `scripts/register_manuscript_sources.py`.
Check: `make registry-check`

**A3. How is the earlier release identified?**
The earlier public base, 202ad5a, is a root commit dated 2026-11-05.
Subsequent commits preserve the changes to the released repository.
The available history does not establish why it was republished or preserve
an independently authenticated earlier history. `RELEASE_MANIFEST.json` and
`SHA256SUMS.txt` verify release bytes; the runtime fingerprint identifies code.
Obsolete base-commit fields are null with an explanatory note, not valid
ancestry references. Neither hashes nor a new commit authenticate old results.
Where: `docs/VALIDATION.json`.
Check: `make verify-manifest`

**A4. Are the bundled example runs real experiments?**
No. `examples/runs/arxiv` and `examples/runs/peptides_func` are synthetic
(`"dataset": "synthetic"` in their `config.json`); they exercise the complete
eight-stage lifecycle so that replay, archiving, ingestion and table building
can be tested without benchmark data. Every tool refuses them as official
evidence.
Where: `examples/README.md`, `scripts/release_audit.py`.
Check: `make reproduce`

## B. Reproducing the results

**B1. Where are the raw results behind Table 1?**
They are the seven full-profile run archives (one per dataset; Roman-empire is
a ten-split campaign) and ten measurement record sets. `runs/index.json` and
`records/index.json` are their indexes; `AVAILABILITY.md` reports how many are
registered and `paper/CLAIMS.md` maps each manuscript display to the evidence
it needs. Registration goes through the preflight, packing, Zenodo deposit and
binding steps of `docs/EVIDENCE_REGISTRATION.md`; `docs/ARRHENIUS_SESSION.md`
orders the session that locates and registers the completed records.
Check: `make scientific-audit`

**B2. Once the archives are registered, how do I rebuild the tables?**
`make paper` builds the run-derived tables and figures from the ingested
archives and record sets; `make compare-runs` places every quantity of a run
beside the printed value and reports exact, interval or outside at printed
precision; `make review REQUIRE_EVIDENCE=1` runs the whole reviewer path
including public-deposit verification.
Where: `scripts/build_paper.py`, `scripts/compare_with_manuscript.py`, `docs/REVIEWER_GUIDE.md`.
Check: `make example-paper`

**B3. Can I run the method myself without the cluster?**
Yes, on a CPU: `make env-cpu` installs the recorded environment,
`make smoke` runs a synthetic lifecycle, and `make train-sample
DATASET=Peptides-func` trains one small profile end to end. Docker and
Apptainer definitions reproduce the CPU environment. Full-profile benchmark
runs use the Slurm wrappers (`run_cross_morse_<dataset>.sh`) and the
Arrhenius profile in `configs/arrhenius-site.env.example`.
Where: `docs/GETTING_STARTED.md`, `docs/HPC.md`, `Dockerfile`, `environment/apptainer.def`.
Check: `make environment-check`

**B4. Which software versions were used?**
`environment/requirements-cpu.lock` pins 53 packages; `environment/check_environment.py`
compares an environment with the lock, and each run records its own
`environment.json`. The GPU wheel used on the cluster is captured by
`environment/arrhenius/capture_environment.sh`.
Check: `make environment-check`

**B5. Are the external baselines reproducible?**
The fourteen dataset/method pairs run through a pinned upstream harness
(`cm_experiments/upstream.py`, `scripts/run_upstream_baseline.py`,
`hpc/run_upstream_baseline_arrhenius.sh`) that records the upstream and
adapter commits, all HPO attempts and the final seeds. Their measurements
enter the `baselines` record set, whose raw-file locators are rechecked by
`make records-check`, and `make degeneracy-check` inspects the deposited
baseline records for repeated values.
Where: `docs/EXPERIMENTS.md`, `records/specs.json`.
Check: `make records-check`

**B6. How are GPU-hours and timings measured?**
Slurm accounting is captured with `hpc/capture_arrhenius_accounting.sh`
(`sacct` export without allocation or host names); inference timing comes
from `cm_campaigns/timing.py` with synchronized repetitions. Table S15b
separates serial cost, nominal wall time and GPU-hours per rerun; the
`nominal_time` record set holds the timing samples.
Where: `docs/EXPERIMENTS.md`, `records/specs.json`.
Check: `make records-check`

## C. Statistics and decisions

**C1. Table S14 prints Holm P values for a family of nine contrasts, but the code applies Holm per dataset. Which was executed?**
The released lifecycle forms a one-sided, within-dataset family from the
selected route and eligible predecessors and applies the practical margin.
The printed S14 summary describes a joint nine-contrast, two-sided family.
These are different analyses. Code review establishes the released procedure,
not which historical procedure generated the printed numbers. Original local
families, raw contrasts, units and the S14 reporting inputs are needed to
resolve that history. The revision must not infer it from the printed adjusted
P values alone.
Where: `cross_morse/statistics.py`, `cross_morse/pipeline.py`, `docs/PAPER_SUPPORT.md`.
Check: `make decision-check DECISION_WARN_ONLY=1`

**C2. Peptides has Holm P = 0.070 in Table S14 and still appears with radius deployed. Why?**
The printed 0.070 is not interchangeable with the runtime's local one-sided
family decision. A radius deployment requires its own eligible selection,
all required confirmation rows, admission and the final policy lock. A passed
radius contrast alone is insufficient. If the verified lifecycle returns the
comparator, that return is the deployed result and the radius effect is an
evaluated-arm result. A downgrade needs its own identity and evidence. The
four outcome states remain in the planning companion until records resolve them.
Where: `scripts/check_decision_consistency.py`, `paper/MANUSCRIPT_REVIEW.md`.
Check: `make decision-check DECISION_WARN_ONLY=1`

**C3. The registered selector prefers radius on Roman, Pascal and COCO, but Table 1 deploys MPC-RH. Why?**
The representative S12 rows prefer radius under the registered reading,
while Table 1 reports MPC-RH. A different result from the complete candidate
ledger is a possible explanation, not a verified fact while the necessary
archives are missing. The complete candidate definitions, feasible masks,
upper limits, simplicity order, predecessor decisions and final lock must
reproduce MPC-RH or the paper must be corrected. A representative-row
sensitivity is retained separately from a full-ledger replay.
Where: `scripts/scientific_evidence_audit.py`, `paper/reported_results.json`.
Check: `make registry-check`

**C4. Is the separation of data roles (60/20/10/5/5) enforced in code or only described?**
Enforced. `scripts/scientific_evidence_audit.py` (`validate_split_evidence`)
requires the five training roles to partition the official training split at
the registered cuts and the official validation and test splits to be
untouched; `cross_morse/role_evidence.py` audits role isolation from the
sealed records; both run inside every preflight. A run with another
allocation is `BLOCKED`.
Where: `docs/EVIDENCE_REGISTRATION.md`, `paper/registered_constants.json` (`data_roles_table_S2`).
Check: `make scientific-audit`

**C5. Which numbers in the paper are measured and which are modelled?**
The registry and `paper/CLAIMS.md` describe reported measurements, derived
arithmetic and analytical sensitivities. Those labels do not independently
verify empirical origin. In particular, the S14/S14b measured-versus-rescaled
ambiguity requires the archived confirmation inputs. `make si-check` verifies
arithmetic against printed inputs; it cannot turn a modeled quantity into an
observation or certify that a timing was measured.
Where: `scripts/check_si_identities.py`, `docs/PAPER_SUPPORT.md`.
Check: `make si-check`

**C6. Are the theorems checked, and what does a passing check mean?**
`cm_theory/` implements every definition, lemma, proposition, theorem and
corollary; `make theory` runs 91 seeded checks on random instances of their
hypotheses, including the executable admission bound and the budget rule of
the runtime. A pass is an implementation check of the stated implication,
not a proof and not a statement that trained plants satisfy the hypotheses.
Where: `docs/THEORY.md`.
Check: `make theory`

**C7. How is admission different on the three multigraph datasets?**
Pascal, COCO and Peptides use the empirical-Bernstein bound of Eq. (27) on
independent validation graphs (`cross_morse/statistics.py`, `empirical_bernstein`);
arxiv, proteins and Roman use a descriptive clipped-loss rule on dependent
cores or official splits. Table 1 prints the regime per dataset and
`admit/decision.json` records the gate values of every run.
Where: `docs/METHOD.md`, `paper/registered_constants.json` (`admission_eq27_theorem2_table_S13`).
Check: `make theory`

## D. Models, data and compute

**D1. Are the seven backbones the ones in Table S5?**
`make model-inventory` instantiates all seven full-profile plants from the
released configurations and compares their parameter counts with Table S5 at
printed precision. Checkpoint identity for the paper's runs comes from the
archives (`train/model_inventory.json` and sealed checkpoint hashes).
Where: `docs/DATASETS.md`, `scripts/model_inventory.py`.
Check: `make model-inventory`

**D2. Where does the data come from, and is any redistributed?**
The seven datasets are downloaded from their official releases by
`cross_morse/datasets.py`; nothing is redistributed. `docs/DATASETS.md` lists
sources, licences, splits and preparation.
Check: `make registry-check`

**D3. How many seeds, and is training deterministic?**
The full profiles declare twelve plant seeds and twelve controller seeds.
Recording seeds and checking numerical tolerances does not guarantee
bit-for-bit retraining across hardware, library versions or nondeterministic
kernels. Each run must retain its effective environment and executed seeds.
Within-run parity checks are separate from across-rerun statistical variation.
Where: `paper/registered_constants.json`, `tests/test_theory_runtime_parity.py`.
Check: `make test`

## E. Current manuscript and evidence status

**E1. Are the paper results described as completed measurements?**
Yes. The author identifies the supplied paper and supplement as the completed
experimental study. The reported numerical targets are transcribed in the
registry. The separate artifact indexes say which original files are available
for reconstruction; an empty index concerns public artifact availability.
Where: `paper/RESULTS.md`, `paper/sources.json`, `AVAILABILITY.md`.
Check: `make paper-report-check`

**E2. Does the support map cover every numbered table and figure?**
Yes: the current source-bound inventory contains two main tables, six main
figures, 62 supplementary tables and 20 supplementary figures. It distinguishes
protocol specifications, mathematical derivations, reported measurements and
analytical scenarios. A listed reconstruction path does not claim that the
required original observations are already deposited.
Where: `paper/display_manifest.json`, `paper/PAPER_SUPPORT.md`.
Check: `make paper-report-check`

**E3. Which statistical qualifications remain relevant?**
The joint confirmation family, complete selector ledger, independent admission
units and the scope of the executed E6 controller remain explicit. Sample
estimates, fitted quantities and modelled cost tails retain their scientific
labels. Peptides evaluated-arm measurements remain separate from the comparator
returned by the printed confirmation rule until the original decision record
resolves that discrepancy.
Where: `paper/MANUSCRIPT_REVIEW.md`, `docs/PAPER_SUPPORT.md`, `scripts/check_decision_consistency.py`.
Check: `make decision-check`

## F. Repository hygiene, licence and citation

**F1. Licence and citation?**
BSD 3-Clause (`LICENSE`, `NOTICE`); citation metadata in `CITATION.cff` and
`codemeta.json`. Deposited archives carry their own licence declared in the
evidence plan.
Check: `make audit`

**F2. Does the repository contain private cluster details or credentials?**
`make audit` refuses allocation identifiers, login names and project-disk
paths that name an allocation; profiles with those values live outside the
checkout (`configs/arrhenius-site.env.example` holds placeholders). Tokens are
read from the environment at deposit time and never stored.
Where: `scripts/release_audit.py`.
Check: `make audit`

**F3. What does continuous integration check?**
`.github/workflows/quality.yml` runs `make verify`, replays the examples,
archives and ingests one, runs the NumPy test suite, the full theory battery
and the supplementary arithmetic report, the reviewer path on the examples
and, in a second job with CPU PyTorch, the complete test suite and one
synthetic end-to-end run. `.github/workflows/container.yml` builds the
container.
Check: `make verify`

**F4. How large are the archives, and where will they live?**
Each run archive holds checkpoints, sealed predictions and all decision
records; sizes are recorded at packing time in the private evidence state and
in `runs/index.json` after registration. Deposits go to Zenodo through
`scripts/deposit_zenodo.py` with the version DOI written into the index;
`make fetch-deposits` verifies the public bytes.
Where: `docs/RUN_ARCHIVES.md`.
Check: `make runs-check-if-present`
