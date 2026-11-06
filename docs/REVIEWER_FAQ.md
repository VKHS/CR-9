# Reviewer questions about this repository

[Documentation](README.md) · [Reviewer guide](REVIEWER_GUIDE.md) · [Reproduce the paper](HPC.md) · [Claims ledger](../paper/CLAIMS.md) · [Availability](../AVAILABILITY.md) · [Paper support](../paper/PAPER_SUPPORT.md)

Each answer names the files that hold the fact and the command that checks it.
`make faq-check` confirms that every referenced file and make target exists.

## A. Provenance and identity

**A1. Which code does a reproduction run, and how is it identified?**
The runtime (`run_cross_morse*.py` and `cross_morse/*.py`) has one SHA-256
fingerprint, `0aba361b54f6561f407f097af257cf27e4bf7f85d1ff5f47a4bcdd8bcda3a4fb`,
recorded in `release/release_scope.json` and `docs/VALIDATION.json`.
`run_cross_morse.sh check` and `reproduce` compute it before anything is
submitted and refuse a different runtime unless it is explicitly labelled a
variant; every run writes the fingerprint it ran with into `run_state.json` and
its sealed stage manifests, and every suite snapshots the source it used.
Where: `scripts/source_fingerprint.py`, `scripts/reproduction.py`.
Check: `make fingerprint`

**A2. Which manuscript version does the registry transcribe?**
`paper/sources.json` identifies the main text and supplement by SHA-256, size
and page count and maps each table to its pages. The PDFs are not in the
repository. `paper/reported_results.json` holds the printed values,
`paper/registered_constants.json` the protocol constants, and
`paper/transcription_check.json` binds the contextual source review to the PDF
bytes and to both registry files. A new PDF is registered through a reviewed
source registration (`make manuscript-sources`, then
`make register-manuscript-sources`), never by overwriting numbers.
Where: `paper/README.md`, `scripts/register_manuscript_sources.py`.
Check: `make registry-check`

**A3. Why does the Git history contain a single commit?**
Each public release is published as one commit. Identity does not depend on
commit ids: `RELEASE_MANIFEST.json` and `SHA256SUMS.txt` list every released
file with its hash, the runtime fingerprint identifies the code, and each
record in `docs/VALIDATION.json` names the fingerprint it validated.
Check: `make verify-manifest`

**A4. Are the bundled example runs real experiments?**
No. `examples/runs/arxiv` and `examples/runs/peptides_func` are synthetic
(`"dataset": "synthetic"` in their `config.json`); they exercise the complete
eight-stage lifecycle so that replay, archiving, table building and review can
be checked without benchmark data. Every comparison with the paper refuses them.
Where: `examples/README.md`, `scripts/release_audit.py`.
Check: `make reproduce`

## B. Reproducing the results

**B1. Where are the raw results behind Table 1?**
They are produced by running the release. This is a code release: it
distributes the implementation, the configurations, the launcher and the
registry of reported values, and no run outputs. `run_cross_morse.sh
reproduce` regenerates every run, and each run directory holds its raw
observations: prepared data and role membership, checkpoints, exploration
trajectories, candidate and final predictions, the selection, confirmation and
admission records and the stage manifests (`select/candidates.json`,
`confirm/decision.json`, `admit/decision.json`, `final/results_summary.json`).
Where: `AVAILABILITY.md`, `docs/HPC.md`.
Check: `make availability-check`

**B2. How do I regenerate all results on my cluster?**
Edit the SITE SETTINGS block of `run_cross_morse.sh` (account, partition, GPU
request, storage, modules, PyTorch wheel), run `bash run_cross_morse.sh check`,
then `bash run_cross_morse.sh reproduce`. The launcher freezes a numbered suite
and submits an environment job, the lifecycle of all seven datasets (Roman-empire
as an array of ten splits) and a report job. `reproduce smoke` exercises the
same orchestration on synthetic data in minutes.
Where: `docs/HPC.md`, `docs/ARRHENIUS_SESSION.md`.
Check: `bash run_cross_morse.sh check`

**B3. How much compute does a full reproduction take?**
The paper reports the GPU time on one GH200 per dataset: plant training per
seed and the qualification increment (Supplementary Tables S5 and S41) and the
GPU-hours per rerun of each method (Table S15b); `check` prints them beside the
job plan, and the HPC guide tabulates them. A job limit only sets the length of
one allocation: run jobs that reach it checkpoint and requeue themselves. The
report records the GPU-hours the suite actually used, from Slurm accounting.
Check: `bash run_cross_morse.sh check full`

**B4. My cluster is not Arrhenius. What do I change?**
Only the SITE SETTINGS block, or a profile selected with `CM_ENV_FILE`
(`configs/public.env.example` is the template), which keeps the launcher
byte-identical. Typical changes: `CM_GPU_REQUEST="--gres=gpu:1"`, empty
`CM_BUILD_MODULE`/`CM_PYTHON_MODULE` to keep the job's environment, empty
`CM_EXPECT_ARCH` on x86_64 nodes, a CUDA wheel matching the node driver in
`CM_TORCH_SPEC`/`CM_TORCH_INDEX`, and `CM_SBATCH_EXTRA` for QoS or constraints.
`site-env` prints the effective values; the auxiliary scripts under `hpc/` read
them too.
Check: `bash run_cross_morse.sh site-env`

**B5. What happens when a job reaches its time limit or a GPU fails?**
Slurm warns the job 180 s before its limit; the runtime stops at its next safe
boundary and the job requeues itself, continuing from the last sealed stage
(at most `CM_MAX_CONTINUATIONS` times). A failed GPU health check or a CUDA or
driver error excludes the node and requeues the job (`CM_MAX_INFRA_REQUEUES`).
A Python exception stops the run and the runtime writes its failure summary. Every event is
recorded in the suite's run registry and the run's launcher records; the report
job waits for requeued runs, and `reproduce resume` continues a suite.
Where: `run_cross_morse.sh`, `tests/test_reproduction_launcher.py`.
Check: `bash run_cross_morse.sh status`

**B6. How do I know whether a regenerated run agrees with the paper?**
`scripts/compare_with_manuscript.py` reads every quantity it can from a run
(Tables 1, S12c, S13, S14, S15 and S20) and reports each at one of three
levels: exact at the printed precision, inside the printed 95% interval, or
outside. The reproduction report runs it for every dataset, puts each deployed
policy and paired effect beside main Table 1 and embeds Tables 1 and 2 rebuilt
from the regenerated runs by `scripts/build_paper.py`.
Check: `make compare RUN_ROOT=/abs/run REPORT_ROOT=/abs/reports`

**B7. Can I run the method without a cluster?**
Yes, on a CPU: `make env-cpu` installs the recorded environment, `make smoke`
runs a synthetic lifecycle, and `make train-sample DATASET=Peptides-func` trains
one small profile end to end. Docker and Apptainer definitions reproduce the CPU
environment. The full profiles need a GPU.
Where: `docs/GETTING_STARTED.md`, `Dockerfile`, `environment/apptainer.def`.
Check: `make train-sample DATASET=Peptides-func`

**B8. Which software versions are used?**
The GPU environment is the SITE SETTINGS recipe: the cluster's NVHPC and Python
modules and the `torch==2.12.1+cu130` wheel on Arrhenius, plus
`requirements.txt`. The setup job records `pip freeze` of the environment, and
every run records its own `environment.json`, package list, module list and
`nvidia-smi` output. The CPU examples use the pinned `environment/requirements-cpu.lock`,
compared with an environment by `environment/check_environment.py`.
Check: `make environment-check`

**B9. Are the external baselines reproducible?**
The fourteen dataset/method pairs run through a pinned upstream harness
(`cm_experiments/upstream.py`, `scripts/run_upstream_baseline.py`,
`hpc/run_upstream_baseline_arrhenius.sh`) that records the upstream and adapter
commits, all HPO attempts and the final seeds. Their measurements enter the
`baselines` record set, whose raw-file locators are rechecked by
`make records-check`; `make degeneracy-check` inspects the records for repeated
values.
Where: `docs/EXPERIMENTS.md`, `records/specs.json`.
Check: `make records-check`

**B10. How are GPU-hours and timings measured?**
The reproduction report sums allocation GPU-hours per run from `sacct` (without
account or node names). Inference timing comes from `cm_campaigns/timing.py`
with synchronized repetitions after warm-up; Table S15b separates serial cost,
nominal wall time and GPU-hours per rerun, and the `nominal_time` record set
holds the timing samples. `hpc/capture_arrhenius_accounting.sh` exports the
accounting of any job list.
Where: `docs/EXPERIMENTS.md`, `records/specs.json`.
Check: `make records-check`

## C. Statistics and decisions

**C1. Table S14 prints Holm P values for a family of nine contrasts, but the code applies Holm per dataset. Which was executed?**
Both, for different purposes. The lifecycle decides deployment on its own
dataset: it tests the contrasts of the selected route and its eligible
predecessors one-sided inside that family and requires the practical margin
(`cross_morse/statistics.py`, `confirmation`; `cross_morse/pipeline.py`).
Table S14 reports the nine prespecified contrasts across datasets as one joint,
two-sided Holm family, a reporting calculation on the nine raw P values that
does not enter the lifecycle decision. `docs/S14_CONFIRMATION.md` specifies the
joint family and its checker; every run records its local family in
`confirm/decision.json`.
Check: `make decision-check DECISION_WARN_ONLY=1`

**C2. Peptides has Holm P = 0.070 in Table S14 and still appears with radius in Table 1. Why?**
The paper states it: the radius values are reported as an evaluated arm, and
under the registered confirmation map the lifecycle returns q_C for Peptides.
The registry encodes exactly that (`registered_confirmation_pass=false`,
`lifecycle_resolution=return_qC`, `evaluation_status=evaluated_arm`), and
`paper/RESULTS.md` shows the evaluated arm and the registered final rule side by
side. A regenerated run records its own decision in `confirm/decision.json`,
and the reproduction report shows the deployed policy beside the paper's row.
Where: `scripts/check_decision_consistency.py`, `paper/reported_results.json`.
Check: `make decision-check DECISION_WARN_ONLY=1`

**C3. The registered selector prefers radius on Roman, Pascal and COCO, but Table 1 deploys MPC-RH. Why?**
Table 1 reports both readings, as the paper explains: the policy the executed
selection chose on the complete candidate ledger (MPC-RH) and the choice of the
registered reading on the representative rows of Table S12c (radius, within
epsilon_eq = 0.002 of MPC-RH and simpler). The representative rows are a subset
of the ledger. A regenerated run holds the complete ledger and the deterministic
selector trace in `select/candidates.json` and `select/decision.json`, and the
reproduction report shows which route it deployed.
Where: `scripts/scientific_evidence_audit.py`, `paper/reported_results.json`.
Check: `make registry-check`

**C4. Is the separation of data roles (60/20/10/5/5) enforced in code or only described?**
Enforced. `cross_morse/split_contract.py` declares the protocol
`fit60-explore20-select10-confirm5-compare5-v1` with its cuts and requires
nonempty, disjoint roles, with D_compare reserved before any fitting; official
validation is the admission role and official test the evaluation role.
`cross_morse/role_evidence.py` audits role isolation from the sealed records, and
the pipeline locks policy-changing stages once test labels have been read. Each run's prepared role membership is extracted by
`scripts/run_manifests.py` into the reproduction report.
Where: `docs/METHOD.md`, `paper/registered_constants.json`.
Check: `make test-no-torch`

**C5. Which numbers in the paper are measured and which are modelled?**
`paper/CLAIMS.md` and `paper/PAPER_SUPPORT.md` label every manuscript display:
measurement, measurement plus derived, derived arithmetic, or analytical
sensitivity. Tables S14b, S34, S42, S44 and S47 are analytical sensitivities
computed from registered formulas; `make si-check` recomputes the derived
supplementary entries from their printed inputs.
Where: `scripts/check_si_identities.py`, `docs/PAPER_SUPPORT.md`.
Check: `make si-check`

**C6. Are the theorems checked, and what does a passing check mean?**
`cm_theory/` implements every definition, lemma, proposition, theorem and
corollary; `make theory` runs seeded checks on random instances of their
hypotheses, including the executable admission bound and the budget rule of the
runtime. A pass is an implementation check of the stated implication; the
proofs are in the paper.
Where: `docs/THEORY.md`.
Check: `make theory`

**C7. How is admission different on the three multigraph datasets?**
Pascal, COCO and Peptides use the empirical-Bernstein bound of Eq. (27) on
independent validation graphs (`cross_morse/statistics.py`,
`empirical_bernstein`); arxiv, proteins and Roman use the descriptive
clipped-loss rule on dependent cores or official splits. Table 1 prints the
regime per dataset and `admit/decision.json` records the gate values of every run.
Where: `docs/METHOD.md`, `paper/registered_constants.json`.
Check: `make theory`

## D. Models, data and seeds

**D1. Are the seven backbones the ones in Table S5?**
`make model-inventory` instantiates all seven full-profile plants from the
released configurations and compares their parameter counts with Table S5 at
printed precision. Each run additionally seals its checkpoints and
`train/model_inventory.json`.
Where: `docs/DATASETS.md`, `scripts/model_inventory.py`.
Check: `make model-inventory`

**D2. Where does the data come from, and is any redistributed?**
The seven datasets are downloaded from their official releases by
`cross_morse/datasets.py` into a shared cache, with file locks for concurrent
jobs; nothing is redistributed. `docs/DATASETS.md` lists sources, licences,
splits and preparation.
Check: `make registry-check`

**D3. How many seeds, and is training deterministic?**
The full profiles declare twelve plant seeds and twelve controller seeds
(`configs/arxiv_full.json` and its siblings). The launcher fixes the
deterministic cuBLAS workspace and disables TF32, and every run records the
seeds it executed. Retraining on other hardware or library versions reproduces
the results within their statistical intervals rather than bit for bit, which
is why the comparison reports exact, interval and outside separately.
Where: `paper/registered_constants.json`, `tests/test_theory_runtime_parity.py`.
Check: `make test`

## E. Release scope

**E1. Why are no run archives included?**
The release is the means to regenerate them: the full archives of seven
datasets hold every checkpoint, trajectory and prediction, and the launcher
regenerates them from the public datasets with the registered seeds, then
compares them with the paper. A suite's results ZIP is the shareable summary,
and `CM_PACK_ARCHIVES=1` packs each run as a sealed archive that
`scripts/archive_run.py --verify` and `scripts/ingest_run.py` accept.
Where: `docs/RUN_ARCHIVES.md`, `runs/README.md`.
Check: `make availability-check`

**E2. Does the availability statement match the repository?**
`AVAILABILITY.md` and the manuscript paragraph `paper/AVAILABILITY_STATEMENT.txt`
are generated from the repository's actual contents and the declared release
scope (`release/release_scope.json`); the checks fail when either text drifts
from the tree.
Check: `make availability-statement-check`

**E3. What is the manuscript review in `paper/`?**
`paper/MANUSCRIPT_REVIEW.md` records the contextual review that compared the
registry with the PDFs, and `paper/transcription_check.json` binds it to the
PDF bytes and both registry files. It establishes transcription, not the
experiments themselves.
Check: `make paper-report-check`

## F. Repository hygiene, licence and citation

**F1. Licence and citation?**
BSD 3-Clause (`LICENSE`, `NOTICE`); citation metadata in `CITATION.cff` and
`codemeta.json`.
Check: `make audit`

**F2. Does the repository contain private cluster details or credentials?**
No. `make audit` refuses allocation identifiers, login names and project-disk
paths that name an allocation. The SITE SETTINGS block leaves the allocation
empty, the private Arrhenius profile is git-ignored, and the results ZIP of a
reproduction redacts the account. Deposit tokens are read from the environment
and never stored.
Where: `scripts/release_audit.py`, `configs/arrhenius-site.env.example`.
Check: `make audit`

**F3. What does continuous integration check?**
`.github/workflows/quality.yml` runs `make verify`, replays the examples,
archives and ingests one and builds its tables, runs the NumPy test suite
(including the launcher on a mock Slurm cluster), the full theory battery and
the supplementary arithmetic report, the reviewer path on the examples and, in
a second job with CPU PyTorch, the complete test suite and one synthetic
end-to-end run. `.github/workflows/container.yml` builds the container.
Check: `make verify`

**F4. How is the launcher tested without a cluster?**
`tests/test_reproduction_launcher.py` runs `run_cross_morse.sh` against mock
`sbatch`, `squeue`, `scontrol`, `sacct` and `srun` with a stub GPU interpreter:
the frozen suite and job graph, the environment job, all seven lifecycles and
the Roman array, the report, time-limit continuation, refused requeue, node
exclusion after a GPU failure, runtime failures and resumption.
`tests/test_reproduction.py` covers the report and its comparison with the paper.
Check: `make test-no-torch`
