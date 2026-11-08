# Producing the measured outputs

[Documentation](README.md) · [Publishing](PUBLICATION.md) · [HPC guide](HPC.md) · [Experiments](EXPERIMENTS.md) · [S14 confirmation](S14_CONFIRMATION.md)

This workflow produces and collects the lifecycle runs, measurement-record sets,
E3–E6 and S14 campaigns, and GPU environment listed in
[`release/paper_artifacts.json`](../release/paper_artifacts.json). Completed runs from
`bash run_cross_morse.sh reproduce` are its inputs; E12 additionally requires the
pinned external study plans and adapter checkouts described below. One command
plans the work, submits its Slurm dependency graph, and writes a verified publication
map when collection finishes:

```bash
bash hpc/produce_paper_outputs_arrhenius.sh "$CM_BASE/outputs/CROSS_MORSE_REPRODUCTION_0001" \
  --baselines /abs/pinned-baseline-study-plans
python3 -B scripts/paper_outputs.py status --work "$CM_BASE/paper-outputs/CROSS_MORSE_REPRODUCTION_0001"
# After the collect job finishes, on a login node with network access:
export GITHUB_TOKEN=...      # fine-grained, contents: write on VKHS/CR-9
bash hpc/publish_paper_artifacts_arrhenius.sh all \
  "$CM_BASE/paper-outputs/CROSS_MORSE_REPRODUCTION_0001/paper-artifacts.map.json"
```

The map names each verified artifact and records unresolved inputs by name under
`_awaiting`. Collection is `COMPLETE` only when every required path verifies.
The publication wrapper requires the complete map. For a deliberate incremental
release, the lower-level `scripts/paper_artifacts.py pack` command accepts selected
artifacts; the availability gate continues to report the scope of that release.

Newly produced artifacts receive `post_result_revision` in the map's `_relationships`.
This records their actual relationship to the manuscript. Importing the original
result-producing archives requires an explicit relationship declaration. Publishing
artifacts does not by itself establish exact agreement with the paper: reconstruction
checks display completeness, numerical comparison coverage, and the published bytes.

## What the command does

1. **Plan** (`scripts/paper_outputs.py plan`, in a short Slurm job because the runners'
   plan steps need the job environment; `CM_PRODUCE_PLAN_LOCAL=1` plans in the current
   shell). The suite's runs are discovered (`runs/NN_<dataset>`; Roman-empire is the
   campaign of its ten split runs) and verified: every stage manifest and sealed file
   (`scripts/archive_run.py`), `COMPLETE` through `evaluate`, the full scientific
   profile, and the runtime fingerprint of this checkout. Then every runner's own
   torch-free `plan` step whose inputs exist is executed, and its jobs size the node's
   Slurm array. `<work>/plan.json` lists every node with its exact command lines, its
   resources, its dependencies and its outputs, and the export of every artifact. The
   plan also pins the auxiliary producer code, configurations and contracts. Submission,
   execution and collection refuse a checkout that changed after planning.
2. **Submit** (`scripts/paper_outputs.py submit`, standard library only, on the login
   node). One batch script per node under `<work>/slurm/`, with the SITE SETTINGS of
   `run_cross_morse.sh` (`bash run_cross_morse.sh site-env`: account, partitions, GPU
   request, threads, memory, modules, environment, data root). Nodes are submitted in
   dependency order with `--dependency=afterok:` (`afterany` where a failed attempt is a
   recorded outcome), then one `collect` job after all of them. Job ids go to
   `<work>/jobs.json`. `--dry-run` prints the `sbatch` commands and submits nothing.
3. **Collect** (the final job). Every export writes its record set into
   `<work>/records/`, each verified value by value with `scripts/records.py`; the E3–E6
   campaign is assembled; the map `<work>/paper-artifacts.map.json` and
   `<work>/collect.json` are written.

## The job graph

Array sizes are the jobs of each runner's own plan; the counts below are those of the
seven full runs (Roman-empire: ten split runs).

| Node | Resource | Tasks | After | Produces |
|---|---|---|---|---|
| `input_gate.run` | GPU | one per admitted run (15) | — | `input_gate` |
| `negative_controls.run` | GPU | one per run of the declared dataset | `input_gate.run` | `negative_controls` |
| `band_audit.run` | GPU | one per run of the declared plant | — | `band_bound_tightness` |
| `inserted_site.run` | GPU | one per multigraph run (3) | — | `inserted_site` |
| `ensemble_grid.run` | GPU | one per admitted run (15) | — | `ensemble_grid`, `ensemble_grid_scores` |
| `cross_backbone.run` | GPU | the plan's substitution lifecycles | — | `cross_backbone` |
| `factorials.lifecycles`, `factorials.cells` | GPU | cell variants, then cells | lifecycles → cells | `factorials` |
| `sensitivity_grids.variants` | GPU | one per variant lifecycle | — | `sensitivity_grids` |
| `horizon_beam_sweep.variants` | GPU | one per variant lifecycle | — | `horizon_beam_sweep` |
| `buffer_sensitivity.run` | CPU | one per parent (arxiv, proteins) | — | `buffer_sensitivity` |
| `module_ablations.plan` → `.lifecycles` → `.measure` | CPU, GPU, GPU | 1; the variant lifecycles; one per run | `input_gate.run`, `negative_controls.run` | `module_ablations` |
| `runtime_scaling.run` | GPU | one per timed instance | — | `runtime_scaling` |
| `timing_anchors.run` | CPU | one per admitted dataset | `runtime_scaling.run` | `timing_anchors` |
| `mechanism_contrasts.run`, `oracle_dispersion.run` | GPU | the plans' jobs | — | `mechanism_contrasts`, `oracle_dispersion` |
| `tube_audit.run`, `depth_diagnostics.run` | GPU | one per run | — | `tube_audit`, `depth_diagnostics` |
| `s14_campaign.run` → `.assemble` | GPU, CPU | one per (run, source); 1 | run → assemble | `campaign:confirmation` |
| `confirmation.plan` → `.run` | CPU, GPU | 1; every trained seed of the three multigraph runs (tasks beyond the plan's jobs exit) | `s14_campaign.assemble` | `confirmation_contrasts`, `confirmation_sensitivity` |
| `decision_audit.plan` → `.run` | CPU, GPU | 1; one per run (16) | `input_gate.run` | `action_gap_audit`, `decision_traces`, `reliability_diagnostics` |
| `budget.parent.<run>` → `budget.studies.<run>` | GPU | 6 stages; 12 controller seeds | parent → studies | `campaign:budget`, `budget_matched` |
| `baselines.internal.run` | GPU | one per run (16; not requeued) | — | `baselines` (internal rows of S15, S15b) |
| `baselines.<study>.hpo` → `.select` → `.final.split<k>` | GPU, CPU, GPU | 48; 1; 12 | `afterany` hpo → select → finals | `baselines` |
| `environment.capture` | GPU | 1 | — | `environment:gpu` |
| `collect` | CPU | 1 | every node | the record sets and the map |

`nominal_time` and `sensor_parameters` need no job: `collect` exports them from the
sealed runs (`scripts/export_records.py from-runs`).

Plans that need another node's output are written inside a job after it: the module
ablations name the sealed input-gate and negative-control runs (comparator branch), the
decision audit reuses the frozen input gates (`--gate-from`), the negative controls run
behind the input gate of their dataset, and E10 reads the assembled S14 campaign. Their
arrays are sized at planning time from a probe of the same plan; a mismatch stops the
plan job instead of skipping work.

## Inputs the runs cannot supply

| Artifact | Default | Replace with |
|---|---|---|
| `campaign:budget`, `budget_matched` (E3–E6) | A new campaign: each admitted run's own configuration is executed again from `prepare` through `confirm` in a fresh directory (the sealed runs carry admission and test decisions, which `cm_experiments/runner.py` and `scripts/arm_campaign.py export --complete` refuse), then twelve controller seeds per run through `hpc/run_budget_arms_arrhenius.sh`; `collect` exports each study's terminal `sacct` record (`hpc/finalize_arm_accounting.sh`) and writes the campaign manifest. Needs a clean git checkout (the campaign names its source commit). | `--budget-campaign DIR` (a complete export) |
| `campaign:confirmation` (S14) | `scripts/build_s14_campaign.py plan\|run\|assemble`; `assemble` stops the graph when `check_campaign` refuses the campaign | `--confirmation-campaign DIR` |
| `baselines` (E12, E17) | awaiting | `--baselines DIR`: the pinned-upstream study plans (`cm-upstream-study-1`) with the upstream and adapter checkouts they name beside them (`source_directory` and `adapter_directory` resolve against each plan file's directory), optional `timing/*.csv` (E17 rows of the external methods) and `accounting/*.psv`; or a finished set with `--artifact records:baselines=DIR`. With `--baselines`, the internal rows of Tables S15 and S15b are measured on the runs (`baselines.internal.run`, `scripts/run_internal_baselines.py`) |
| `environment:gpu` | a GPU job runs `environment/arrhenius/capture_environment.sh` | `--environment DIR` |
| any artifact | — | `--artifact ID=DIR` (verified by `collect`) |

Scheduler accounting is exported by `collect` inside its job, after the jobs it accounts
for are terminal: the E17 timing allocations (`hpc/capture_arrhenius_accounting.sh`, bound
to the timing rows by `runtime_scaling` export), each E3–E6 study, and every E12 attempt and
internal-baselines job.

The `baselines` export (`scripts/export_records.py baselines`) receives the E12 studies, the
internal-baselines output (`--internal`) and the `--baselines DIR` as `--checkouts`. For each of
the fourteen external comparisons it writes the validated study plan and `upstream.json`
(upstream repository URL from the source checkout's `origin`, source and adapter commits, the
adapter archive's SHA-256) to `raw/<dataset>/<method>/`, and archives the adapter checkout at its
pinned commit (`git archive`) to `code/adapters/<dataset>/<method>/adapter-<commit>.tar`; the
upstream third-party source is referenced by URL and commit, never copied. Each checkout's commit
must equal the plan's. The internal rows are the per-seed scores and GPU-hours of every internal
route and the E17 serial cost and wall time of q_C and the deployed rule; their timing rows must
bind to completed GPU allocations of the captured `sacct` export. A set the record audit does not
accept as complete (internal rows of the seven datasets, the fourteen comparisons with their study
code) is refused (`--allow-partial` writes it to another root only).

## Declared choices

The manuscript does not fix these; the plan records them under `declared_choices`.

| Choice | Default | Option |
|---|---|---|
| Dataset of the Table S19 negative controls and the Figure S16d stress reference | PascalVOC-SP | `--negative-controls-dataset` |
| Plant and nominal baseline of the Figure S18c bound tightness | ogbn-arxiv, `shared` (A0) | `--band-audit-dataset`, `--band-audit-baseline` |
| E3–E6 controller-optimization seeds | 0–11 (the campaign template) | `--arm-seed` (twelve times) |
| Roman-empire run of `nominal_time` and `sensor_parameters` | `split_00` (as the E17 timing records) | — |
| Internal rows of Tables S15 and S15b: route configurations, scores, timing inputs, GPU-hours | the definitions of `cm_experiments/internal_baselines.py` (recorded in its plan and in the set's provenance); E17 warm-up 2 and 5 measured repeats | `scripts/run_internal_baselines.py plan --warmup/--repetitions/--graph-quantiles` (outside this workflow) |
| Rule of the action-gap audit | the deployed rule | `--provisional DATASET` (a second audit of q_s) |

## Following and recovering

```bash
python3 -B scripts/paper_outputs.py status --work WORK          # sacct/squeue, else the task markers
python3 -B scripts/paper_outputs.py submit --work WORK --resume # after fixing a failure: unfinished nodes only
python -B scripts/paper_outputs.py collect --work WORK           # again, e.g. on a login node with NumPy
python -B scripts/paper_outputs.py collect --work WORK --only input_gate
```

Every task writes `<work>/state/<node>/task-<k>.json`. A runner that stops at a safe
boundary before the time limit (exit 75) requeues its job (`CM_MAX_CONTINUATIONS`) and
resumes; group-B runs and E3–E6 studies are not resumable, so a failed attempt is moved
to `<dir>.attempt-<n>` and the task starts again in a fresh directory. A failed node
cancels its dependents (`--kill-on-invalid-dep=yes`) and the collect job; `submit
--resume` checks both live queue and accounting state, retains active jobs, and retries
terminal unfinished jobs. If scheduler state is unavailable, it stops before submitting
a duplicate. Completion receipts bind the plan, node definition and task-specific
output seals; a changed or missing seal is incomplete. Submissions, collection and
individual tasks use separate process locks. `collect` exports only what verifies; an export
whose nodes have not completed stays `awaiting` (`--ignore-markers` tries it anyway).

Settings beyond the SITE SETTINGS: `CM_PRODUCE_GPU_TIME` (default `CM_TIME` or
`CM_TIME_FULL`), `CM_PRODUCE_CPU_TIME` (04:00:00), `CM_PRODUCE_COLLECT_TIME` (24:00:00),
`CM_PRODUCE_PLAN_TIME` (02:00:00), `CM_PRODUCE_ARRAY_PARALLEL` (array throttle). CPU
nodes use `CM_REPORT_PARTITION` and `CM_REPORT_GPU_REQUEST`; set the latter empty where
the site allows CPU-only jobs.

## Work directory

```text
<work>/plan.json  jobs.json  collect.json  paper-artifacts.map.json
<work>/slurm/<node>.sbatch, slurm/logs/      state/<node>/task-<k>.json
<work>/experiments/<experiment>/             the runners' plan and job directories
<work>/budget/parents/, budget/studies/, budget/campaign.json
<work>/campaigns/budget/, campaigns/confirmation/   accounting/   environment/gpu/
<work>/records/<set>/                        verified record sets (records/index.json layout)
```

## Without Slurm, and software checks

The nodes are ordinary commands: `python -B scripts/paper_outputs.py execute --work WORK
--node NODE --task K` runs one task in any allocation, in the order of `plan.json`, and
`collect` needs no scheduler (the accounting exports then fail and their artifacts stay
awaiting). `--allow-fixture` plans a smoke suite (`bash run_cross_morse.sh reproduce
smoke`) end to end as a software check; its outputs are synthetic and `pack` refuses them.
`--only SET ...` plans a subset with the nodes it needs.

## Model and prediction delivery

The E12 exporter runs with both `--copy-studies` and `--copy-predictions`. Its record
asset includes each complete sealed external study, its per-file inventory, required
final checkpoints and prediction files, and the pinned adapter source. The small
`experiments/baselines/` view committed by publication contains the plans, adapter
sources and study inventories, with hash-bound links to the downloadable study
archives. Checkpoints remain in release assets rather than the Git history.
