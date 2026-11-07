# Reproduce the paper on a Slurm cluster

[Documentation](README.md) · [Getting started](GETTING_STARTED.md) · [Arrhenius walkthrough](ARRHENIUS_SESSION.md) · [Run archives](RUN_ARCHIVES.md) · [Reviewer guide](REVIEWER_GUIDE.md)

`run_cross_morse.sh` executes the registered dataset lifecycles on a Slurm
GPU cluster. One command submits an environment job, the
sealed eight-stage lifecycle of all seven datasets (Roman-empire on its ten
official splits) and a report job that rebuilds the paper's run-derived tables
from the regenerated runs and places every regenerated value beside the value
printed in the paper. The launcher is the only file that knows about the
cluster; everything it runs is the released Python code, whose fingerprint it
checks before submitting and which every run seals.

Auxiliary experiments use the separate drivers in [Experiments](EXPERIMENTS.md).
After their records are registered, [the reconstruction command](RECONSTRUCTION.md)
combines available run outputs and supplementary analyses into one workspace.
Its per-display report identifies columns and partial panels; completing the
dataset lifecycles does not by itself reconstruct every printed display.

```bash
git clone https://github.com/VKHS/CR-9.git && cd CR-9
$EDITOR run_cross_morse.sh            # SITE SETTINGS: account, partition, GPU request, storage, modules
bash run_cross_morse.sh check         # settings, runtime identity and job plan; creates and submits nothing
bash run_cross_morse.sh reproduce     # setup, seven lifecycles, report
bash run_cross_morse.sh status        # progress of the latest suite
```

## 1. Site settings

The SITE SETTINGS block at the top of `run_cross_morse.sh` is the only part
that differs between clusters. Its defaults are the authors' setting on the
NAISS Arrhenius cluster (one NVIDIA GH200 96 GB per job, aarch64 nodes); the
allocation and the storage project are left empty.

| Setting | Arrhenius default | Purpose |
|---|---|---|
| `CM_ACCOUNT` | empty (required) | Slurm account passed to `sbatch --account` |
| `CM_PARTITION` | `gpu` | GPU partition of every job |
| `CM_GPU_REQUEST` | `--gpus=1` | how the site requests one GPU, e.g. `--gres=gpu:1` |
| `CM_THREADS` | `16` | CPU threads per job (`--cpus-per-task`) |
| `CM_MEM` | empty | memory per job (`--mem`); per dataset `CM_MEM_PRODUCTS` and so on |
| `CM_SBATCH_EXTRA` | empty | further options for every job, e.g. `--qos=... --constraint=...` |
| `CM_STORAGE_PROJECT` | empty | Arrhenius project disk; sets `CM_BASE` to `/nobackup/proj/disk/<project>/cross_morse` |
| `CM_BASE` | `$HOME/cross_morse_work` without a storage project | suites, logs, caches, datasets and the environment |
| `CM_BUILD_MODULE`, `CM_PYTHON_MODULE` | `GPU/buildenv-nvhpc/25.9-cu13.0`, `GPU/Python/3.13.5-bare-gcc-2025b-eb` | modules loaded by every job; empty keeps the job's environment |
| `CM_EXPECT_ARCH` | `aarch64` | compute-node architecture the jobs require; empty accepts any |
| `CM_VENV_NAME` | `venv_arrhenius_gh200` | environment created once by the setup job under `CM_BASE/code` |
| `CM_TORCH_SPEC`, `CM_TORCH_INDEX` | `torch==2.12.1+cu130`, PyTorch `cu130` index | wheel the setup job installs when no PyTorch is importable |
| `CM_TIME_FULL`, `CM_TIME_PILOT`, `CM_TIME_SMOKE` | `24:00:00`, `04:00:00`, `01:00:00` | limit of one run job; per dataset `CM_TIME_PRODUCTS` and so on |
| `CM_MAX_CONTINUATIONS` | `12` | checkpointed requeues of one run at its time limit |
| `CM_MAX_INFRA_REQUEUES` | `3` | requeues after a GPU or driver failure, excluding the failed node |
| `CM_REPORT_PARTITION`, `CM_REPORT_GPU_REQUEST` | the run settings | where the report job runs; empty GPU request for a CPU-only job |
| `CM_PACK_ARCHIVES` | `0` | `1` also packs every run as a sealed, shareable `.tar.gz` archive |

There are two equivalent ways to set them: edit the block, or keep the
launcher byte-identical to the release and put the values in a profile,
`export CM_ENV_FILE=/absolute/profile.env`. `configs/public.env.example` is the
template for another cluster; `configs/arrhenius-site.env.example` holds the two
Arrhenius identifiers for a git-ignored private copy,
`configs/arrhenius.env.example`, which the launcher finds by itself. Precedence
is command-line routing, then exported `CM_*` variables, then the profile, then
the SITE SETTINGS block; an explicitly empty value is kept.
`bash run_cross_morse.sh site-env` prints the effective values, and the
auxiliary Slurm scripts under `hpc/` read the same settings through it.

## 2. Check

```bash
bash run_cross_morse.sh check            # full profile; `check pilot` or `check smoke` for the others
```

`check` runs on the login node and creates nothing. It reports:

- the runtime fingerprint of `run_cross_morse*.py` and `cross_morse/*.py`
  against the released `0aba361b54f6561f407f097af257cf27e4bf7f85d1ff5f47a4bcdd8bcda3a4fb`;
- every configuration the profile will use, with its seeds, epochs and precision;
- `sbatch`, the account, the partition (`sinfo`), the per-job request, the
  storage base and its free space, the environment and the modules;
- the complete job plan with each job's limit and dependency, and the GPU-hours
  the paper reports per dataset.

Each line is `ok`, `warn` or `FAIL`; any `FAIL` makes the exit status nonzero.
A different runtime fingerprint is a failure: a reproduction uses the released
runtime unchanged. `CM_ALLOW_SOURCE_CHANGE=1` runs a modified runtime, and the
suite records itself as a variant.

## 3. Reproduce

```bash
bash run_cross_morse.sh reproduce            # full profile, all seven datasets
bash run_cross_morse.sh reproduce smoke      # synthetic data: the whole orchestration in minutes
bash run_cross_morse.sh reproduce pilot      # official data, reduced compute
bash run_cross_morse_peptides_func.sh reproduce full           # one dataset
CM_REPRODUCE_DATASETS="ogbn-arxiv,COCO-SP" bash run_cross_morse.sh reproduce
```

`reproduce` first freezes a numbered suite under `$CM_BASE/outputs`: every
run directory receives a hashed snapshot of the source, the resolved
configuration and the effective settings, and the Roman-empire campaign one
common snapshot for its ten splits. Only then are the jobs submitted:

| Job | Dependency | What it does |
|---|---|---|
| `setup` | none | creates the environment once (idempotent), installs the PyTorch wheel and `requirements.txt`, checks the architecture and runs the CUDA health check |
| one per dataset; Roman-empire as an array of ten | `afterok` setup | loads the modules, verifies the snapshot, records modules, packages, `nvidia-smi` and the node, runs the GPU health check and the eight lifecycle stages |
| `report` | `afterany` every run | waits for any run that was requeued, collects scheduler accounting and the ordered logs, and writes the report and the results ZIP |

```text
$CM_BASE/outputs/CROSS_MORSE_REPRODUCTION_0001/
  000_MANIFEST/   suite.json, plan.tsv, job_ids.tsv, run_registry.tsv, suite.env, WHERE_ARE_MY_FILES.txt
  source/         the release tree the setup and report jobs use (hash inventory in 000_MANIFEST)
  runs/01_arxiv/  02_proteins/  03_products/  04_roman_empire/split_00..09/  05_pascalvoc_sp/  06_coco_sp/  07_peptides_func/
  999_REPORT/     comparisons/, paper/tables, paper/figures, manifests/, logs/, scheduler_accounting.psv
  REPRODUCTION_REPORT.md   SUITE_STATUS.json   CMR_0001_RESULTS.zip
```

### Compute

The paper reports the GPU time on one GH200 per dataset (Supplementary Tables
S5, S15b and S41):

| Dataset | Plant training per seed, GPU-h | Qualification increment, GPU-h | Deployed method per rerun, GPU-h (S15b) |
|---|---:|---:|---:|
| ogbn-arxiv | 1.8 | 1.92 | 12.8 |
| ogbn-proteins | 2.4 | 2.01 | 13.4 |
| ogbn-products | 4.8 | 2.28 | 15.2 (q_C returned) |
| Roman-empire | 0.45 | 1.38 | 9.2 |
| PascalVOC-SP | 1.1 | 1.59 | 10.6 |
| COCO-SP | 1.3 | 1.81 | 12.1 |
| Peptides-func | 0.9 | 1.69 | 11.3 |

A job limit is the length of one allocation, not of the run: a run job that
reaches its limit checkpoints and requeues itself, so the default 24-hour
limit completes every dataset. The report records the GPU-hours actually used.

## 4. Follow and recover

```bash
bash run_cross_morse.sh status                        # the latest suite
bash run_cross_morse.sh status /abs/suite             # a given suite, with each job's queue state
tail -n 100 -F "$CM_BASE/logs/CROSS_MORSE_LIVE.out" "$CM_BASE/logs/CROSS_MORSE_LIVE.err"
```

Every job mirrors its output into the live pair; each job also writes
`$CM_BASE/logs/CMR_<n>_<run>_<job>.out/.err`, and each run keeps
`launcher/stdout.log` and `launcher/stderr.log`. `000_MANIFEST/run_registry.tsv`
lists every transition with its job, array task, restart count and node.

| Event | What the launcher does | Record |
|---|---|---|
| Time limit | Slurm sends USR1 180 s before the limit; the runtime stops at its next safe boundary (within `CM_CHECKPOINT_GRACE`), the job requeues itself, and the run continues from its last sealed stage | `REQUEUE_CONTINUATION`; `launcher/continuation_count` |
| GPU or driver failure | a failed health check or a CUDA/driver error in this attempt excludes the node and requeues the job | `REQUEUE_INFRASTRUCTURE`; `launcher/infrastructure_count` |
| Requeue not permitted by the site | a continuation job is submitted after this one | `continuation` in `job_ids.tsv` |
| Python exception | the run stops; the runtime writes `failure_summary.json` | `FAILED_RUNTIME`; `launcher/terminal.json` |
| Budgets exhausted | the run is recorded as incomplete | `INCOMPLETE`, `TIME_BUDGET` or `INFRASTRUCTURE` |

`bash run_cross_morse.sh reproduce resume /abs/suite` resubmits the runs of a
suite that are neither complete nor queued; completed stages are verified and
reused. `bash run_cross_morse.sh report /abs/suite` rebuilds the report in a
short job. Earlier attempts' records are kept under `launcher/attempts/`.

| Exit code | Meaning |
|---|---|
| 0 | complete, or requeued to continue |
| 2 | usage or settings |
| 33 | prerequisite: modules, interpreter, architecture or CUDA |
| 70 | a launcher step failed |
| 74 | the runtime returned without completing the lifecycle |
| 75 | checkpointed; the continuation budget is exhausted |
| 76 | GPU, driver or node failure; the requeue budget is exhausted |
| other | the Python runtime's own exit status |

## 5. The report

`REPRODUCTION_REPORT.md` states the profile, the runtime fingerprint and
whether every run sealed it, the execution environment, the GPU-hours from
Slurm accounting and the state of every planned run. For the full profile it
places each dataset's deployed policy and paired loss effect beside main
Table 1 and summarises `scripts/compare_with_manuscript.py`, which compares
every quantity it can read from a run (Tables 1, S12c, S13, S14, S15 and S20) at
three levels: exact at the printed precision, inside the printed 95% interval,
or outside. The per-dataset comparisons are in `999_REPORT/comparisons/`.

Before analysis, the report verifies the frozen report tools and run-source
snapshots against their inventories. A moved suite reads its own copied source
and runs. Completed-run identities must agree with the suite and sealed stage
records; historical example outputs cannot be relabeled as a new execution.

`SUITE_STATUS.json` records `report_status`, `report_errors` and the explicit
`report_scope`. The Python report command exits `0` for `PASS`, `1` for an
unfinished suite (`INCOMPLETE`) and `2` for failed analysis or verification
(`FAIL`). A valid numerical disagreement with the paper remains visible in the
comparison and is distinct from a failed analysis command. Successful report
construction covers the requested lifecycles and their supported outputs.

The report job also rebuilds the paper's run-derived tables (and figures)
from the regenerated runs with `scripts/build_paper.py`, embeds Tables 1 and 2,
and extracts each run's data-role membership, seeds and environment with
`scripts/run_manifests.py`. `CMR_<n>_RESULTS.zip` holds the report, the
manifests, the comparisons, the accounting and every run's decision records,
without checkpoints or prediction arrays; with `CM_PACK_ARCHIVES=1` each run is
also packed as a sealed archive (`scripts/archive_run.py`) that
`scripts/archive_run.py --verify` and `scripts/ingest_run.py` accept.

`bash hpc/produce_paper_outputs_arrhenius.sh <suite>` produces every other measured
output from the suite's runs as one job graph ([Producing the measured outputs](PRODUCTION.md)),
and `bash hpc/publish_paper_artifacts_arrhenius.sh all <work>/paper-artifacts.map.json`
publishes the runs with the record sets and campaigns as the assets of this
repository's release `paper-artifacts`, from which `make paper` rebuilds every table
and figure ([Publishing the measured outputs](PUBLICATION.md)).

On `reproduce resume`, the launcher restores the saved suite settings while
preserving explicit environment overrides. It queries the scheduler before
resubmission, leaves active jobs alone and submits only unfinished Roman split
indices. An unavailable scheduler query stops resubmission so active work is not
duplicated. These recovery paths are covered by the simulated-Slurm tests.

## Single runs and Roman-empire campaigns

The same launcher runs one dataset, profile or split:

```bash
bash run_cross_morse_proteins.sh dry-run pilot          # print the plan; no writes
bash run_cross_morse_proteins.sh submit setup           # the environment, once
bash run_cross_morse_proteins.sh submit pilot
bash run_cross_morse.sh --dataset Roman-empire --split-index 3 submit full
bash run_cross_morse_roman_empire.sh submit campaign full /abs/new-campaign
```

To resume a run, use its captured source and profile:

```bash
CM_SUITE_PATH=/abs/existing-suite
CM_ENV_FILE="$CM_SUITE_PATH/source/configs/active.env.example" \
CM_PROJECT_DIR="$CM_SUITE_PATH/source" \
bash "$CM_SUITE_PATH/source/run_cross_morse.sh" submit resume "$CM_SUITE_PATH"
```

Source, configuration, dataset and split identities and completed stages are
verified; a lock prevents concurrent writers. `CM_STAGE=prepare` stops a new
run after data preparation; `CM_STAGE=all` continues it. Inside an existing
GPU allocation the modes run without `submit`. For direct `sbatch`, pass the
account, partition, GPU request and limit yourself.

After a campaign, `python -B scripts/summarize_roman_splits.py CAMPAIGN --output
REPORT.json` writes the paired split summary; the reproduction report runs it
for you. The ten masks share one graph, so the split interval is descriptive.

## Another scheduler

The runtime and the report tools do not depend on Slurm. One lifecycle is one
call of a dataset entry point, which `--resume` continues after an
interruption:

```bash
python -B run_cross_morse_arxiv.py --profile full --device cuda \
  --data-root /abs/data --output-dir /abs/runs/arxiv [--resume]
python -B run_cross_morse_roman_empire.py --profile full --split-index 3 --device cuda \
  --data-root /abs/data --output-dir /abs/runs/roman_empire/split_03 [--resume]
make compare RUN_ROOT=/abs/runs/arxiv REPORT_ROOT=/abs/reports
python -B scripts/build_paper.py all --runs-root /abs/runs --label mine --output-dir /abs/paper
```

`run_cross_morse.sh` adds the suite, the job graph and the recovery around
these calls through `sbatch`, `squeue`, `sacct` and `scontrol`; replacing those
calls ports the orchestration to another scheduler.

## Environment

The jobs use `$CM_VENV_DIR/bin/python`; activating another environment in the
login shell does not select it. The setup job creates the environment with the
module's Python if it is missing, moves an incomplete one aside, installs the
PyTorch wheel only when none is importable, then `requirements.txt` and
Matplotlib, runs `pip check`, checks the architecture and runs the CUDA
sparse/dense health check. Every run records its own `environment.json`,
package list, module list and `nvidia-smi` output.
`environment/arrhenius/capture_environment.sh` captures the same information
outside a run. BF16 profiles require BF16 support.

## Experiments outside the lifecycle

The budget-matched arms (E3–E6), the retrained external baselines and the E17
timing protocol have their own Slurm scripts under `hpc/`; they read the same
SITE SETTINGS through `run_cross_morse.sh site-env`.
`bash hpc/produce_paper_outputs_arrhenius.sh <suite>` submits every auxiliary
experiment as one job graph ([Producing the measured outputs](PRODUCTION.md)).
[Experiments](EXPERIMENTS.md) gives their inputs and commands, and
`hpc/capture_arrhenius_accounting.sh` exports the scheduler accounting of any
job list without account or node names.
