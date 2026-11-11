# Reproduce the paper on Arrhenius

[Documentation](README.md) · [HPC guide](HPC.md) · [Reviewer guide](REVIEWER_GUIDE.md) · [Experiments](EXPERIMENTS.md)

The paper's experiments ran on the NAISS Arrhenius cluster, one NVIDIA GH200
96 GB per job. `run_cross_morse.sh` ships with that setting in its SITE SETTINGS
block: partition `gpu`, `--gpus=1`, 16 CPU threads, aarch64 nodes, the modules
`GPU/buildenv-nvhpc/25.9-cu13.0` and `GPU/Python/3.13.5-bare-gcc-2025b-eb`, the
persistent environment `venv_arrhenius_gh200` with the `torch==2.12.1+cu130`
wheel, and the project-disk layout. This page is the complete session on
Arrhenius; the [HPC guide](HPC.md) explains every step for any other cluster.

## 1. Clone and add the allocation

Log in to an Arrhenius login node, change to the project disk and clone:

```bash
cd /nobackup/proj/disk/<storage project>
git clone https://github.com/VKHS/CROSS-MORSE.git
cd CROSS-MORSE
cp configs/arrhenius-site.env.example configs/arrhenius.env.example   # git-ignored
$EDITOR configs/arrhenius.env.example
```

Fill the two values from the project's NAISS page: `CM_ACCOUNT`, the GPU
allocation (the project id with its `-gpu` suffix), and `CM_STORAGE_PROJECT`,
the project that owns the disk. Suites, logs, caches, datasets and the
environment then live under `/nobackup/proj/disk/<storage project>/cross_morse`.
The launcher finds the private copy by itself and stays byte-identical to the
released file.

## 2. Check the setting

```bash
bash run_cross_morse.sh check
```

Expect `ok` for the runtime fingerprint (`9f7393f7…`, the released runtime),
the seven full-profile configurations, `sbatch`, the account, the `gpu`
partition, the project disk and the two modules, and the plan of nine jobs:
setup, seven runs (Roman-empire as an array of ten) and the report. The
environment is reported as absent until the first setup job has run. Nothing
is created or submitted.

## 3. Exercise the cluster path in minutes

```bash
bash run_cross_morse.sh reproduce smoke
bash run_cross_morse.sh status
```

The smoke profile uses synthetic data with the real orchestration: the setup
job builds the environment once (the step that takes longest the first time),
then every dataset and all ten Roman-empire splits pass through the eight
stages, and the report job writes `REPRODUCTION_REPORT.md`. Its numbers carry no
scientific meaning; it confirms modules, environment, GPU access, logs and the
Slurm chain before the full study is spent.

## 4. Regenerate the paper

```bash
bash run_cross_morse.sh reproduce
tail -n 100 -F /nobackup/proj/disk/<storage project>/cross_morse/logs/CROSS_MORSE_LIVE.out
bash run_cross_morse.sh status
```

The datasets download from their official providers into the shared cache on
first use. Jobs that reach the 24-hour limit checkpoint and requeue themselves;
a GH200 that fails its health check is excluded and the job requeued. When a run
needs attention, `status` names it and `launcher/stderr.log` in its directory
holds the reason; `bash run_cross_morse.sh reproduce resume <suite>` resubmits
what is unfinished.

## 5. Read and keep the result

The report job writes, in the suite directory:

- `REPRODUCTION_REPORT.md`: every dataset's deployed policy and paired loss
  effect beside main Table 1, the comparison levels of every quantity the
  comparison tool reads, the rebuilt Tables 1 and 2, the GPU-hours from Slurm
  accounting and the state of every run;
- `SUITE_STATUS.json`: the same in machine-readable form;
- `CMR_<n>_RESULTS.zip`: report, manifests, comparisons, rebuilt tables,
  accounting and every run's decision records, ready to copy off the cluster.

`CM_PACK_ARCHIVES=1 bash run_cross_morse.sh report <suite>` additionally packs
each run as a sealed archive with its checkpoints for sharing.

## 6. Experiments outside the lifecycle

The budget-matched arms (E3–E6), the retrained external baselines and the E17
timing protocol use `hpc/run_budget_arms_arrhenius.sh`,
`hpc/run_upstream_baseline_arrhenius.sh` and
`hpc/capture_arrhenius_accounting.sh`. They read the same SITE SETTINGS through
`run_cross_morse.sh site-env`; [Experiments](EXPERIMENTS.md) lists their inputs.
