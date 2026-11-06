# HPC, Slurm and Arrhenius

[Documentation](README.md) · [Getting started](GETTING_STARTED.md) · [Run archives](RUN_ARCHIVES.md)

The shared launcher runs one Python process per Slurm task, with one GPU and 16
CPU threads by default. Dataset wrappers use the same launcher. Each suite stores
a source/config snapshot, effective cluster profile and launcher logs. Keep data,
environments and suites on shared project storage visible from compute nodes.

## Select a cluster profile

For Arrhenius GH200 nodes, copy the supplied profile outside the checkout:

```bash
mkdir -p "$HOME/cross_morse_profiles"
cp configs/arrhenius-site.env.example "$HOME/cross_morse_profiles/arrhenius.env"
export CM_ENV_FILE="$HOME/cross_morse_profiles/arrhenius.env"
$EDITOR "$CM_ENV_FILE"
bash run_cross_morse_arxiv.sh dry-run pilot
```

Fill in `CM_ACCOUNT` and `CM_STORAGE_PROJECT` from the allocation you are using.
The profile records the `gpu` partition, `aarch64` architecture check and project
storage layout. For another cluster, start with `configs/public.env.example`.
The profiles are trusted sourced Bash files.

| Setting | Purpose |
|---|---|
| `CM_ACCOUNT`, `CM_PARTITION` | Slurm allocation and partition |
| `CM_BASE` | Shared work root for outputs, logs and caches |
| `CM_VENV_DIR` | Environment whose `bin/python` will execute the job |
| `CM_DATA_ROOT` | Shared official raw-data cache |
| `CM_THREADS` | Numerical threads; must not exceed allocated CPUs |
| `CM_EXPECT_ARCH` | Optional host/interpreter architecture requirement |
| `CM_BUILD_MODULE`, `CM_PYTHON_MODULE` | Optional modules loaded by the launcher |
| `CM_STAGE` | `all` or an individual lifecycle stage |
| `CM_TIME` | Wrapper walltime override |
| `CM_TORCH_SPEC`, `CM_TORCH_INDEX` | Explicit setup-only PyTorch package/index |

Routing flags override corresponding settings; exported `CM_*` variables override
the selected profile. `CM_ENV_FILE` selects one complete profile. Without it the
launcher checks its configured source locations for `active.env.example`,
`arrhenius.env.example` and `public.env.example`. The private Arrhenius copy is
ignored by Git. `dry-run` prints the effective plan without submitting or creating
outputs. Check the printed paths and account before submission.

## GPU environment

The job uses `$CM_VENV_DIR/bin/python`; activating another login-shell environment
does not select it. Use a compute-node-compatible CUDA PyTorch environment with
[requirements.txt](../requirements.txt). If it is already ready, skip setup.
Otherwise set the intended CUDA wheel explicitly in the profile and run:

```bash
bash run_cross_morse_arxiv.sh submit setup
```

Setup installs missing dependencies inside a GPU job. It preserves an importable
PyTorch installation and then checks CUDA operations; it does not guess a
replacement wheel. Normal experiment modes do not install packages. A CPU-only
wheel is unsuitable for these GPU jobs. BF16 profiles additionally require BF16
support. Empty module settings retain the job's current module environment.

Inside an allocated GPU shell, record the observed environment:

```bash
bash environment/arrhenius/capture_environment.sh \
  /absolute/path/to/gpu-environment/bin/python ../cm-gpu-environment
```

The capture writes the package lock, environment details, GPU information and
module list. Each run also records its own `environment.json`.

## Submit and inspect a run

```bash
bash run_cross_morse_proteins.sh dry-run pilot
bash run_cross_morse_proteins.sh submit pilot
```

Inspect the pilot's memory, timing and output records before requesting a full
profile:

```bash
bash run_cross_morse_proteins.sh submit full
```

To stop a new suite after data preparation, set `CM_STAGE=prepare` for submission.
Then resume that suite with `CM_STAGE=all`. Default wrapper requests are one hour
for setup/smoke, four hours for pilot and 24 hours for full; these are allocation
limits, not predicted completion times. Slurm smoke uses a GPU; use the direct
Python smoke command for a local CPU check.

The submission output identifies the suite and logs. Run locations are recorded
explicitly; do not choose a directory by modification time when jobs overlap.
Profile variables loaded inside the launcher do not automatically become
variables in the interactive shell.

## Resume

Use the captured source and profile of the existing suite:

```bash
CM_SUITE_PATH="/absolute/path/to/existing-suite"
CM_ENV_FILE="$CM_SUITE_PATH/source/configs/active.env.example" \
CM_PROJECT_DIR="$CM_SUITE_PATH/source" \
bash "$CM_SUITE_PATH/source/run_cross_morse.sh" submit resume "$CM_SUITE_PATH"
```

Source/config/dataset/split identities and completed stages are verified. Keep the
captured source unchanged. The suite lock prevents concurrent writers. Slurm is
asked to signal 180 seconds before the time limit; Python checkpoints at a safe
boundary when possible. An interrupted operation may not reach that boundary in
time. Requeue eligibility is enabled, while actual requeue behavior depends on
the scheduler. Explicit resume remains available.

## Roman-empire campaign

A single split uses `--split-index 0` through `9`. For the full ten-mask protocol,
leave any single-split override unset and choose a fresh campaign path:

```bash
CM_CAMPAIGN_ROOT="/absolute/path/to/new-roman-campaign"
bash run_cross_morse_roman_empire.sh submit campaign full "$CM_CAMPAIGN_ROOT"
```

The launcher freezes a common source/config before submitting array tasks 0–9.
Each task writes `split_00` through `split_09`. To resume the same campaign:

```bash
CM_ENV_FILE="$CM_CAMPAIGN_ROOT/source/configs/active.env.example" \
CM_PROJECT_DIR="$CM_CAMPAIGN_ROOT/source" \
bash "$CM_CAMPAIGN_ROOT/source/run_cross_morse.sh" --dataset Roman-empire \
  submit campaign resume "$CM_CAMPAIGN_ROOT"
```

After all ten complete, write a report outside the campaign:

```bash
python -B scripts/summarize_roman_splits.py "$CM_CAMPAIGN_ROOT" \
  --output ../cm-reports/roman-summary.json
```

The paired split summary is descriptive for overlapping masks on one graph.

## Preserve the completed evidence

Use [Run archives](RUN_ARCHIVES.md) for sealing, packing and depositing the suite.
For completed official runs, [evidence registration](EVIDENCE_REGISTRATION.md)
provides a private plan and `hpc/pack_evidence_arrhenius.sh` to run preflight and
packing as a Slurm array. This job does not train new models or publish deposits.
The [Arrhenius evidence-registration guide](ARRHENIUS_SESSION.md) adds
`hpc/discover_evidence_arrhenius.sh` for locating complete original runs, private
Slurm log routing, and the preflight, packing and registration sequence.
Capture the original terminal Slurm accounting with
[`hpc/capture_arrhenius_accounting.sh`](../hpc/capture_arrhenius_accounting.sh).
Process duration, allocation GPU-hours and inference wall time are different
quantities; [Experiments](EXPERIMENTS.md) describes their required records.

Check the target environment with the GPU health check and pilot before a full
campaign. [Validation](VALIDATION.md) lists the software checks.
