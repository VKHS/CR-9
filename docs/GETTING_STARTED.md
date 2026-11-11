# Getting started

[Documentation](README.md) · [Datasets](DATASETS.md) · [HPC](HPC.md)

Run commands from the checkout root. Keep environments, downloaded data, new runs
and reports outside the checkout. The included arxiv-shaped and Peptides-shaped
examples exercise the workflow using generated inputs.

## 1. Install the recorded CPU environment

The recorded environment uses Linux x86_64 and Python 3.12. The installer checks
the host and interpreter before installing the pinned packages:

```bash
bash environment/create_cpu_env.sh --python python3.12 --venv ../cm-cpu
source ../cm-cpu/bin/activate
python -B environment/check_environment.py
```

Use a fresh environment path. The checker reports Python patch-version agreement
separately from package agreement. Details are in [environment/README.md](../environment/README.md).
GPU execution needs a host-compatible CUDA PyTorch installation; the recorded
CPU wheel is intended for local checks and examples. Follow [HPC](HPC.md) for GPU setup.

The source-tree commands do not require an editable package installation. The
supported Python minimum is declared in [pyproject.toml](../pyproject.toml).

## 2. Verify and replay the included examples

```bash
make verify PYTHON=python
make test PYTHON=python
make reproduce PYTHON=python REPORT_ROOT=../cm-reports/examples
make example-paper PYTHON=python PAPER_EXAMPLES_ROOT=../cm-reports/example-paper
```

`make verify` checks source identities, manifests, registry arithmetic and the
bundled examples. `make reproduce` recomputes saved decisions and metrics from
observations, without training or loading checkpoints. The outputs include
CSV/Markdown metrics, JSON replay records and figures. They retain their synthetic
labels. `make example-paper` exercises the table/figure builders on those runs.

To check actual checkpoint execution, use the separate PyTorch path:

```bash
make reexecute PYTHON=python RUN_ROOT=examples/runs/arxiv \
  REPORT_ROOT=../cm-reports/arxiv-checkpoints DEVICE=cpu THREADS=2
```

Checkpoint reexecution uses the preserved source identity of the historical
example. A saved-data replay and a new run of the current code answer different
questions; [Run archives](RUN_ARCHIVES.md) explains the distinction.

## 3. Select the dataset and profile

| Dataset | Wrapper/config stem | Task and benchmark metric |
|---|---|---|
| `ogbn-arxiv` | `arxiv` | Multiclass nodes; accuracy |
| `ogbn-products` | `products` | Multiclass nodes; accuracy |
| `ogbn-proteins` | `proteins` | Multilabel nodes; ROC-AUC |
| `Roman-empire` | `roman_empire` | Multiclass nodes; accuracy over declared split masks |
| `PascalVOC-SP` | `pascalvoc_sp` | Superpixel node classification; macro-F1 |
| `COCO-SP` | `coco_sp` | Superpixel node classification; macro-F1 |
| `Peptides-func` | `peptides_func` | Multilabel graph classification; average precision |

Every dataset has three configuration profiles:

| Profile | Intended use |
|---|---|
| `smoke` | Generated fixture; small CPU software run; downloads disabled |
| `pilot` | Official dataset with smaller model and training budgets; integration and resource measurement |
| `full` | Configured full workload and twelve optimization seeds |

A pilot retains the complete official graph or graph corpus, so it can still
require substantial memory. Its results correspond to the selected pilot
settings; full-profile results use their own run records. A comparator return
is a valid lifecycle outcome.

Inspect a configuration without downloading or training:

```bash
python -B run_cross_morse.py --dataset Roman-empire --profile full \
  --split-index 3 --print-config
```

`--config PATH` selects a custom JSON configuration. Any accompanying dataset or
profile arguments must agree with it. Dataset wrappers call the same shared CLI.

## 4. Execute a fresh synthetic run

```bash
python -B run_cross_morse.py --dataset ogbn-arxiv --profile smoke \
  --device cpu --threads 2 --data-root ../cm-data \
  --output-dir ../cm-runs/arxiv-smoke --stage all

python -B scripts/validate_run_manifest.py ../cm-runs/arxiv-smoke \
  --require-complete --json

python -B scripts/reproduce_results.py ../cm-runs/arxiv-smoke \
  --output ../cm-reports/arxiv-smoke-replay.json
```

Choose fresh run and single-run report paths. The smoke profiles cover all seven
task adapters even though only two historical runs are committed. To execute all
seven synthetic task paths, use `make train-samples WORK_ROOT=../cm-smokes THREADS=2`.

## 5. Prepare and run official data

Prepare a dataset without training:

```bash
make prepare-data PYTHON=python DATASET=Roman-empire PROFILE=pilot \
  DATA_ROOT=../cm-data RUN_ROOT=../cm-runs/roman-prepared DEVICE=cpu
```

This target may download missing official files. Provider locations, cache
layouts and data roles are in [Datasets](DATASETS.md). Before a GPU run, activate
the intended environment and check it:

```bash
python -B run_cross_morse.py --gpu-health-check
python -B run_cross_morse_proteins.py --profile pilot --device cuda \
  --threads 16 --data-root ../cm-data --output-dir ../cm-runs/proteins-pilot
```

The real profiles use BF16 except Roman-empire, which uses FP32. For BF16 profiles,
check both the health-check status and BF16 support. Changing the device does not
change the configured precision. Inspect the pilot's records before allocating a
full run at a fresh output path with `--profile full`.

Roman-empire accepts `--split-index 0` through `9`. The [HPC guide](HPC.md#roman-empire-campaign)
describes the complete ten-split campaign. The [Experiments guide](EXPERIMENTS.md)
covers E3–E6, baselines and measurements outside the core lifecycle.

## 6. Resume and locate outputs

Reuse the original source, configuration, dataset and output directory:

```bash
python -B run_cross_morse.py --dataset ogbn-arxiv --profile smoke \
  --device cpu --threads 2 --data-root ../cm-data \
  --output-dir ../cm-runs/arxiv-smoke --stage all --resume
```

The stages are `prepare`, `train`, `explore`, `fit`, `select`, `confirm`, `admit`
and `evaluate`. `all` runs them in order. Completed stages are verified before
reuse; changed computation source or configuration is rejected. `train` fits the
predictor, while `fit` fits controller-side models after exploration.

| Run location | Contents |
|---|---|
| `config.json`, `environment.json`, `prepared/` | Resolved settings, environment, graph data and exact split memberships |
| `train/` | Predictor checkpoints, histories and model inventory |
| `explore/`, `fit/` | Logged trajectories and fitted controller models |
| `select/`, `confirm/`, `admit/`, `policy_lock.json` | Candidate ledger, decisions and locked predictor |
| `final/` | Predictions, paired units and computed metrics |
| `provenance/stages/` | Hash-linked stage records |

Use [Run archives](RUN_ARCHIVES.md) to package a completed run and connect its
results to a paper. `make help` lists the supported commands.
