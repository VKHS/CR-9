# CPU demonstration

This directory produces and documents a complete, separately published demonstration of
the CROSS-MORSE pipeline: the seven lifecycle runs, the measurement-record sets and
campaigns that the production graph derives from them, their checkpoints, predictions,
propensity-logged trajectories, split and seed manifests, observed environment and
statistical outputs, and every table and figure of the paper rebuilt from them with one
command.

**It is not the paper's evidence.** Every artifact was produced by this repository's own
code on public benchmark data, on GitHub-hosted CPU runners, at reduced scale. The values
are not the paper's measurements and are not expected to equal the printed values. The
demonstration has its own release (`demo-artifacts`), its own manifest
([release/demo_artifacts.json](../release/demo_artifacts.json)) and its own views
([demo/published/](published/)); the paper's manifest refuses demonstration artifacts, and no
status in [STATEMENT_COMPLIANCE.md](../STATEMENT_COMPLIANCE.md) depends on it (its separate
section reports what the demonstration shows for each item).

## Rebuild every table and figure from the demonstration

```bash
make demo-paper        # download and verify the release demo-artifacts, rebuild all 90 displays
make demo-status       # what the release holds
```

`make demo-paper` runs `scripts/reconstruct_paper.py --demo`: the same verification, display
generators and printed-value comparison as `make paper`, on the demonstration's manifest. The
workspace (`../cm-demo-paper`) is labelled as the demonstration and its report opens with the
scope above. [RESULTS.md](RESULTS.md) records what the published demonstration contains.

## Scale

| Dataset | Data | Why |
|---|---|---|
| Roman-empire | complete official graph, all ten official splits | small enough for the complete lifecycle on a 4-CPU runner |
| ogbn-arxiv | the nodes of one part of an 8-way METIS partition, every induced edge | the 24-trajectory exploration and the repeated selection statistics on the complete graph exceed a runner's time limit |
| ogbn-products | one part of a 128-way METIS partition | as above; the complete graph also exceeds a runner's memory |
| ogbn-proteins | for each official role (species), the part of a 96-way METIS partition holding most of its nodes | a single part can miss a species, and with it an official role |
| PascalVOC-SP, COCO-SP | the first 480 / 60 / 60 graphs of the official train / validation / test splits | as above |
| Peptides-func | the first 1,200 / 150 / 150 graphs | as above |

[data.py](data.py) loads the official data with the runtime's own adapters (download
authorized) and writes each subset as the trusted raw graph NPZ that the run's configuration
names in `data.graph_npz`; the run records the NPZ's SHA-256, and `<stem>_demo.json` beside it
records the rule, the sizes and the official source files. The runtime admits a benchmark run
only on the complete official graph, so a subset runs as a declared fixture routed to its
benchmark (`dataset: synthetic`, `benchmark_dataset: <dataset>`, `profile: smoke`). Every
run uses the dataset's pilot settings ([configure.py](configure.py): 30 plant epochs, two
plant seeds, 24 exploration trajectories, 500 bootstrap draws) with the fp32 plant that CPU
execution requires.

## How it was produced

1. [demo-runs.yml](../.github/workflows/demo-runs.yml) builds the reduced graphs and runs each
   dataset's eight stages (`run_cross_morse.py --device cpu`), Roman-empire as its ten split
   runs. Each sealed run is kept as a workflow artifact.
2. [demo-graph.yml](../.github/workflows/demo-graph.yml) plans the repository's production graph
   on those runs (`scripts/paper_outputs.py plan --allow-fixture --device cpu`), executes it
   level by level ([execute.py](execute.py), the same `execute` entry point as the Slurm
   workflow), collects the record sets, campaigns and environment, packs them
   (`scripts/paper_artifacts.py pack --demo`), publishes the release `demo-artifacts`
   (`publish --demo`), rebuilds every display from the release (`make demo-paper`), installs the
   views under [published/](published/) (`derive --demo`) and writes [RESULTS.md](RESULTS.md).

The same steps run on any machine with the recorded CPU environment:

```bash
python -B demo/data.py --dataset ogbn-arxiv --data-root DATA --output GRAPHS/arxiv_demo.npz
python -B demo/configure.py --variant demo --dataset ogbn-arxiv --graph-npz GRAPHS/arxiv_demo.npz --output CONFIGS
python -B run_cross_morse.py --config CONFIGS/arxiv_demo.json --dataset ogbn-arxiv --data-root DATA \
    --output-dir RUNS/arxiv --device cpu --threads 4
# ... every dataset; Roman-empire: --split-index 0..9 into RUNS/roman_empire/split_NN, no subset
python -B scripts/paper_outputs.py plan --runs RUNS --work WORK --allow-fixture --device cpu --data-root DATA
python -B demo/execute.py --work WORK --parallel 2
python -B scripts/paper_outputs.py collect --work WORK
```

## What the demonstration cannot show

[RESULTS.md](RESULTS.md) lists every artifact the demonstration did not produce, with the
production graph's reason. Three need the author's resources by design: the fourteen external
comparisons of E12 (`records:baselines`: the pinned upstream studies and adapters the author
supplies), the E17 timing records (`records:runtime_scaling`, `records:timing_anchors`: Slurm
accounting of GPU allocations, which hosted runners do not have; the repository does not
emulate scheduler records) and a GPU environment (the demonstration records its CPU
environment).
