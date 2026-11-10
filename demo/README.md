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
   on those runs (`scripts/paper_outputs.py plan --allow-fixture --device cpu`) and executes its first
   dependency level on sixteen runners ([execute.py](execute.py), the same `execute` entry point as
   the Slurm workflow; tasks are shared among runners by interleaving the unfinished ones).
3. [demo-continue.yml](../.github/workflows/demo-continue.yml) executes the later levels and the
   exports. The work directory outgrows one runner's disk, so every job restores from the earlier
   jobs' archives only the outputs its own tasks depend on ([restore.py](restore.py)); the producer
   code is pinned to the commit the graph was planned at, which `execute` and `collect` verify.
4. [demo-grid.yml](../.github/workflows/demo-grid.yml) completes the one task longer than a runner's
   six hours: the PascalVOC-SP job of E14 trains eighteen additional plants. Its unfinished seeds
   train on several runners with the job's own training call ([grid_seeds.py](grid_seeds.py)), and
   the graph task then finds them complete and computes the member predictions.
5. [demo-sweep.yml](../.github/workflows/demo-sweep.yml) plans and runs the Table S29 sweep as a
   graph of its own (`--only horizon_beam_sweep`, below).
6. [demo-publish.yml](../.github/workflows/demo-publish.yml) adds the sweep's record set to the main
   map ([merge_maps.py](merge_maps.py)), packs every artifact on its own
   ([pack_each.py](pack_each.py), `scripts/paper_artifacts.py pack --demo`), publishes the release
   `demo-artifacts` (`publish --demo`), rebuilds every display from the release (`make demo-paper`),
   installs the views under [published/](published/) (`derive --demo`) and writes
   [RESULTS.md](RESULTS.md).

The same steps run on any machine with the recorded CPU environment:

```bash
python -B demo/data.py --dataset ogbn-arxiv --data-root DATA --output GRAPHS/arxiv_demo.npz
python -B demo/configure.py --variant demo --dataset ogbn-arxiv --graph-npz GRAPHS/arxiv_demo.npz --output CONFIGS
python -B run_cross_morse.py --config CONFIGS/arxiv_demo.json --dataset ogbn-arxiv --data-root DATA \
    --output-dir RUNS/arxiv --device cpu --threads 4
# ... every dataset; Roman-empire: --split-index 0..9 into RUNS/roman_empire/split_NN, no subset
python -B scripts/paper_outputs.py plan --runs RUNS --work WORK --allow-fixture --device cpu --data-root DATA \
    --s14-exclude Roman-empire --horizon-sweep-profile full --horizon-sweep-ensemble 1
python -B demo/execute.py --work WORK --parallel 2
python -B scripts/paper_outputs.py collect --work WORK
# Table S29, as a graph of its own
python -B scripts/paper_outputs.py plan --runs RUNS --work WORK_S29 --allow-fixture --device cpu --data-root DATA \
    --only horizon_beam_sweep --horizon-sweep-profile full --horizon-sweep-ensemble 1 --horizon-sweep-beam 16
python -B demo/execute.py --work WORK_S29 && python -B scripts/paper_outputs.py collect --work WORK_S29
python -B demo/merge_maps.py WORK/paper-artifacts.map.json WORK_S29/paper-artifacts.map.json
```

On one machine no task needs to be split: `demo/execute.py` runs the long E14 job like any other.

## Declared demonstration choices

The production graph records these in its plan (`declared_choices`):

- **S14 family without Roman-empire** (`--s14-exclude Roman-empire`). The joint S14 family needs
  the ten Roman-empire split runs to share one route, and it cannot join benchmark and fixture
  parents. The ten demonstration splits selected different routes, and Roman-empire is the only
  dataset run on its complete official graph, so the family's test jobs run on the six reduced
  datasets. The campaign is not assembled from them: `assemble_s14` refuses a partial family and
  `check_campaign` admits only a non-synthetic campaign of all seven datasets
  (`cm_campaigns/s14_builder.py`, `cm_campaigns/confirmation.py`).
- **Swept configuration of Table S29** (`--horizon-sweep-profile full --horizon-sweep-ensemble 1
  --horizon-sweep-beam 16`). The demonstration's PascalVOC-SP run deploys no MPC-RH rule, so the
  sweep names the configuration it sweeps. That run was planned with the pilot planner (beam 4,
  horizons 1 and 4), which holds neither the registered horizon 16 nor the registered beam width
  16, so the sweep holds the registered beam width: each of its seven points is a variant
  lifecycle of the run's configuration, and no replicate of the run's own configuration is
  planned.

## What the demonstration cannot show

[RESULTS.md](RESULTS.md) lists every artifact the demonstration does not hold, with the production
graph's reason. By design they need the author's resources or benchmark runs: the fourteen
external comparisons of E12 (`records:baselines`: the pinned upstream studies and adapters the
author supplies), the E17 timing records (`records:runtime_scaling`, `records:timing_anchors`:
Slurm accounting of GPU allocations, which hosted runners do not have; the repository does not
emulate scheduler records), the S14 campaign and the confirmation record sets that read it
(`campaign:confirmation`, `records:confirmation_contrasts`, `records:confirmation_sensitivity`;
above) and a GPU environment (`environment:gpu` holds the capture of the CPU runner).
