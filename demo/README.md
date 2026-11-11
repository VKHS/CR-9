# Published execution package

This directory documents CROSS-MORSE lifecycle runs, measurement records and
campaigns, together with their checkpoints, predictions, propensity-logged
trajectories, split and seed manifests, environment records and statistical
outputs. It also documents reconstruction of the displays supported by those inputs.

**Recorded execution scope.** The artifacts were produced by this repository's
code on GitHub-hosted CPU runners using the exact dataset selections and settings
listed below: all ten Roman-empire splits and specified subsets of the other
official datasets. The existing [release manifest](../release/demo_artifacts.json)
and [derived views](published/) retain their original identities and download links.

Measurements and manuscript-comparison outcomes are preserved. The package command
verifies available inputs and renders supported displays; successful rendering
alone does not establish complete agreement with the manuscript. The strict
`make paper` and availability checks retain their acceptance rules.

## Verify and reconstruct the published package

```bash
make demo-paper        # verify existing release assets and invoke the display generators
make demo-status       # inspect the package inventory
```

Existing command and release identifiers are retained for compatibility. The
package command calls `scripts/reconstruct_paper.py --demo` and writes its workspace
to `../cm-demo-paper`. Machine-readable classifications and data memberships remain
unchanged; a dataset subset is not relabelled as a full benchmark execution.

Read [RESULTS.md](RESULTS.md) and the generated status report for input coverage,
complete display content and manuscript comparisons. Package-command success is
not the strict full-manuscript acceptance test.

## What the release holds

The release `demo-artifacts` holds 31 of the 37 artifacts the paper's manifest names (38.55 GiB,
28,913 files): the seven lifecycle runs, 22 record sets, the E3-E6 campaign and the environment
capture of the CPU runner. On it, `make demo-paper` builds 56 of the 90 displays, 35 from the
release and 21 from the repository's own sources; 1,263 printed values are equal, 1,202 differ and
248 are missing. The other 34 displays wait for an input the package does not hold or a cell
its runs do not reach (for instance, the ogbn-arxiv subset run deploys no feedback rule with a
positive gain over q_C); [RESULTS.md](RESULTS.md) lists each with what it lacks.

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

1. [demo-runs.yml](../.github/workflows/demo-runs.yml) builds the dataset subsets and runs each
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
5. Four groups of artifacts come from graphs of their own, each planned for its artifacts alone
   (`--only`) in its own work directory, because the main graph was planned before the code paths
   below existed: the Table S29 sweep ([demo-sweep.yml](../.github/workflows/demo-sweep.yml), below),
   the E3-E6 campaign, the E17 timing records, and the decision-log and mechanism record sets
   ([demo-graph.yml](../.github/workflows/demo-graph.yml) with `plan_options`). On the CPU device
   the planner runs each E3-E6 study through `scripts/run_budget_study.py` (the study command of the
   Slurm batch script, with the same job evidence, recording that no scheduler allocation exists),
   and in a fixture plan every exporter that can write the measured part of an incomplete set does
   so (`--allow-partial`, never into `records/`), the set stating the cells it lacks.
6. [demo-publish.yml](../.github/workflows/demo-publish.yml) adds those graphs' artifacts to the main
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

## Recorded execution choices

The production graph records these in its plan (`declared_choices`):

- **S14 family without Roman-empire** (`--s14-exclude Roman-empire`). The joint S14 family needs
  the ten Roman-empire split runs to share one route, and it cannot join benchmark and fixture
  parents. The ten package splits selected different routes, and Roman-empire is the only
  dataset run on its complete official graph, so the family's test jobs run on the six dataset
  subsets. The campaign is not assembled from them: `assemble_s14` refuses a partial family and
  `check_campaign` admits only a non-synthetic campaign of all seven datasets
  (`cm_campaigns/s14_builder.py`, `cm_campaigns/confirmation.py`).
- **Swept configuration of Table S29** (`--horizon-sweep-profile full --horizon-sweep-ensemble 1
  --horizon-sweep-beam 16`). The package's PascalVOC-SP run deploys no MPC-RH rule, so the
  sweep names the configuration it sweeps. That run was planned with the pilot planner (beam 4,
  horizons 1 and 4), which holds neither the registered horizon 16 nor the registered beam width
  16, so the sweep holds the registered beam width: each of its seven points is a variant
  lifecycle of the run's configuration, and no replicate of the run's own configuration is
  planned.

## Coverage and input requirements

[RESULTS.md](RESULTS.md) lists every artifact the package does not hold, with the production
graph's reason. By design they need the author's resources or benchmark runs: the fourteen
external comparisons of E12 (`records:baselines`: the pinned upstream studies and adapters the
author supplies), the S14 campaign and the confirmation record sets that read it
(`campaign:confirmation`, `records:confirmation_contrasts`, `records:confirmation_sensitivity`;
above) and a GPU environment (`environment:gpu` holds the capture of the CPU runner). The
action-gap audit (`records:action_gap_audit`) scores the decisions of a closed-loop deployed rule;
every package run deployed an open-loop rule, so the audit has no decision to score and the
exporter writes no set. Fifteen of the sixteen measurement jobs of the S17 module ablations
(`records:module_ablations`) completed; the remaining job reads the sealed variant lifecycle runs of
its parent, each opened and verified through its confirm stage with its prediction arrays, and
restoring them filled a hosted runner's disk twice. The module-ablation exporter writes only the
complete set.

Several sets are present in a declared partial form. The E3-E6 campaign (`campaign:budget`,
`records:budget_matched`) is the partial campaign export: its studies ran on fixture parents
without scheduler allocations, so `cm_experiments/campaign.export_campaign` records each study's
qualification blockers (software fixture, missing accounting) and the record set is marked
synthetic. The E17 timing records (`records:runtime_scaling`, `records:timing_anchors`) time the
deployed rule on CPU without GPU allocations; each set records why its measurement is incomplete
(`incomplete_measurement`). The mechanism contrasts and the decision-log sets hold the cells the
package's runs measure; a cell without a valid measurement (e.g. a state contrast without
matched donors) is listed in the set instead of a value. The repository does not emulate scheduler
records.
