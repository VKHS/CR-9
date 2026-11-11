# Two complete software examples

[Repository](../README.md) · [Recomputed results](generated/metrics.md) · [Environment](../environment/README.md)

Two archived **synthetic** runs demonstrate the complete artifact format without
requiring a benchmark download. Together they cover node prediction on a shared
graph and multilabel prediction on independent graphs. All seven dataset adapters
and their smoke, pilot, and full configurations remain available for fresh runs.

| Task path | Archive | Inventory | Generated input |
|---|---|---|---|
| ogbn-arxiv | [arxiv](runs/arxiv) | [arxiv inventory](inventory/arxiv.md) | 160 nodes in one graph |
| Peptides-func | [Peptides](runs/peptides_func) | [Peptides inventory](inventory/peptides_func.md) | 80 graphs, eight nodes each |

Each archive retains all eight stages: prepared inputs and role IDs, two training
seeds, model inventories and checkpoints, controller fitting state, exploration
trajectories, predictions, statistical decisions, and parent-linked stage manifests.
The examples use four generated classes or labels and three training epochs.

These historical runs retain their original source identity. The exact runtime
is stored in [`source_snapshots/`](source_snapshots) for replay. They demonstrate
the recorded lifecycle and its comparator-return decisions; they do not establish
independent-role execution by the current runtime or performance on official data.
The official evidence status is stated in [AVAILABILITY.md](../AVAILABILITY.md).

## Replay the saved observations

From the repository root, in the installed environment:

```bash
make reproduce REPORT_ROOT=../cm-reports/examples
make example-paper
```

`make reproduce` verifies every declared example artifact and recomputes the
statistical decisions and scores from saved observations. It performs no training
or checkpoint loading. Its report contains an eight-row predictor table, decision
records, and a labeled synthetic score figure. Committed copies are under
[`generated/`](generated); `make example-views-check` verifies their numeric contents.

`make example-paper` rebuilds the supported table and figure formats from these
two archives. The outputs retain a synthetic-data label. They are examples of the
build process, not reconstructed benchmark results.

The two small [`records/`](records) sets demonstrate measurement-record export:
`sensor_parameters` and `nominal_time`. They include their own raw observations,
provenance, and manifests. `make example-records-check` verifies their values.

## Reexecute a checkpoint

For these included, source-identified checkpoints:

```bash
python -B scripts/evaluate_checkpoints.py examples/runs/arxiv \
  --trust-local-checkpoints --device cpu --threads 2 \
  --output ../cm-reports/arxiv-checkpoints.json
```

This runs the four recorded predictors and compares their probabilities with the
saved outputs. It requires PyTorch and explicit checkpoint trust. Keep the run
archives unchanged and write new reports outside them; artifact edits invalidate
the recorded hashes and stage ancestry.

## Execute fresh examples

```bash
make train-sample DATASET=Peptides-func \
  RUN_ROOT=../cm-fresh/peptides DATA_ROOT=../cm-data

# All seven task configurations, using generated data.
make train-samples WORK_ROOT=../cm-fresh-all
```

Fresh runs exercise the current implementation and create their own source,
configuration, split, and artifact identities. Use a new output directory for a
new run. Official datasets require the `pilot` or `full` profiles described in the
[dataset source guide](../data/README.md).
