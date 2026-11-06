# Recomputed example results

Synthetic software examples; not official benchmark scores or manuscript reproduction.

Every value below is recomputed from the sealed predictions and labels. Selection, confirmation and admission are also replayed. Metric scores are fractions in [0, 1], including AP; loss is in nats.

| Task configuration | Metric | q0 | qN | qC | qD | qD − qC loss | Admission |
|---|---|---:|---:|---:|---:|---:|---|
| ogbn-arxiv | accuracy | 0.156250 | 0.406250 | 0.281250 | 0.281250 | +0.00000000 | comparator retained |
| Peptides-func | ap | 0.787494 | 0.545922 | 0.561291 | 0.561291 | +0.00000000 | comparator retained |

## How to interpret the examples

Generated fixtures have 4 classes or labels and reduced training budgets. They demonstrate transductive node and independent-graph task paths, numerical review and checkpoint replay against their archived runtime. Historical role assignments do not establish fresh validation of the current separated-role implementation. They do not estimate performance on the named official datasets.

`make paper` requires separately deposited official run archives. This report does not compare synthetic scores with the manuscript registry.

`metrics.csv` contains all 8 predictor rows, losses, paired effects and source/configuration/unit identities. `replay.json` retains every recomputed decision and verified numeric input. The original probability arrays, splits, checkpoint files and trajectories remain in the input run directories.
