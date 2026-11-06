# Datasets and task semantics

[Documentation](README.md) · [Getting started](GETTING_STARTED.md) · [Method](METHOD.md)

The seven adapters share the CLI and lifecycle. Official datasets are downloaded
from their providers or read from an existing raw-data cache; they are not
redistributed as benchmark evidence in this repository. Source URLs and metadata
are registered in [data/sources.json](../data/sources.json).

## Tasks and full-profile configuration

| Dataset | Execution unit | Metric | Width / control depth | Native family |
|---|---|---|---|---|
| ogbn-arxiv | One transductive graph | Accuracy | 512 / 256 | Four-layer encoder, shared update |
| ogbn-products | One transductive graph | Accuracy | 512 / 64 | Three-layer encoder, shared update |
| ogbn-proteins | One transductive graph | ROC-AUC | 512 / 128 | Edge-weighted shared update |
| Roman-empire | One graph per official mask | Accuracy | 384 / 256 | Separate ego/neighbor maps |
| PascalVOC-SP | One independent graph | Macro-F1 | 128 / 128 | Six GPS-style native blocks |
| COCO-SP | One independent graph | Macro-F1 | 128 / 128 | Six GPS-style native blocks |
| Peptides-func | One independent graph | Average precision | 128 / 64 | Eight linear-GINE-style native blocks |

Each run emits its actual checkpoint inventory. All full profiles register twelve
optimization seeds.
The real profiles use BF16 except Roman-empire, which uses FP32. See each
[`configs/*_full.json`](../configs/) for optimizer, patience, exploration and
ensemble settings. `pilot` reduces computation on the actual dataset; `smoke`
uses generated fixtures and remains labelled synthetic.

Multiclass tasks use softmax and cross-entropy. Multilabel tasks use independent
sigmoids and masked binary cross-entropy; fully unlabelled samples do not become
zero-loss observations. Peptides pools within each graph and scores graph labels.
Superpixel tasks predict node labels but retain complete-graph evaluation and
resampling units. Official metrics are computed with their task-specific evaluators.

## Model parameter inventory

Each released count is the exact number of parameters in one predictor
constructed by the corresponding `configs/*_full.json` at the listed
input/output dimensions. The full profiles set explicit internal widths and
layer counts so their totals agree with the rounded benchmark counts in Table
S5. These settings define the released architectures; count agreement alone
does not identify a unique historical benchmark model. Archived configurations
and checkpoint inventories establish that identity.

`hidden_dim` is the propagated-state width and final encoder output width;
`encoder_hidden` sizes earlier encoder layers. GPS blocks use
`feedforward_hidden`; linear-GINE blocks use `block_mlp_hidden` and
`block_mlp_layers`. A positive `head_hidden` selects a two-layer prediction head.
`make model-inventory` reports the actual model class, each named parameter and
the component subtotals that produce these counts.

<!-- BEGIN MODEL PARAMETER INVENTORY -->
| Dataset | Table S5 (M) | Released full profile | Inputs / outputs | Explicit internal widths |
|---|---:|---:|---:|---|
| ogbn-arxiv | 3.1 | 3,116,632 | 128 / 40 | 4-layer encoder, width 1040; linear head |
| ogbn-proteins | 3.52 | 3,517,068 | 8 / 112 | 4-layer encoder, width 1140; linear head |
| ogbn-products | 3.36 | 3,358,895 | 100 / 47 | 3-layer encoder, width 1472; linear head |
| Roman-empire | 2.52 | 2,524,946 | 300 / 18 | 4-layer encoder, width 896; linear head |
| PascalVOC-SP | 3.14 | 3,140,277 | 14 / 21 | 6 GPS blocks, feed-forward 1584; head 640 |
| COCO-SP | 3.18 | 3,178,737 | 14 / 81 | 6 GPS blocks, feed-forward 1584; head 640 |
| Peptides-func | 5.1 | 5,114,130 | 174 / 10 | 8 linear-GINE blocks, 3-layer MLP 672; head 640 |
<!-- END MODEL PARAMETER INVENTORY -->

Counts include encoder, native-update and prediction-head parameters for one
predictor; separate controllers are excluded. Ensemble size, optimization seeds
and repeated propagation depth do not multiply this per-model count. Freezing
weights preserves their parameter count. The model-inventory command reports
non-parameter buffers separately. Peptides' nine categorical atom fields expand
to 174 one-hot inputs.

Omitting the optional width keys preserves the earlier tensor layouts: encoder
hidden width equals `hidden_dim`, the head is linear, GPS feed-forward width is
twice `hidden_dim`, and linear-GINE uses a two-layer update MLP of width
`hidden_dim`. Load an older checkpoint with its saved configuration. To inspect
this compatibility path for arxiv, run `python scripts/model_inventory.py
--config configs/arxiv_full.json --default-widths`; it has the 1,138,216 parameters
recorded for the default plant in Table S5d.

Run `make model-inventory` to instantiate all seven released full configurations
and report their counts without downloading data or training. `make compare-runs`
checks complete registered official archives, including checkpoint identities and
actual parameter counts, against the manuscript targets. Each run records its
checkpoint inventory in `train/model_inventory.json`, including its actual model
class, parameter and buffer counts, and the number of parameters optimized by
each fit. The `q0` checkpoint stores the plant schema but optimizes only the
encoder and head; its inactive propagation weights are not additional learned
parameters of the no-propagation baseline.

## Split and label roles

Official training is partitioned by seeded permutation and cumulative floor
boundaries at 0.60, 0.80, 0.90 and 0.95:

| Role | Share / source | Use |
|---|---|---|
| `D_fit` | 60% of official train | Predictor fit; internal checkpoint-validation subset |
| `D_explore` | 20% | Exploration, controller fitting, radius calibration and assessment |
| `D_select` | 10% | Comparator and candidate selection |
| `D_confirm` | 5% | Prespecified mechanism confirmation |
| `D_compare` | Remaining 5% | Independent E3–E6 comparison |
| `D_admit` | Official validation | Admission of the fixed eligible policy |
| `D_test` | Official test | Final evaluation after policy lock |

Graph collections partition complete graph IDs. All trajectories and seeds for a
graph remain in the same exploration fitting/calibration/assessment group. Each
execution episode resets its state, sensor history, observer and recovery anchor.
Attention and pooling never span examples. The exact membership arrays and hashes
are saved under `prepared/`; a proportion alone is insufficient provenance.

OGB and Roman remain transductive: graph structure/features are shared while label
roles are separated. Buffered cores summarize losses; they are not independent
input graphs. Roman-empire supplies ten official masks. A ten-split campaign
reports variation across those masks rather than independent-graph coverage.

## Raw caches and preparation

Use `--data-root` for a shared cache. Downloads occur only when enabled in the
configuration and required raw files are absent.

| Dataset | Required raw layout beneath the cache |
|---|---|
| ogbn-arxiv | `ogbn_arxiv/raw/`, `ogbn_arxiv/split/time/` |
| ogbn-products | `ogbn_products/raw/`, `ogbn_products/split/sales_ranking/` |
| ogbn-proteins | `ogbn_proteins/raw/`, `ogbn_proteins/split/species/` |
| Roman-empire | `roman_empire/raw/roman_empire.npz` |
| PascalVOC-SP | `pascalvoc-sp/raw/{train,val,test}.pickle` |
| COCO-SP | `coco-sp/raw/{train,val,test}.pickle` |
| Peptides-func | `peptides-func/raw/{train,val,test}.pt` |

An OGB processed-pickle cache alone does not replace its raw files. To prepare a
pilot without training:

```bash
make prepare-data DATASET=ogbn-arxiv PROFILE=pilot DEVICE=cpu \
  DATA_ROOT=../cm-data RUN_ROOT=../cm-runs/arxiv-prepared
```

For offline execution, prepare the raw cache first, copy the configuration outside
the checkout, set `data.download=false`, and supply that copy through `--config`.
Use provider files from trusted sources because some formats use pickle.

## Graph representation and resource scope

The adapter builds nonnegative row-normalized sparse transport with its recorded
edge preprocessing and self-loop convention. Proteins scalar weights use the mean
of the supplied edge-feature channels. Peptides uses `1 + mean(bond categorical
codes)` as an explicit scalar-weight reconstruction choice; this is not the
standard nonlinear GINE message function. Prepared metadata records these choices.

Products uses full-graph sparse propagation and dense/readout calls chunked at
65,536 rows. Chunking does not replace the graph with partitions and does not bound
total hidden-state memory. Independent-graph training accumulates gradients over
the configured graph batch while executing complete graph episodes.

Measure host/GPU memory, work and wall time on the intended hardware before a full
campaign. [HPC](HPC.md) describes execution and environment capture.
