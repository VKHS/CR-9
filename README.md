# CROSS-MORSE

**Inference-time control of message passing in frozen graph neural networks**

Abdolvahab Khalili Sadaghiani and Jose Nunez-Yanez · Linköping University

[![Software checks](https://github.com/VKHS/CR-9/actions/workflows/quality.yml/badge.svg)](https://github.com/VKHS/CR-9/actions/workflows/quality.yml)
[![Evidence coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FVKHS%2FCR-9%2Fmain%2Fdocs%2Fbadges%2Fevidence.json)](paper/CLAIMS.md)
[Reviewer guide](docs/REVIEWER_GUIDE.md) · [Reviewer FAQ](docs/REVIEWER_FAQ.md) · [Method](docs/METHOD.md) · [Analytical statements](docs/THEORY.md) · [Getting started](docs/GETTING_STARTED.md) · [Datasets](docs/DATASETS.md) · [Availability](AVAILABILITY.md) · [Paper support](paper/PAPER_SUPPORT.md) · [Reported results](paper/RESULTS.md) · [Claims ledger](paper/CLAIMS.md) · [Paper references](paper/README.md)

CROSS-MORSE attaches an external controller to a frozen graph predictor. At
inference, intermediate representations guide whether and how propagation
continues. Learned predictor weights remain fixed; open-loop schedules and
feedback policies are compared on the same predictor through a declared
selection, confirmation, and admission protocol.

![Frozen predictor and external feedback controller](docs/assets/architecture.svg)

## For reviewers

```bash
make review REVIEW_OFFLINE=1 REPORT_ROOT=../cm-reports
```

This checks the checkout and replays the bundled examples. It writes
`../cm-reports/review/REVIEW_REPORT.md` with a one-line verdict
(`EVIDENCE COMPLETE`, `EVIDENCE INCOMPLETE`, `SOFTWARE-ONLY` or `SOFTWARE FAILURE`), the
status of each check and paths to its output. Omit `REVIEW_OFFLINE=1` to fetch
registered run deposits, rebuild supported outputs and audit the official evidence.
Downloads and verification can take substantial time for full benchmark archives.
The [reviewer guide](docs/REVIEWER_GUIDE.md) maps each concern to its evidence and
explains the additional inputs needed for a complete check; the
[reviewer FAQ](docs/REVIEWER_FAQ.md) answers the questions a reader of this
repository is likely to ask, each with the file that holds the fact and the
command that checks it. The
[claim-to-evidence ledger](paper/CLAIMS.md) lists all 34 manuscript table groups,
their registered evidence and builder coverage. The complete display inventory also covers the protocol tables and method figures outside those numerical groups. These statuses describe automated
checks of the supplied artifacts; they do not certify every scientific claim.
`make docker-review` provides the same entry point in a CPU container.

## Paper and supplementary results

This repository supports the completed experimental study **CROSS-MORSE:
Cross-Spectral Control of Message Passing in Frozen Graph Neural Networks**.
The authors identify the supplied 16-page paper and 38-page supplement as the
completed measured study. Their exact document identities, protocol constants
and reported numerical values are recorded in `paper/`.

- [Reported results](paper/RESULTS.md) presents the seven-dataset loss, official-metric
  and cost results, including the distinction between an evaluated arm and the
  policy returned by the registered confirmation rule.
- [Complete paper support inventory](paper/PAPER_SUPPORT.md) maps all **90 numbered
  tables and figures** to source pages, numerical registries, implementation,
  reconstruction commands and the underlying evidence required.
- [Manuscript review](paper/MANUSCRIPT_REVIEW.md) records the contextual review of
  the current PDFs and the precise decisions that need reconciliation with the
  original experiment records.

```bash
make paper-report-check    # document identities, complete display map and current reports
make verify                # implementation, reference arithmetic and sealed example checks
make paper                 # rebuild supported outputs from registered official observations
make scientific-audit      # check the original experiment evidence and decision records
```

The reported measurements and the availability of their original artifacts are
tracked separately. The registration counts below describe what a reader can
currently retrieve through this repository. Mathematical bounds, resampled
uncertainty, fitted cost laws and analytical sensitivities retain their scientific
meaning in the paper. The earlier development workspace is retained in
[Git history](https://github.com/VKHS/CR-9/tree/89bbd136625d581027a84755cef0715cf2231a5f).

## Try the software

Use Python 3.12 on Linux. The recorded CPU environment installs PyTorch and the
dependencies needed for verification and a small synthetic run.

```bash
git clone --depth 1 https://github.com/VKHS/CR-9.git
cd CR-9
make env-cpu PYTHON=python3.12 VENV_DIR=../cm-cpu
source ../cm-cpu/bin/activate
make verify
make test
make reproduce REPORT_ROOT=../cm-reports
```

The repository includes two complete, sealed synthetic examples: **arxiv** for
transductive node classification and **Peptides** for independent-graph multilabel
prediction. They contain checkpoints, predictions, trajectories, and decision
records. `make reproduce` numerically replays those records; `make example-paper`
rebuilds their tables and figures. The examples are software demonstrations.

To execute a new small lifecycle with the current source:

```bash
make train-sample DATASET=Peptides-func RUN_ROOT=../cm-work/peptides-smoke DATA_ROOT=../cm-work/data
make replay RUN_ROOT=../cm-work/peptides-smoke REPORT_ROOT=../cm-reports
```

## Run an experiment

All seven dataset adapters and all 21 profiles are included. `smoke` uses
synthetic data, `pilot` uses official data with reduced computation, and `full`
uses the registered full execution budget and explicit model widths. The seven
full-profile parameter totals agree with Supplementary Table S5 at its printed
precision; the [parameter inventory](docs/DATASETS.md#model-parameter-inventory)
records their architecture settings and counting scope.
`make model-inventory` instantiates and counts them; `make compare-runs` checks
registered official archives against the manuscript targets. Pilot and full
runs may download data and require substantial compute; see the
[dataset guide](docs/DATASETS.md) and [HPC instructions](docs/HPC.md).

| Dataset | Task | Entry point |
|---|---|---|
| ogbn-arxiv | Node classification | `run_cross_morse_arxiv.py` |
| ogbn-products | Node classification | `run_cross_morse_products.py` |
| ogbn-proteins | Multilabel node prediction | `run_cross_morse_proteins.py` |
| Roman-empire | Node classification; ten official splits | `run_cross_morse_roman_empire.py` |
| PascalVOC-SP | Node classification on independent graphs | `run_cross_morse_pascalvoc_sp.py` |
| COCO-SP | Node classification on independent graphs | `run_cross_morse_coco_sp.py` |
| Peptides-func | Multilabel graph prediction | `run_cross_morse_peptides_func.py` |

For example, run an arxiv pilot on a CUDA device:

```bash
python run_cross_morse_arxiv.py --profile pilot --device cuda \
  --data-root ../cm-data --output-dir ../cm-runs/arxiv-pilot
```

Training data are divided into fitting, exploration, selection, confirmation,
and comparison roles (60/20/10/5/5). Official validation is reserved for admission;
test labels are used after policy lock. Supported operators, statistical
assumptions, and the exact implemented variants are explained in the
[method guide](docs/METHOD.md).

## Reproduction status

<!-- availability:start -->
**Registered:** 0 of 7 official benchmark archives, 0 of 10 auxiliary measurement sets, and 2 complete synthetic examples. See [availability and reconstruction scope](AVAILABILITY.md).
<!-- availability:end -->

`make verify` checks the software package and bundled examples.
`make scientific-audit` checks the official experiment evidence and returns a
nonzero status while required records are missing. The [paper reference guide](paper/README.md)
explains the numerical registry, source identities and result builders.

The [archive guide](docs/RUN_ARCHIVES.md) covers packing, resumable Zenodo uploads,
publication and ingestion. Draft DOI reservations are kept separate from published
records; reviewer downloads must match the registered filename, byte count and
SHA-256. `make availability-statement` generates a manuscript paragraph from the
registered state. `make degeneracy-check` identifies patterns in baseline resource
measurements for inspection, and `make decision-check` checks confirmation-rule
consistency. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) maps the reproducibility
checklist to implementation files and the records needed to verify results.

Authors with completed runs can use the [evidence-registration guide](docs/EVIDENCE_REGISTRATION.md).
`make evidence-discover EVIDENCE_ROOT=/abs/export EVIDENCE_PLAN=/abs/new-plan.json`
inspects an existing export tree and proposes unambiguous complete full-profile
run paths. It reports rejected, incomplete and duplicate candidates before any
deposit or registration. The [paper-support contract](docs/PAPER_SUPPORT.md)
maps E1–E20 to the observations, runtime checks and manuscript decisions needed
to support each claim.
One private plan drives preflight, packing, resumable deposits, registration and
manuscript bindings, including the ten-split Roman campaign. Record mappings trace
summary values and declared calculations to copied original files. PDF registration
records document identities and a named contextual source review; the current review
is AI-assisted and bound to the PDF and registry hashes. Numeric text searches are
advisory. `make readiness` checks the current evidence and review report for outstanding
requirements.

## Source and reference checks

`make registry-check` checks executable constants, analytical reference formulas
and arithmetic within the transcribed tables:

| Check | Inputs |
|---|---|
| Actions, sensor settings, thresholds, seeds, plant settings and controlled sites | Source literals and full-profile configurations against the protocol registry |
| Kernel coefficients, band sensitivities, sparse-to-dense ratio, sensing shares and estimates of nominal propagation work | Registered formulas, published dataset summaries, widths and depths |
| Cost ratios, sensing residuals, admissible sketch ratios, expected costs and cross-table results | Transcribed measurements and the formulas relating them |

`make theory` and `make si-check` are part of `make verify` and need only NumPy
and the Python standard library:

| Check | Inputs |
|---|---|
| Definitions 1–2, Lemmas 1–3, Propositions 1–11, Theorems 1–2 and Corollaries 1–2 on seeded random instances of their hypotheses | Reference implementations in `cm_theory/`, the executable action registry, `cross_morse.statistics` (Theorem 2) and `cm_experiments.budget` (Algorithm S3) |
| Holm family, margin and two-way rescaling of the mechanism tests, factorial effects, selection readings, ensemble law, power, retained-share, attribution, tail-cost and amortization entries | The transcribed supplementary tables and the formulas the supplement states for them |

[Analytical statements](docs/THEORY.md) maps every statement to its reference
implementation, runtime counterpart and checks. `make si-report` writes the
recomputed supplementary values with their notes to `REPORT_ROOT`.

`make model-inventory` separately constructs all seven full-profile models and
checks their actual parameter counts. These checks establish configuration and
arithmetic consistency. Reconstructing benchmark losses, metrics, timings and
GPU-hours requires the corresponding run archives and measurement records:
`make paper` builds results from those observations, and `make compare-runs`
compares them with the manuscript targets.

## Repository guide

| Location | Purpose |
|---|---|
| `cross_morse/` | Models, transport, sensing, policies, statistics, and the eight-stage lifecycle |
| `configs/`, `run_cross_morse*.py`, `run_cross_morse*.sh` | Dataset profiles and local/Slurm entry points |
| `cm_experiments/`, `cm_campaigns/`, `hpc/` | Budget-matched arms, external-baseline harness, campaigns, and measurement capture |
| `cm_theory/` | Reference implementations and numerical verification of the analytical statements |
| `examples/` | Two complete synthetic runs and their replay outputs |
| `paper/` | Source metadata, numerical comparison targets, and result builders |
| `runs/`, `records/`, `manifests/` | Official evidence indexes, record schemas, and split/seed specifications |
| `scripts/`, `tests/`, `environment/` | Verification, packaging, tests, and environment specifications |
| `docs/` | Practical guides for use, method, experiments, archives, validation and review |
| `Dockerfile`, `environment/apptainer.def` | CPU containers using recorded Python package versions (`make docker-verify`, `make docker-review`) |

For the next step, use [experiment instructions](docs/EXPERIMENTS.md),
[archive and result reconstruction](docs/RUN_ARCHIVES.md), or
[validation commands](docs/VALIDATION.md). `make help` lists the common commands.

## Citation and license

Use [CITATION.cff](CITATION.cff) and identify the source commit used in an
experiment. The accompanying manuscript is under review; this repository does
not assert publication or a public result-archive DOI. Software is licensed
under [BSD-3-Clause](LICENSE); manuscript and figure rights are described in
[NOTICE](NOTICE). Contributions should follow [CONTRIBUTING.md](CONTRIBUTING.md).
