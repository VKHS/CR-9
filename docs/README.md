# Documentation

[Repository home](../README.md) · [Reported results](../paper/RESULTS.md) · [Availability](../AVAILABILITY.md)

## Reproduce the paper

| Guide | Purpose |
|---|---|
| [HPC and reproduction](HPC.md) | `run_cross_morse.sh`: site settings, `check`, `reproduce`, recovery, the report and the compute per dataset |
| [Arrhenius walkthrough](ARRHENIUS_SESSION.md) | The complete session on the authors' cluster |
| [Getting started](GETTING_STARTED.md) | Install, replay examples, execute a smoke or benchmark profile without Slurm |
| [Experiments](EXPERIMENTS.md) | E3–E6 budget-matched arms, upstream baselines and measurement capture |
| [Run archives](RUN_ARCHIVES.md) | Archive schemas, checksums, provenance, ingestion and the table builders |

## Read and review the paper

| Guide | Purpose |
|---|---|
| [Reviewer guide](REVIEWER_GUIDE.md) | Software check without a GPU, regeneration on a cluster, and each review concern mapped to code, commands and records |
| [Reviewer FAQ](REVIEWER_FAQ.md) | Provenance, reproduction, statistics, models, release scope and hygiene, each with its file and check |
| [Complete paper support inventory](../paper/PAPER_SUPPORT.md) | All 90 numbered tables and figures, source pages, implementations and builders |
| [Reported results](../paper/RESULTS.md) | Seven-dataset results with evaluated-arm and registered-lifecycle interpretation |
| [Manuscript review](../paper/MANUSCRIPT_REVIEW.md) | Contextual review that binds the registry to the paper and supplement |
| [Paper support criteria](PAPER_SUPPORT.md) | E1–E20 implementation and scientific evidence requirements |
| [Claims ledger](../paper/CLAIMS.md) | Registered numerical groups, record sets and builders |
| [Reproducibility checklist](../REPRODUCIBILITY.md) | Each reproducibility requirement mapped to code and records |

## Method and checks

| Guide | Purpose |
|---|---|
| [Method](METHOD.md) | Frozen predictor, supported actions, feedback, selection, confirmation and admission |
| [Analytical statements](THEORY.md) | Definitions, lemmas, propositions, theorems and corollaries mapped to reference and runtime checks |
| [Datasets](DATASETS.md) | Seven dataset adapters, sources, metrics, splits and model inventories |
| [Validation](VALIDATION.md) | Executed software checks and their scope |
| [Joint S14 confirmation](S14_CONFIRMATION.md) | The nine-test family, paired observations, frozen decisions and its checker |
| [Evidence registration](EVIDENCE_REGISTRATION.md) | Publishing run archives into a run index: preflight, archive, deposit, register and bind |
| [Paper references](../paper/README.md) | Current PDF identities and numerical comparison registries |

The [two bundled examples](../examples/README.md) are complete synthetic runs for
software verification. [runs/index.json](../runs/index.json) and
[records/index.json](../records/index.json) accept registered archives and record
sets; this release registers none.
