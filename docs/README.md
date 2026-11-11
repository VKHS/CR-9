# Documentation

[Repository home](../README.md) · [Reported results](../paper/RESULTS.md) · [Availability](../AVAILABILITY.md) · [Statement compliance](../STATEMENT_COMPLIANCE.md)

## Artifact reconstruction and publication

| Guide | Purpose |
|---|---|
| [Reconstruction](RECONSTRUCTION.md) | `make paper`: download and verify the published outputs, rebuild all 90 displays with their source data, compare with the printed values |
| [Paper references and reconstruction evidence](PAPER_REFERENCE_VALUES.md) | PDF-bound comparison targets, required cells and panels, source-data provenance and the final publication gate |
| [Figure reference coverage plan](FIGURE_REFERENCE_COVERAGE.md) | Every remaining figure contract, exact source pages, available targets and the specific original data or scenario declarations required |
| [Implementation and reconstruction review](AVAILABILITY_REVIEW.md) | Implementation decisions, source references, comparison findings and artifact requirements |
| [Producing the measured outputs](PRODUCTION.md) | One command on the cluster: every runner of `release/experiment_code.json` as one Slurm job graph from the seven completed runs, ending with the map that publication reads |
| [Publishing the measured outputs](PUBLICATION.md) | Local original-folder preflight, followed by verified publication of runs, record sets, campaigns and environments in `paper-artifacts` |
| [Statement compliance](../STATEMENT_COMPLIANCE.md) | Each item of the registered availability statement and each claim of the availability text printed in the manuscript (main page 15, Methods, SI page 30): where the repository fulfils it, and what awaits publication |

## Publication planning references

| Reference | Purpose |
|---|---|
| [Publication layout and storage estimates](FINAL_DESTINATION.md) | Planning calculations, intended archive organization and publication steps; measured results remain identified by their execution records. |
| [Publication-layout templates](../destination/README.md) | Templates for release manifests, executed manifests, environments, E12 study views and source data. These planning files remain distinct from execution artifacts; `make final-destination-converge` compares the two. |

## Regenerate the results on a cluster

| Guide | Purpose |
|---|---|
| [HPC and reproduction](HPC.md) | `run_cross_morse.sh`: site settings, `check`, `reproduce`, recovery, the report and the compute per dataset |
| [Arrhenius walkthrough](ARRHENIUS_SESSION.md) | The complete session on the authors' cluster |
| [Getting started](GETTING_STARTED.md) | Install, replay examples, execute a smoke or benchmark profile without Slurm |
| [Experiments](EXPERIMENTS.md) | E3–E6 budget-matched arms, upstream baselines and measurement capture |
| [Run archives](RUN_ARCHIVES.md) | Archive schemas, checksums, provenance, ingestion and the table builders |
| [Supplementary builders](SUPPLEMENT_BUILD.md) | Confirmation, factorial, ensemble and budget analyses from their actual inputs |
| [Artifact catalog](ARTIFACT_CATALOG.md) | Every availability category, bundled files, data providers and producing commands |

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
[records/index.json](../records/index.json) accept separately registered archives
and record sets. The release publication path uses
[release/paper_artifacts.json](../release/paper_artifacts.json); the
[availability page](../AVAILABILITY.md) identifies each artifact's current status.
