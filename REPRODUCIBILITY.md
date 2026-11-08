# Reproducibility checklist

This file maps the main topics in the machine-learning reproducibility checklist
reviewed by [Pineau et al. (JMLR 22(164), 2021)](https://jmlr.org/papers/v22/20-303.html)
to this repository. A file describing a protocol is distinct from evidence that
an experiment followed it. The publication workflow distributes measured outputs
through the repository's release `paper-artifacts`
([publication guide](docs/PUBLICATION.md)); `make paper` verifies them and rebuilds
every table and figure, and `bash run_cross_morse.sh reproduce` regenerates the
lifecycle results of all seven datasets on a Slurm cluster ([HPC guide](docs/HPC.md)).
The right-hand column names the records that support each topic. The lifecycle
records are written by every run; the auxiliary records come from the experiment
code in [Experiments](docs/EXPERIMENTS.md) and
[`release/experiment_code.json`](release/experiment_code.json).
[Statement compliance](STATEMENT_COMPLIANCE.md) evaluates each item of the
registered availability statement, [AVAILABILITY.md](AVAILABILITY.md) states the
release scope and [paper/CLAIMS.md](paper/CLAIMS.md) maps every registered
numerical group to its builder.

The [artifact catalog](docs/ARTIFACT_CATALOG.md) maps each availability category
to actual bundled files, external dataset sources or a producing command.
`make reconstruct` combines supplied run and auxiliary artifacts into a
workspace whose report accounts for every numbered display and records the
coverage of each generated output. The [reconstruction guide](docs/RECONSTRUCTION.md)
documents byte verification, atomic builds and the optional full-display gate.

| Topic | Implementation or declared protocol | Records that support the manuscript results |
|---|---|---|
| Models, algorithms and supported operators | [Method](docs/METHOD.md), `cross_morse/core.py`, `cross_morse/policies.py`, [implementation notes](paper/IMPLEMENTATION_NOTES.md) | Each run's resolved configuration, checkpoint tensor inventory and nominal-identity check; auxiliary `cross_backbone` and `inserted_site` records for those comparisons |
| Assumptions and mathematical guarantees | The manuscript and supplement identified by `paper/sources.json`; executable thresholds and formulas in `paper/registered_constants.json`; reference implementations of every analytical statement in `cm_theory/` with seeded numerical checks (`make theory`, [Analytical statements](docs/THEORY.md)); the derived supplementary entries recomputed from their printed inputs (`make si-check`) | Matching PDF copies verify document identity. The numerical checks test each stated implication on random instances of its hypotheses and the arithmetic of derived table entries; they do not replace the proofs, test whether trained backbones satisfy the hypotheses, or guarantee predictive improvement |
| Dataset sources and preprocessing | [Datasets](docs/DATASETS.md), `data/sources.json`, `cross_morse/datasets.py`, `cross_morse/data.py` | Prepared-data identities and hashes in each run's `prepared/` files and preparation-stage manifest; source attribution and applicable dataset terms |
| Data roles and evaluation separation | `cross_morse/split_contract.py`, `manifests/registered/`; training roles 60/20/10/5/5, official validation for admission, test after policy lock | Actual unit memberships and their hashes, stage ordering, selection and policy-lock records. Declared percentages alone do not establish separation |
| Hyperparameters, search and optimization seeds | Full-profile configurations and registered manifests | Each run's `config.json`, histories, complete candidate ledger and checkpoint inventory; the recorded search must satisfy the declared budget |
| Software and execution environment | `pyproject.toml`, `environment/requirements-cpu.lock`, `environment/observed_environment.json`; Docker and Apptainer recipes; the SITE SETTINGS of `run_cross_morse.sh` (modules and PyTorch wheel of the GPU environment) | Per-run source identity, resolved configuration and `environment.json`; the job's package list, module list and `nvidia-smi` output in the run's `launcher/` directory. The CPU package lock is not a record of every GPU or native-library environment |
| Training and evaluation | Dataset entry points and the eight stages in `cross_morse/pipeline.py`; task metrics in `cross_morse/evaluation.py` | Sealed checkpoints, predictions and paired observations. `scripts/reproduce_results.py` recomputes statistics; `scripts/evaluate_checkpoints.py` separately re-executes trusted models |
| Number of runs, seeds and splits | Registered full profiles, including Roman-empire's ten official splits; `run_cross_morse.sh reproduce` submits every dataset and all ten splits | The complete runs of a reproduction suite and its run registry; the audit checks applicable coverage and identities rather than inferring execution from the protocol |
| Metrics, uncertainty and decision rules | `cross_morse/statistics.py`, `paper/registered_constants.json`, result builders | Paired units, confirmation contrasts, admission records and locked policy. Builders expose the quantities available in those records; analytical scenarios remain separately identified |
| Runtime, cost and computing infrastructure | `cm_campaigns/accounting.py`, `hpc/capture_arrhenius_accounting.sh`, E17 timing/export tools; the job table and scheduler accounting of a reproduction suite | Actual timings, scheduler exports and environment details in runs and auxiliary records; the GPU-hours of every run in the suite's report. Repeated ratios are diagnostic patterns, not a substitute for checking these records |
| Figures, tables and manuscript correspondence | [Display support inventory](paper/PAPER_SUPPORT.md), [Claims ledger](paper/CLAIMS.md), one generator per numbered display in `paper/build/displays/`, `scripts/records.py`, `paper/bindings/`, `scripts/compare_with_manuscript.py` | Every display rebuilt by `make paper` from the published outputs with its source data (`paper/source_data/`) and each computed value beside the printed one; the same comparison for regenerated runs in a suite's `REPRODUCTION_REPORT.md`; summary values traced to raw files and result bindings |
| Access to artifacts and licensing | This repository: code, configurations, launcher, registry, and the release `paper-artifacts` holding the measured outputs ([publication guide](docs/PUBLICATION.md), [`release/paper_artifacts.json`](release/paper_artifacts.json)); [Availability](AVAILABILITY.md), [Statement compliance](STATEMENT_COMPLIANCE.md), [LICENSE](LICENSE), [NOTICE](NOTICE) | Every published run, record set and campaign with the SHA-256 of each part, archive and file; `make paper` verifies them before use. Dataset terms apply to the downloaded benchmarks |
| One-command reconstruction | `make paper` (`scripts/reconstruct_paper.py`), one generator per numbered display in `paper/build/displays/` | Every table and figure with its source data and a per-cell comparison with the printed value (`displays/compare/`); exit status 0 only when every display is complete and every value agrees |

## ML Code Completeness Checklist

The five items of the [ML Code Completeness Checklist](https://github.com/paperswithcode/releasing-research-code)
(Papers with Code; part of the NeurIPS 2020 code-submission process):

| Item | In this repository |
|---|---|
| Specification of dependencies | `pyproject.toml`, `requirements.txt`, `environment/requirements-cpu.lock`, `Dockerfile`, `environment/apptainer.def`, the GPU modules and PyTorch wheel in the SITE SETTINGS of `run_cross_morse.sh`; the observed environment of every reported run in `environment/observed/` |
| Training code | `run_cross_morse.py` and the dataset entry points `run_cross_morse_<dataset>.py`, `cross_morse/`; the experiments outside the lifecycle in `cm_experiments/`, `cm_campaigns/` and [`release/experiment_code.json`](release/experiment_code.json) |
| Evaluation code | `cross_morse/evaluation.py`, `cross_morse/statistics.py`, `scripts/reproduce_results.py`, `scripts/evaluate_checkpoints.py`, the display generators in `paper/build/displays/` |
| Pre-trained models | The checkpoints of every reported run: complete tier of the release `paper-artifacts` ([`release/paper_artifacts.json`](release/paper_artifacts.json), `make paper-artifacts-status`) |
| Results: a table of the main results and the command that reproduces them | [README](README.md) (`make paper`, `bash run_cross_morse.sh reproduce`), [paper/RESULTS.md](paper/RESULTS.md), [Reconstruction](docs/RECONSTRUCTION.md) |

## Executable entry points

```bash
make paper                                    # every table and figure from the published outputs
make statement-check                          # each item of the registered availability statement
make verify                                   # software, registry, sealed examples; no GPU
make test
make review REVIEW_OFFLINE=1 REPORT_ROOT=../cm-reports
make reconstruction-example REPORT_ROOT=../cm-reports
make artifact-catalog-check
bash run_cross_morse.sh check                 # on a Slurm cluster: settings, runtime identity, plan
bash run_cross_morse.sh reproduce             # regenerate all seven datasets and the report
bash hpc/produce_paper_outputs_arrhenius.sh <suite>   # every other measured output from those runs
```

`make paper` downloads and verifies the published measured outputs and rebuilds every
numbered display ([Reconstruction](docs/RECONSTRUCTION.md)). The next three exercise the
software and the two bundled synthetic examples.
The launcher regenerates the lifecycle results; the E3–E6 campaign, the external
baselines and the other auxiliary record sets run from their drivers
([Experiments](docs/EXPERIMENTS.md)), all submitted by one command
([Producing the measured outputs](docs/PRODUCTION.md)), and the cross-dataset S14
family is built from the sealed runs and audited by [its checker](docs/S14_CONFIRMATION.md). The
[reviewer guide](docs/REVIEWER_GUIDE.md) explains each verdict. A passing
automated report establishes the checks it enumerates; it does not resolve every
scientific judgment in peer review.
