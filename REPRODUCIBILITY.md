# Reproducibility checklist

This file maps the main topics in the machine-learning reproducibility checklist
reviewed by [Pineau et al. (JMLR 22(164), 2021)](https://jmlr.org/papers/v22/20-303.html)
to this repository. A file describing a protocol is distinct from evidence that
an experiment followed it. This is a code release: `bash run_cross_morse.sh
reproduce` regenerates the lifecycle results of all seven datasets on a Slurm
cluster ([HPC guide](docs/HPC.md)), and the right-hand column names the records
that support each topic. The lifecycle records are written by every regenerated
run; the auxiliary records come from the drivers in
[Experiments](docs/EXPERIMENTS.md). [AVAILABILITY.md](AVAILABILITY.md) states the
release scope and [paper/CLAIMS.md](paper/CLAIMS.md) maps every registered
numerical group to its builder.

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
| Figures, tables and manuscript correspondence | [Display support inventory](paper/PAPER_SUPPORT.md), [Claims ledger](paper/CLAIMS.md), `paper/build/`, `scripts/records.py`, `paper/bindings/`, `scripts/compare_with_manuscript.py` | Displays rebuilt from the regenerated runs and every compared quantity beside the printed value in the suite's `REPRODUCTION_REPORT.md`; summary values traced to raw files and result bindings |
| Access to artifacts and licensing | This repository (code, configurations, launcher and registry), [Availability](AVAILABILITY.md), [Archive guide](docs/RUN_ARCHIVES.md), [LICENSE](LICENSE), [NOTICE](NOTICE) | A reproduction suite regenerates every run; with `CM_PACK_ARCHIVES=1` it also packs sealed archives whose filename, size and SHA-256 an index can register. Dataset terms apply to the downloaded benchmarks |

## Executable entry points

```bash
make verify                                   # software, registry, sealed examples; no GPU
make test
make review REVIEW_OFFLINE=1 REPORT_ROOT=../cm-reports
bash run_cross_morse.sh check                 # on a Slurm cluster: settings, runtime identity, plan
bash run_cross_morse.sh reproduce             # regenerate all seven datasets and the report
```

The first three exercise the software and the two bundled synthetic examples.
The launcher regenerates the lifecycle results; the E3–E6 campaign, the external
baselines and the other auxiliary record sets run from their drivers
([Experiments](docs/EXPERIMENTS.md)), and the cross-dataset S14 family is audited
by [its checker](docs/S14_CONFIRMATION.md). The
[reviewer guide](docs/REVIEWER_GUIDE.md) explains each verdict. A passing
automated report establishes the checks it enumerates; it does not resolve every
scientific judgment in peer review.
