# Reproducibility checklist

This file maps the main topics in the machine-learning reproducibility checklist
reviewed by [Pineau et al. (JMLR 22(164), 2021)](https://jmlr.org/papers/v22/20-303.html)
to this repository. A file describing a protocol is distinct from evidence that
an experiment followed it. Current registrations are generated in
[AVAILABILITY.md](AVAILABILITY.md) and [paper/CLAIMS.md](paper/CLAIMS.md);
`make review` checks the available artifacts and reports remaining requirements.

| Topic | Implementation or declared protocol | Evidence needed for the manuscript results |
|---|---|---|
| Models, algorithms and supported operators | [Method](docs/METHOD.md), `cross_morse/core.py`, `cross_morse/policies.py`, [implementation notes](paper/IMPLEMENTATION_NOTES.md) | Each run's resolved configuration, checkpoint tensor inventory and nominal-identity check; auxiliary `cross_backbone` and `inserted_site` records for those comparisons |
| Assumptions and mathematical guarantees | The manuscript and supplement identified by `paper/sources.json`; executable thresholds and formulas in `paper/registered_constants.json`; reference implementations of every analytical statement in `cm_theory/` with seeded numerical checks (`make theory`, [Analytical statements](docs/THEORY.md)); the derived supplementary entries recomputed from their printed inputs (`make si-check`) | Matching PDF copies verify document identity. The numerical checks test each stated implication on random instances of its hypotheses and the arithmetic of derived table entries; they do not replace the proofs, test whether trained backbones satisfy the hypotheses, or guarantee predictive improvement |
| Dataset sources and preprocessing | [Datasets](docs/DATASETS.md), `data/sources.json`, `cross_morse/datasets.py`, `cross_morse/data.py` | Prepared-data identities and hashes in each run's `prepared/` files and preparation-stage manifest; source attribution and applicable dataset terms |
| Data roles and evaluation separation | `cross_morse/split_contract.py`, `manifests/registered/`; training roles 60/20/10/5/5, official validation for admission, test after policy lock | Actual unit memberships and their hashes, stage ordering, selection and policy-lock records. Declared percentages alone do not establish separation |
| Hyperparameters, search and optimization seeds | Full-profile configurations and registered manifests | Each run's `config.json`, histories, complete candidate ledger and checkpoint inventory; the recorded search must satisfy the declared budget |
| Software and execution environment | `pyproject.toml`, `environment/requirements-cpu.lock`, `environment/observed_environment.json`; Docker and Apptainer recipes | Per-run source identity, resolved configuration, package and hardware records. The CPU package lock is not a record of every GPU or native-library environment |
| Training and evaluation | Dataset entry points and the eight stages in `cross_morse/pipeline.py`; task metrics in `cross_morse/evaluation.py` | Sealed checkpoints, predictions and paired observations. `scripts/reproduce_results.py` recomputes statistics; `scripts/evaluate_checkpoints.py` separately re-executes trusted models |
| Number of runs, seeds and splits | Registered full profiles, including Roman-empire's ten official splits | Complete run/campaign archives; the audit checks applicable coverage and identities rather than inferring execution from the protocol |
| Metrics, uncertainty and decision rules | `cross_morse/statistics.py`, `paper/registered_constants.json`, result builders | Paired units, confirmation contrasts, admission records and locked policy. Builders expose the quantities available in those records; analytical scenarios remain separately identified |
| Runtime, cost and computing infrastructure | `cm_campaigns/accounting.py`, `hpc/capture_arrhenius_accounting.sh`, E17 timing/export tools | Actual timings, scheduler exports and environment details in runs and auxiliary records. Repeated ratios are diagnostic patterns, not a substitute for checking these records |
| Figures, tables and manuscript correspondence | [Display support inventory](paper/PAPER_SUPPORT.md), [Claims ledger](paper/CLAIMS.md), `paper/build/`, `scripts/records.py`, `paper/bindings/` | Supported displays rebuilt from observations, summary values traced to raw files, result bindings and comparison against the transcribed manuscript targets |
| Access to artifacts and licensing | [Archive guide](docs/RUN_ARCHIVES.md), `runs/index.json`, `records/index.json`, [LICENSE](LICENSE), [NOTICE](NOTICE) | Published archive records whose filename, size and SHA-256 match the index; complete auxiliary sets and applicable redistribution terms |

## Executable entry points

```bash
make verify
make test
make review REVIEW_OFFLINE=1 REPORT_ROOT=../cm-reports
```

These exercise the software and the two bundled synthetic examples. Official
benchmark reconstruction additionally requires the seven full run archives,
ten auxiliary record sets, the E3–E6 campaign, the original joint S14
confirmation campaign and manuscript bindings. The
[reviewer guide](docs/REVIEWER_GUIDE.md) gives the evidence-mode command and explains
its verdicts. A passing automated report establishes the checks it enumerates;
it does not independently establish the historical origin of observations or
resolve every scientific judgment in peer review.
