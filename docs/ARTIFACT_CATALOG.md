# Artifact catalog

[Availability](../AVAILABILITY.md) · [Reconstruction guide](RECONSTRUCTION.md) · [Display support](../paper/PAPER_SUPPORT.md) · [Machine-readable catalog](../release/artifact_catalog.json)

This catalog maps each item in the availability statement to shipped files, dataset providers, execution products and producing commands. Bundled files have SHA-256 and byte identities in the JSON catalog. A generated-file contract describes what an execution produces; it does not represent a file already available for download. Paths use the indicated root: the checkout, one run, the lifecycle suite, one imported record set, or the relevant command's report/output directory.

## Artifact types

- **Implementation and registered design:** executable code, configurations, prescribed seeds, split rules and environments.
- **Reported numerical references:** transcribed paper values and protocol constants with source provenance.
- **Synthetic outputs:** complete software examples with their own historical source identities.
- **New execution products:** checkpoints, predictions, logs and analyses created by the declared producer.
- **Registered experiment artifacts:** original or subsequent runs distinguished by an explicit manuscript relationship tied to source, configuration and unit identities.

## Availability map

| Item | Shipped material | Execution products | Scope |
|---|---|---|---|
| Experimental implementation | Implementation: `cross_morse/*.py`, `run_cross_morse*.py`, `cm_experiments/*.py`, `cm_campaigns/*.py`, `release/experiment_code.json`; Implementation: `run_cross_morse.sh`, `scripts/reproduction.py` | `suite/` `000_MANIFEST/suite.json`, `000_MANIFEST/source_sha256.json`, `runs/**/run_state.json`, `REPRODUCTION_REPORT.md`, `SUITE_STATUS.json` | Core and auxiliary experiment implementations are included; release/experiment_code.json maps every experiment behind a reported display to its code, command and outputs. |
| Configurations | Configuration: `configs/*_smoke.json`, `configs/*_pilot.json`, `configs/*_full.json`, `configs/public.env.example`, `configs/arrhenius-site.env.example` | `run/` `config.json`; `suite/` `runs/*/launcher/run_manifest.json`, `runs/*/launcher/source_sha256.json`, `runs/*/source/configs/submitted.json`, `runs/*/source/configs/resolved.json`, `runs/*/source/configs/active.env.example` | Smoke, pilot and full profiles plus portable site templates. Model counts identify released configurations; original checkpoint identity requires the associated run. |
| Source data and reported values | Reported numerical reference: `paper/reported_results.json`, `paper/registered_constants.json`, `paper/sources.json`; Synthetic example output: `examples/runs/*/prepared/graph.npz`, `examples/runs/*/prepared/metadata.json` | Official data from the providers below; `run/` `prepared/graph.npz`, `prepared/metadata.json`, `prepared/official_splits.npz`, `prepared/splits.npz` | Official provider data, reported numerical references and experimental observations have separate identities. The source data of every display is generated from the published measured outputs (paper/source_data/, written by the publication step); transcribed values never become reconstructed measurements. |
| Split and seed manifests | Registered design: `manifests/registered/*.json`; Synthetic example output: `examples/runs/*/prepared/splits.npz`, `examples/runs/*/prepared/official_splits.npz`, `examples/runs/*/config.json` | `run/` `prepared/splits.npz`, `prepared/official_splits.npz`, `prepared/graph_splits.npz`, `prepared/clusters.npz`; `checkout/` `manifests/official/*.json`, `manifests/official/*.npz` | Registered manifests prescribe splits/seeds; manifests/published/ records the memberships and seeds that every published run used. Membership and optimization-seed evidence stay separate. |
| Environment specifications | Environment specification: `environment/requirements-*.lock`, `environment/observed_environment.json`, `environment/apptainer.def`, `Dockerfile`; Synthetic example output: `examples/runs/*/environment.json` | `run/` `environment.json` | Released lock files and containers specify environments; each run carries its actual execution environment, and environment/observed/ summarizes it for every published run. |
| Trained checkpoints and model inventories | Implementation: `cross_morse/core.py`, `scripts/model_inventory.py`; Synthetic example output: `examples/runs/*/train/*_best.pt`, `examples/runs/*/train/model_inventory.json` | `run/` `train/*_best.pt`, `train/model_inventory.json`, `provenance/frozen_models.json` | The checkpoints of the reported runs are assets of the release paper-artifacts (complete tier; release/paper_artifacts.json). Bundled checkpoint files are synthetic examples. |
| Predictions and paired observations | Synthetic example output: `examples/runs/*/final/test_predictions.npz`, `examples/runs/*/final/paired_test_units.npz`, `examples/runs/*/final/q*_probabilities.npy` | `run/` `final/test_predictions.npz`, `final/paired_test_units.npz`, `final/q*_probabilities.npy`, `select/unit_scores.npz` | The predictions of the reported runs are assets of the release paper-artifacts (complete tier); paired observations are in the analysis tier. Numerical agreement with a manuscript value does not establish that a new run produced the original paper. |
| Exploration and propensity logs | Synthetic example output: `examples/runs/*/explore/trajectory_*.json`, `examples/runs/*/explore/trajectory_*.npz`, `examples/runs/*/explore/manifest.json` | `run/` `explore/trajectory_*.json`, `explore/trajectory_*.npz`, `explore/manifest.json`, `fit/*/audit.json` | The propensity-logged exploration trajectories of the reported runs are assets of the release paper-artifacts (both tiers); bundled synthetic examples show the format. |
| Statistical analyses | Implementation: `cross_morse/statistics.py`, `cm_theory/*.py`, `scripts/check_si_identities.py`, `cm_campaigns/confirmation.py`; Synthetic example output: `examples/runs/*/select/decision.json`, `examples/runs/*/confirm/*.json`, `examples/runs/*/admit/*.json`, `examples/runs/*/final/results_summary.json` | `run/` `select/decision.json`, `confirm/decision.json`, `confirm/contrast_registry.json`, `admit/pre_admission_lock.json`, `admit/decision.json`, `final/results_summary.json`; `record_set/` `summary.csv`, `raw/*`, `code/*`, `provenance.json`, `MANIFEST.json`; `report/` `scientific-evidence-audit.json`; `report/` `tables/S14_joint.csv`, `tables/S16.csv`, `tables/S16b.csv`, `tables/factorial_point_contrasts.csv`, `tables/S26b_fit.csv`, `tables/S46b_comparison.csv` | Statistical code and arithmetic checks are included; the selection, confirmation, admission and evaluation records of every reported run and every record set and campaign behind a display are assets of the release paper-artifacts. |
| Table and figure generators | Analysis or display builder: `paper/build/*.py`, `scripts/build_paper.py`, `scripts/plot_registry_figures.py`, `scripts/draw_diagrams.py`, `scripts/reconstruct.py`, `scripts/build_supplement.py`, `paper/build/displays/*.py`, `scripts/reconstruct_paper.py`; Reported numerical reference: `paper/display_manifest.json` | `report/` `RECONSTRUCTION_REPORT.md`, `reconstruction.json`, `FILES.json`; `report/` `runs/tables/*`, `runs/figures/*`, `runs/evidence/*`, `records/*`, `supplement/tables/*`, `supplement/figures/*`, `supplement/data/*`, `supplement/BUILD.json`; `report/` `tables/*`, `figures/*`, `data/*`, `BUILD.json` | paper/build/displays/ holds one generator per numbered display; make paper runs them on the published outputs and compares every computed value with the printed one. |

## Producing and checking commands

Paths in angle brackets are inputs supplied by the operator. These recipes are listed for inspection; catalog generation runs none of them. `make reconstruct` reports the supported outputs of each display; it does not turn related plots into exact replicas of every printed panel.

| Command | Purpose | Required inputs |
|---|---|---|
| `make verify` | Check implementation, sealed examples, source references and generated artifacts. | Recorded CPU or NumPy environment |
| `bash run_cross_morse.sh reproduce` | Execute the registered seven-dataset lifecycle suite and compare supported run outputs with the paper. | Reviewed cluster/site settings; Provider datasets or permitted downloads; Execution environment and compute allocation |
| `bash run_cross_morse.sh reproduce smoke` | Exercise the launcher on synthetic data; outputs remain synthetic. | Cluster/site settings; Execution environment |
| `make model-inventory` | Instantiate the released model configurations and count their named parameters. | CPU PyTorch environment |
| `python -B scripts/run_manifests.py registered` | Write prescribed split/seed/configuration manifests; these describe the design. | Dataset configurations |
| `python -B scripts/run_manifests.py runs --index runs/index.json --label official --with-indices` | Extract actual roles, seeds, environment and membership arrays from registered runs. | Validated run archives |
| `make registry-check` | Check reported reference values, protocol constants and recorded PDF identities within their stated scope. | Checkout |
| `python -B environment/check_environment.py` | Inspect the installed execution environment. | Checkout |
| `python -B scripts/arm_campaign.py execute --manifest '<campaign.json>' --trust-local-checkpoints` | Execute the declared E3–E6 controller-seed studies; the supplied plan identifies paid work and parent runs. | Completed parent runs; Prespecified campaign; Reviewed local checkpoints; Compute allocation |
| `python -B scripts/run_upstream_baseline.py --plan '<baseline-plan.json>' --output '<study>' --phase hpo --index '<trial-index>' --trust-upstream-code` | Execute one declared pinned-upstream HPO attempt; selection and final-seed phases follow the baseline guide. | Reviewed pinned upstream code and plan; Training/validation data; Compute allocation |
| `python -B scripts/export_records.py from-mapping --mapping '<measurement-mapping.json>'` | Import existing measurements and declared calculations with raw-file locators; never fill them from manuscript targets. | Actual original or reproduced outputs; Explicit mapping and provenance |
| `python -B scripts/archive_run.py '<run>' --output '<archive.tar.gz>'` | Pack a complete sealed run with its byte inventory and identities. | Complete run directory |
| `python -B scripts/ingest_run.py '<archive.tar.gz>'` | Verify and register a supplied archive; manuscript relationship remains a separate author declaration. | Archive with sealed outputs |
| `make fetch-deposits 'REPORT_ROOT=<reports>'` | Retrieve registered public run deposits and verify their pinned archive bytes. | Complete run-index registration; Network access |
| `make records-check` | Recheck record-set file hashes and trace each summary value to its source. | Indexed measurement sets |
| `make scientific-audit 'BUDGET_CAMPAIGN=<e3-e6>' 'CONFIRMATION_CAMPAIGN=<s14>' 'REPORT_ROOT=<reports>'` | Check original experiment bindings, joint confirmation, budget and auxiliary evidence within the enumerated audit scope. | Registered runs and records; Original S14 family; Complete E3–E6 campaign |
| `make theory-full 'REPORT_ROOT=<reports>'` | Run numerical checks of the stated analytical implications; these are not experimental observations. | Checkout |
| `make si-report 'REPORT_ROOT=<reports>'` | Recompute supplementary relationships from printed inputs with precision and interpretation notes. | Checkout |
| `make reconstruct 'REPORT_ROOT=<reports>'` | Build supported numerical outputs and enumerate the scope of every numbered display. | Registered original or reproduced run archives; Applicable measurement sets/campaigns |
| `make reconstruction-example 'REPORT_ROOT=<reports>'` | Build the unified report using bundled synthetic observations; no benchmark measurements are implied. | Checkout |
| `python -B scripts/build_supplement.py --records-index records/index.json --confirmation-campaign '<s14>' --budget-campaign '<e3-e6>' --output-dir '<supplement-output>'` | Build supported factorial, joint-confirmation, ensemble-fit and arm-comparison outputs from validated inputs. | Applicable measurement sets; Original S14 family for confirmation outputs; E3–E6 campaign for arm outputs |
| `make paper` | Download and verify the measured outputs published with the release paper-artifacts, rebuild every numbered table and figure with its source data, and compare every computed value with the printed one. | Published release assets (release/paper_artifacts.json); Network access; Recorded CPU or NumPy environment |
| `bash hpc/publish_paper_artifacts_arrhenius.sh publish` | Upload the packed outputs of the reported runs, record sets, campaigns and environment to the release paper-artifacts, record their SHA-256 in release/paper_artifacts.json, download them back, verify them, rebuild every display and write the split/seed manifests, observed environments and source data to commit. | Completed reported runs, record sets and campaigns on the cluster; GITHUB_TOKEN with contents:write |
| `make statement-check` | Evaluate each item of the registered availability statement against the repository and the published manifest. | Checkout |

## Dataset providers

URLs and adapter identities come from [data/sources.json](../data/sources.json). The catalog verifies the recorded loader hashes, without making network requests or asserting provider uptime.

| Dataset | Provider documentation | Statistical unit |
|---|---|---|
| ogbn-arxiv | [https://ogb.stanford.edu/docs/nodeprop/](https://ogb.stanford.edu/docs/nodeprop/) | METIS core |
| ogbn-products | [https://ogb.stanford.edu/docs/nodeprop/](https://ogb.stanford.edu/docs/nodeprop/) | METIS core |
| ogbn-proteins | [https://ogb.stanford.edu/docs/nodeprop/](https://ogb.stanford.edu/docs/nodeprop/) | degree-stratified core |
| Roman-empire | [https://github.com/yandex-research/heterophilous-graphs](https://github.com/yandex-research/heterophilous-graphs) | METIS core |
| PascalVOC-SP | [https://github.com/vijaydwivedi75/lrgb](https://github.com/vijaydwivedi75/lrgb) | complete graph |
| COCO-SP | [https://github.com/vijaydwivedi75/lrgb](https://github.com/vijaydwivedi75/lrgb) | complete graph |
| Peptides-func | [https://github.com/vijaydwivedi75/lrgb](https://github.com/vijaydwivedi75/lrgb) | complete graph |

## Execution identity and data roles

Every run uses source, resolved-configuration and unit-membership identities. The prepared split arrays record the executed memberships; registered design manifests specify the intended seeds and rules. The roles are `fit`, `explore`, `select`, `confirm`, `compare`, `admit` and `test`. Optimization seeds, ensemble members, resampling seeds, evaluator and statistical unit must remain attached to each relevant result. [The registration guide](EVIDENCE_REGISTRATION.md) explains the original-run declarations.

Current runtime identity: `d96876520ad80478dddcc71f8fdc1c1f277b7734da87681f047f4192f037c651`.

| Bundled synthetic run | Source identity | Sealed stages |
|---|---|---|
| `examples/runs/arxiv` | `b5005fc53d31dfcb430c3b8f82a1fcbb7f97c2e8cb41eefbcb4a1f87e7a4bb5e` | 8 |
| `examples/runs/peptides_func` | `b5005fc53d31dfcb430c3b8f82a1fcbb7f97c2e8cb41eefbcb4a1f87e7a4bb5e` | 8 |

## Registered experimental artifacts

[runs/index.json](../runs/index.json) and [records/index.json](../records/index.json) are the registration sources. Present archive payloads are checked against their sealed stage chains and file inventories; measurement sets are checked against their manifests and raw-value traces. An absent archive with a deposit reference requires a matching committed registration manifest. Public retrieval remains a separate check.

No run archive is registered in runs/index.json. The measured outputs of the reported experiments are listed in [release/paper_artifacts.json](../release/paper_artifacts.json) as assets of the release paper-artifacts; original completed runs or new reproductions can also be registered here with their actual source identities and an explicit relationship to the paper.

An `exact_result_producing_run` declaration identifies an author-declared original run; `post_result_revision` identifies a later reproduction, and `refactor_with_prediction_equivalence` identifies the declared equivalence case. This catalog checks declaration identities. It does not authenticate historical chronology, recompute result bindings or replace the scientific evidence audit.

## Regeneration and verification

```bash
python -B scripts/artifact_catalog.py
python -B scripts/artifact_catalog.py --check
python -B scripts/artifact_catalog.py --output-dir /abs/catalog-review
```

The specification is [artifact_catalog_spec.json](../release/artifact_catalog_spec.json). `--check` validates its types, required coverage, safe paths, producer references, file hashes and registrations, then checks that both generated files are current. It executes no training, shell recipe or downloaded code.
