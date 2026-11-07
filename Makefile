PYTHON ?= python3
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

.PHONY: scientific-audit
scientific-audit:
	"$(PYTHON)" -B scripts/scientific_evidence_audit.py --index "$(RUNS_INDEX)" --records "$(RECORDS_ROOT)" \
	  --output "$(REPORT_ROOT)/scientific-evidence-audit.json" \
	  $(if $(BUDGET_CAMPAIGN),--budget-campaign "$(BUDGET_CAMPAIGN)",) \
	  $(if $(CONFIRMATION_CAMPAIGN),--confirmation-campaign "$(CONFIRMATION_CAMPAIGN)",) \
	  $(if $(MANUSCRIPT_PDF),--manuscript-pdf "$(MANUSCRIPT_PDF)",) \
	  $(if $(SUPPLEMENTARY_PDF),--supplementary-pdf "$(SUPPLEMENTARY_PDF)",)

# ---------------------------------------------------------------- locations
WORK_ROOT ?= ../cm_work
RUN_ROOT ?= $(WORK_ROOT)/runs/sample
REPORT_ROOT ?= $(WORK_ROOT)/reports
DATA_ROOT ?= $(WORK_ROOT)/data
DATASET ?= ogbn-arxiv
PROFILE ?= pilot
DEVICE ?= cpu
THREADS ?= 2
EXAMPLE_RUNS_ROOT ?= examples/runs
EXAMPLE_INVENTORY_ROOT ?= examples/inventory
EXAMPLE_GENERATED_ROOT ?= examples/generated
DATASET_STEMS := arxiv products proteins roman_empire pascalvoc_sp coco_sp peptides_func
EXAMPLE_STEMS := arxiv peptides_func
RUNS_INDEX ?= runs/index.json
RECORDS_ROOT ?= records
RECORDS_INDEX ?= $(RECORDS_ROOT)/index.json
RECONSTRUCTION_ROOT ?= $(REPORT_ROOT)/reconstruction
# One-command reconstruction from the outputs published with the repository's release.
ARTIFACTS_ROOT ?= ../cm-artifacts
PAPER_ROOT ?= ../cm-paper
PAPER_TIER ?= analysis
PUBLISH_MAP ?=
RECONSTRUCTION_EXAMPLES_ROOT ?= $(REPORT_ROOT)/reconstruction-examples
SUPPLEMENT_ROOT ?= $(REPORT_ROOT)/supplement
REQUIRE_ALL_DISPLAYS ?= 0
EXAMPLE_RECORDS_ROOT ?= examples/records
RECORDS_SPECS ?= records/specs.json
# Out-of-lifecycle experiments (E3-E6); see docs/EXPERIMENTS.md.
ARM_RUN ?=
ARM_OUTPUT ?= ../cm_budget_arms
ARM_EPOCHS ?= 100
ARM_COST_BOUNDS ?=
ARM_THREADS ?= 2
ARM_EXPECTED_RATIO ?=
ARM_HARD_CAP_RATIO ?=
ARM_SEED ?= 17
ARM_DATASET ?= ogbn-arxiv
ARM_PROFILE ?= pilot
ARM_CAMPAIGN ?= ../arm_campaign.json
ARM_EXPORT ?= ../budget_matched
TRUST_CHECKPOINTS ?= 0
PAPER_EXAMPLES_ROOT ?= paper/generated/examples
PAPER_OFFICIAL_ROOT ?= paper/generated/official
MANIFEST_ROOT ?= manifests
REGISTRY_FIGURE_ROOT ?= docs/assets
DIAGRAM_ROOT ?= docs/assets
VENV_DIR ?= ../cm_cpu_env
PRIVATE_ZIP ?= ../cross_morse_private.zip
PUBLIC_ZIP ?= ../cross_morse_public.zip
VALIDATION_RECORD ?= docs/VALIDATION.json
ARCHIVE ?=
OUTPUT ?=
DOI ?=
# Reviewer workflow, deposits and containers
REVIEW_ROOT ?= $(REPORT_ROOT)/review
REVIEW_OFFLINE ?=
REQUIRE_EVIDENCE ?=
BUDGET_CAMPAIGN ?=
CONFIRMATION_CAMPAIGN ?=
MANUSCRIPT_PDF ?=
SUPPLEMENTARY_PDF ?=
DECISION_WARN_ONLY ?=
DEPOSITS_DIR ?= ../cm-deposits
PUBLISH ?=
SANDBOX ?=
CODE_DOI ?=
DEPOSIT_LICENSE ?=
DOCKER_IMAGE ?= cross-morse:verify
# Evidence registration (authors): plan-driven preflight, packing, deposit, registration and bindings
EVIDENCE_PLAN ?=
EVIDENCE_DATASET ?=
EVIDENCE_ROOT ?=
DISCOVERY_OUTPUT ?= $(REPORT_ROOT)/evidence-discovery.json
DISCOVERY_MAX_DEPTH ?= 6
DISCOVERY_MAX_DIRECTORIES ?= 10000
FULL_AUDIT ?= auto
ALLOW_BLOCKED ?=
ALLOW_INTEGRITY_ONLY ?=
RELATIONSHIP ?=
SET ?=
MAPPING ?=
REPLACE ?=
REVIEW_REPORT ?=
READINESS_OUTPUT ?=
REVIEW_NOTES ?=
REVIEW_TEMPLATE ?=

# Publication and review switches accept only empty/0/1; nonempty is not consent.
define check-boolean
@case "$($(1))" in ""|0|1) ;; *) printf '%s\n' '$(1) must be 0 or 1' >&2; exit 2 ;; esac
endef

.PHONY: help verify check test test-no-torch test-release \
 tables figures paper paper-offline paper-from-index paper-from-runs fetch-artifacts paper-artifacts-status \
 paper-artifacts-plan paper-artifacts-check statement statement-check example-paper example-paper-check \
 manifests manifests-check availability availability-check \
 records-index records-check records-tables example-records example-records-check \
 arm-help arm-run arm-check arm-parent arm-replay arm-campaign-template arm-campaign-plan arm-campaign-run \
 arm-export arm-export-partial arm-campaign-replay \
 archive-run archive-campaign ingest-run runs-check compare compare-runs model-inventory \
 reproduce audit-artifacts example-views-check audit fingerprint source-id \
 registry-check registry-views diagrams diagrams-png diagrams-check \
 manifest verify-manifest release-private release-public release \
 prepare-data train-sample train-samples sample samples smoke replay reexecute review-bundle run-figures \
 env-cpu environment-check \
 review fetch-deposits deposit deposit-plan deposit-code claims claims-check \
 degeneracy-check degeneracy-check-registry decision-check availability-statement availability-statement-check \
 docker-build docker-verify docker-review \
 evidence-discover evidence-plan evidence-preflight evidence-pack evidence-deposit evidence-publish evidence-register \
 evidence-bind evidence-records evidence-status campaign-binding record-mapping-template record-from-mapping \
 manuscript-sources register-manuscript-sources readiness \
 theory theory-full theory-doc si-check si-report \
 faq-check paper-report paper-report-check confirmation-audit \
 artifact-catalog artifact-catalog-check reconstruct reconstruction-example reconstruction-check supplement

help:
	@printf '%s\n' \
	  'CROSS-MORSE: start with docs/GETTING_STARTED.md' \
	  '' \
	  'Install and check:' \
	  '  make env-cpu              create the recorded CPU environment' \
	  '  make verify               verify source, sealed examples and generated views; no training' \
	  '  make test                 run the full test suite (PyTorch required)' \
	  '  make test-no-torch        run the numeric/release tests without PyTorch' \
	  '  make model-inventory     count all seven released full-profile plants (PyTorch; no data download)' \
	  '  make theory              verify every analytical statement numerically (NumPy; seconds)' \
	  '  make si-check            recompute the derived entries of the supplementary tables' \
	  '  make paper-report        render the paper results and all table/figure support mappings' \
	  '  make paper-report-check  check current manuscript, display inventory and report freshness' \
	  '  make artifact-catalog-check  verify the typed artifact catalog and source identities' \
	  '  make faq-check           check that docs/REVIEWER_FAQ.md names only existing files and targets' \
	  '' \
	  'Reproduce the paper on a Slurm GPU cluster (docs/HPC.md; edit SITE SETTINGS in run_cross_morse.sh):' \
	  '  bash run_cross_morse.sh check             settings, runtime fingerprint and job plan; submits nothing' \
	  '  bash run_cross_morse.sh reproduce smoke   the whole job graph on synthetic data, in minutes' \
	  '  bash run_cross_morse.sh reproduce         all seven datasets; the suite report compares them with the paper' \
	  '  bash run_cross_morse.sh status | report   follow the latest suite; rebuild its report' \
	  '' \
	  'Run and replay locally:' \
	  '  make reproduce            replay the two bundled synthetic examples into REPORT_ROOT' \
	  '  make example-paper        rebuild tables and figures from those examples' \
	  '  make reconstruction-example  build one hashed workspace with coverage of every display' \
	  '  make train-sample DATASET=Peptides-func   execute a fresh synthetic CPU lifecycle' \
	  '  make prepare-data DATASET=ogbn-arxiv PROFILE=pilot   prepare official data; may download' \
	  '  make replay / reexecute   replay statistics or checkpoints from RUN_ROOT' \
	  '' \
	  'Registered evidence (run archives and record sets registered in runs/ and records/):' \
	  '  make paper                build supported tables/figures from official archives and records' \
	  '  make reconstruct          build a traced workspace from registered runs, records and optional campaigns' \
	  '  make reconstruction-check verify the exported workspace file hashes' \
	  '  make supplement           build factorial, ensemble, confirmation and budget outputs from supplied records' \
	  '  make compare-runs         compare official model inventories and results with the manuscript' \
	  '  make scientific-audit     report missing evidence and model/decision mismatches; fails if incomplete' \
	  '  make confirmation-audit CONFIRMATION_CAMPAIGN=...  verify the original joint S14 family' \
	  '' \
	  'Reviewers:' \
	  '  make review               fetch registered deposits, rebuild displays, audit evidence, write REVIEW_REPORT.md' \
	  '  make review REVIEW_OFFLINE=1   the same on the bundled synthetic examples; no network' \
	  '  make fetch-deposits       fetch and verify registered deposits into an external review directory' \
	  '  make claims               regenerate paper/CLAIMS.md (claim-to-evidence ledger) and the coverage badge' \
	  '' \
	  'Authors (deposit and guards):' \
	  '  make deposit ARCHIVE=/abs/x.tar.gz DEPOSIT_LICENSE=... [PUBLISH=1] [SANDBOX=1]   draft by default' \
	  '  make deposit-code [PUBLISH=1]                                deposit git archive HEAD as software' \
	  '  make degeneracy-check / decision-check                       guards on Table S15b and Table 1' \
	  '  make availability-statement                                  journal availability paragraph' \
	  '  make docker-verify                                           verify and test inside the CPU container' \
	  '' \
	  'Evidence registration (docs/EVIDENCE_REGISTRATION.md; EVIDENCE_PLAN=/abs/plan.json):' \
	  '  make evidence-discover EVIDENCE_ROOT=/abs/export [EVIDENCE_PLAN=/abs/new-plan.json]   inspect completed runs' \
	  '  make evidence-plan EVIDENCE_PLAN=...   write an empty plan; then evidence-preflight, -pack, -deposit,' \
	  '  -publish PUBLISH=1, -register, -bind, -records and -status in that order' \
	  '  make readiness [MANUSCRIPT_PDF=... SUPPLEMENTARY_PDF=... BUDGET_CAMPAIGN=... CONFIRMATION_CAMPAIGN=... REVIEW_REPORT=...]' \
	  '  make manuscript-sources MANUSCRIPT_PDF=... SUPPLEMENTARY_PDF=... REVIEW_TEMPLATE=...   PDF identity and review aid' \
	  '' \
	  'Advanced workflows: docs/EXPERIMENTS.md, docs/RUN_ARCHIVES.md, docs/HPC.md.' \
	  'Common variables: PYTHON, DATASET, PROFILE, RUN_ROOT, REPORT_ROOT, DATA_ROOT, DEVICE, THREADS.'

# ---------------------------------------------------------------- paper from runs
tables:
	"$(PYTHON)" -B scripts/build_paper.py tables --index "$(RUNS_INDEX)" --label official --output-dir "$(PAPER_OFFICIAL_ROOT)"

figures:
	"$(PYTHON)" -B scripts/build_paper.py figures --index "$(RUNS_INDEX)" --label official --output-dir "$(PAPER_OFFICIAL_ROOT)"

paper-from-runs:
	"$(PYTHON)" -B scripts/build_paper.py all --index "$(RUNS_INDEX)" --label official --output-dir "$(PAPER_OFFICIAL_ROOT)"

# The one command of the availability statement: download and verify the published outputs,
# rebuild every numbered table and figure with its source data, compare with the printed values.
paper:
	"$(PYTHON)" -B scripts/reconstruct_paper.py --artifacts "$(ARTIFACTS_ROOT)" --output-dir "$(PAPER_ROOT)" \
	  --tier "$(PAPER_TIER)"

paper-offline:
	"$(PYTHON)" -B scripts/reconstruct_paper.py --offline --artifacts "$(ARTIFACTS_ROOT)" --output-dir "$(PAPER_ROOT)"

paper-from-index: paper-from-runs records-tables

fetch-artifacts:
	"$(PYTHON)" -B scripts/paper_artifacts.py fetch --dest "$(ARTIFACTS_ROOT)" --tier "$(PAPER_TIER)"

paper-artifacts-status:
	"$(PYTHON)" -B scripts/paper_artifacts.py status

paper-artifacts-plan:
	"$(PYTHON)" -B scripts/paper_artifacts.py plan

paper-artifacts-check:
	"$(PYTHON)" -B scripts/paper_artifacts.py plan --check

statement:
	"$(PYTHON)" -B scripts/statement_check.py --write

statement-check:
	"$(PYTHON)" -B scripts/statement_check.py --check

# A single workspace of the supported outputs, with every manuscript display accounted for.
reconstruct:
	$(call check-boolean,REQUIRE_ALL_DISPLAYS)
	"$(PYTHON)" -B scripts/reconstruct.py --index "$(RUNS_INDEX)" --records-index "$(RECORDS_INDEX)" \
	  --output-dir "$(RECONSTRUCTION_ROOT)" --label registered-inputs \
	  $(if $(CONFIRMATION_CAMPAIGN),--confirmation-campaign "$(CONFIRMATION_CAMPAIGN)",) \
	  $(if $(BUDGET_CAMPAIGN),--budget-campaign "$(BUDGET_CAMPAIGN)",) \
	  $(if $(filter 1,$(REQUIRE_ALL_DISPLAYS)),--require-all-displays,)

reconstruction-example:
	"$(PYTHON)" -B scripts/reconstruct.py --runs-root "$(EXAMPLE_RUNS_ROOT)" \
	  --records-index "$(EXAMPLE_RECORDS_ROOT)/index.json" --label synthetic-examples \
	  --output-dir "$(RECONSTRUCTION_EXAMPLES_ROOT)"

reconstruction-check:
	"$(PYTHON)" -B scripts/reconstruct.py --verify "$(RECONSTRUCTION_ROOT)"

supplement:
	"$(PYTHON)" -B scripts/build_supplement.py --records-index "$(RECORDS_INDEX)" \
	  --output-dir "$(SUPPLEMENT_ROOT)" \
	  $(if $(CONFIRMATION_CAMPAIGN),--confirmation-campaign "$(CONFIRMATION_CAMPAIGN)",) \
	  $(if $(BUDGET_CAMPAIGN),--budget-campaign "$(BUDGET_CAMPAIGN)",)

artifact-catalog:
	"$(PYTHON)" -B scripts/artifact_catalog.py

artifact-catalog-check:
	"$(PYTHON)" -B scripts/artifact_catalog.py --check

example-paper:
	"$(PYTHON)" -B scripts/build_paper.py all --runs-root "$(EXAMPLE_RUNS_ROOT)" --label examples --output-dir "$(PAPER_EXAMPLES_ROOT)"

example-paper-check:
	"$(PYTHON)" -B scripts/build_paper.py all --runs-root "$(EXAMPLE_RUNS_ROOT)" --label examples --output-dir "$(PAPER_EXAMPLES_ROOT)" --check

compare-runs:
	"$(PYTHON)" -B scripts/compare_registered_runs.py --index "$(RUNS_INDEX)" --output-dir "$(REPORT_ROOT)/compare"

model-inventory:
	"$(PYTHON)" -B scripts/model_inventory.py --all-full --output "$(REPORT_ROOT)/model-inventory.json"

compare:
	"$(PYTHON)" -B scripts/compare_with_manuscript.py "$(RUN_ROOT)" --output "$(REPORT_ROOT)/manuscript_comparison.json"

# ---------------------------------------------------------------- run archives
archive-run:
	@test -n "$(RUN_ROOT)" -a -n "$(OUTPUT)" || { printf '%s\n' 'usage: make archive-run RUN_ROOT=/abs/suite OUTPUT=/abs/archive.tar.gz' >&2; exit 2; }
	"$(PYTHON)" -B scripts/archive_run.py "$(RUN_ROOT)" --output "$(OUTPUT)"

archive-campaign:
	@test -n "$(RUN_ROOT)" -a -n "$(OUTPUT)" || { printf '%s\n' 'usage: make archive-campaign RUN_ROOT=/abs/campaign OUTPUT=/abs/archive.tar.gz' >&2; exit 2; }
	"$(PYTHON)" -B scripts/archive_run.py "$(RUN_ROOT)" --campaign --output "$(OUTPUT)"

ingest-run:
	@test -n "$(ARCHIVE)" || { printf '%s\n' 'usage: make ingest-run ARCHIVE=/abs/archive.tar.gz [DOI=10.5281/zenodo.N]' >&2; exit 2; }
	@test "$$(realpath -m -- "$(RUNS_INDEX)")" = "$$(realpath -m -- runs/index.json)" -o "$$(realpath -m -- "$(MANIFEST_ROOT)")" != "$$(realpath -m -- manifests)" || { printf '%s\n' 'An external RUNS_INDEX requires a separate MANIFEST_ROOT; canonical manifests are preserved.' >&2; exit 2; }
	"$(PYTHON)" -B scripts/ingest_run.py "$(ARCHIVE)" --index "$(RUNS_INDEX)" $(if $(DOI),--doi "$(DOI)",)
	"$(PYTHON)" -B scripts/run_manifests.py runs --index "$(RUNS_INDEX)" --label official --output-root "$(MANIFEST_ROOT)" --with-indices --allow-deposited-absent
	@if [ "$$(realpath -m -- "$(RUNS_INDEX)")" = "$$(realpath -m -- runs/index.json)" ]; then \
	  "$(PYTHON)" -B scripts/availability.py; \
	  "$(PYTHON)" -B scripts/claims_ledger.py; \
	  "$(PYTHON)" -B scripts/manuscript_availability.py; \
	else printf '%s\n' 'External index registered; canonical availability views were preserved. Use claims_ledger.py --index/--output and manuscript_availability.py --index/--output for scoped reports.'; fi

records-index:
	"$(PYTHON)" -B scripts/records.py index
	"$(PYTHON)" -B scripts/availability.py
	"$(PYTHON)" -B scripts/claims_ledger.py
	"$(PYTHON)" -B scripts/manuscript_availability.py

records-check:
	"$(PYTHON)" -B scripts/records.py check

records-tables:
	"$(PYTHON)" -B scripts/records.py tables --output-dir "$(PAPER_OFFICIAL_ROOT)/records"

example-records:
	"$(PYTHON)" -B scripts/records.py tables --root "$(EXAMPLE_RECORDS_ROOT)" --specs "$(RECORDS_SPECS)" --output-dir "$(PAPER_EXAMPLES_ROOT)/records"

example-records-check:
	"$(PYTHON)" -B scripts/records.py check --root "$(EXAMPLE_RECORDS_ROOT)" --specs "$(RECORDS_SPECS)"
	@tmp="$$(mktemp -d)"; trap 'rm -rf "$$tmp"' EXIT; \
	 "$(PYTHON)" -B scripts/records.py tables --root "$(EXAMPLE_RECORDS_ROOT)" --specs "$(RECORDS_SPECS)" --output-dir "$$tmp" >/dev/null; \
	 diff -r "$$tmp" "$(PAPER_EXAMPLES_ROOT)/records" >/dev/null || { echo 'paper/generated/examples/records differs from a fresh build (make example-records)' >&2; exit 1; }; \
	 echo 'example record tables match a fresh build'

arm-help:
	"$(PYTHON)" -B scripts/run_budget_arms.py --help

arm-run:
	@test -n "$(ARM_RUN)" || { echo 'Set ARM_RUN to a trusted, confirmed source run' >&2; exit 2; }
	@test "$(TRUST_CHECKPOINTS)" = 1 || { echo 'Set TRUST_CHECKPOINTS=1 for trusted checkpoints' >&2; exit 2; }
	"$(PYTHON)" -B scripts/run_budget_arms.py --run "$(ARM_RUN)" --output "$(ARM_OUTPUT)" --device "$(DEVICE)" --threads "$(ARM_THREADS)" --epochs "$(ARM_EPOCHS)" --seed "$(ARM_SEED)" --trust-local-checkpoints $(if $(ARM_EXPECTED_RATIO),--budget-ratio "$(ARM_EXPECTED_RATIO)",) $(if $(ARM_HARD_CAP_RATIO),--hard-cap-ratio "$(ARM_HARD_CAP_RATIO)",) $(if $(ARM_COST_BOUNDS),--cost-bounds "$(ARM_COST_BOUNDS)",)

arm-check:
	"$(PYTHON)" -B scripts/run_budget_arms.py --verify "$(ARM_OUTPUT)"

arm-parent:
	@test -n "$(ARM_RUN)" || { echo 'Set a fresh ARM_RUN or a pre-admission parent' >&2; exit 2; }
	@test ! -f "$(ARM_RUN)/admit/decision.json" && test ! -f "$(ARM_RUN)/final/results_summary.json" || { echo 'Parent already used admission/test; use a fresh directory' >&2; exit 2; }
	@set -e; for stage in prepare train explore fit select confirm; do "$(PYTHON)" -B run_cross_morse.py --dataset "$(ARM_DATASET)" --profile "$(ARM_PROFILE)" --device "$(DEVICE)" --threads "$(ARM_THREADS)" --data-root "$(DATA_ROOT)" --output-dir "$(ARM_RUN)" --stage "$$stage" --resume; done

arm-replay:
	@test -n "$(ARM_RUN)" || { echo 'Set ARM_RUN to the immutable parent' >&2; exit 2; }
	@test "$(TRUST_CHECKPOINTS)" = 1 || { echo 'Explicit checkpoint trust required' >&2; exit 2; }
	"$(PYTHON)" -B scripts/run_budget_arms.py --replay "$(ARM_OUTPUT)" --run "$(ARM_RUN)" --device "$(DEVICE)" --threads "$(ARM_THREADS)" --trust-local-checkpoints

arm-campaign-template:
	"$(PYTHON)" -B scripts/arm_campaign.py template --output "$(ARM_CAMPAIGN)"

arm-campaign-plan:
	"$(PYTHON)" -B scripts/arm_campaign.py execute --manifest "$(ARM_CAMPAIGN)" --device "$(DEVICE)" --threads "$(ARM_THREADS)" --epochs "$(ARM_EPOCHS)" --dry-run

arm-campaign-run:
	@test "$(TRUST_CHECKPOINTS)" = 1 || { echo 'Explicit checkpoint trust required' >&2; exit 2; }
	"$(PYTHON)" -B scripts/arm_campaign.py execute --manifest "$(ARM_CAMPAIGN)" --device "$(DEVICE)" --threads "$(ARM_THREADS)" --epochs "$(ARM_EPOCHS)" --trust-local-checkpoints

arm-export:
	"$(PYTHON)" -B scripts/arm_campaign.py export --manifest "$(ARM_CAMPAIGN)" --output "$(ARM_EXPORT)" --complete

arm-campaign-replay:
	@test "$(TRUST_CHECKPOINTS)" = 1 || { echo 'Explicit checkpoint trust required' >&2; exit 2; }
	"$(PYTHON)" -B scripts/arm_campaign.py replay --directory "$(ARM_EXPORT)" --device "$(DEVICE)" --trust-local-checkpoints

arm-export-partial:
	"$(PYTHON)" -B scripts/arm_campaign.py export --manifest "$(ARM_CAMPAIGN)" --output "$(ARM_EXPORT)"

runs-check:
	"$(PYTHON)" -B scripts/ingest_run.py --check --index "$(RUNS_INDEX)" --manifests-dir "$(MANIFEST_ROOT)/official"

# ---------------------------------------------------------------- manifests, availability
manifests:
	"$(PYTHON)" -B scripts/run_manifests.py registered --output-root "$(MANIFEST_ROOT)"
	"$(PYTHON)" -B scripts/run_manifests.py runs --runs-root "$(EXAMPLE_RUNS_ROOT)" --label examples --output-root "$(MANIFEST_ROOT)"
	@if [ -f "$(RUNS_INDEX)" ]; then "$(PYTHON)" -B scripts/run_manifests.py runs --index "$(RUNS_INDEX)" --label official --output-root "$(MANIFEST_ROOT)" --with-indices --allow-empty --allow-deposited-absent; fi

manifests-check:
	"$(PYTHON)" -B scripts/run_manifests.py registered --output-root "$(MANIFEST_ROOT)" --check
	"$(PYTHON)" -B scripts/run_manifests.py runs --runs-root "$(EXAMPLE_RUNS_ROOT)" --label examples --output-root "$(MANIFEST_ROOT)" --check
	@if [ -f "$(RUNS_INDEX)" ]; then "$(PYTHON)" -B scripts/run_manifests.py runs --index "$(RUNS_INDEX)" --label official --output-root "$(MANIFEST_ROOT)" --check --allow-empty --allow-deposited-absent; fi

availability:
	"$(PYTHON)" -B scripts/availability.py

availability-check:
	"$(PYTHON)" -B scripts/availability.py --check \
	  $(if $(BUDGET_CAMPAIGN),--budget-campaign "$(BUDGET_CAMPAIGN)",) \
	  $(if $(CONFIRMATION_CAMPAIGN),--confirmation-campaign "$(CONFIRMATION_CAMPAIGN)",)

# ---------------------------------------------------------------- bundled examples
audit-artifacts:
	"$(PYTHON)" -B scripts/artifact_inventory.py --check-static
	@for stem in $(EXAMPLE_STEMS); do \
	  "$(PYTHON)" -B scripts/artifact_inventory.py "$(EXAMPLE_RUNS_ROOT)/$$stem" \
	    --output "$(EXAMPLE_INVENTORY_ROOT)/$$stem.json" --check; \
	done

# Recompute decisions from observations. No PyTorch/checkpoint loading or fitting.
reproduce: audit-artifacts
	"$(PYTHON)" -B scripts/reproduce_examples.py --runs-root "$(EXAMPLE_RUNS_ROOT)" \
	  --output-dir "$(REPORT_ROOT)" --figures

example-views-check:
	"$(PYTHON)" -B scripts/reproduce_examples.py --runs-root "$(EXAMPLE_RUNS_ROOT)" \
	  --output-dir "$(EXAMPLE_GENERATED_ROOT)" --check

# ---------------------------------------------------------------- manuscript registry (comparison only)
registry-check:
	"$(PYTHON)" -B scripts/check_manuscript_sources.py
	"$(PYTHON)" -B scripts/check_paper_constants.py
	"$(PYTHON)" -B scripts/model_inventory.py --check-docs
	"$(PYTHON)" -B scripts/render_paper_results.py --validate-only
	"$(PYTHON)" -B scripts/draw_diagrams.py --check --png

registry-views:
	"$(PYTHON)" -B scripts/render_paper_results.py --output-dir "$(REPORT_ROOT)/registry"
	"$(PYTHON)" -B scripts/plot_paper_summary.py --output-dir "$(REGISTRY_FIGURE_ROOT)"
	"$(PYTHON)" -B scripts/plot_registry_figures.py --output-dir "$(REGISTRY_FIGURE_ROOT)/registry"
	"$(PYTHON)" -B scripts/draw_diagrams.py --output-dir "$(DIAGRAM_ROOT)"

diagrams:
	"$(PYTHON)" -B scripts/draw_diagrams.py --output-dir "$(DIAGRAM_ROOT)"

diagrams-png:
	"$(PYTHON)" -B scripts/draw_diagrams.py --output-dir "$(DIAGRAM_ROOT)" --png

diagrams-check:
	"$(PYTHON)" -B scripts/draw_diagrams.py --check --png

# ---------------------------------------------------------------- analytical statements and supplementary arithmetic
# NumPy and the standard library only. docs/THEORY.md maps every statement to its code and checks.
theory:
	"$(PYTHON)" -B scripts/verify_theory.py --check-doc

theory-full:
	"$(PYTHON)" -B scripts/verify_theory.py --full --json-output "$(REPORT_ROOT)/theory-full.json" --markdown "$(REPORT_ROOT)/theory-full.md"

theory-doc:
	"$(PYTHON)" -B scripts/verify_theory.py --write-doc

si-check:
	"$(PYTHON)" -B scripts/check_si_identities.py --quiet

si-report:
	"$(PYTHON)" -B scripts/check_si_identities.py --json-output "$(REPORT_ROOT)/si-identities.json" --markdown "$(REPORT_ROOT)/si-identities.md"

# ---------------------------------------------------------------- whole-tree checks
verify: registry-check theory si-check paper-report-check artifact-catalog-check faq-check fingerprint audit-artifacts example-views-check example-paper-check manifests-check records-check example-records-check availability-check claims-check availability-statement-check paper-artifacts-check statement-check runs-check-if-present audit
check: verify

.PHONY: runs-check-if-present
runs-check-if-present:
	@if [ -f "$(RUNS_INDEX)" ]; then "$(PYTHON)" -B scripts/ingest_run.py --check --index "$(RUNS_INDEX)" --manifests-dir "$(MANIFEST_ROOT)/official" --allow-deposited-absent; else printf '%s\n' 'selected runs index absent: no official run archives registered yet'; fi

audit:
	"$(PYTHON)" -B scripts/release_audit.py --root . --strict

fingerprint:
	"$(PYTHON)" -B scripts/source_fingerprint.py --validation-record "$(VALIDATION_RECORD)"

source-id:
	"$(PYTHON)" -B scripts/source_fingerprint.py

test:
	"$(PYTHON)" -B -m unittest discover -s tests -v

NUMPY_TESTS := test_paper_registry test_records test_export_records test_availability test_paper_build test_run_archives \
 test_artifact_inventory test_reproduce_examples test_statistics test_evaluation test_compare_with_manuscript \
 test_release_tooling test_release_launcher test_source_fingerprint_cli test_draw_diagrams test_make_workflows \
 test_dataset_entrypoints test_environment_spec \
 test_experiment_accounting test_experiment_budget test_experiment_revision test_experiment_upstream \
 test_cr8_availability test_cr8_source_and_coverage \
 test_cr8_role_evidence test_cr8_model_parity test_cr8_source_snapshots test_cr8_empty_manifests \
 test_uplift_tools test_uplift_make_workflows test_deposit_transport \
 test_evidence_registration test_registration_pipeline_safety test_registration_sources_safety test_registration_clone_safety \
 test_evidence_discovery \
 test_theory_battery test_theory_formulas test_theory_review_regressions test_si_identities \
 test_arrhenius_discovery test_reviewer_faq test_paper_support_report test_review_confirmation test_s14_confirmation test_selection_figure \
 test_reproduction test_reproduction_launcher test_artifact_catalog test_reconstruction test_supplement_build \
 test_paper_artifacts test_reconstruct_paper test_statement_check \
 test_displays_main_paper test_displays_run_audits test_displays_records_campaigns test_displays_protocol_tables \
 test_displays_analytical test_displays_experiments test_sweeps test_experiments_runners_a test_experiments_runners_b \
 test_timing_records test_confirmation_analysis

test-no-torch:
	"$(PYTHON)" -B -m unittest $(addprefix tests.,$(NUMPY_TESTS)) -v

test-release:
	"$(PYTHON)" -B -m unittest discover -s tests -p 'test_release_*.py' -v
	"$(PYTHON)" -B -m unittest discover -s tests -p test_make_workflows.py -v
	"$(PYTHON)" -B -m unittest discover -s tests -p test_environment_spec.py -v

manifest:
	"$(PYTHON)" -B scripts/create_release_manifest.py --root . --output RELEASE_MANIFEST.json

verify-manifest:
	"$(PYTHON)" -B scripts/create_release_manifest.py --root . --verify RELEASE_MANIFEST.json

release-private:
	"$(PYTHON)" -B scripts/build_release.py --source . --profile private --output "$(PRIVATE_ZIP)"

release-public:
	"$(PYTHON)" -B scripts/build_release.py --source . --profile public --output "$(PUBLIC_ZIP)"

release: release-public

# ---------------------------------------------------------------- current paper support
confirmation-audit:
	@test -n "$(CONFIRMATION_CAMPAIGN)" || { printf '%s\n' 'usage: make confirmation-audit CONFIRMATION_CAMPAIGN=/abs/campaign' >&2; exit 2; }
	"$(PYTHON)" -B scripts/audit_s14_confirmation.py "$(CONFIRMATION_CAMPAIGN)" --output "$(REPORT_ROOT)/confirmation-audit.json"

paper-report:
	"$(PYTHON)" -B scripts/paper_support_report.py

paper-report-check:
	"$(PYTHON)" -B scripts/paper_support_report.py --check

faq-check:
	"$(PYTHON)" -B scripts/check_reviewer_faq.py

# ---------------------------------------------------------------- reviewer workflow, deposits, guards
review:
	$(call check-boolean,REVIEW_OFFLINE)
	$(call check-boolean,REQUIRE_EVIDENCE)
	"$(PYTHON)" -B scripts/review.py --report-root "$(REVIEW_ROOT)" --index "$(RUNS_INDEX)" --records "$(RECORDS_ROOT)" \
	  --deposits-dir "$(DEPOSITS_DIR)" \
	  $(if $(filter 1,$(REVIEW_OFFLINE)),--offline,) $(if $(filter 1,$(REQUIRE_EVIDENCE)),--require-evidence,) \
	  $(if $(BUDGET_CAMPAIGN),--budget-campaign "$(BUDGET_CAMPAIGN)",) \
	  $(if $(CONFIRMATION_CAMPAIGN),--confirmation-campaign "$(CONFIRMATION_CAMPAIGN)",) \
	  $(if $(MANUSCRIPT_PDF),--manuscript-pdf "$(MANUSCRIPT_PDF)",) \
	  $(if $(SUPPLEMENTARY_PDF),--supplementary-pdf "$(SUPPLEMENTARY_PDF)",)

fetch-deposits:
	"$(PYTHON)" -B scripts/fetch_deposits.py --index "$(RUNS_INDEX)" --deposits-dir "$(DEPOSITS_DIR)" --output "$(REPORT_ROOT)/fetch-deposits.json"

deposit-plan:
	@test -n "$(ARCHIVE)" || { printf '%s\n' 'usage: make deposit-plan ARCHIVE=/abs/archive.tar.gz' >&2; exit 2; }
	"$(PYTHON)" -B scripts/deposit_zenodo.py plan "$(ARCHIVE)" $(if $(CODE_DOI),--code-doi "$(CODE_DOI)",) $(if $(DEPOSIT_LICENSE),--license "$(DEPOSIT_LICENSE)",)

deposit:
	$(call check-boolean,PUBLISH)
	$(call check-boolean,SANDBOX)
	@test -n "$(ARCHIVE)" || { printf '%s\n' 'usage: make deposit ARCHIVE=/abs/archive.tar.gz DEPOSIT_LICENSE=... [PUBLISH=1] [SANDBOX=1] [CODE_DOI=10.5281/zenodo.N]' >&2; exit 2; }
	"$(PYTHON)" -B scripts/deposit_zenodo.py archive "$(ARCHIVE)" $(if $(filter 1,$(PUBLISH)),--publish,) $(if $(filter 1,$(SANDBOX)),--sandbox,) $(if $(CODE_DOI),--code-doi "$(CODE_DOI)",) $(if $(DEPOSIT_LICENSE),--license "$(DEPOSIT_LICENSE)",)

deposit-code:
	$(call check-boolean,PUBLISH)
	$(call check-boolean,SANDBOX)
	"$(PYTHON)" -B scripts/deposit_zenodo.py code --output-dir "$(DEPOSITS_DIR)" $(if $(filter 1,$(PUBLISH)),--publish,) $(if $(filter 1,$(SANDBOX)),--sandbox,)

claims:
	"$(PYTHON)" -B scripts/claims_ledger.py

claims-check:
	"$(PYTHON)" -B scripts/claims_ledger.py --check

degeneracy-check:
	"$(PYTHON)" -B scripts/check_measurement_degeneracy.py --records "$(RECORDS_ROOT)" --output "$(REPORT_ROOT)/degeneracy-check.json"

degeneracy-check-registry:
	"$(PYTHON)" -B scripts/check_measurement_degeneracy.py --registry --output "$(REPORT_ROOT)/degeneracy-check-registry.json"

decision-check:
	$(call check-boolean,DECISION_WARN_ONLY)
	"$(PYTHON)" -B scripts/check_decision_consistency.py --output "$(REPORT_ROOT)/decision-consistency.json" $(if $(filter 1,$(DECISION_WARN_ONLY)),--warn-only,)

availability-statement:
	"$(PYTHON)" -B scripts/manuscript_availability.py

availability-statement-check:
	"$(PYTHON)" -B scripts/manuscript_availability.py --check

docker-build:
	docker build -t "$(DOCKER_IMAGE)" .

docker-verify: docker-build
	docker run --rm --network none "$(DOCKER_IMAGE)" make verify test PYTHON=python

docker-review: docker-build
	mkdir -p "$(REPORT_ROOT)"
	@reports="$$(cd "$(REPORT_ROOT)" && pwd)"; \
	 docker run --rm --network none --user "$$(id -u):$$(id -g)" \
	  -e HOME=/tmp -e MPLCONFIGDIR=/tmp/matplotlib -v "$$reports":/reports \
	  "$(DOCKER_IMAGE)" make review PYTHON=python REVIEW_OFFLINE=1 REVIEW_ROOT=/reports/review

# ---------------------------------------------------------------- evidence registration (authors)
define require-plan
@test -n "$(EVIDENCE_PLAN)" || { printf '%s\n' 'set EVIDENCE_PLAN=/abs/plan.json (make evidence-plan EVIDENCE_PLAN=...)' >&2; exit 2; }
endef
EVIDENCE_SELECT = $(if $(EVIDENCE_DATASET),--dataset "$(EVIDENCE_DATASET)",)

evidence-discover:
	@test -n "$(EVIDENCE_ROOT)" || { printf '%s\n' 'usage: make evidence-discover EVIDENCE_ROOT=/abs/export [EVIDENCE_PLAN=/abs/new-plan.json]' >&2; exit 2; }
	"$(PYTHON)" -B scripts/evidence_pipeline.py discover --root "$(EVIDENCE_ROOT)" --output "$(DISCOVERY_OUTPUT)" \
	  --max-depth "$(DISCOVERY_MAX_DEPTH)" --max-directories "$(DISCOVERY_MAX_DIRECTORIES)" \
	  $(if $(EVIDENCE_PLAN),--plan-output "$(EVIDENCE_PLAN)" --deposits-dir "$(DEPOSITS_DIR)",)

evidence-plan:
	$(call require-plan)
	"$(PYTHON)" -B scripts/evidence_pipeline.py template --output "$(EVIDENCE_PLAN)"

evidence-preflight:
	$(call require-plan)
	"$(PYTHON)" -B scripts/evidence_pipeline.py preflight --plan "$(EVIDENCE_PLAN)" $(EVIDENCE_SELECT) --full-audit "$(FULL_AUDIT)"

evidence-pack:
	$(call require-plan)
	$(call check-boolean,ALLOW_BLOCKED)
	"$(PYTHON)" -B scripts/evidence_pipeline.py pack --plan "$(EVIDENCE_PLAN)" $(EVIDENCE_SELECT) \
	  $(if $(filter 1,$(ALLOW_BLOCKED)),--allow-blocked,)

evidence-deposit:
	$(call require-plan)
	$(call check-boolean,SANDBOX)
	"$(PYTHON)" -B scripts/evidence_pipeline.py deposit --plan "$(EVIDENCE_PLAN)" $(EVIDENCE_SELECT) \
	  $(if $(filter 1,$(SANDBOX)),--sandbox,)

evidence-publish:
	$(call require-plan)
	$(call check-boolean,PUBLISH)
	$(call check-boolean,SANDBOX)
	$(call check-boolean,ALLOW_BLOCKED)
	$(call check-boolean,ALLOW_INTEGRITY_ONLY)
	@test "$(PUBLISH)" = 1 || { printf '%s\n' 'evidence-publish creates permanent public records; repeat with PUBLISH=1' >&2; exit 2; }
	@test "$(ALLOW_BLOCKED)" != 1 || { printf '%s\n' 'ALLOW_BLOCKED is limited to evidence-pack; sandbox rehearsal uses SANDBOX=1.' >&2; exit 2; }
	"$(PYTHON)" -B scripts/evidence_pipeline.py publish --plan "$(EVIDENCE_PLAN)" $(EVIDENCE_SELECT) --confirm \
	  $(if $(filter 1,$(SANDBOX)),--sandbox,) \
	  $(if $(filter 1,$(ALLOW_INTEGRITY_ONLY)),--allow-integrity-only,)

evidence-register:
	$(call require-plan)
	"$(PYTHON)" -B scripts/evidence_pipeline.py register --plan "$(EVIDENCE_PLAN)" $(EVIDENCE_SELECT)

evidence-bind:
	$(call require-plan)
	"$(PYTHON)" -B scripts/evidence_pipeline.py bind --plan "$(EVIDENCE_PLAN)" $(EVIDENCE_SELECT)

evidence-records:
	$(call require-plan)
	$(call check-boolean,REPLACE)
	"$(PYTHON)" -B scripts/evidence_pipeline.py records --plan "$(EVIDENCE_PLAN)" \
	  $(if $(SET),--set "$(SET)",) $(if $(filter 1,$(REPLACE)),--replace,)

evidence-status:
	$(call require-plan)
	"$(PYTHON)" -B scripts/evidence_pipeline.py status --plan "$(EVIDENCE_PLAN)"

campaign-binding:
	@test -n "$(RUN_ROOT)" -a -n "$(RELATIONSHIP)" -a -n "$(OUTPUT)" || { printf '%s\n' 'usage: make campaign-binding RUN_ROOT=/abs/campaign RELATIONSHIP=exact_result_producing_run OUTPUT=/abs/binding.json' >&2; exit 2; }
	"$(PYTHON)" -B scripts/emit_campaign_binding.py "$(RUN_ROOT)" --relationship "$(RELATIONSHIP)" --output "$(OUTPUT)" --check

record-mapping-template:
	@test -n "$(SET)" -a -n "$(OUTPUT)" || { printf '%s\n' 'usage: make record-mapping-template SET=factorials OUTPUT=/abs/factorials.mapping.json' >&2; exit 2; }
	"$(PYTHON)" -B scripts/export_records.py mapping-template --set "$(SET)" --output "$(OUTPUT)"

record-from-mapping:
	$(call check-boolean,REPLACE)
	@test -n "$(MAPPING)" || { printf '%s\n' 'usage: make record-from-mapping MAPPING=/abs/set.mapping.json [REPLACE=1]' >&2; exit 2; }
	"$(PYTHON)" -B scripts/export_records.py from-mapping --mapping "$(MAPPING)" $(if $(filter 1,$(REPLACE)),--replace,)
	"$(PYTHON)" -B scripts/availability.py
	"$(PYTHON)" -B scripts/claims_ledger.py
	"$(PYTHON)" -B scripts/manuscript_availability.py

manuscript-sources:
	@test -n "$(MANUSCRIPT_PDF)" -a -n "$(SUPPLEMENTARY_PDF)" || { printf '%s\n' 'usage: make manuscript-sources MANUSCRIPT_PDF=/abs/main.pdf SUPPLEMENTARY_PDF=/abs/si.pdf' >&2; exit 2; }
	"$(PYTHON)" -B scripts/register_manuscript_sources.py --manuscript-pdf "$(MANUSCRIPT_PDF)" --supplementary-pdf "$(SUPPLEMENTARY_PDF)" \
	  $(if $(REVIEW_TEMPLATE),--review-template "$(REVIEW_TEMPLATE)",)

register-manuscript-sources:
	@test -n "$(MANUSCRIPT_PDF)" -a -n "$(SUPPLEMENTARY_PDF)" -a -n "$(REVIEW_NOTES)" || { printf '%s\n' 'usage: make register-manuscript-sources MANUSCRIPT_PDF=/abs/main.pdf SUPPLEMENTARY_PDF=/abs/si.pdf REVIEW_NOTES=/abs/review.json' >&2; exit 2; }
	"$(PYTHON)" -B scripts/register_manuscript_sources.py --manuscript-pdf "$(MANUSCRIPT_PDF)" --supplementary-pdf "$(SUPPLEMENTARY_PDF)" --register \
	  --review-notes "$(REVIEW_NOTES)"

readiness:
	"$(PYTHON)" -B scripts/readiness.py --index "$(RUNS_INDEX)" --records "$(RECORDS_ROOT)" --manifests-root "$(MANIFEST_ROOT)" \
	  $(if $(MANUSCRIPT_PDF),--manuscript-pdf "$(MANUSCRIPT_PDF)",) \
	  $(if $(SUPPLEMENTARY_PDF),--supplementary-pdf "$(SUPPLEMENTARY_PDF)",) \
	  $(if $(BUDGET_CAMPAIGN),--budget-campaign "$(BUDGET_CAMPAIGN)",) \
	  $(if $(CONFIRMATION_CAMPAIGN),--confirmation-campaign "$(CONFIRMATION_CAMPAIGN)",) \
	  $(if $(REVIEW_REPORT),--review-report "$(REVIEW_REPORT)",) \
	  $(if $(READINESS_OUTPUT),--output "$(READINESS_OUTPUT)",)

# ---------------------------------------------------------------- execution
prepare-data:
	@case "$(PROFILE)" in pilot|full) ;; *) printf '%s\n' 'prepare-data requires PROFILE=pilot or PROFILE=full; synthetic smoke is not official data.' >&2; exit 2 ;; esac
	"$(PYTHON)" -B run_cross_morse.py --dataset "$(DATASET)" --profile "$(PROFILE)" \
	  --device "$(DEVICE)" --threads "$(THREADS)" --stage prepare --data-root "$(DATA_ROOT)" --output-dir "$(RUN_ROOT)"

train-sample:
	"$(PYTHON)" -B run_cross_morse.py --dataset "$(DATASET)" --profile smoke \
	  --device cpu --threads "$(THREADS)" --stage all --data-root "$(DATA_ROOT)" --output-dir "$(RUN_ROOT)"

train-samples:
	@for stem in $(DATASET_STEMS); do \
	  "$(PYTHON)" -B "run_cross_morse_$$stem.py" --profile smoke --device cpu --threads "$(THREADS)" \
	    --stage all --data-root "$(DATA_ROOT)" --output-dir "$(WORK_ROOT)/runs/$$stem"; \
	done

sample smoke: train-sample
samples: train-samples

replay:
	"$(PYTHON)" -B scripts/reproduce_results.py "$(RUN_ROOT)" --output "$(REPORT_ROOT)/replay.json"

reexecute:
	"$(PYTHON)" -B scripts/evaluate_checkpoints.py "$(RUN_ROOT)" --trust-local-checkpoints \
	  --device "$(DEVICE)" --threads "$(THREADS)" --output "$(REPORT_ROOT)/checkpoints.json"

review-bundle:
	"$(PYTHON)" -B scripts/build_review_bundle.py "$(RUN_ROOT)" --output "$(REPORT_ROOT)/review_evidence.zip"

run-figures:
	"$(PYTHON)" -B scripts/plot_run_figures.py "$(RUN_ROOT)" --output-dir "$(REPORT_ROOT)/run-figures"

env-cpu:
	bash environment/create_cpu_env.sh --python "$(PYTHON)" --venv "$(VENV_DIR)"

environment-check:
	"$(PYTHON)" -B environment/check_environment.py
