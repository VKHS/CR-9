#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-6%3
#SBATCH --job-name=cm_pack_evidence
# Preflight and pack the official run archives on the cluster, one dataset per array task.
#
#   sbatch --account=<allocation> --partition=<partition> [--gpus=1 if the partition requires it] \
#     --export=ALL,CM_SOURCE_ROOT=/abs/CR-9,CM_ENV_FILE=/abs/arrhenius.env,CM_EVIDENCE_PLAN=/abs/plan.json \
#     hpc/pack_evidence_arrhenius.sh
#
# Packing is CPU and storage work; no GPU is used. Pass --account and --partition explicitly, as for
# the other Arrhenius scripts. Array index i examines the i-th dataset of the fixed order below; a
# dataset absent from the plan, or still without a run_root, ends its task successfully. The preflight
# runs the full scientific audit (PyTorch is in the site environment); any nonzero preflight
# exit stops the task before packing. Reports, archives and the shared state file go to the plan's
# deposits_dir; concurrent tasks serialize their state updates. Nothing is deposited or published here.
set -Eeuo pipefail
: "${CM_SOURCE_ROOT:?Set the reviewed repository directory}"
: "${CM_ENV_FILE:?Set the trusted Arrhenius environment profile}"
: "${CM_EVIDENCE_PLAN:?Set the evidence plan (scripts/evidence_pipeline.py template)}"
[[ -f "$CM_ENV_FILE" && -d "$CM_SOURCE_ROOT" && -f "$CM_EVIDENCE_PLAN" ]] || { echo 'Missing source, profile or plan' >&2; exit 2; }
source "$CM_ENV_FILE"
if [[ -n "${CM_BUILD_MODULE:-}${CM_PYTHON_MODULE:-}" ]]; then
  module purge
  # shellcheck disable=SC2086
  module load ${CM_BUILD_MODULE:-} ${CM_PYTHON_MODULE:-}
fi
PYTHON="${CM_PYTHON:-${CM_VENV_DIR:?}/bin/python}"
[[ -x "$PYTHON" ]] || { echo 'Configured interpreter missing' >&2; exit 2; }
cd "$CM_SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
DATASETS=(ogbn-arxiv ogbn-proteins ogbn-products Roman-empire PascalVOC-SP COCO-SP Peptides-func)
index="${SLURM_ARRAY_TASK_ID:-${CM_DATASET_INDEX:?Run as a Slurm array task or set CM_DATASET_INDEX}}"
[[ "$index" =~ ^[0-6]$ ]] || { echo "Array index must be 0-6, got $index" >&2; exit 2; }
dataset="${DATASETS[$index]}"
# Only an explicitly unconfigured dataset is a successful no-op. Invalid JSON,
# missing dependencies and malformed plans must fail the Slurm task visibly.
set +e
"$PYTHON" - "$CM_EVIDENCE_PLAN" "$dataset" <<'PY'
import sys
from pathlib import Path
from scripts.evidence_pipeline import load_plan
plan = load_plan(Path(sys.argv[1]))
entry = plan["datasets"].get(sys.argv[2])
raise SystemExit(0 if entry and entry.get("run_root") else 3)
PY
plan_status=$?
set -e
case "$plan_status" in
  0) ;;
  3) echo "$dataset is not in the plan or has no run_root yet; nothing to do"; exit 0 ;;
  *) echo "Cannot validate the evidence plan for $dataset (exit $plan_status)" >&2; exit "$plan_status" ;;
esac
"$PYTHON" -c 'import torch; print("torch", torch.__version__)' || { echo 'PyTorch is required for the full audit' >&2; exit 2; }
echo "dataset=$dataset job=${SLURM_JOB_ID:-local} task=$index source=$("$PYTHON" -B scripts/source_fingerprint.py 2>/dev/null | tail -1)"
"$PYTHON" -B scripts/evidence_pipeline.py preflight --plan "$CM_EVIDENCE_PLAN" --dataset "$dataset" --full-audit always
"$PYTHON" -B scripts/evidence_pipeline.py pack --plan "$CM_EVIDENCE_PLAN" --dataset "$dataset"
"$PYTHON" -B scripts/evidence_pipeline.py status --plan "$CM_EVIDENCE_PLAN" > /dev/null
echo "done: $dataset; see the plan's deposits_dir/preflight/PREFLIGHT.md and EVIDENCE_STATUS.md"
