#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=cm_upstream
set -Eeuo pipefail
: "${CM_SOURCE_ROOT:?Set the reviewed repository}"
: "${CM_ENV_FILE:?Set the private site environment}"
: "${CM_BASELINE_PLAN:?Set the pinned upstream experiment plan}"
: "${CM_BASELINE_OUTPUT:?Set the output study root}"
: "${CM_BASELINE_PHASE:?Set hpo or final}"
[[ "${CM_TRUST_UPSTREAM:-0}" == 1 ]] || { echo 'Explicit adapter execution consent required' >&2; exit 2; }
source "$CM_ENV_FILE"
module purge
module load "${CM_BUILD_MODULE:?}" "${CM_PYTHON_MODULE:?}"
PYTHON="${CM_PYTHON:-${CM_VENV_DIR:?}/bin/python}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
cd "$CM_SOURCE_ROOT"
"$PYTHON" -c 'import platform,torch; assert platform.machine()=="aarch64"; assert torch.cuda.is_available()'
# Use --array=0-47 for HPO, --array=0-11 for final fits. A final Roman
# campaign submits all ten official splits separately; accounting is not reused.
JOB_ID="${SLURM_JOB_ID:?An allocated Slurm job is required}"
JOB_ROOT="${CM_BASELINE_JOB_EVIDENCE:-${CM_BASELINE_OUTPUT}.job-${JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}}"
[[ ! -e "$JOB_ROOT" ]] || { echo 'Use a fresh job-evidence directory; preserve previous attempts' >&2; exit 2; }
mkdir -p "$JOB_ROOT"
printf '%s\n' "$JOB_ID" > "$JOB_ROOT/job_id.txt"
module list > "$JOB_ROOT/modules.txt" 2>&1
"$PYTHON" -m pip freeze --all > "$JOB_ROOT/pip-freeze.txt"
nvidia-smi -q > "$JOB_ROOT/nvidia-smi.txt"
uname -a > "$JOB_ROOT/uname.txt"
set +e
srun "$PYTHON" -B scripts/run_upstream_baseline.py --plan "$CM_BASELINE_PLAN" \
  --output "$CM_BASELINE_OUTPUT" --phase "$CM_BASELINE_PHASE" \
  --index "${SLURM_ARRAY_TASK_ID:-0}" --split "${CM_BASELINE_SPLIT:-0}" --trust-upstream-code \
  > "$JOB_ROOT/stdout.log" 2> "$JOB_ROOT/stderr.log"
rc=$?
set -e
printf '%s\n' "$rc" > "$JOB_ROOT/process_exit_code.txt"
printf 'bash hpc/finalize_arm_accounting.sh %q %q\n' "$JOB_ID" "$JOB_ROOT" > "$JOB_ROOT/finalize_accounting_command.txt"
# Capture the terminal allocation using hpc/finalize_arm_accounting.sh after
# the job ends. Never write a synthetic sacct record or treat CPU walltime as GPU-hours.

exit "$rc"
