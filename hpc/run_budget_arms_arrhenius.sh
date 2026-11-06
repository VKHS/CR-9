#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=cm_budget_arms
# Submit from a reviewed source snapshot. Pass --account explicitly to sbatch.
# CM_ENV_FILE is a trusted local file, not a public credential store.
set -Eeuo pipefail
: "${CM_SOURCE_ROOT:?Set the reviewed repository directory}"
: "${CM_ENV_FILE:?Set the trusted Arrhenius environment profile}"
: "${CM_BASE_RUN:?Set the immutable confirmed run directory}"
: "${CM_ARM_OUTPUT:?Set a fresh experiment output directory}"
[[ "${CM_TRUST_CHECKPOINTS:-0}" == 1 ]] || { echo 'Explicit checkpoint trust required' >&2; exit 2; }
[[ -f "$CM_ENV_FILE" && -d "$CM_SOURCE_ROOT" && -d "$CM_BASE_RUN" ]] || { echo 'Missing source/profile/base run' >&2; exit 2; }
[[ ! -e "$CM_ARM_OUTPUT" ]] || { echo 'Experiment output exists; never overwrite original records' >&2; exit 2; }
source "$CM_ENV_FILE"
module purge
module load "${CM_BUILD_MODULE:?}" "${CM_PYTHON_MODULE:?}"
PYTHON="${CM_PYTHON:-${CM_VENV_DIR:?}/bin/python}"
[[ -x "$PYTHON" ]] || { echo 'Configured interpreter missing' >&2; exit 2; }
cd "$CM_SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
"$PYTHON" -c 'import platform,torch; assert platform.machine()=="aarch64"; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name())'
JOB_ROOT="${CM_ARM_OUTPUT}.job-${SLURM_JOB_ID:?Run inside an allocated Slurm job}"
mkdir -m 700 "$JOB_ROOT"
bash environment/arrhenius/capture_environment.sh "$PYTHON" "$JOB_ROOT/environment"
module list > "$JOB_ROOT/modules.txt" 2>&1
scontrol show job "$SLURM_JOB_ID" > "$JOB_ROOT/scontrol_at_start.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$JOB_ROOT/job_id.txt"
"$PYTHON" -m pip freeze --all > "$JOB_ROOT/pip-freeze.txt"
cmd=("$PYTHON" -B scripts/run_budget_arms.py --run "$CM_BASE_RUN" --output "$CM_ARM_OUTPUT"
     --device cuda --threads "$OMP_NUM_THREADS" --trust-local-checkpoints
     --epochs "${CM_ARM_EPOCHS:-100}" --patience "${CM_ARM_PATIENCE:-15}"
     --seed "${CM_ARM_SEED:-17}")
[[ -z "${CM_ARM_BUDGET_RATIO:-}" ]] || cmd+=(--budget-ratio "$CM_ARM_BUDGET_RATIO")
[[ -z "${CM_ARM_HARD_CAP_RATIO:-}" ]] || cmd+=(--hard-cap-ratio "$CM_ARM_HARD_CAP_RATIO")
[[ -z "${CM_ARM_COST_BOUNDS:-}" ]] || cmd+=(--cost-bounds "$CM_ARM_COST_BOUNDS")
# Explicit post-hoc analysis is recorded; it never restores pre-admission validity.
[[ "${CM_ALLOW_POSTHOC:-0}" != 1 ]] || cmd+=(--allow-posthoc)
printf '%q ' "${cmd[@]}" > "$JOB_ROOT/command.txt"; printf '\n' >> "$JOB_ROOT/command.txt"
set +e
srun --ntasks=1 "${cmd[@]}" > "$JOB_ROOT/stdout.log" 2> "$JOB_ROOT/stderr.log"
rc=$?
set -e
printf '%s\n' "$rc" > "$JOB_ROOT/process_exit_code.txt"
# A sacct export taken here would precede the terminal Slurm state. Request it
# after job completion instead of presenting a RUNNING record as final evidence.
printf 'bash hpc/finalize_arm_accounting.sh %q %q\n' "$SLURM_JOB_ID" "$JOB_ROOT" > "$JOB_ROOT/finalize_accounting_command.txt"
if [[ "$rc" -eq 0 ]]; then
  "$PYTHON" -B scripts/run_budget_arms.py --verify "$CM_ARM_OUTPUT" --report "$JOB_ROOT/numerical_replay.json"
fi
exit "$rc"
