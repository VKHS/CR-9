#!/usr/bin/env bash
# Produce every measured output of the paper from a completed reproduction suite, as one Slurm job graph.
#
#   bash hpc/produce_paper_outputs_arrhenius.sh SUITE [--work DIR] [--dry-run] [plan options ...]
#
# SUITE is the suite of `bash run_cross_morse.sh reproduce` (runs/NN_<dataset>; Roman-empire: the campaign of
# its ten split runs), or a directory holding one sealed run per dataset. The script
#   1. plans every producing step (scripts/paper_outputs.py plan) in a short Slurm job, because the runners'
#      plan steps need the job environment (NumPy); CM_PRODUCE_PLAN_LOCAL=1 plans in this shell instead;
#   2. submits one job per node (an array where a runner's plan has several jobs) with --dependency=afterok
#      chains, and a final job that runs `scripts/paper_outputs.py collect` after every node succeeded;
#   3. prints the one follow-up command that publishes the outputs (it needs GITHUB_TOKEN and the network,
#      so it runs on the login node):
#        bash hpc/publish_paper_artifacts_arrhenius.sh all <work>/paper-artifacts.map.json
#
# Options after SUITE: --work DIR (default: $CM_BASE/paper-outputs/<suite name>), --dry-run (plan, then print
# the sbatch commands without submitting), and every option of `scripts/paper_outputs.py plan`, for example
# --budget-campaign DIR, --confirmation-campaign DIR, --baselines DIR, --environment DIR, --only SET ...,
# --artifact ID=DIR. --baselines DIR holds the E12 study plans with their upstream and adapter checkouts beside
# them (the baselines export archives each adapter at its pinned commit from there, as --checkouts DIR) and plans
# the internal rows of Tables S15 and S15b on the runs. Follow the jobs with:
# python3 -B scripts/paper_outputs.py status --work <work>.
# Cluster settings come from the SITE SETTINGS block of run_cross_morse.sh (bash run_cross_morse.sh site-env).
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
cd "${ROOT}"
die() { printf '[ERROR] %s\n' "$*" >&2; exit 2; }
note() { printf '[produce] %s\n' "$*"; }

[[ "$#" -ge 1 && "$1" != -* ]] || { sed -n '2,23p' "$0"; exit 2; }
SUITE="$(readlink -f "$1")"; shift
[[ -d "${SUITE}" ]] || die "not a directory: ${SUITE}"
WORK="" DRY_RUN=0 PLAN_ARGS=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --work) [[ "$#" -ge 2 && -n "$2" ]] || die "--work needs a directory"; WORK="$2"; shift 2 ;;
        --work=*) WORK="${1#*=}"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) PLAN_ARGS+=("$1"); shift ;;
    esac
done
eval "$(bash "${ROOT}/run_cross_morse.sh" site-env)" || die "cannot read the site settings of run_cross_morse.sh"
WORK="$(readlink -m "${WORK:-${CM_BASE}/paper-outputs/$(basename "${SUITE}")}")"

load_modules() {
    [[ -n "${CM_BUILD_MODULE}${CM_PYTHON_MODULE}" ]] || return 0
    if ! type module >/dev/null 2>&1 && [[ -f "${MODULESHOME:-/nonexistent}/init/bash" ]]; then
        # shellcheck source=/dev/null
        source "${MODULESHOME}/init/bash"
    fi
    set +u
    module --force purge >/dev/null 2>&1 || module purge >/dev/null 2>&1 || true
    # shellcheck disable=SC2086
    module load ${CM_BUILD_MODULE} ${CM_PYTHON_MODULE} || { set -u; die "cannot load ${CM_BUILD_MODULE} ${CM_PYTHON_MODULE}"; }
    set -u
}
plan_here() {
    local python=python3
    if [[ -x "${CM_PYTHON:-/nonexistent}" ]] && "${CM_PYTHON}" -c 'import numpy' >/dev/null 2>&1; then
        python="${CM_PYTHON}"
    fi
    export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
    unset PYTHONPATH || true
    note "planning with ${python}"
    "${python}" -B "${ROOT}/scripts/paper_outputs.py" plan --runs "${SUITE}" --work "${WORK}" "${PLAN_ARGS[@]}"
}

# Inside the planning job submitted below.
if [[ "${CM_PRODUCE_PLAN_JOB:-0}" == 1 ]]; then
    load_modules
    plan_here
    exit 0
fi

[[ ! -e "${WORK}/plan.json" ]] || die "${WORK}/plan.json exists; give a new --work directory, or resubmit unfinished \
nodes with: python3 -B scripts/paper_outputs.py submit --work ${WORK} --resume"
mkdir -p "${WORK}/slurm/logs"
if [[ "${CM_PRODUCE_PLAN_LOCAL:-0}" == 1 ]]; then
    plan_here
else
    command -v sbatch >/dev/null 2>&1 || die "sbatch is not available; plan inside an allocation with CM_PRODUCE_PLAN_LOCAL=1"
    [[ -n "${CM_ACCOUNT}" ]] || die "set CM_ACCOUNT in the SITE SETTINGS of run_cross_morse.sh"
    read -r -a gpu_request <<< "${CM_REPORT_GPU_REQUEST}"
    read -r -a extra <<< "${CM_SBATCH_EXTRA}"
    memory=()
    [[ -z "${CM_MEM}" ]] || memory=(--mem="${CM_MEM}")
    note "planning in a Slurm job (waits for it; log ${WORK}/slurm/logs/plan_<job>.out)"
    if ! job="$(sbatch --parsable --wait --account="${CM_ACCOUNT}" --partition="${CM_REPORT_PARTITION}" \
            "${gpu_request[@]}" "${memory[@]}" "${extra[@]}" --cpus-per-task="$(( CM_THREADS < 8 ? CM_THREADS : 8 ))" \
            --time="${CM_PRODUCE_PLAN_TIME:-02:00:00}" --job-name=cmpo_plan \
            --output="${WORK}/slurm/logs/plan_%j.out" --error="${WORK}/slurm/logs/plan_%j.err" \
            --export=ALL,CM_PRODUCE_PLAN_JOB=1 "${ROOT}/hpc/produce_paper_outputs_arrhenius.sh" "${SUITE}" \
            --work "${WORK}" "${PLAN_ARGS[@]}")"; then
        tail -n 20 "${WORK}"/slurm/logs/plan_*.err 2>/dev/null >&2 || true
        die "the planning job failed; its log is under ${WORK}/slurm/logs/"
    fi
    job="${job%%;*}"
    cat "${WORK}/slurm/logs/plan_${job}.out" 2>/dev/null || true
fi
[[ -f "${WORK}/plan.json" ]] || die "no plan was written to ${WORK}/plan.json"

submit=(python3 -B "${ROOT}/scripts/paper_outputs.py" submit --work "${WORK}")
[[ "${DRY_RUN}" == 0 ]] || submit+=(--dry-run)
"${submit[@]}"

cat <<EOF

The job graph ends with the collect job, which writes ${WORK}/paper-artifacts.map.json.
  follow    python3 -B ${ROOT}/scripts/paper_outputs.py status --work ${WORK}
  resubmit  python3 -B ${ROOT}/scripts/paper_outputs.py submit --work ${WORK} --resume   (after fixing a failed node)
  publish   export GITHUB_TOKEN=...; bash ${ROOT}/hpc/publish_paper_artifacts_arrhenius.sh all ${WORK}/paper-artifacts.map.json
EOF
