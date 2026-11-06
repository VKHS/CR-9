#!/bin/bash -l
#SBATCH --job-name=cross_morse
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=cross_morse_%j.out
#SBATCH --error=cross_morse_%j.err
# =============================================================================
#  CROSS-MORSE reproduction launcher
#  CROSS-MORSE: Cross-Spectral Control of Message Passing in Frozen Graph
#  Neural Networks -- Khalili Sadaghiani and Nunez-Yanez, Linkoping University
# -----------------------------------------------------------------------------
#  Regenerates every benchmark result of the paper on a Slurm GPU cluster: the
#  sealed eight-stage lifecycle on all seven datasets (Roman-empire on its ten
#  official splits), followed by one report that places each regenerated value
#  beside the value printed in the paper.
#
#    1. Edit the SITE SETTINGS block below for your cluster.
#    2. bash run_cross_morse.sh check       settings, runtime identity and the
#                                           complete job plan; submits nothing
#    3. bash run_cross_morse.sh reproduce   submit the complete study
#
#  Follow:  bash run_cross_morse.sh status
#           tail -F <CM_BASE>/logs/CROSS_MORSE_LIVE.out
#  Result:  <suite>/REPRODUCTION_REPORT.md and <suite>/CMR_<n>_RESULTS.zip
#
#  `bash run_cross_morse.sh --help` lists every mode: single datasets, the
#  Roman-empire campaign, resumption and environment setup. docs/HPC.md is the
#  complete guide. The Python runtime is never modified by this file; its
#  fingerprint is checked before a reproduction and sealed into every run.
# =============================================================================

# =============================================================================
#  SITE SETTINGS -- the block to edit for another cluster
# -----------------------------------------------------------------------------
#  The defaults are the authors' setting: the NAISS Arrhenius cluster, one
#  NVIDIA GH200 96 GB GPU per job on aarch64 nodes, the cluster's NVHPC and
#  Python modules, and a persistent virtual environment holding the CUDA 13
#  PyTorch wheel. The allocation and the storage project are left empty.
#  An exported CM_* variable, or one set in a profile named by CM_ENV_FILE,
#  takes precedence over this block; an explicitly empty value is kept.
# =============================================================================
site_settings() {
    # -- scheduler ----------------------------------------------------------------
    : "${CM_ACCOUNT=}"                           # Slurm account (sbatch --account); required to submit
    : "${CM_PARTITION=gpu}"                      # GPU partition
    : "${CM_GPU_REQUEST=--gpus=1}"               # one GPU per job; elsewhere e.g. "--gres=gpu:1"
    : "${CM_THREADS=16}"                         # CPU threads per job (sbatch --cpus-per-task)
    : "${CM_MEM=}"                               # memory per job, e.g. 120G; empty: the site default
    : "${CM_SBATCH_EXTRA=}"                      # further options, e.g. "--qos=normal --constraint=gh200"
    # -- storage --------------------------------------------------------------------
    : "${CM_STORAGE_PROJECT=}"                   # Arrhenius project-disk name
    if [[ -z "${CM_BASE+set}" ]]; then           # outputs, logs, caches, datasets and the environment
        if [[ -n "${CM_STORAGE_PROJECT}" ]]; then
            CM_BASE="/nobackup/proj/disk/${CM_STORAGE_PROJECT}/cross_morse"
        else
            CM_BASE="${HOME}/cross_morse_work"
        fi
    fi
    # -- software -------------------------------------------------------------------
    : "${CM_BUILD_MODULE=GPU/buildenv-nvhpc/25.9-cu13.0}"    # empty: keep the job's environment
    : "${CM_PYTHON_MODULE=GPU/Python/3.13.5-bare-gcc-2025b-eb}"
    : "${CM_EXPECT_ARCH=aarch64}"                # compute-node architecture; empty: any
    : "${CM_VENV_NAME=venv_arrhenius_gh200}"     # created once by `setup` under CM_BASE/code
    : "${CM_TORCH_SPEC=torch==2.12.1+cu130}"     # PyTorch wheel `setup` installs when none is importable
    : "${CM_TORCH_INDEX=https://download.pytorch.org/whl/cu130}"
    # -- wall time and resilience -----------------------------------------------------
    : "${CM_TIME=}"                              # one limit for every run job; empty: by profile
    : "${CM_TIME_SMOKE=01:00:00}"
    : "${CM_TIME_PILOT=04:00:00}"
    : "${CM_TIME_FULL=24:00:00}"                 # per dataset: CM_TIME_ARXIV, CM_TIME_PRODUCTS, ...
    : "${CM_SETUP_TIME=01:00:00}"
    : "${CM_MAX_CONTINUATIONS=12}"               # checkpointed requeues of one run at its time limit
    : "${CM_MAX_INFRA_REQUEUES=3}"               # requeues after a GPU or driver failure, node excluded
    : "${CM_GPU_HEALTH_RETRIES=3}"
    : "${CM_CHECKPOINT_GRACE=150}"               # seconds to reach a safe boundary after the warning
    : "${CM_ARRAY_PARALLEL=}"                    # concurrent Roman-empire splits; empty: all ten
    # -- report job -----------------------------------------------------------------
    : "${CM_REPORT_PARTITION=${CM_PARTITION}}"
    : "${CM_REPORT_GPU_REQUEST=${CM_GPU_REQUEST}}"   # empty where a CPU-only job is allowed
    : "${CM_REPORT_TIME=02:00:00}"
    : "${CM_PACK_ARCHIVES=0}"                    # 1: also pack each run as a sealed .tar.gz archive
}

# =============================================================================
#  Everything below is the same on every cluster.
# =============================================================================
set -Eeuo pipefail
umask 002
# Original descriptors: log tees attached later write through them exactly once.
exec 7>&1 8>&2

RUNNER="run_cross_morse.py"
SHELL_NAME="run_cross_morse.sh"
RUNTIME_SHELL="$(readlink -f "${BASH_SOURCE[0]}")"
ALL_DATASETS=(ogbn-arxiv ogbn-proteins ogbn-products Roman-empire PascalVOC-SP COCO-SP Peptides-func)
EXIT_USAGE=2
EXIT_PREREQUISITE=33        # environment, modules, interpreter or CUDA missing
EXIT_ORCHESTRATION=70       # launcher step failed
EXIT_ARTIFACT_MISSING=74    # the runner returned 0 without completing the requested stage
EXIT_CHECKPOINTED=75        # the runner stopped at a safe boundary (time limit); the run continues
EXIT_INFRASTRUCTURE=76      # GPU, driver or node failure; requeued on another node

die() { printf '[ERROR] %s\n' "$*" >&2; exit "${EXIT_USAGE}"; }
fail() { local code="$1"; shift; printf '[ERROR] %s\n' "$*" >&2; exit "${code}"; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# ---------------------------------------------------------------- routing options
CLI_DATASET="" CLI_SPLIT="" CLI_DATA_ROOT="" CLI_CONFIG=""
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dataset|--split-index|--data-root|--config)
            [[ "$#" -ge 2 && -n "$2" ]] || die "$1 needs a value."
            case "$1" in
                --dataset)
                    [[ -z "${CLI_DATASET}" || "${CLI_DATASET}" == "$2" ]] || die "Conflicting --dataset arguments."
                    CLI_DATASET="$2" ;;
                --split-index) CLI_SPLIT="$2" ;;
                --data-root) CLI_DATA_ROOT="$2" ;;
                --config) CLI_CONFIG="$2" ;;
            esac
            shift 2 ;;
        --dataset=*)
            value="${1#*=}"
            [[ -n "${value}" && ( -z "${CLI_DATASET}" || "${CLI_DATASET}" == "${value}" ) ]] || die "Empty/conflicting --dataset."
            CLI_DATASET="${value}"; shift ;;
        --split-index=*) CLI_SPLIT="${1#*=}"; [[ -n "${CLI_SPLIT}" ]] || die "Empty --split-index."; shift ;;
        --data-root=*) CLI_DATA_ROOT="${1#*=}"; [[ -n "${CLI_DATA_ROOT}" ]] || die "Empty --data-root."; shift ;;
        --config=*) CLI_CONFIG="${1#*=}"; [[ -n "${CLI_CONFIG}" ]] || die "Empty --config."; shift ;;
        -h|--help) POSITIONAL=(--help); break ;;
        --*) die "Unknown launcher option: $1" ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]}"

# ---------------------------------------------------------------- settings
# Precedence: command-line routing > exported CM_* > profile (CM_ENV_FILE, or a
# run snapshot's configs/active.env.example, or the git-ignored private
# configs/arrhenius.env.example) > SITE SETTINGS.
declare -A cm_prior_env=()
while IFS= read -r cm_name; do
    [[ "${cm_name}" == CM_* ]] || continue
    cm_prior_env["${cm_name}"]="${!cm_name}"
done < <(compgen -e)
PROFILE_SOURCE=""
if [[ -n "${CM_ENV_FILE:-}" ]]; then
    PROFILE_SOURCE="$(readlink -f "${CM_ENV_FILE}")" || die "Cannot resolve CM_ENV_FILE: ${CM_ENV_FILE}"
    [[ -f "${PROFILE_SOURCE}" ]] || die "CM_ENV_FILE does not exist: ${CM_ENV_FILE}"
else
    for cm_candidate in "${CM_PROJECT_DIR:-/nonexistent}" "$(dirname "${RUNTIME_SHELL}")" "${SLURM_SUBMIT_DIR:-/nonexistent}"; do
        for cm_profile in active.env.example arrhenius.env.example; do
            if [[ -f "${cm_candidate}/configs/${cm_profile}" ]]; then
                PROFILE_SOURCE="$(readlink -f "${cm_candidate}/configs/${cm_profile}")"
                break 2
            fi
        done
    done
fi
if [[ -n "${PROFILE_SOURCE}" ]]; then
    # shellcheck source=/dev/null
    source "${PROFILE_SOURCE}"
fi
for cm_name in "${!cm_prior_env[@]}"; do
    printf -v "${cm_name}" '%s' "${cm_prior_env[${cm_name}]}"
    export "${cm_name}"
done
unset cm_prior_env cm_name cm_candidate cm_profile
site_settings
[[ -z "${CLI_DATASET}" ]] || CM_DATASET="${CLI_DATASET}"
[[ -z "${CLI_SPLIT}" ]] || CM_SPLIT_INDEX="${CLI_SPLIT}"
[[ -z "${CLI_DATA_ROOT}" ]] || CM_DATA_ROOT="${CLI_DATA_ROOT}"
[[ -z "${CLI_CONFIG}" ]] || CM_CONFIG="${CLI_CONFIG}"

BASE="$(readlink -m "${CM_BASE}")"
CODE_DIR="${BASE}/code"
DATA_ROOT="$(readlink -m "${CM_DATA_ROOT:-${BASE}/datasets/ogb_cache}")"
CACHE_DIR="${BASE}/cache"
LOGS_DIR="${BASE}/logs"
OUTPUTS_DIR="${BASE}/outputs"
VENV_DIR="$(readlink -m "${CM_VENV_DIR:-${CODE_DIR}/${CM_VENV_NAME:-venv_cross_morse}}")"
PYTHON="${VENV_DIR}/bin/python"
BUILD_MODULE="${CM_BUILD_MODULE}"
PYTHON_MODULE="${CM_PYTHON_MODULE}"
EXPECTED_ARCH="${CM_EXPECT_ARCH}"
THREADS="${CM_THREADS}"
STAGE="${CM_STAGE:-all}"
DATASET="${CM_DATASET:-ogbn-arxiv}"
SPLIT_INDEX="${CM_SPLIT_INDEX:-}"
SPLIT_EXPLICIT=0
[[ -z "${SPLIT_INDEX}" ]] || SPLIT_EXPLICIT=1
LIVE_OUT="${LOGS_DIR}/CROSS_MORSE_LIVE.out"
LIVE_ERR="${LOGS_DIR}/CROSS_MORSE_LIVE.err"
REPRO_SUITE="${CM_REPRODUCTION_SUITE:-}"
RUN_KEY="${CM_RUN_KEY:-}"
SUITE="" CHILD_PID=""
SIGNAL_RECEIVED=0 WAIT_INTERRUPTED=0
EXECUTION_STATUS="" FAILURE_CLASS="" REQUEUE_REQUESTED=0

usage() {
    cat <<'USAGE'
CROSS-MORSE reproduction launcher (Slurm; one GPU per job)

Reproduce the paper (login node; edit the SITE SETTINGS block first):
  bash run_cross_morse.sh check [full|pilot|smoke]     settings, runtime identity, job plan; submits nothing
  bash run_cross_morse.sh reproduce [full|pilot|smoke] submit the complete study as one numbered suite
  bash run_cross_morse.sh reproduce resume SUITE       resubmit the unfinished runs of a suite
  bash run_cross_morse.sh status [SUITE]               progress of every run (default: the latest suite)
  bash run_cross_morse.sh report [SUITE]               rebuild the report and results ZIP in a short job

  full uses the paper's registered budget; pilot uses official data with reduced
  compute; smoke uses synthetic data and checks the whole orchestration in
  minutes. A dataset wrapper limits a suite to its dataset, for example
    bash run_cross_morse_peptides_func.sh reproduce full
  and CM_REPRODUCE_DATASETS="ogbn-arxiv,COCO-SP" selects several.

One dataset (login node):
  bash run_cross_morse.sh --dataset ogbn-arxiv dry-run pilot
  bash run_cross_morse.sh --dataset ogbn-products submit setup
  bash run_cross_morse.sh --dataset ogbn-proteins submit smoke
  bash run_cross_morse.sh --dataset COCO-SP submit full
  bash run_cross_morse.sh --dataset Peptides-func submit resume /absolute/suite
  bash run_cross_morse.sh --dataset Roman-empire --split-index 3 submit full
  bash run_cross_morse_roman_empire.sh submit campaign full /absolute/campaign
  bash run_cross_morse_roman_empire.sh submit campaign resume /absolute/campaign

Inside an allocation the same modes run without `submit`. Canonical datasets:
ogbn-arxiv, ogbn-proteins, ogbn-products, Roman-empire, PascalVOC-SP, COCO-SP,
Peptides-func; the generic default is ogbn-arxiv.

Options (before or after the mode):
  --dataset NAME     overrides CM_DATASET; must match the configuration/wrapper
  --split-index 0..9 overrides CM_SPLIT_INDEX; Roman-empire only
  --data-root PATH   overrides CM_DATA_ROOT
  --config PATH      overrides CM_CONFIG for a new run

Settings: the SITE SETTINGS block at the top of this file. Precedence:
command-line routing > exported CM_* > profile > SITE SETTINGS. The profile is
CM_ENV_FILE (one trusted Bash file) or, without it, the first of
configs/active.env.example (run snapshots) and configs/arrhenius.env.example
(git-ignored private copy) in CM_PROJECT_DIR, this file's directory or
SLURM_SUBMIT_DIR. `bash run_cross_morse.sh site-env` prints the effective values.

Resilience: a job that reaches its time limit receives USR1 180 s earlier,
checkpoints at a safe boundary and requeues itself (CM_MAX_CONTINUATIONS); a
GPU or driver failure requeues the job on another node (CM_MAX_INFRA_REQUEUES).
Every job mirrors its output into <CM_BASE>/logs/CROSS_MORSE_LIVE.out/.err.
Exit codes: 2 usage, 33 prerequisite, 70 orchestration, 74 missing artifact,
75 checkpointed, 76 infrastructure; other codes come from the Python runtime.

`setup` creates the selected environment only if missing, preserves an
importable PyTorch and installs requirements.txt. Other modes never install.
Default limits: setup/smoke 01:00:00, pilot 04:00:00, full 24:00:00.
For direct sbatch supply --account, --partition, the GPU request and --time:
  sbatch --account=ALLOCATION --partition=gpu --gpus=1 run_cross_morse.sh --dataset ogbn-arxiv pilot
USAGE
}

# ---------------------------------------------------------------- dataset routing
mode="${1:-}"
case "${mode}" in help|-h|--help) usage; exit 0 ;; esac
if [[ -z "${mode}" ]]; then
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then usage; exit 0; fi
    mode=pilot
fi
stem_of() {
    case "$1" in
        ogbn-arxiv) echo arxiv ;; ogbn-products) echo products ;; ogbn-proteins) echo proteins ;;
        Roman-empire) echo roman_empire ;; PascalVOC-SP) echo pascalvoc_sp ;; COCO-SP) echo coco_sp ;;
        Peptides-func) echo peptides_func ;; *) return 1 ;;
    esac
}
DATASET_STEM="$(stem_of "${DATASET}")" || die "Unknown canonical dataset: ${DATASET}"
DATASET_LOG="${DATASET_STEM^^}"
if [[ "${DATASET}" == Roman-empire ]]; then
    [[ -z "${SPLIT_INDEX}" || "${SPLIT_INDEX}" =~ ^[0-9]$ ]] || die "Roman-empire split index must be 0 through 9."
else
    [[ -z "${SPLIT_INDEX}" ]] || die "Split indices are supported only for Roman-empire."
fi
[[ "${THREADS}" =~ ^[1-9][0-9]*$ ]] || die "CM_THREADS must be a positive integer."
case "${STAGE}" in prepare|train|explore|fit|select|confirm|admit|evaluate|all) ;; *) die "Invalid CM_STAGE: ${STAGE}" ;; esac

resolve_project() {
    local candidate own_dir
    own_dir="$(dirname "${RUNTIME_SHELL}")"
    if [[ -n "${CM_PROJECT_DIR:-}" ]]; then
        [[ -f "${CM_PROJECT_DIR}/${RUNNER}" && -f "${CM_PROJECT_DIR}/${SHELL_NAME}" ]] || die "CM_PROJECT_DIR lacks the shared Python/shell entry points."
        PROJECT_DIR="$(readlink -f "${CM_PROJECT_DIR}")"
        return
    fi
    for candidate in "${own_dir}" "${SLURM_SUBMIT_DIR:-/nonexistent}" "${CODE_DIR}/CROSS-MORSE" "${CODE_DIR}/cross_morse_arxiv" "${CODE_DIR}"; do
        if [[ -f "${candidate}/${RUNNER}" && -f "${candidate}/${SHELL_NAME}" ]]; then
            PROJECT_DIR="$(readlink -f "${candidate}")"
            return
        fi
    done
    die "Cannot find source; set CM_PROJECT_DIR to the unpacked repository."
}
resolve_project

# ---------------------------------------------------------------- shared helpers
select_config() {
    PROFILE="$1"
    case "${PROFILE}" in smoke|pilot|full) ;; *) die "Profile must be smoke, pilot or full." ;; esac
    CONFIG_SOURCE="${CM_CONFIG:-${PROJECT_DIR}/configs/${DATASET_STEM}_${PROFILE}.json}"
    [[ -f "${CONFIG_SOURCE}" ]] || die "Config does not exist: ${CONFIG_SOURCE}"
    CONFIG_SOURCE="$(readlink -f "${CONFIG_SOURCE}")"
}
check_config() {
    SPLIT_INDEX="$(python3 -B - "${PROJECT_DIR}" "${CONFIG_SOURCE}" "${DATASET}" "${PROFILE}" "${SPLIT_INDEX}" <<'PYCONFIGCHECK'
import sys
sys.path.insert(0, sys.argv[1])
from run_cross_morse import resolve_configuration
config = resolve_configuration(config_path=sys.argv[2], dataset=sys.argv[3], profile=sys.argv[4],
                               split_index=int(sys.argv[5]) if sys.argv[5] else None)
print(config['data'].get('split_index', ''))
PYCONFIGCHECK
)"
}
export_effective_environment() {
    export CM_BASE="${BASE}" CM_DATA_ROOT="${DATA_ROOT}" CM_VENV_DIR="${VENV_DIR}"
    export CM_ACCOUNT="${SLURM_JOB_ACCOUNT:-${CM_ACCOUNT:-}}" CM_STORAGE_PROJECT="${CM_STORAGE_PROJECT:-}"
    export CM_PARTITION="${CM_PARTITION:-gpu}" CM_THREADS="${THREADS}"
    export CM_BUILD_MODULE="${BUILD_MODULE}" CM_PYTHON_MODULE="${PYTHON_MODULE}" CM_EXPECT_ARCH="${EXPECTED_ARCH}"
    export CM_TORCH_SPEC="${CM_TORCH_SPEC:-}" CM_TORCH_INDEX="${CM_TORCH_INDEX:-}"
    export CM_DATASET="${DATASET}" CM_SPLIT_INDEX="${SPLIT_INDEX}"
}
# Every SITE SETTINGS value, exported for snapshots, suites and child jobs.
SITE_KEYS=(CM_ACCOUNT CM_PARTITION CM_GPU_REQUEST CM_THREADS CM_MEM CM_SBATCH_EXTRA CM_STORAGE_PROJECT CM_BASE
           CM_DATA_ROOT CM_VENV_DIR CM_VENV_NAME CM_BUILD_MODULE CM_PYTHON_MODULE CM_EXPECT_ARCH CM_TORCH_SPEC
           CM_TORCH_INDEX CM_TIME CM_TIME_SMOKE CM_TIME_PILOT CM_TIME_FULL CM_SETUP_TIME CM_MAX_CONTINUATIONS
           CM_MAX_INFRA_REQUEUES CM_GPU_HEALTH_RETRIES CM_CHECKPOINT_GRACE CM_ARRAY_PARALLEL CM_REPORT_PARTITION
           CM_REPORT_GPU_REQUEST CM_REPORT_TIME CM_PACK_ARCHIVES)
export_site_settings() {
    local key
    CM_BASE="${BASE}" CM_DATA_ROOT="${DATA_ROOT}" CM_VENV_DIR="${VENV_DIR}"
    for key in "${SITE_KEYS[@]}"; do export "${key}=${!key-}"; done
    while IFS= read -r key; do export "${key}"; done < <(compgen -v | grep -E '^CM_(TIME|MEM)_[A-Z_]+$' || true)
}
read_suite_context() {
    local result
    result="$(python3 -B - "${PROJECT_DIR}" "$1" "${DATASET}" "${SPLIT_INDEX}" "${SPLIT_EXPLICIT}" <<'PYCONTEXT'
import sys
sys.path.insert(0, sys.argv[1])
from run_cross_morse import verify_launcher_snapshot
context = verify_launcher_snapshot(sys.argv[2])
if context['dataset'] != sys.argv[3]:
    raise SystemExit('Dataset does not match the saved suite; use its dataset wrapper/profile.')
if sys.argv[5] == '1' and str(context['split_index']) != sys.argv[4]:
    raise SystemExit('Requested split does not match the saved suite; a resume cannot change splits.')
print(context['data_root'])
print(context['profile'])
print('-' if context['split_index'] is None else context['split_index'])
PYCONTEXT
)"
    mapfile -t saved_context <<< "${result}"
    DATA_ROOT="${saved_context[0]}"
    PROFILE="${saved_context[1]}"
    SPLIT_INDEX="${saved_context[2]}"
    [[ "${SPLIT_INDEX}" != - ]] || SPLIT_INDEX=""
}
read_campaign_profile() {
    python3 -B - "${PROJECT_DIR}" "$1" <<'PYCAMPAIGNCHECK'
import sys
sys.path.insert(0, sys.argv[1])
from run_cross_morse import verify_roman_campaign
print(verify_roman_campaign(sys.argv[2])['profile'])
PYCAMPAIGNCHECK
}
profile_time() {   # profile_time PROFILE STEM -> wall-time limit of one run job
    local profile="$1" stem="${2^^}" per
    per="CM_TIME_${stem}"
    if [[ -n "${CM_TIME:-}" ]]; then printf '%s\n' "${CM_TIME}"; return; fi
    if [[ "${profile}" == full && -n "${!per:-}" ]]; then printf '%s\n' "${!per}"; return; fi
    case "${profile}" in
        smoke|setup) printf '%s\n' "${CM_TIME_SMOKE}" ;;
        pilot) printf '%s\n' "${CM_TIME_PILOT}" ;;
        *) printf '%s\n' "${CM_TIME_FULL}" ;;
    esac
}
# Fill SBATCH_ARGS with the site's scheduler settings for one job.
sbatch_site_args() {   # sbatch_site_args [report] [STEM]
    local kind="${1:-run}" stem="${2:-}" words=() mem per
    if [[ "${kind}" == report ]]; then
        SBATCH_ARGS=(--account="${CM_ACCOUNT}" --partition="${CM_REPORT_PARTITION}" --cpus-per-task="$(( THREADS < 8 ? THREADS : 8 ))")
        read -r -a words <<< "${CM_REPORT_GPU_REQUEST}"
    else
        SBATCH_ARGS=(--account="${CM_ACCOUNT}" --partition="${CM_PARTITION}" --cpus-per-task="${THREADS}")
        read -r -a words <<< "${CM_GPU_REQUEST}"
    fi
    SBATCH_ARGS+=("${words[@]}")
    mem="${CM_MEM:-}"
    if [[ -n "${stem}" ]]; then per="CM_MEM_${stem^^}"; mem="${!per:-${mem}}"; fi
    [[ -z "${mem}" ]] || SBATCH_ARGS+=(--mem="${mem}")
    read -r -a words <<< "${CM_SBATCH_EXTRA}"
    SBATCH_ARGS+=("${words[@]}")
}
reproduction_tool() {   # reproduction_tool INTERPRETER SOURCE_DIR ARGS...
    local interpreter="$1" source_dir="$2"; shift 2
    "${interpreter}" -B "${source_dir}/scripts/reproduction.py" "$@"
}
latest_suite() {
    local latest="${OUTPUTS_DIR}/CROSS_MORSE_REPRODUCTION_LATEST"
    if [[ -e "${latest}" ]]; then readlink -f "${latest}"; return; fi
    [[ -s "${CACHE_DIR}/cross_morse/latest_reproduction.txt" ]] && { head -n1 "${CACHE_DIR}/cross_morse/latest_reproduction.txt"; return; }
    return 1
}
selected_datasets() {   # fills REPRO_DATASETS
    local spec item
    REPRO_DATASETS=()
    if [[ -n "${CLI_DATASET}" ]]; then REPRO_DATASETS=("${CLI_DATASET}"); return; fi
    spec="${CM_REPRODUCE_DATASETS:-}"
    if [[ -z "${spec}" ]]; then REPRO_DATASETS=("${ALL_DATASETS[@]}"); return; fi
    for item in ${spec//,/ }; do
        stem_of "${item}" >/dev/null || die "Unknown dataset in CM_REPRODUCE_DATASETS: ${item}"
        REPRO_DATASETS+=("${item}")
    done
}
ensure_module_command() {
    command -v module >/dev/null 2>&1 && return 0
    local init
    for init in "${MODULESHOME:-/nonexistent}/init/bash" /usr/share/lmod/lmod/init/bash /etc/profile.d/modules.sh; do
        [[ -f "${init}" ]] || continue
        set +u
        # shellcheck source=/dev/null
        source "${init}" >/dev/null 2>&1 || true
        set -u
        command -v module >/dev/null 2>&1 && return 0
    done
    return 1
}
load_site_modules() {
    [[ -n "${BUILD_MODULE}${PYTHON_MODULE}" ]] || return 0
    ensure_module_command || fail "${EXIT_PREREQUISITE}" "The SITE SETTINGS request modules, but no module command is available; empty CM_BUILD_MODULE and CM_PYTHON_MODULE to keep the job's environment."
    set +u
    module --force purge >/dev/null 2>&1 || module purge >/dev/null 2>&1 || true
    if [[ -n "${BUILD_MODULE}" ]] && ! module load "${BUILD_MODULE}"; then set -u; fail "${EXIT_PREREQUISITE}" "Cannot load module ${BUILD_MODULE}"; fi
    if [[ -n "${PYTHON_MODULE}" ]] && ! module load "${PYTHON_MODULE}"; then set -u; fail "${EXIT_PREREQUISITE}" "Cannot load module ${PYTHON_MODULE}"; fi
    set -u
}

# =============================================================================
#  Login-node modes
# =============================================================================
mode_dry_run() {
    [[ "$#" -le 1 ]] || die "dry-run takes one profile."
    select_config "${1:-pilot}"
    check_config
    printf 'account=%s\npartition=%s\nbase=%s\nproject=%s\ndata_root=%s\npython=%s\nenv_file=%s\n' \
        "${CM_ACCOUNT:-UNSET}" "${CM_PARTITION:-gpu}" "${BASE}" "${PROJECT_DIR}" "${DATA_ROOT}" "${PYTHON}" \
        "${PROFILE_SOURCE:-none (SITE SETTINGS of ${SHELL_NAME})}"
    printf 'dataset=%s\nconfig=%s\nsplit_index=%s\nmodules=%s %s\nprofile=%s\nstage=%s\ngpu_request=%s\ntime=%s\n' \
        "${DATASET}" "${CONFIG_SOURCE}" "${SPLIT_INDEX:--}" "${BUILD_MODULE}" "${PYTHON_MODULE}" "${PROFILE}" "${STAGE}" \
        "${CM_GPU_REQUEST}" "$(profile_time "${PROFILE}" "${DATASET_STEM}")"
    cmd=("${PYTHON}" -u "SUITE/source/${RUNNER}" --dataset "${DATASET}" --config SUITE/source/configs/resolved.json
         --data-root "${DATA_ROOT}" --output-dir SUITE --device cuda --threads "${THREADS}" --stage "${STAGE}")
    [[ -z "${SPLIT_INDEX}" ]] || cmd+=(--split-index "${SPLIT_INDEX}")
    printf 'Python command after source snapshot:'; printf ' %q' "${cmd[@]}"; printf '\n'
    printf 'No directories, packages or jobs were created.\n'
}

mode_submit() {
    [[ "$#" -gt 0 ]] || set -- pilot
    [[ -z "${SLURM_JOB_ID:-}" ]] || die "submit must run outside a job; invoke the mode directly inside an allocation."
    command -v sbatch >/dev/null 2>&1 || die "sbatch is not available in this shell."
    [[ -n "${CM_ACCOUNT:-}" ]] || die "Set CM_ACCOUNT in the SITE SETTINGS, the environment or CM_ENV_FILE before submitting."
    local submit_mode="$1" default_time run_args=() batch_extra=() routing=() log_pattern
    case "${submit_mode}" in
        setup)
            [[ "$#" == 1 ]] || die "setup takes no positional arguments."
            default_time="${CM_SETUP_TIME}"; run_args=(setup) ;;
        smoke|pilot|full)
            [[ "$#" == 1 ]] || die "${submit_mode} takes no positional arguments."
            select_config "${submit_mode}"; check_config
            default_time="$(profile_time "${submit_mode}" "${DATASET_STEM}")"
            run_args=(--config "${CONFIG_SOURCE}" "${submit_mode}") ;;
        resume)
            [[ "$#" == 2 && "$2" == /* ]] || die "resume requires one absolute suite path."
            read_suite_context "$2"
            default_time="$(profile_time "${PROFILE}" "${DATASET_STEM}")"; run_args=(resume "$2") ;;
        campaign)
            [[ "${DATASET}" == Roman-empire ]] || die "campaign is available only for Roman-empire."
            [[ "$#" == 3 && "$3" == /* ]] || die "Use submit campaign smoke|pilot|full|resume /absolute/campaign."
            [[ "${SPLIT_EXPLICIT}" == 0 ]] || die "Campaigns select all ten splits; omit CM_SPLIT_INDEX/--split-index."
            CAMPAIGN_ROOT="$3"
            if [[ "$2" == resume ]]; then
                PROFILE="$(read_campaign_profile "${CAMPAIGN_ROOT}")"
            else
                select_config "$2"; check_config
                (
                    export_effective_environment
                    python3 -B - "${PROJECT_DIR}" "${CAMPAIGN_ROOT}" "${CONFIG_SOURCE}" "${PROFILE}" "${DATA_ROOT}" "${RUNTIME_SHELL}" <<'PYCAMPAIGNCREATE'
import sys
sys.path.insert(0, sys.argv[1])
from run_cross_morse import create_roman_campaign
print('[CAMPAIGN]', create_roman_campaign(*sys.argv[1:]))
PYCAMPAIGNCREATE
                )
            fi
            read_suite_context "${CAMPAIGN_ROOT}"
            PROJECT_DIR="${CAMPAIGN_ROOT}/source"
            PROFILE_SOURCE="${PROJECT_DIR}/configs/active.env.example"
            default_time="$(profile_time "${PROFILE}" "${DATASET_STEM}")"
            batch_extra=(--array="0-9${CM_ARRAY_PARALLEL:+%${CM_ARRAY_PARALLEL}}")
            run_args=(campaign "${PROFILE}" "${CAMPAIGN_ROOT}") ;;
        *) die "Unknown submit mode: ${submit_mode}" ;;
    esac
    mkdir -p "${LOGS_DIR}" "${OUTPUTS_DIR}" "${CACHE_DIR}/cross_morse"
    export_site_settings
    export CM_PROJECT_DIR="${PROJECT_DIR}"
    if [[ -n "${PROFILE_SOURCE}" ]]; then export CM_ENV_FILE="${PROFILE_SOURCE}"; else unset CM_ENV_FILE; fi
    routing=(--dataset "${DATASET}" --data-root "${DATA_ROOT}")
    if [[ "${submit_mode}" != campaign && -n "${SPLIT_INDEX}" ]]; then routing+=(--split-index "${SPLIT_INDEX}"); fi
    log_pattern="CROSS_MORSE_${DATASET_LOG}_%j"
    [[ "${submit_mode}" != campaign ]] || log_pattern="CROSS_MORSE_${DATASET_LOG}_%A_%a"
    sbatch_site_args run "${DATASET_STEM}"
    exec sbatch "${SBATCH_ARGS[@]}" --time="${CM_TIME:-${default_time}}" "${batch_extra[@]}" \
        --output="${LOGS_DIR}/${log_pattern}.out" --error="${LOGS_DIR}/${log_pattern}.err" \
        "${PROJECT_DIR}/${SHELL_NAME}" "${routing[@]}" "${run_args[@]}"
}

mode_site_env() {
    local key
    CM_BASE="${BASE}" CM_DATA_ROOT="${DATA_ROOT}" CM_VENV_DIR="${VENV_DIR}"
    printf '# Effective CROSS-MORSE site settings (%s)\n' "${PROFILE_SOURCE:-SITE SETTINGS of ${RUNTIME_SHELL}}"
    for key in "${SITE_KEYS[@]}"; do printf 'export %s=%q\n' "${key}" "${!key-}"; done
    printf 'export CM_PYTHON=%q\n' "${CM_PYTHON:-${PYTHON}}"
}

print_plan() {   # print_plan PROFILE -> the jobs `reproduce PROFILE` submits
    local profile="$1" dataset stem n=0 reference
    reference="$(reproduction_tool python3 "${PROJECT_DIR}" reference --root "${PROJECT_DIR}" "${REPRO_DATASETS[@]}" 2>/dev/null || true)"
    printf '\n  %-7s %-14s %-8s %-6s %-10s %-22s %s\n' job dataset profile splits limit "depends on" "paper GPU-hours (Tables S5, S41)"
    printf '  %-7s %-14s %-8s %-6s %-10s %-22s\n' setup - - - "${CM_SETUP_TIME}" -
    for dataset in "${REPRO_DATASETS[@]}"; do
        stem="$(stem_of "${dataset}")"; n=$((n+1))
        local splits=- paper=-
        [[ "${dataset}" != Roman-empire ]] || splits=0-9
        if [[ -n "${reference}" && "${profile}" == full ]]; then
            paper="$(python3 -c 'import json,sys; r=json.loads(sys.argv[1])[sys.argv[2]]; s=r["training_GPU_h_per_seed"]; q=r["qualification_increment_GPU_h"]; print(("%s per seed, %s qualification" % (s, q)) if s is not None else "-")' "${reference}" "${dataset}" 2>/dev/null || echo -)"
        fi
        printf '  %-7s %-14s %-8s %-6s %-10s %-22s %s\n' "run ${n}" "${dataset}" "${profile}" "${splits}" \
            "$(profile_time "${profile}" "${stem}")" "setup" "${paper}"
    done
    printf '  %-7s %-14s %-8s %-6s %-10s %-22s\n' report - - - "${CM_REPORT_TIME}" "all runs (afterany)"
    printf '\n  A run job that reaches its limit checkpoints and requeues itself (up to %s times);\n' "${CM_MAX_CONTINUATIONS}"
    printf '  the limit sets the length of one job, not of the run.\n'
}

mode_check() {
    local profile="${1:-full}" problems=0 warnings=0 ident actual registered matches parent module_name
    case "${profile}" in smoke|pilot|full) ;; *) die "check takes full, pilot or smoke." ;; esac
    [[ "$#" -le 1 ]] || die "check takes one profile."
    selected_datasets
    ok() { printf '  [ ok ] %s\n' "$*"; }
    warn() { printf '  [warn] %s\n' "$*"; warnings=$((warnings+1)); }
    bad() { printf '  [FAIL] %s\n' "$*"; problems=$((problems+1)); }
    printf 'CROSS-MORSE reproduction check (%s, %s)\n\n' "${profile}" "$(utc)"
    printf 'Source and identity\n'
    if command -v python3 >/dev/null 2>&1; then ok "python3 $(python3 -c 'import platform; print(platform.python_version())') on the login node"
    else bad "python3 is required on the login node"; fi
    ok "source ${PROJECT_DIR}"
    if ident="$(reproduction_tool python3 "${PROJECT_DIR}" identity --root "${PROJECT_DIR}" 2>&1)"; then
        actual="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["runtime_sha256"])' "${ident}")"
        registered="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["registered_sha256"] or "")' "${ident}")"
        matches="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["matches_release"])' "${ident}")"
        if [[ "${matches}" == True ]]; then ok "runtime fingerprint ${actual} (the released runtime)"
        elif [[ "${CM_ALLOW_SOURCE_CHANGE:-0}" == 1 ]]; then warn "runtime fingerprint ${actual} differs from the released ${registered}; CM_ALLOW_SOURCE_CHANGE=1 labels the suite a variant"
        else bad "runtime fingerprint ${actual} differs from the released ${registered}; restore run_cross_morse*.py and cross_morse/*.py"; fi
    else bad "runtime identity: ${ident}"; fi
    ok "launcher $(sha256sum "${RUNTIME_SHELL}" | cut -c1-16)... (${RUNTIME_SHELL})"
    if out="$(reproduction_tool python3 "${PROJECT_DIR}" configs --root "${PROJECT_DIR}" --profile "${profile}" "${REPRO_DATASETS[@]}" 2>&1)"; then
        while IFS=$'\t' read -r name file seeds epochs precision; do ok "${name}: ${file} (${seeds}, ${epochs}, ${precision})"; done <<< "${out}"
    else bad "configurations: ${out}"; fi

    printf '\nScheduler\n'
    if command -v sbatch >/dev/null 2>&1; then ok "sbatch available"; else bad "sbatch is not available on this node"; fi
    if [[ -n "${CM_ACCOUNT}" ]]; then ok "account ${CM_ACCOUNT}"; else bad "CM_ACCOUNT is empty; set it in the SITE SETTINGS or export it"; fi
    if command -v sinfo >/dev/null 2>&1; then
        if [[ -n "$(sinfo -h -p "${CM_PARTITION}" -o %P 2>/dev/null | head -n1)" ]]; then ok "partition ${CM_PARTITION}"
        else warn "partition ${CM_PARTITION} is not listed by sinfo"; fi
    else warn "sinfo unavailable; partition ${CM_PARTITION} not verified"; fi
    ok "per job: ${CM_GPU_REQUEST:-no GPU request}, ${THREADS} CPUs${CM_MEM:+, ${CM_MEM}}${CM_SBATCH_EXTRA:+, ${CM_SBATCH_EXTRA}}"

    printf '\nStorage\n'
    parent="${BASE}"; while [[ ! -e "${parent}" ]]; do parent="$(dirname "${parent}")"; done
    if [[ -d "${parent}" && -w "${parent}" ]]; then
        ok "base ${BASE}$([[ "${parent}" == "${BASE}" ]] || printf ' (to be created under %s)' "${parent}")"
        ok "free space at ${parent}: $(df -Ph "${parent}" 2>/dev/null | awk 'NR==2{print $4}')"
    else bad "base ${BASE}: ${parent} is not writable"; fi
    ok "datasets ${DATA_ROOT} (official releases are downloaded on first use)"
    if [[ -x "${PYTHON}" ]]; then ok "environment ${VENV_DIR} (the setup job verifies PyTorch and the GPU)"
    else warn "environment ${VENV_DIR} is absent; the setup job creates it (${CM_TORCH_SPEC:-no PyTorch wheel set})"; fi

    printf '\nSoftware\n'
    if [[ -n "${BUILD_MODULE}${PYTHON_MODULE}" ]]; then
        if ensure_module_command; then
            for module_name in ${BUILD_MODULE} ${PYTHON_MODULE}; do
                if (set +u; module is-avail "${module_name}") >/dev/null 2>&1; then ok "module ${module_name}"
                else warn "module ${module_name} is not reported available here; the jobs load it on the compute node"; fi
            done
        else warn "no module command on this node; the jobs load ${BUILD_MODULE} ${PYTHON_MODULE}"; fi
    else ok "no modules (the jobs keep their environment)"; fi
    ok "compute-node architecture ${EXPECTED_ARCH:-any}; PyTorch ${CM_TORCH_SPEC:-preinstalled}${CM_TORCH_INDEX:+ from ${CM_TORCH_INDEX}}"

    printf '\nPlan (bash run_cross_morse.sh reproduce %s)' "${profile}"
    print_plan "${profile}"
    printf '\n%d problem(s), %d warning(s). Nothing was created or submitted.\n' "${problems}" "${warnings}"
    (( problems == 0 )) || exit "${EXIT_USAGE}"
}

submit_job() {   # submit_job ROLE KEY [sbatch options...] -- SCRIPT ARGS... -> prints the job id
    local role="$1" key="$2" id; shift 2
    id="$(sbatch --parsable "$@")" || fail "${EXIT_ORCHESTRATION}" "sbatch refused the ${role} job ${key}; submitted jobs are listed in ${REPRO_SUITE}/000_MANIFEST/job_ids.tsv"
    id="${id%%;*}"
    printf '%s\t%s\t%s\t%s\n' "${role}" "${key}" "${id}" "$(utc)" >> "${REPRO_SUITE}/000_MANIFEST/job_ids.tsv"
    printf '%s\n' "${id}"
}

# Submit setup, every unfinished run and the report job of an existing suite.
submit_suite_jobs() {   # submit_suite_jobs SUITE PROFILE [only-unfinished]
    local suite="$1" profile="$2" only_unfinished="${3:-}" tag key dataset kind path launcher stem source
    local setup_id run_ids=() id time array source_env label states=""
    tag="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tag"])' "${suite}/000_MANIFEST/suite.json")"
    label="$(printf '%s' "${tag}" | tr -d '_' | tr '[:upper:]' '[:lower:]')"
    source="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["report_source"])' "${suite}/000_MANIFEST/suite.json")"
    REPRO_SUITE="${suite}"
    mkdir -p "${LOGS_DIR}"
    if [[ -n "${only_unfinished}" ]]; then
        local squeue_file
        squeue_file="$(mktemp)"
        if command -v squeue >/dev/null 2>&1; then squeue -h -u "${USER:-$(id -un)}" -o '%i %T' > "${squeue_file}" 2>/dev/null || true; fi
        states="$(reproduction_tool python3 "${source}" status "${suite}" --json --squeue "${squeue_file}")"
        rm -f "${squeue_file}"
    fi
    export_site_settings
    unset CM_DATASET CM_SPLIT_INDEX CM_CONFIG CM_STAGE
    export CM_REPRODUCTION_SUITE="${suite}"
    sbatch_site_args run
    setup_id="$(CM_PROJECT_DIR="${source}" CM_RUN_KEY=00_setup submit_job setup 00_setup "${SBATCH_ARGS[@]}" \
        --time="${CM_SETUP_TIME}" --job-name="${label}_setup" \
        --output="${LOGS_DIR}/${tag}_00_setup_%j.out" --error="${LOGS_DIR}/${tag}_00_setup_%j.err" \
        "${source}/${SHELL_NAME}" setup)"
    printf '[SETUP] job %s\n' "${setup_id}"
    while IFS=$'\t' read -r key dataset kind path launcher; do
        [[ "${key}" != key ]] || continue
        if [[ -n "${only_unfinished}" ]]; then
            state="$(python3 -c 'import json,sys; print(next(r["state"] for r in json.loads(sys.argv[1])["runs"] if r["key"]==sys.argv[2]))' "${states}" "${key}")"
            case "${state}" in
                COMPLETE) printf '[SKIP] %s complete\n' "${key}"; continue ;;
                QUEUED|RUNNING) printf '[SKIP] %s already %s\n' "${key}" "${state,,}"; continue ;;
            esac
        fi
        stem="$(stem_of "${dataset}")"
        time="$(profile_time "${profile}" "${stem}")"
        source_env="${path}/source/configs/active.env.example"
        sbatch_site_args run "${stem}"
        if [[ "${kind}" == campaign ]]; then
            array="0-9${CM_ARRAY_PARALLEL:+%${CM_ARRAY_PARALLEL}}"
            id="$(CM_PROJECT_DIR="${path}/source" CM_ENV_FILE="${source_env}" CM_RUN_KEY="${key}" submit_job run "${key}" \
                "${SBATCH_ARGS[@]}" --time="${time}" --job-name="${label}_${key}" --array="${array}" \
                --dependency="afterok:${setup_id}" --kill-on-invalid-dep=yes \
                --output="${LOGS_DIR}/${tag}_${key}_%A_%a.out" --error="${LOGS_DIR}/${tag}_${key}_%A_%a.err" \
                "${launcher}" --dataset "${dataset}" --data-root "${DATA_ROOT}" campaign "${profile}" "${path}")"
        else
            id="$(CM_PROJECT_DIR="${path}/source" CM_ENV_FILE="${source_env}" CM_RUN_KEY="${key}" submit_job run "${key}" \
                "${SBATCH_ARGS[@]}" --time="${time}" --job-name="${label}_${key}" \
                --dependency="afterok:${setup_id}" --kill-on-invalid-dep=yes \
                --output="${LOGS_DIR}/${tag}_${key}_%j.out" --error="${LOGS_DIR}/${tag}_${key}_%j.err" \
                "${launcher}" --dataset "${dataset}" --data-root "${DATA_ROOT}" resume "${path}")"
        fi
        run_ids+=("${id}")
        printf '[RUN] %-18s %-14s job %s (limit %s)\n' "${key}" "${dataset}" "${id}" "${time}"
    done < "${suite}/000_MANIFEST/plan.tsv"
    local dependency="afterany:${setup_id}"
    (( ${#run_ids[@]} == 0 )) || dependency="afterany:$(IFS=:; printf '%s' "${run_ids[*]}")"
    sbatch_site_args report
    id="$(CM_PROJECT_DIR="${source}" CM_RUN_KEY=99_report submit_job report 99_report "${SBATCH_ARGS[@]}" \
        --time="${CM_REPORT_TIME}" --job-name="${label}_report" --dependency="${dependency}" \
        --output="${LOGS_DIR}/${tag}_99_report_%j.out" --error="${LOGS_DIR}/${tag}_99_report_%j.err" \
        "${source}/${SHELL_NAME}" finalize "${suite}")"
    printf '[REPORT] job %s (after every run)\n' "${id}"
}

write_where_files() {   # write_where_files SUITE TAG
    local suite="$1" tag="$2"
    {
        printf 'SUITE=%s\n' "${suite}"
        printf 'STATUS_COMMAND=bash %q status %q\n' "${RUNTIME_SHELL}" "${suite}"
        printf 'LIVE_LOG_OUT=%s\nLIVE_LOG_ERR=%s\n' "${LIVE_OUT}" "${LIVE_ERR}"
        printf 'JOB_LOGS=%s/%s_*.out (.err)\n' "${LOGS_DIR}" "${tag}"
        printf 'RUNS=%s/runs/<run>/ (each with launcher/stdout.log and launcher/stderr.log)\n' "${suite}"
        printf 'REPORT=%s/REPRODUCTION_REPORT.md\n' "${suite}"
        printf 'RESULTS_ZIP=%s/%s_RESULTS.zip\n' "${suite}" "${tag}"
        printf 'MACHINE_STATUS=%s/SUITE_STATUS.json\n' "${suite}"
        printf 'RESUME_COMMAND=bash %q reproduce resume %q\n' "${RUNTIME_SHELL}" "${suite}"
    } > "${suite}/000_MANIFEST/WHERE_ARE_MY_FILES.txt"
}

mode_reproduce() {
    [[ -z "${SLURM_JOB_ID:-}" ]] || die "reproduce submits jobs; run it on a login node."
    command -v sbatch >/dev/null 2>&1 || die "sbatch is not available in this shell."
    command -v python3 >/dev/null 2>&1 || die "python3 is required on the login node."
    [[ -n "${CM_ACCOUNT:-}" ]] || die "Set CM_ACCOUNT in the SITE SETTINGS, the environment or CM_ENV_FILE before submitting."
    if [[ "${1:-}" == resume ]]; then
        [[ "$#" == 2 && "$2" == /* ]] || die "Use: reproduce resume /absolute/suite"
        local suite="$2" profile
        [[ -f "${suite}/000_MANIFEST/suite.json" ]] || die "Not a reproduction suite: ${suite}"
        profile="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["profile"])' "${suite}/000_MANIFEST/suite.json")"
        printf '[RESUME] %s (%s): resubmitting the unfinished runs\n' "${suite}" "${profile}"
        submit_suite_jobs "${suite}" "${profile}" only-unfinished
        return
    fi
    local profile="${1:-full}" suite tag allow=()
    case "${profile}" in smoke|pilot|full) ;; *) die "reproduce takes full, pilot or smoke, or resume /absolute/suite." ;; esac
    [[ "$#" -le 1 ]] || die "reproduce takes one profile."
    selected_datasets
    [[ "${CM_ALLOW_SOURCE_CHANGE:-0}" != 1 ]] || allow=(--allow-source-change)
    mkdir -p "${LOGS_DIR}" "${OUTPUTS_DIR}" "${CACHE_DIR}/cross_morse" "${DATA_ROOT}"
    export_site_settings
    unset CM_DATASET CM_SPLIT_INDEX CM_CONFIG
    suite="$(reproduction_tool python3 "${PROJECT_DIR}" create --outputs "${OUTPUTS_DIR}" \
        --state-dir "${CACHE_DIR}/cross_morse" --source "${PROJECT_DIR}" --profile "${profile}" \
        --data-root "${DATA_ROOT}" --runtime-shell "${RUNTIME_SHELL}" "${allow[@]}" "${REPRO_DATASETS[@]}")" \
        || fail "${EXIT_ORCHESTRATION}" "The suite could not be created; nothing was submitted."
    tag="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tag"])' "${suite}/000_MANIFEST/suite.json")"
    write_where_files "${suite}" "${tag}"
    printf '[SUITE] %s\n[TAG] %s\n[PROFILE] %s\n[DATASETS] %s\n' "${suite}" "${tag}" "${profile}" "${REPRO_DATASETS[*]}"
    submit_suite_jobs "${suite}" "${profile}"
    cat <<EOF

Submitted. The suite and every run directory are frozen; the jobs start when the scheduler allows.
  progress   bash ${RUNTIME_SHELL} status ${suite}
  live log   tail -n 100 -F ${LIVE_OUT} ${LIVE_ERR}
  report     ${suite}/REPRODUCTION_REPORT.md   (written by the report job)
  results    ${suite}/${tag}_RESULTS.zip
  resume     bash ${RUNTIME_SHELL} reproduce resume ${suite}
EOF
}

mode_status() {
    local suite="${1:-}" squeue_file=""
    [[ -n "${suite}" ]] || suite="$(latest_suite)" || die "No reproduction suite found under ${OUTPUTS_DIR}; pass its path."
    [[ -f "${suite}/000_MANIFEST/suite.json" ]] || die "Not a reproduction suite: ${suite}"
    if command -v squeue >/dev/null 2>&1; then
        squeue_file="$(mktemp)"
        squeue -h -u "${USER:-$(id -un)}" -o '%i %T' > "${squeue_file}" 2>/dev/null || true
    fi
    reproduction_tool python3 "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["report_source"])' "${suite}/000_MANIFEST/suite.json")" \
        status "${suite}" ${squeue_file:+--squeue "${squeue_file}"}
    [[ -z "${squeue_file}" ]] || rm -f "${squeue_file}"
    [[ ! -f "${suite}/REPRODUCTION_REPORT.md" ]] || printf '\nreport: %s\n' "${suite}/REPRODUCTION_REPORT.md"
}

mode_report() {
    local suite="${1:-}" source tag label id
    [[ -z "${SLURM_JOB_ID:-}" ]] || { job_finalize "${suite}"; return; }
    [[ -n "${suite}" ]] || suite="$(latest_suite)" || die "No reproduction suite found under ${OUTPUTS_DIR}; pass its path."
    [[ -f "${suite}/000_MANIFEST/suite.json" ]] || die "Not a reproduction suite: ${suite}"
    command -v sbatch >/dev/null 2>&1 || die "sbatch is not available in this shell."
    [[ -n "${CM_ACCOUNT:-}" ]] || die "Set CM_ACCOUNT before submitting."
    source="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["report_source"])' "${suite}/000_MANIFEST/suite.json")"
    tag="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tag"])' "${suite}/000_MANIFEST/suite.json")"
    label="$(printf '%s' "${tag}" | tr -d '_' | tr '[:upper:]' '[:lower:]')"
    REPRO_SUITE="${suite}"
    mkdir -p "${LOGS_DIR}"
    export_site_settings
    export CM_REPRODUCTION_SUITE="${suite}"
    sbatch_site_args report
    id="$(CM_PROJECT_DIR="${source}" CM_RUN_KEY=99_report submit_job report 99_report "${SBATCH_ARGS[@]}" \
        --time="${CM_REPORT_TIME}" --job-name="${label}_report" \
        --output="${LOGS_DIR}/${tag}_99_report_%j.out" --error="${LOGS_DIR}/${tag}_99_report_%j.err" \
        "${source}/${SHELL_NAME}" finalize "${suite}")"
    printf '[REPORT] job %s; it waits for any run of the suite that is still queued or running\n' "${id}"
}

# =============================================================================
#  Job-side helpers (inside an allocation)
# =============================================================================
attach_logs() {   # attach_logs LABEL EXACT_OUT EXACT_ERR: tee into exact files and the live pair
    local label="$1" out="$2" err="$3"
    mkdir -p "$(dirname "${out}")" "$(dirname "${err}")" "${LOGS_DIR}"
    touch "${out}" "${err}" "${LIVE_OUT}" "${LIVE_ERR}"
    printf '\n===== %s | job=%s | task=%s | restart=%s | node=%s | %s =====\n' "${label}" "${SLURM_JOB_ID:-local}" \
        "${SLURM_ARRAY_TASK_ID:-NA}" "${SLURM_RESTART_COUNT:-0}" "${SLURMD_NODENAME:-$(hostname -s)}" "$(utc)" >> "${LIVE_OUT}"
    exec > >(tee -a "${out}" "${LIVE_OUT}" >&7) 2> >(tee -a "${err}" "${LIVE_ERR}" >&8)
}
job_target() {
    if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        printf '%s_%s\n' "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}"
    else
        printf '%s\n' "${SLURM_JOB_ID:-local}"
    fi
}
registry() {   # registry STATUS [DETAIL]: timestamped line in the suite's run registry
    [[ -n "${REPRO_SUITE}" && -d "${REPRO_SUITE}/000_MANIFEST" ]] || return 0
    local file="${REPRO_SUITE}/000_MANIFEST/run_registry.tsv" fd
    exec {fd}>>"${file}"
    flock -x "${fd}"
    [[ -s "${file}" ]] || printf 'utc\tjob\tarray_task\trestart\trun\tstatus\tdetail\n' >&"${fd}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(utc)" "${SLURM_JOB_ID:-local}" "${SLURM_ARRAY_TASK_ID:-NA}" \
        "${SLURM_RESTART_COUNT:-0}" "${RUN_KEY:-${SUITE##*/}}" "$1" "${2:-}" >&"${fd}"
    exec {fd}>&-
}
json_python() { if [[ -x "${PYTHON}" ]]; then printf '%s\n' "${PYTHON}"; else command -v python3; fi; }
write_terminal() {   # write_terminal STATUS CLASS REASON EXIT_CODE
    [[ -n "${SUITE}" && -d "${SUITE}/launcher" ]] || return 0
    "$(json_python)" - "${SUITE}/launcher/terminal.json" "$1" "$2" "$3" "$4" "$(job_target)" \
        "${SLURM_RESTART_COUNT:-0}" "${SLURMD_NODENAME:-$(hostname -s)}" <<'PYTERMINAL' || true
import datetime, json, os, sys
from pathlib import Path
path, status, kind, reason, code, job, restart, node = sys.argv[1:]
state = {}
try:
    state = json.loads((Path(path).parents[1] / "run_state.json").read_text())
except Exception:
    pass
record = {"status": status, "failure_class": kind or None, "reason": reason, "exit_code": int(code), "job": job,
          "restart_count": int(restart or 0), "node": node, "last_complete_stage": state.get("last_complete_stage"),
          "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
tmp = path + ".tmp"
with open(tmp, "w") as handle:
    json.dump(record, handle, indent=2)
    handle.write("\n")
os.replace(tmp, path)
PYTERMINAL
}
bump_counter() {   # bump_counter NAME -> prints the incremented count stored in SUITE/launcher
    local file="${SUITE}/launcher/${1}_count" fd count=0
    exec {fd}>>"${SUITE}/launcher/counters.lock"
    flock -x "${fd}"
    [[ -s "${file}" ]] && count="$(<"${file}")"
    [[ "${count}" =~ ^[0-9]+$ ]] || count=0
    count=$((count + 1))
    printf '%s\n' "${count}" > "${file}"
    exec {fd}>&-
    printf '%s\n' "${count}"
}
infrastructure_failure() {   # infrastructure_failure STDERR_FILE BYTE_OFFSET
    local pattern='cudaErrorDevicesUnavailable|CUDA-capable device\(s\) is/are busy or unavailable|uncorrectable ECC error|CUDA driver initialization failed|CUDA initialization error|NVIDIA driver is shutting down|GPU has fallen off the bus|device is lost|system not yet initialized|failed call to cuInit|driver/library version mismatch|CUDA error: unspecified launch failure|NCCL error|Bus error'
    # grep reads the whole stream (no -q), so pipefail never sees an early SIGPIPE.
    if [[ -f "$1" ]] && tail -c "+$(( ${2:-0} + 1 ))" "$1" | grep -E "${pattern}" >/dev/null; then return 0; fi
    if [[ -f "${SUITE}/failure_summary.json" ]] && grep -E "${pattern}" "${SUITE}/failure_summary.json" >/dev/null; then return 0; fi
    return 1
}
submit_continuation_job() {   # fallback when the site does not allow `scontrol requeue`
    local extra=("$@") id stem
    stem="$(stem_of "${DATASET}")"
    sbatch_site_args run "${stem}"
    id="$(sbatch --parsable "${SBATCH_ARGS[@]}" "${extra[@]}" --time="$(profile_time "${PROFILE}" "${stem}")" \
        --job-name="${SLURM_JOB_NAME:-cross_morse}" --dependency="afterany:${SLURM_JOB_ID}" \
        --output="${LOGS_DIR}/CROSS_MORSE_${DATASET_LOG}_%j.out" --error="${LOGS_DIR}/CROSS_MORSE_${DATASET_LOG}_%j.err" \
        "${SUITE}/source/${SHELL_NAME}" --dataset "${DATASET}" --data-root "${DATA_ROOT}" resume "${SUITE}")" || return 1
    id="${id%%;*}"
    [[ -z "${REPRO_SUITE}" ]] || printf 'continuation\t%s\t%s\t%s\n' "${RUN_KEY}" "${id}" "$(utc)" >> "${REPRO_SUITE}/000_MANIFEST/job_ids.tsv"
    printf '[CONTINUE] submitted continuation job %s\n' "${id}"
}
requeue_run() {   # requeue_run continuation|infrastructure REASON: continue this run in a new allocation
    local kind="$1" reason="$2" limit count target node
    if [[ "${kind}" == infrastructure ]]; then limit="${CM_MAX_INFRA_REQUEUES}"; else limit="${CM_MAX_CONTINUATIONS}"; fi
    count="$(bump_counter "${kind}")"
    if (( count > limit )); then
        printf '[%s] budget exhausted (%s of %s): %s\n' "${kind^^}" "$((count - 1))" "${limit}" "${reason}" >&2
        return 1
    fi
    target="$(job_target)"
    node="${SLURMD_NODENAME:-$(hostname -s)}"
    EXECUTION_STATUS="REQUEUED_${kind^^}" REQUEUE_REQUESTED=1
    registry "REQUEUE_${kind^^}" "${count}/${limit}: ${reason}"
    write_terminal REQUEUED "${kind^^}" "${reason}" 0
    printf '[%s] %s of %s: %s\n' "${kind^^}" "${count}" "${limit}" "${reason}"
    if [[ "${kind}" == infrastructure ]]; then
        scontrol update JobId="${target}" ExcNodeList="${node}" >/dev/null 2>&1 || true
    fi
    if scontrol requeue "${target}" >/dev/null 2>&1; then
        printf '[%s] job %s requeued; the run continues from its last sealed stage\n' "${kind^^}" "${target}"
        exit 0
    fi
    printf '[%s] scontrol requeue was refused; submitting a continuation job instead\n' "${kind^^}" >&2
    if [[ "${kind}" == infrastructure ]]; then submit_continuation_job --exclude="${node}" && exit 0
    else submit_continuation_job && exit 0; fi
    return 1
}
gpu_health() {   # GPU health check with retries; result kept in the run's launcher records
    local attempt rc=1 record="${SUITE}/launcher/gpu_health_${JOB_KEY}.json"
    for ((attempt = 1; attempt <= CM_GPU_HEALTH_RETRIES; attempt++)); do
        if "${PYTHON}" "${SUITE}/source/${RUNNER}" --gpu-health-check > "${record}.tmp"; then
            mv "${record}.tmp" "${record}"
            printf '[GPU] %s\n' "$(cat "${record}")"
            return 0
        else
            rc=$?
        fi
        printf '[GPU] health check %s of %s failed (rc=%s)\n' "${attempt}" "${CM_GPU_HEALTH_RETRIES}" "${rc}" >&2
        (( attempt == CM_GPU_HEALTH_RETRIES )) || sleep 10
    done
    rm -f "${record}.tmp"
    return "${rc}"
}
on_job_error() {   # ERR trap inside a run job: a machine-readable record before exit
    local rc="$1" line="$2" command="$3"
    trap - ERR
    FAILURE_CLASS="ORCHESTRATION"
    printf '[FATAL] launcher step failed: rc=%s line=%s command=%s\n' "${rc}" "${line}" "${command}" >&2
    registry "FAILED_ORCHESTRATION" "line ${line}: ${command}"
    write_terminal FAILED ORCHESTRATION "launcher line ${line}: ${command}" "${rc}"
    exit "${rc}"
}
record_exit() {
    local rc=$?
    trap - EXIT
    if [[ -n "${SUITE}" && -d "${SUITE}/launcher" ]]; then
        local status="${EXECUTION_STATUS}"
        [[ -n "${status}" ]] || status="$([[ "${rc}" == 0 ]] && echo COMPLETED || { [[ "${SIGNAL_RECEIVED}" == 1 ]] && echo INTERRUPTED || echo FAILED; })"
        "$(json_python)" - "${SUITE}" "${rc}" "${JOB_KEY:-local}" "${SIGNAL_RECEIVED}" "${STAGE}" "${status}" \
            "${FAILURE_CLASS}" "${SLURM_RESTART_COUNT:-0}" "${SLURMD_NODENAME:-$(hostname -s)}" <<'PYSTATUS' || true
import datetime, json, pathlib, sys
root, code, job, signal, stage, status, kind, restart, node = sys.argv[1:]
data = {'job_id': job, 'stage': stage, 'exit_code': int(code), 'execution_status': status,
        'failure_class': kind or None, 'checkpoint_signal_received': signal == '1',
        'restart_count': int(restart or 0), 'node': node,
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'note': 'Process completion alone does not establish statistical qualification.'}
path = pathlib.Path(root) / 'launcher' / ('status_' + job + ('_r' + restart if restart != '0' else '') + '.json')
temporary = path.with_suffix('.json.tmp')
temporary.write_text(json.dumps(data, indent=2) + '\n')
temporary.replace(path)
PYSTATUS
        if [[ "${rc}" != 0 && "${REQUEUE_REQUESTED}" != 1 ]]; then
            printf '[RESUME] CM_ENV_FILE=%q CM_PROJECT_DIR=%q bash %q submit resume %q\n' \
                "${SUITE}/source/configs/active.env.example" "${SUITE}/source" "${SUITE}/source/${SHELL_NAME}" "${SUITE}" >&2
        fi
    fi
    exit "${rc}"
}
request_checkpoint() {
    SIGNAL_RECEIVED=1; WAIT_INTERRUPTED=1
    printf '[SIGNAL] Time limit approaching: requesting a checkpoint at the next safe Python boundary.\n' >&2
    [[ -z "${CHILD_PID}" ]] || kill -USR1 "${CHILD_PID}" 2>/dev/null || true
}

# =============================================================================
#  Job modes
# =============================================================================
job_setup() {
    local lock building_note=""
    mkdir -p "${CACHE_DIR}/cross_morse" "$(dirname "${VENV_DIR}")"
    exec {lock}>"${CACHE_DIR}/cross_morse/venv_setup.lock"
    flock -x "${lock}"
    if [[ ! -x "${PYTHON}" ]]; then
        if [[ -e "${VENV_DIR}" ]]; then
            building_note="${VENV_DIR}.incomplete.${SLURM_JOB_ID:-$$}"
            mv "${VENV_DIR}" "${building_note}"
            printf '[SETUP] moved an incomplete environment aside: %s\n' "${building_note}"
        fi
        printf '[SETUP] creating %s with %s\n' "${VENV_DIR}" "$(command -v python3)"
        python3 -m venv "${VENV_DIR}"
        "${PYTHON}" -m pip install --upgrade pip setuptools wheel
    fi
    if ! "${PYTHON}" -c 'import torch' >/dev/null 2>&1; then
        [[ -n "${CM_TORCH_SPEC:-}" ]] || fail "${EXIT_PREREQUISITE}" "PyTorch is missing; set CM_TORCH_SPEC (and CM_TORCH_INDEX) in the SITE SETTINGS."
        torch_install=("${PYTHON}" -m pip install "${CM_TORCH_SPEC}")
        [[ -z "${CM_TORCH_INDEX:-}" ]] || torch_install+=(--index-url "${CM_TORCH_INDEX}")
        "${torch_install[@]}"
    fi
    # requirements.txt plus Matplotlib, with which the report job also rebuilds the paper's figures.
    "${PYTHON}" -m pip install -r "${PROJECT_DIR}/requirements.txt" 'matplotlib>=3.8,<4'
    "${PYTHON}" -m pip check
    flock -u "${lock}"
    if [[ -n "${EXPECTED_ARCH}" ]]; then
        [[ "$("${PYTHON}" -c 'import platform; print(platform.machine())')" == "${EXPECTED_ARCH}" ]] \
            || fail "${EXIT_PREREQUISITE}" "The environment's interpreter is not ${EXPECTED_ARCH}."
    fi
    "${PYTHON}" "${PROJECT_DIR}/${RUNNER}" --gpu-health-check
    "${PYTHON}" -m pip freeze --all > "${VENV_DIR}/cross_morse_requirements.lock" 2>/dev/null || true
    registry SETUP_COMPLETE "${VENV_DIR}"
    printf '[SETUP COMPLETE] %s\n' "${VENV_DIR}"
}

job_finalize() {   # the report job of a reproduction suite
    local suite="${1:?finalize needs the suite path}" source tag squeue_file active sacct_file ids deferrals
    suite="$(readlink -f "${suite}")"
    [[ -f "${suite}/000_MANIFEST/suite.json" ]] || die "Not a reproduction suite: ${suite}"
    REPRO_SUITE="${suite}" RUN_KEY="99_report"
    source="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["report_source"])' "${suite}/000_MANIFEST/suite.json")"
    tag="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tag"])' "${suite}/000_MANIFEST/suite.json")"
    mkdir -p "${suite}/999_REPORT/logs"
    attach_logs "REPORT_${tag}" "${suite}/999_REPORT/logs/report_${SLURM_JOB_ID:-local}.out" "${suite}/999_REPORT/logs/report_${SLURM_JOB_ID:-local}.err"
    load_site_modules
    [[ -x "${PYTHON}" ]] || fail "${EXIT_PREREQUISITE}" "Missing ${PYTHON}; the setup job did not complete."
    export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
    unset PYTHONPATH || true
    # Runs that were requeued or continued are still on the scheduler: wait for them.
    if command -v squeue >/dev/null 2>&1; then
        squeue_file="$(mktemp)"
        squeue -h -u "${USER:-$(id -un)}" -o '%i %T' > "${squeue_file}" 2>/dev/null || true
        active="$(reproduction_tool "${PYTHON}" "${source}" active "${suite}" --squeue "${squeue_file}" --exclude "${SLURM_JOB_ID:-0}")"
        rm -f "${squeue_file}"
        if [[ -n "${active}" ]]; then
            deferrals="$(SUITE="${suite}/999_REPORT"; mkdir -p "${SUITE}/launcher"; bump_counter report_deferral)"
            if (( deferrals <= 50 )); then
                sbatch_site_args report
                id="$(sbatch --parsable "${SBATCH_ARGS[@]}" --time="${CM_REPORT_TIME}" --job-name="${SLURM_JOB_NAME:-cross_morse_report}" \
                    --dependency="afterany:${active// /:}" \
                    --output="${LOGS_DIR}/${tag}_99_report_%j.out" --error="${LOGS_DIR}/${tag}_99_report_%j.err" \
                    "${source}/${SHELL_NAME}" finalize "${suite}")"
                printf 'report\t99_report\t%s\t%s\n' "${id%%;*}" "$(utc)" >> "${suite}/000_MANIFEST/job_ids.tsv"
                registry REPORT_DEFERRED "waiting for ${active}"
                printf '[REPORT] runs still active (%s); report job %s will run after them\n' "${active}" "${id%%;*}"
                return 0
            fi
        fi
    fi
    # Slurm accounting of every job of the suite (no account or node names).
    sacct_file="${suite}/999_REPORT/scheduler_accounting.psv"
    ids="$(awk -F '\t' 'NR>1 && $3!="" {sub(/_.*/, "", $3); print $3}' "${suite}/000_MANIFEST/job_ids.tsv" | sort -u | paste -sd, -)"
    if command -v sacct >/dev/null 2>&1 && [[ -n "${ids}" ]]; then
        sacct --jobs "${ids}" --allocations --duplicates --parsable2 \
            --format=JobIDRaw,JobID,JobName%80,Partition,State,ExitCode,ElapsedRaw,Elapsed,Timelimit,AllocTRES%300,Submit,Start,End \
            > "${sacct_file}" 2>/dev/null || true
    fi
    # Ordered copies of every job log of the suite.
    local n=0 log
    while IFS= read -r log; do
        n=$((n + 1)); cp -f "${log}" "${suite}/999_REPORT/logs/$(printf '%04d' "${n}")_$(basename "${log}")" 2>/dev/null || true
    done < <(find "${LOGS_DIR}" -maxdepth 1 -type f \( -name "${tag}_*.out" -o -name "${tag}_*.err" \) | sort -V)
    reproduction_tool "${PYTHON}" "${source}" report "${suite}" ${sacct_file:+--sacct "${sacct_file}"} \
        $([[ "${CM_PACK_ARCHIVES}" != 1 ]] || printf '%s' --pack-archives)
    registry REPORT_WRITTEN "${suite}/REPRODUCTION_REPORT.md"
    ln -sfn "${suite}/${tag}_RESULTS.zip" "${OUTPUTS_DIR}/CROSS_MORSE_LATEST_RESULTS.zip" 2>/dev/null || true
    printf '[REPORT] %s\n[RESULTS] %s\n' "${suite}/REPRODUCTION_REPORT.md" "${suite}/${tag}_RESULTS.zip"
}

job_run() {   # setup, smoke|pilot|full, resume SUITE, campaign PROFILE CAMPAIGN
    local mode="$1"; shift
    case "${mode}" in
        setup|smoke|pilot|full) [[ "$#" -le 0 ]] || die "${mode} takes no positional arguments." ;;
        resume) [[ "$#" == 1 && "$1" == /* ]] || die "resume requires an absolute suite path." ;;
        campaign) [[ "$#" == 2 && "$2" == /* ]] || die "campaign needs profile and absolute campaign path." ;;
        *) die "Unknown mode: ${mode}. Use --help." ;;
    esac
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "Use submit, sbatch, or run inside an allocated GPU job."
    if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${THREADS}" -gt "${SLURM_CPUS_PER_TASK}" ]]; then
        die "CM_THREADS exceeds allocated --cpus-per-task (${SLURM_CPUS_PER_TASK})."
    fi
    if [[ -n "${EXPECTED_ARCH}" && "$(uname -m)" != "${EXPECTED_ARCH}" ]]; then
        fail "${EXIT_PREREQUISITE}" "This site requires ${EXPECTED_ARCH}; the job runs on $(uname -m)."
    fi
    load_site_modules
    mkdir -p "${LOGS_DIR}" "${OUTPUTS_DIR}" "${CACHE_DIR}/cross_morse" "${DATA_ROOT}"
    export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
    export_effective_environment
    unset PYTHONPATH || true
    export OMP_NUM_THREADS="${THREADS}" OPENBLAS_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"
    export TORCH_HOME="${CACHE_DIR}/torch" XDG_CACHE_HOME="${CACHE_DIR}/xdg" PIP_CACHE_DIR="${CACHE_DIR}/pip" OGB_HOME="${DATA_ROOT}"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    mkdir -p "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}"

    if [[ "${mode}" == setup ]]; then
        attach_logs "SETUP" "${LOGS_DIR}/setup_${SLURM_JOB_ID}.out" "${LOGS_DIR}/setup_${SLURM_JOB_ID}.err"
        job_setup
        return 0
    fi
    [[ -x "${PYTHON}" ]] || fail "${EXIT_PREREQUISITE}" "Missing ${PYTHON}; run the setup mode or select an existing environment."
    "${PYTHON}" - <<'PYENV' || fail "${EXIT_PREREQUISITE}" "The job environment is not usable (see above)."
import os, platform
import numpy, scipy, torch
expected = os.environ.get('CM_EXPECT_ARCH', '')
if expected and platform.machine() != expected:
    raise SystemExit('Wrong Python interpreter architecture.')
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable; check the GPU allocation, modules and PyTorch build.')
print('[ENV]', platform.machine(), 'torch', torch.__version__, 'CUDA', torch.version.cuda,
      'GPU', torch.cuda.get_device_name(0), 'NumPy', numpy.__version__, 'SciPy', scipy.__version__)
PYENV

    JOB_KEY="${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+_${SLURM_ARRAY_TASK_ID}}"
    local job_map="${CACHE_DIR}/cross_morse/job_${JOB_KEY}.suite" resume=0 campaign_root restart="${SLURM_RESTART_COUNT:-0}"
    if [[ "${mode}" == campaign ]]; then
        [[ "${DATASET}" == Roman-empire ]] || die "campaign is Roman-empire only."
        [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]$ ]] || die "Campaign requires a Slurm array task from 0 through 9."
        campaign_root="$2"
        PROFILE="$(read_campaign_profile "${campaign_root}")"
        [[ "$1" == "${PROFILE}" ]] || die "Campaign profile disagrees with its frozen record."
        read_suite_context "${campaign_root}"
        PROJECT_DIR="${campaign_root}/source"
        CONFIG_SOURCE="${PROJECT_DIR}/configs/resolved.json"
        SPLIT_INDEX="${SLURM_ARRAY_TASK_ID}"
        SPLIT_EXPLICIT=1
        printf -v split_dir 'split_%02d' "${SPLIT_INDEX}"
        SUITE="${campaign_root}/${split_dir}"
        [[ -z "${RUN_KEY}" ]] || RUN_KEY="${RUN_KEY}/${split_dir}"
        if [[ -f "${SUITE}/launcher/run_manifest.json" ]]; then
            resume=1
        elif [[ -d "${SUITE}/source" ]]; then
            die "Incomplete campaign split snapshot; inspect ${SUITE} before continuing."
        else
            mkdir -p "${SUITE}"
        fi
    elif [[ "${mode}" == resume ]]; then
        SUITE="$(readlink -f "$1")"; resume=1
    elif [[ "${restart}" -gt 0 && -s "${job_map}" ]]; then
        SUITE="$(cat "${job_map}")"; resume=1
    else
        select_config "${mode}"
        SUITE="$(mktemp -d "${OUTPUTS_DIR}/CROSS_MORSE_${DATASET_STEM}_${PROFILE}_$(date -u +%Y%m%dT%H%M%SZ)_J${JOB_KEY}_XXXXXX")"
    fi
    [[ -d "${SUITE}" ]] || die "Suite directory does not exist: ${SUITE}"
    mkdir -p "${SUITE}/launcher/attempts"
    local run_lock
    exec {run_lock}>"${SUITE}/launcher/run.lock"
    flock -n "${run_lock}" || die "Another process is already writing to ${SUITE}."
    printf '%s\n' "${SUITE}" > "${job_map}"
    attach_logs "RUN ${RUN_KEY:-${SUITE##*/}} (${DATASET})" "${SUITE}/launcher/stdout.log" "${SUITE}/launcher/stderr.log"
    trap record_exit EXIT
    trap 'on_job_error "$?" "${LINENO}" "${BASH_COMMAND}"' ERR
    printf '[SUITE] %s\n[DATASET] %s\n[STAGE] %s\n[RESTART] %s\n' "${SUITE}" "${DATASET}" "${STAGE}" "${restart}"
    registry RUNNING "node ${SLURMD_NODENAME:-$(hostname -s)}"
    # Keep the records of an earlier attempt; a resumed run starts from its sealed stages.
    local previous
    for previous in failure_summary.json interruption.json launcher/terminal.json; do
        [[ ! -f "${SUITE}/${previous}" ]] || mv "${SUITE}/${previous}" "${SUITE}/launcher/attempts/${JOB_KEY}_r${restart}_${previous##*/}"
    done
    cp "${RUNTIME_SHELL}" "${SUITE}/launcher/runtime_shell_${JOB_KEY}.sh"
    if [[ "${resume}" == 0 ]]; then
        export_effective_environment
        "${PYTHON}" - "${PROJECT_DIR}" "${SUITE}" "${CONFIG_SOURCE}" "${PROFILE}" "${DATA_ROOT}" "${RUNTIME_SHELL}" "${DATASET}" "${SPLIT_INDEX}" <<'PYSNAPSHOT'
import sys
sys.path.insert(0, sys.argv[1])
from run_cross_morse import create_launcher_snapshot
source, suite, config, profile, data_root, runtime_shell, dataset, split = sys.argv[1:]
create_launcher_snapshot(source, suite, config, profile, data_root, runtime_shell,
                         dataset=dataset, split_index=int(split) if split else None)
print('[SNAPSHOT] Captured source, effective configuration and environment.')
PYSNAPSHOT
    else
        read_suite_context "${SUITE}"
    fi
    "${PYTHON}" - "${PROJECT_DIR}" "${SUITE}" <<'PYVERIFY'
import sys
sys.path.insert(0, sys.argv[1])
from run_cross_morse import verify_launcher_snapshot
record = verify_launcher_snapshot(sys.argv[2])
print('[SNAPSHOT VERIFIED]', record['dataset'], 'split', record['split_index'])
PYVERIFY

    if command -v module >/dev/null 2>&1; then (set +u; module list) > "${SUITE}/launcher/modules_${JOB_KEY}.txt" 2>&1 || true
    else printf 'No environment module system selected.\n' > "${SUITE}/launcher/modules_${JOB_KEY}.txt"; fi
    "${PYTHON}" -m pip freeze > "${SUITE}/launcher/packages_${JOB_KEY}.txt" 2>/dev/null || true
    nvidia-smi > "${SUITE}/launcher/nvidia_smi_${JOB_KEY}.txt" 2>&1 || true
    uname -a > "${SUITE}/launcher/uname_${JOB_KEY}.txt"
    "${PYTHON}" - "${SUITE}/launcher/job_${JOB_KEY}.json" "${JOB_KEY}" "${restart}" "${SLURMD_NODENAME:-$(hostname -s)}" \
        "${SLURM_JOB_PARTITION:-}" "${RUNTIME_SHELL}" "${RUN_KEY}" "${REPRO_SUITE}" <<'PYJOB' || true
import datetime, hashlib, json, os, sys
path, job, restart, node, partition, shell, key, suite = sys.argv[1:]
record = {"job": job, "restart_count": int(restart or 0), "node": node, "partition": partition, "run_key": key or None,
          "reproduction_suite": suite or None, "launcher_sha256": hashlib.sha256(open(shell, "rb").read()).hexdigest(),
          "slurm": {k: v for k, v in os.environ.items() if k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                    "SLURM_CPUS_PER_TASK", "SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES")},
          "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
open(path, "w").write(json.dumps(record, indent=2) + "\n")
PYJOB
    if ! gpu_health; then
        registry INFRASTRUCTURE_HEALTH_FAILURE "GPU health check failed on ${SLURMD_NODENAME:-$(hostname -s)}"
        requeue_run infrastructure "GPU health check failed" || true
        FAILURE_CLASS=INFRASTRUCTURE EXECUTION_STATUS=FAILED
        write_terminal FAILED INFRASTRUCTURE "GPU health check failed; requeue budget exhausted or refused" "${EXIT_INFRASTRUCTURE}"
        exit "${EXIT_INFRASTRUCTURE}"
    fi

    trap request_checkpoint USR1
    local cmd stderr_offset rc deadline
    cmd=("${PYTHON}" -u "${SUITE}/source/${RUNNER}" --dataset "${DATASET}"
         --config "${SUITE}/source/configs/resolved.json" --data-root "${DATA_ROOT}" --output-dir "${SUITE}"
         --device cuda --threads "${THREADS}" --stage "${STAGE}")
    [[ "${resume}" == 0 ]] || cmd+=(--resume)
    printf '[COMMAND]'; printf ' %q' "${cmd[@]}"; printf '\n'
    stderr_offset="$(stat -c %s "${SUITE}/launcher/stderr.log" 2>/dev/null || echo 0)"
    cd "${SUITE}/source"
    srun --ntasks=1 --cpus-per-task="${THREADS}" --unbuffered "${cmd[@]}" &
    CHILD_PID=$!
    while :; do
        WAIT_INTERRUPTED=0
        if wait "${CHILD_PID}"; then rc=0; else rc=$?; fi
        if [[ "${WAIT_INTERRUPTED}" == 1 ]]; then
            # The time limit is near: allow the runner to reach a safe boundary.
            deadline=$(( $(date +%s) + CM_CHECKPOINT_GRACE ))
            while kill -0 "${CHILD_PID}" 2>/dev/null && (( $(date +%s) < deadline )); do sleep 2; done
            if kill -0 "${CHILD_PID}" 2>/dev/null; then
                printf '[SIGNAL] No safe boundary within %ss; continuing from the last sealed stage.\n' "${CM_CHECKPOINT_GRACE}" >&2
                rc="${EXIT_CHECKPOINTED}"
                break
            fi
            continue
        fi
        break
    done
    CHILD_PID=""
    trap - ERR
    printf '[PROCESS EXIT] rc=%s suite=%s\n' "${rc}" "${SUITE}"
    if [[ "${rc}" == 0 ]]; then
        if [[ "${STAGE}" == all ]] && ! grep -q '"status": "COMPLETE"' "${SUITE}/run_state.json" 2>/dev/null; then
            FAILURE_CLASS=ARTIFACT_MISSING EXECUTION_STATUS=FAILED
            registry FAILED_ARTIFACT_MISSING "runner returned 0 without a complete lifecycle"
            write_terminal FAILED ARTIFACT_MISSING "runner returned 0 without a complete lifecycle" "${EXIT_ARTIFACT_MISSING}"
            exit "${EXIT_ARTIFACT_MISSING}"
        fi
        EXECUTION_STATUS=COMPLETED
        registry COMPLETED "stage ${STAGE}"
        write_terminal COMPLETE "" "lifecycle stage ${STAGE} complete" 0
        exit 0
    fi
    if [[ "${rc}" == "${EXIT_CHECKPOINTED}" ]]; then
        requeue_run continuation "checkpointed at the time limit" || true
        FAILURE_CLASS=TIME_BUDGET EXECUTION_STATUS=INTERRUPTED
        registry INCOMPLETE "continuation budget exhausted or refused"
        write_terminal INCOMPLETE TIME_BUDGET "continuation budget (${CM_MAX_CONTINUATIONS}) exhausted or requeue refused" "${rc}"
        exit "${rc}"
    fi
    if infrastructure_failure "${SUITE}/launcher/stderr.log" "${stderr_offset}"; then
        registry INFRASTRUCTURE_FAILURE "rc=${rc}"
        requeue_run infrastructure "GPU or driver failure (rc=${rc})" || true
        FAILURE_CLASS=INFRASTRUCTURE EXECUTION_STATUS=FAILED
        write_terminal FAILED INFRASTRUCTURE "GPU or driver failure (rc=${rc}); requeue budget exhausted or refused" "${rc}"
        exit "${rc}"
    fi
    FAILURE_CLASS=RUNTIME EXECUTION_STATUS=FAILED
    registry FAILED_RUNTIME "rc=${rc}; see ${SUITE}/failure_summary.json"
    write_terminal FAILED RUNTIME "the Python runtime exited with ${rc}; see failure_summary.json" "${rc}"
    exit "${rc}"
}

# =============================================================================
#  Dispatch
# =============================================================================
shift || true
case "${mode}" in
    dry-run) mode_dry_run "$@" ;;
    submit) mode_submit "$@" ;;
    check) mode_check "$@" ;;
    reproduce) mode_reproduce "$@" ;;
    status) mode_status "$@" ;;
    report) mode_report "$@" ;;
    site-env) mode_site_env ;;
    finalize)
        [[ -n "${SLURM_JOB_ID:-}" ]] || die "finalize runs inside the report job; use \`report\` on a login node."
        job_finalize "$@" ;;
    setup|smoke|pilot|full|resume|campaign) job_run "${mode}" "$@" ;;
    *) die "Unknown mode: ${mode}. Use --help." ;;
esac
