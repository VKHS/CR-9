#!/bin/bash
# The acceptance test of the availability statement, as a reviewer runs it: clone the published repository at one
# revision into a new directory, retrieve the published artifacts there, rebuild every table and figure with the
# complete tier and require every registered item to be fulfilled. Nothing from the author's working tree or local
# artifact caches is used.
#
#   bash scripts/independent_verification.sh --work /abs/new-directory [--revision <commit>] [--repository <url>]
#   make independent-verification VERIFY_WORK=/abs/new-directory [VERIFY_REVISION=<commit>]
#
# --dry-run prints the commands without cloning or downloading.
set -euo pipefail

REPOSITORY="https://github.com/VKHS/CROSS-MORSE.git"
REVISION=""
WORK=""
DRY_RUN=0
PYTHON="${PYTHON:-python3}"

usage() { sed -n '2,10p' "$0" >&2; exit 2; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --work) WORK="${2:-}"; shift 2 ;;
        --revision) REVISION="${2:-}"; shift 2 ;;
        --repository) REPOSITORY="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done
[[ -n "${WORK}" ]] || { echo "--work names a new, empty directory" >&2; usage; }
[[ "${WORK}" = /* ]] || { echo "--work must be an absolute path" >&2; exit 2; }
if [[ -z "${REVISION}" ]]; then
    REVISION="$(git -C "$(dirname "$0")/.." rev-parse HEAD)"
fi

CHECKOUT="${WORK}/checkout"
ARTIFACTS="${WORK}/artifacts"
PAPER="${WORK}/paper"
steps=(
    "git clone --quiet ${REPOSITORY} ${CHECKOUT}"
    "git -C ${CHECKOUT} checkout --quiet ${REVISION}"
    "make -C ${CHECKOUT} paper PYTHON=${PYTHON} PAPER_TIER=complete ARTIFACTS_ROOT=${ARTIFACTS} PAPER_ROOT=${PAPER}"
    "make -C ${CHECKOUT} statement-fulfilled PYTHON=${PYTHON} ARTIFACTS_ROOT=${ARTIFACTS} PAPER_ROOT=${PAPER}"
    "make -C ${CHECKOUT} final-destination-converge PYTHON=${PYTHON}"
)
if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%s\n' "${steps[@]}"
    exit 0
fi
if [[ -e "${WORK}" ]] && [[ -n "$(ls -A "${WORK}")" ]]; then
    echo "${WORK} must be new or empty: the verification never reuses an earlier checkout or artifact cache" >&2
    exit 2
fi
mkdir -p "${WORK}"
for step in "${steps[@]}"; do
    printf '[verify] %s\n' "${step}"
    # shellcheck disable=SC2086
    ${step}
done
printf '[verify] revision %s: every registered item fulfilled from a fresh clone against the published artifacts\n' "${REVISION}"
