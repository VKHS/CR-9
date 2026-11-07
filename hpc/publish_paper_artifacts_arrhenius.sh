#!/usr/bin/env bash
# Publish the measured outputs behind the paper into this repository's GitHub release.
#
#   bash hpc/publish_paper_artifacts_arrhenius.sh template [MAP]   # MAP: JSON {artifact id: directory}
#   bash hpc/publish_paper_artifacts_arrhenius.sh pack MAP          # Slurm job: verify, archive, split
#   bash hpc/publish_paper_artifacts_arrhenius.sh publish           # login node: upload, fetch back, derive
#   bash hpc/publish_paper_artifacts_arrhenius.sh all MAP           # pack in this shell, then publish
#
# `template` lists every artifact the paper's tables and figures need (from the display
# generators). Fill in the directory of each completed run, record set, campaign and the
# environment capture. `pack` checks each directory with the repository's own verifiers
# (sealed eight-stage chain, record-set traces) and writes deterministic archives below
# GitHub's 2 GiB asset limit. `publish` needs a token with contents:write on the repository
# (export GITHUB_TOKEN=...), uploads the parts to the release `paper-artifacts`, records their
# SHA-256 in release/paper_artifacts.json, downloads everything back and verifies it, rebuilds
# every table and figure with `scripts/reconstruct_paper.py`, and writes the split/seed manifests, observed
# environments, source data and the E12 study plans and adapter code (experiments/baselines/) to commit.
# Set CM_PUBLISH_COMMIT=1 to commit and push them.
#
# Cluster settings come from the SITE SETTINGS block of run_cross_morse.sh.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
cd "${ROOT}"
eval "$(bash "${ROOT}/run_cross_morse.sh" site-env)"
# The environment created by `run_cross_morse.sh submit setup` when it exists; any Python 3.11+
# with NumPy and Matplotlib otherwise.
PYTHON=python3
[[ -x "${CM_PYTHON:-/nonexistent}" ]] && PYTHON="${CM_PYTHON}"
STAGE="${CM_PUBLISH_ROOT:-${CM_BASE}/paper-artifacts}"
STAGED="${STAGE}/staged"
FETCHED="${STAGE}/fetched"
WORKSPACE="${STAGE}/paper"
MODE="${1:-}"
MAP="${2:-}"

die() { printf '[ERROR] %s\n' "$*" >&2; exit 2; }
note() { printf '[publish] %s\n' "$*"; }

pack_here() {
    [[ -f "${MAP}" ]] || die "map file not found: ${MAP} (create it with: $0 template)"
    mkdir -p "${STAGED}"
    note "packing into ${STAGED}"
    "${PYTHON}" -B scripts/paper_artifacts.py pack --map "${MAP}" --output-dir "${STAGED}"
}

publish_here() {
    [[ -n "${GITHUB_TOKEN:-${GH_TOKEN:-}}" ]] || command -v gh >/dev/null 2>&1 \
        || die "export GITHUB_TOKEN with contents:write on the repository (or log in with gh)"
    compgen -G "${STAGED}/*.entry.json" >/dev/null || die "nothing packed under ${STAGED}; run: $0 pack MAP"
    note "uploading to the release"
    "${PYTHON}" -B scripts/paper_artifacts.py publish --staged "${STAGED}"
    note "downloading every published artifact back and verifying it"
    rm -rf "${FETCHED}"
    "${PYTHON}" -B scripts/paper_artifacts.py fetch --dest "${FETCHED}" --tier analysis || true
    note "rebuilding every table and figure from the downloaded artifacts"
    "${PYTHON}" -B scripts/reconstruct_paper.py --offline --artifacts "${FETCHED}" --output-dir "${WORKSPACE}" || true
    note "writing split/seed manifests, observed environments, source data and the E12 study plans into the checkout"
    "${PYTHON}" -B scripts/paper_artifacts.py derive --fetched "${FETCHED}" --workspace "${WORKSPACE}"
    note "refreshing the statement compliance, availability page, claim ledger and paper reports"
    "${PYTHON}" -B scripts/statement_check.py --write || true
    "${PYTHON}" -B scripts/availability.py
    "${PYTHON}" -B scripts/claims_ledger.py
    "${PYTHON}" -B scripts/manuscript_availability.py
    "${PYTHON}" -B scripts/paper_support_report.py
    "${PYTHON}" -B scripts/create_release_manifest.py --root . --output RELEASE_MANIFEST.json
    local changed=(release/paper_artifacts.json release/release_scope.json manifests/published environment/observed
                   paper/source_data STATEMENT_COMPLIANCE.md AVAILABILITY.md README.md paper/CLAIMS.md
                   paper/AVAILABILITY_STATEMENT.txt paper/PAPER_SUPPORT.md paper/RESULTS.md docs/badges/evidence.json
                   RELEASE_MANIFEST.json SHA256SUMS.txt)
    # E12 study plans and adapter code, copied from the published baselines record set by derive.
    [[ -d experiments/baselines ]] && changed+=(experiments/baselines)
    if [[ "${CM_PUBLISH_COMMIT:-0}" == "1" ]]; then
        git add "${changed[@]}"
        git commit -m "Publish the measured outputs of the paper as release assets"
        git push
    else
        note "review and commit: git add ${changed[*]}"
    fi
}

case "${MODE}" in
    template)
        MAP="${MAP:-${STAGE}/paper-artifacts.map.json}"
        mkdir -p "$(dirname "${MAP}")"
        "${PYTHON}" -B scripts/paper_artifacts.py map-template --output "${MAP}"
        note "edit ${MAP}, then: $0 pack ${MAP}" ;;
    pack)
        [[ -n "${MAP}" ]] || die "usage: $0 pack MAP"
        if [[ "${CM_PUBLISH_LOCAL:-0}" == "1" ]]; then
            pack_here
        else
            [[ -n "${CM_ACCOUNT}" ]] || die "set CM_ACCOUNT in the SITE SETTINGS of run_cross_morse.sh"
            mkdir -p "${STAGE}/logs"
            read -r -a gpu_request <<< "${CM_REPORT_GPU_REQUEST}"
            read -r -a extra <<< "${CM_SBATCH_EXTRA}"
            sbatch --account="${CM_ACCOUNT}" --partition="${CM_REPORT_PARTITION}" "${gpu_request[@]}" "${extra[@]}" \
                --job-name=cm-pack-artifacts --time="${CM_PACK_TIME:-12:00:00}" --cpus-per-task=4 \
                --output="${STAGE}/logs/pack-%j.out" --error="${STAGE}/logs/pack-%j.err" \
                --export=ALL,CM_PUBLISH_LOCAL=1 "${ROOT}/hpc/publish_paper_artifacts_arrhenius.sh" pack "$(readlink -f "${MAP}")"
            note "after the job completes: $0 publish"
        fi ;;
    publish) publish_here ;;
    all)
        [[ -n "${MAP}" ]] || die "usage: $0 all MAP"
        pack_here
        publish_here ;;
    *) sed -n '2,20p' "$0"; exit 2 ;;
esac
