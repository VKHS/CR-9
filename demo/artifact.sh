#!/usr/bin/env bash
# Download one workflow artifact of this repository by name (GitHub REST API; GH_TOKEN with actions:read).
#   bash demo/artifact.sh NAME DEST [RUN_ID]     RUN_ID defaults to the current workflow run
set -euo pipefail
name="${1:?artifact name}"; dest="${2:?destination}"; run="${3:-${GITHUB_RUN_ID:?}}"
api="https://api.github.com/repos/${GITHUB_REPOSITORY:?}"
auth=(-H "Authorization: Bearer ${GH_TOKEN:?}" -H "Accept: application/vnd.github+json")
id="$(curl -fsSL --retry 8 --retry-delay 15 --retry-all-errors "${auth[@]}" "${api}/actions/runs/${run}/artifacts?per_page=100&name=${name}" \
      | python3 -c 'import json,sys; a=[x for x in json.load(sys.stdin)["artifacts"] if not x["expired"]]; print(a[0]["id"] if a else "")')"
[[ -n "${id}" ]] || { echo "no artifact ${name} in run ${run}" >&2; exit 1; }
mkdir -p "${dest}"
curl -fsSL --retry 8 --retry-delay 15 --retry-all-errors "${auth[@]}" -o "${dest}/.artifact.zip" "${api}/actions/artifacts/${id}/zip"
python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "${dest}/.artifact.zip" "${dest}"
rm -f "${dest}/.artifact.zip"
echo "downloaded ${name} (run ${run}) into ${dest}"
