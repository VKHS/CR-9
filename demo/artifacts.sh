#!/usr/bin/env bash
# Download every workflow artifact whose name starts with PREFIX into DEST/<name>/ (GH_TOKEN with actions:read).
#   bash demo/artifacts.sh PREFIX DEST [RUN_ID]     RUN_ID defaults to the current workflow run
set -euo pipefail
prefix="${1:?artifact name prefix}"; dest="${2:?destination}"; run="${3:-${GITHUB_RUN_ID:?}}"
api="https://api.github.com/repos/${GITHUB_REPOSITORY:?}"
auth=(-H "Authorization: Bearer ${GH_TOKEN:?}" -H "Accept: application/vnd.github+json")
mkdir -p "${dest}"
page=1
while :; do
  listing="$(curl -fsSL "${auth[@]}" "${api}/actions/runs/${run}/artifacts?per_page=100&page=${page}")"
  rows="$(printf '%s' "${listing}" | python3 -c 'import json,sys
p=sys.argv[1]
for a in json.load(sys.stdin)["artifacts"]:
    if a["name"].startswith(p) and not a["expired"]: print(a["id"], a["name"])' "${prefix}")"
  count="$(printf '%s' "${listing}" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["artifacts"]))')"
  while read -r id name; do
    [[ -n "${id:-}" ]] || continue
    mkdir -p "${dest}/${name}"
    curl -fsSL "${auth[@]}" -o "${dest}/${name}/.artifact.zip" "${api}/actions/artifacts/${id}/zip"
    python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "${dest}/${name}/.artifact.zip" "${dest}/${name}"
    rm -f "${dest}/${name}/.artifact.zip"
    echo "downloaded ${name}"
  done <<< "${rows}"
  [[ "${count}" -lt 100 ]] && break
  page=$((page + 1))
done
