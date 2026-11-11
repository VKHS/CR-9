#!/usr/bin/env bash
# Place the sealed CPU runs and subset graphs of "CPU lifecycle runs" workflow runs at the fixed paths that
# their sealed configurations name ($DEMO/runs/<stem>, Roman-empire as $DEMO/runs/roman_empire/split_NN;
# $DEMO/graphs/<stem>_demo.npz).
#   bash demo/fetch_inputs.sh RUN_ID [STEM=RUN_ID ...]     a later run of one dataset overrides RUN_ID
set -euo pipefail
default="${1:?the run id of the Demo runs workflow}"; shift
DEMO="${DEMO:?}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
declare -A source=()
for spec in "$@"; do source["${spec%%=*}"]="${spec#*=}"; done
run_of() { echo "${source[$1]:-${default}}"; }
for stem in arxiv proteins products pascalvoc_sp coco_sp peptides_func; do
  bash "${here}/artifact.sh" "demo-graph-${stem}" "${DEMO}/graphs" "$(run_of "${stem}")"
  bash "${here}/artifact.sh" "demo-run-${stem}" "${DEMO}/runs/${stem}" "$(run_of "${stem}")"
done
for split in 00 01 02 03 04 05 06 07 08 09; do
  bash "${here}/artifact.sh" "demo-run-roman_empire_split_${split}" "${DEMO}/runs/roman_empire/split_${split}" \
    "$(run_of roman_empire)"
done
# A workflow artifact drops hidden files; restore the empty ones the seals declare (demo/restore_hidden.py).
python3 -B "${here}/restore_hidden.py" "${DEMO}"/runs/*
du -sh "${DEMO}/runs"/* "${DEMO}/graphs"
