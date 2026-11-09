#!/usr/bin/env bash
# Place the demonstration runs and reduced graphs of one "Demo runs" workflow run at the fixed paths
# that their sealed configurations name ($DEMO/runs/<stem>, Roman-empire as $DEMO/runs/roman_empire/split_NN;
# $DEMO/graphs/<stem>_demo.npz).
#   bash demo/fetch_inputs.sh RUNS_RUN_ID
set -euo pipefail
run="${1:?the run id of the Demo runs workflow}"
DEMO="${DEMO:?}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for stem in arxiv proteins products pascalvoc_sp coco_sp peptides_func; do
  bash "${here}/artifact.sh" "demo-graph-${stem}" "${DEMO}/graphs" "${run}"
  bash "${here}/artifact.sh" "demo-run-${stem}" "${DEMO}/runs/${stem}" "${run}"
done
for split in 00 01 02 03 04 05 06 07 08 09; do
  bash "${here}/artifact.sh" "demo-run-roman_empire_split_${split}" "${DEMO}/runs/roman_empire/split_${split}" "${run}"
done
du -sh "${DEMO}/runs"/* "${DEMO}/graphs"
