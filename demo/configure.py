#!/usr/bin/env python3
"""Write the CPU demonstration configurations from the supplied pilot configurations.

The demonstration executes the complete eight-stage lifecycle of every dataset on a
CPU at reduced scale. Each demonstration configuration is the dataset's pilot
configuration (configs/<stem>_pilot.json) with the overrides of VARIANTS applied and
nothing else changed. CPU execution requires the fp32 plant (cross_morse/common.py,
precision_context). The configurations keep profile "pilot" and scientific_run false,
so no demonstration run can be mistaken for a full scientific run. A dataset that runs
on a reduced official-split graph (demo/data.py, ``--graph-npz``) is declared as the
runtime requires for anything but the complete benchmark: dataset "synthetic" with
benchmark_dataset routing and profile "smoke".

    python -B demo/configure.py --variant demo --output DIR      # DIR/<stem>_demo.json
    python -B demo/configure.py --variant probe --dataset ogbn-arxiv --output DIR
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STEMS = {"ogbn-arxiv": "arxiv", "ogbn-proteins": "proteins", "ogbn-products": "products",
         "Roman-empire": "roman_empire", "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp",
         "Peptides-func": "peptides_func"}

# Overrides by variant: (dotted key, value). "probe" measures stage costs with few epochs.
COMMON = (("model.precision", "fp32"),)
VARIANTS = {
    "probe": COMMON + (("training.epochs", 3), ("training.patience", 3),
                       ("controller.epochs", 3), ("controller.patience", 2)),
    "demo": COMMON,
}
# Dataset-specific overrides of a variant (applied after the variant's own).
DATASET_OVERRIDES: dict[str, dict[str, tuple]] = {"probe": {}, "demo": {}}


def _set(config: dict, dotted: str, value) -> None:
    *parents, leaf = dotted.split(".")
    node = config
    for key in parents:
        if not isinstance(node.get(key), dict):
            raise KeyError(f"{dotted}: {key!r} is not an object of the pilot configuration")
        node = node[key]
    if leaf not in node:
        raise KeyError(f"{dotted}: the pilot configuration has no {leaf!r}")
    node[leaf] = value


def demo_config(dataset: str, variant: str, *, graph_npz: str | None = None, split_index: int | None = None) -> dict:
    pilot = json.loads((ROOT / "configs" / f"{STEMS[dataset]}_pilot.json").read_text(encoding="utf-8"))
    if pilot.get("profile") != "pilot" or pilot.get("scientific_run") is not False:
        raise ValueError(f"{dataset}: the pilot configuration is not a non-scientific pilot")
    config = copy.deepcopy(pilot)
    for dotted, value in VARIANTS[variant] + DATASET_OVERRIDES[variant].get(dataset, ()):
        _set(config, dotted, value)
    if graph_npz is not None:
        # The runtime admits a benchmark run only on the complete official graph
        # (cross_morse/data.py); a subset runs as a declared fixture routed to the benchmark.
        config["dataset"] = "synthetic"
        config["benchmark_dataset"] = dataset
        config["profile"] = "smoke"
        config["data"]["graph_npz"] = graph_npz
        config["data"]["download"] = False
    if split_index is not None:
        if dataset != "Roman-empire":
            raise ValueError("only Roman-empire has split runs")
        config["data"]["split_index"] = int(split_index)
    return config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="demo")
    parser.add_argument("--dataset", action="append", choices=sorted(STEMS), default=None)
    parser.add_argument("--graph-npz", default=None, help="reduced official-split graph (demo/data.py)")
    parser.add_argument("--split-index", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    for dataset in args.dataset or list(STEMS):
        config = demo_config(dataset, args.variant, graph_npz=args.graph_npz, split_index=args.split_index)
        path = args.output / f"{STEMS[dataset]}_{args.variant}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
