#!/usr/bin/env python3
"""Plan, run and export experiment group B against a sealed CROSS-MORSE run.

    # declare: a fresh directory holding plan.json with the parent's identities
    python -B scripts/run_experiment_b.py predictability plan --parent RUN --output DIR [options]
    # execute the released runtime into that directory and seal every output
    python -B scripts/run_experiment_b.py predictability run --plan DIR/plan.json --device cuda --trust-local-checkpoints
    # recompute the values from the sealed raw files and write the record set
    python -B scripts/run_experiment_b.py predictability export --results DIR --hardware "..." --period "..."
    # E18 only: records:band_bound_tightness (Fig. S18c) of one declared plant from the same sealed run
    python -B scripts/run_experiment_b.py band-audit export-tightness --results DIR --dataset NAME --baseline shared \
        --hardware "..." --period "..."

Experiments: predictability (E2), search-enumeration (E9), band-audit (E18),
negative-controls (Supplementary Table S19) and input-gate (Tables S23/S37).
The parent run is never written. ``export`` verifies the set against its
contract in records/specs.json and installs it, with its index entry, under
records/<set>/ by default (or under --root).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENTS = {
    "predictability": ("cm_experiments.predictability", "predictability"),
    "search-enumeration": ("cm_experiments.search_enumeration", "search_enumeration"),
    "band-audit": ("cm_experiments.band_audit", "band_audit"),
    "negative-controls": ("cm_experiments.negative_controls", "negative_controls"),
    "input-gate": ("cm_experiments.input_gate", "input_gate"),
}
CONTROL_NAMES = {"graph-irrelevant": "graph irrelevant", "misleading-graph": "misleading graph",
                 "sensor-bias": "sensor bias", "support-shift": "support shift"}


def _plan_options(name, p):
    p.add_argument("--parent", type=Path, required=True, help="sealed lifecycle run directory (read only)")
    p.add_argument("--output", type=Path, required=True, help="new directory for plan.json and every output")
    p.add_argument("--allow-fixture", action="store_true", help="accept a synthetic smoke parent (software test)")
    if name == "predictability":
        p.add_argument("--allow-posthoc", action="store_true")
        p.add_argument("--max-graphs", type=int, default=0)
        p.add_argument("--training-seed", type=int, default=17)
        p.add_argument("--epochs", type=int, default=100)
        p.add_argument("--patience", type=int, default=15)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--capacity-tolerance", type=float, default=.01)
    elif name == "search-enumeration":
        p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4])
        p.add_argument("--planner", choices=("mpc_rh", "mpc_ol"), default="mpc_rh")
        p.add_argument("--profile", default="full")
        p.add_argument("--role", choices=("explore", "select", "confirm", "compare", "admit", "test"), default=None)
        p.add_argument("--depths", type=int, nargs="+", default=None)
        p.add_argument("--members", type=int, nargs="+", default=None)
        p.add_argument("--max-graphs", type=int, default=16)
        p.add_argument("--confidence", type=float, default=.95)
        p.add_argument("--max-sequences", type=int, default=20000)
    elif name == "band-audit":
        p.add_argument("--max-trajectories", type=int, default=8)
        p.add_argument("--trajectories", type=int, nargs="+", default=None)
        p.add_argument("--depth-stride", type=int, default=1)
        p.add_argument("--lambda0", type=float, nargs="+", default=[.1, .3, 1.])
        p.add_argument("--exact-max-nodes", type=int, default=4096)
        p.add_argument("--grid-points", type=int, default=4097)
    elif name == "negative-controls":
        p.add_argument("--role", choices=("compare", "admit", "test"), default="test")
        p.add_argument("--controls", nargs="+", choices=tuple(CONTROL_NAMES), default=list(CONTROL_NAMES))
        p.add_argument("--realizations", type=int, default=1)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--max-graphs", type=int, default=0)
        p.add_argument("--gate-alpha", type=float, default=.05)
        p.add_argument("--gate-profile", default="full")
        p.add_argument("--gate-from", type=Path, default=None, help="sealed input-gate run whose frozen gate is reused")
        p.add_argument("--bias-offset", type=float, default=1.5)
        p.add_argument("--bias-noise", type=float, default=.5)
        p.add_argument("--degree-factor", type=float, default=3.)
    elif name == "input-gate":
        p.add_argument("--role", choices=("compare", "admit", "test"), default="test")
        p.add_argument("--alpha", type=float, default=.05)
        p.add_argument("--profile", default="full")
        p.add_argument("--warmup", type=int, default=3)
        p.add_argument("--repetitions", type=int, default=10)
        p.add_argument("--max-graphs", type=int, default=0)


def _plan(name, module, a):
    if name == "predictability":
        return module.plan_predictability(a.parent, a.output, allow_fixture=a.allow_fixture,
                                          allow_posthoc=a.allow_posthoc, max_graphs=a.max_graphs,
                                          training_seed=a.training_seed, epochs=a.epochs, patience=a.patience,
                                          lr=a.lr, batch_size=a.batch_size, capacity_tolerance=a.capacity_tolerance)
    if name == "search-enumeration":
        return module.plan_search_enumeration(a.parent, a.output, horizons=a.horizons, planner=a.planner,
                                              profile=a.profile, role=a.role, depths=a.depths, members=a.members,
                                              max_graphs=a.max_graphs, confidence=a.confidence,
                                              max_sequences=a.max_sequences, allow_fixture=a.allow_fixture)
    if name == "band-audit":
        return module.plan_band_audit(a.parent, a.output, max_trajectories=a.max_trajectories,
                                      trajectories=a.trajectories, depth_stride=a.depth_stride, lambda0=a.lambda0,
                                      exact_max_nodes=a.exact_max_nodes, grid_points=a.grid_points,
                                      allow_fixture=a.allow_fixture)
    if name == "negative-controls":
        return module.plan_negative_controls(a.parent, a.output, role=a.role,
                                             controls=[CONTROL_NAMES[c] for c in a.controls],
                                             realizations=a.realizations, seed=a.seed, max_graphs=a.max_graphs,
                                             gate_alpha=a.gate_alpha, gate_profile=a.gate_profile,
                                             gate_from=a.gate_from, bias_offset=a.bias_offset,
                                             bias_noise=a.bias_noise, degree_factor=a.degree_factor,
                                             allow_fixture=a.allow_fixture)
    return module.plan_input_gate(a.parent, a.output, role=a.role, alpha=a.alpha, profile=a.profile,
                                  warmup=a.warmup, repetitions=a.repetitions, max_graphs=a.max_graphs,
                                  allow_fixture=a.allow_fixture)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    experiments = parser.add_subparsers(dest="experiment", required=True)
    for name in EXPERIMENTS:
        actions = experiments.add_parser(name).add_subparsers(dest="action", required=True)
        _plan_options(name, actions.add_parser("plan"))
        q = actions.add_parser("run")
        q.add_argument("--plan", type=Path, required=True)
        q.add_argument("--device", default="cuda")
        q.add_argument("--threads", type=int, default=2)
        q.add_argument("--trust-local-checkpoints", action="store_true")
        q = actions.add_parser("export")
        q.add_argument("--results", type=Path, action="append", required=True,
                       help="sealed run directory; repeat for the official splits of one dataset")
        q.add_argument("--root", type=Path, default=None, help="records root (default records/)")
        q.add_argument("--hardware", required=True)
        q.add_argument("--period", required=True)
        q.add_argument("--replace", action="store_true")
        q.add_argument("--allow-private-patterns", action="store_true")
        q.add_argument("--allow-truncated", action="store_true")
        if name == "band-audit":
            q = actions.add_parser("export-tightness",
                                   help="write records:band_bound_tightness (Fig. S18c) of one declared plant")
            q.add_argument("--results", type=Path, action="append", required=True,
                           help="sealed band-audit run of the plant; repeated runs must hold identical values")
            q.add_argument("--dataset", required=True, help="dataset of the plant (Figure S18c names none)")
            q.add_argument("--baseline", required=True,
                           help="nominal transport the actions are measured against: shared (A0), native_mean "
                                "or inserted_identity (Figure S18c names none)")
            q.add_argument("--root", type=Path, default=None, help="records root (default records/)")
            q.add_argument("--hardware", required=True)
            q.add_argument("--period", required=True)
            q.add_argument("--replace", action="store_true")
            q.add_argument("--allow-private-patterns", action="store_true")
    a = parser.parse_args(argv)
    import importlib
    module_name, key = EXPERIMENTS[a.experiment]
    module = importlib.import_module(module_name)
    from cm_experiments.predictability import ExperimentError
    try:
        if a.action == "plan":
            path = _plan(a.experiment, module, a)
            print(json.dumps({"plan": str(path), "experiment": key, "executed": False}, indent=2))
            return 0
        if a.action == "run":
            if a.threads < 1:
                raise ExperimentError("a positive thread count is required")
            import torch
            torch.set_num_threads(a.threads)
            out = getattr(module, f"run_{key}")(a.plan, device=a.device, trust=a.trust_local_checkpoints)
            print(json.dumps({"output": str(out), "experiment": key, "sealed": True, "record_set_written": False},
                             indent=2))
            return 0
        command = ["scripts/run_experiment_b.py", *(argv if argv is not None else sys.argv[1:])]
        if a.action == "export-tightness":
            result = module.export_tightness(a.results, dataset=a.dataset, baseline=a.baseline, records_root=a.root,
                                             hardware=a.hardware, period=a.period, replace=a.replace,
                                             allow_private=a.allow_private_patterns, argv=command)
        else:
            result = module.export_records(a.results, records_root=a.root, hardware=a.hardware, period=a.period,
                                           replace=a.replace, allow_private=a.allow_private_patterns,
                                           allow_truncated=a.allow_truncated, argv=command)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except (ExperimentError, ValueError, OSError, RuntimeError, ImportError) as exc:
        if a.action == "run":
            directory = Path(a.plan).resolve().parent
            if (directory / "experiment.json").is_file() and not (directory / "ARTIFACT_MANIFEST.json").exists():
                from cm_experiments.artifacts import write_json
                failure = directory / "FAILURE.json"
                if not failure.exists():
                    write_json(failure, {"status": "FAILED", "exception": type(exc).__name__, "message": str(exc),
                                         "scope": "partial experiment; no value is exported from it"})
        print("ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
