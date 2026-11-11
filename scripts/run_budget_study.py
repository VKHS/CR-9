#!/usr/bin/env python3
"""Execute one E3-E6 arm study outside a Slurm allocation, with the job evidence the campaign export reads.

``hpc/run_budget_arms_arrhenius.sh`` runs ``scripts/run_budget_arms.py`` inside a GPU allocation of the
authors' cluster and keeps, beside the study, a job-evidence directory ``<study>.job-<id>``: the observed
environment, the installed packages, the command, the standard output and error, the process exit status
and the numerical replay of the finished study; after the allocation ends, ``hpc/finalize_arm_accounting.sh``
adds the scheduler's accounting. This runner keeps the same directory for an execution without a scheduler
(a workstation or the CPU execution): ``<study>.job-local-<time>-<pid>``, whose ``scheduler.json`` records
that no allocation and no accounting exist. ``scripts/paper_outputs.py`` uses it for plans of the CPU device;
an export of such studies is a partial campaign (missing accounting), never the original GPU campaign.

The settings are the batch script's environment variables: CM_SOURCE_ROOT, CM_BASE_RUN, CM_ARM_OUTPUT,
CM_TRUST_CHECKPOINTS=1, CM_ARM_SEED, CM_ARM_EPOCHS, CM_ARM_PATIENCE, CM_ARM_BUDGET_RATIO,
CM_ARM_HARD_CAP_RATIO, CM_ARM_COST_BOUNDS and CM_ALLOW_POSTHOC.

    CM_SOURCE_ROOT=. CM_BASE_RUN=PARENT CM_ARM_OUTPUT=STUDY CM_TRUST_CHECKPOINTS=1 CM_ARM_SEED=3 \\
        python -B scripts/run_budget_study.py --device cpu --threads 4 [--allow-fixture]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def setting(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None or value == "":
        raise SystemExit(f"Set {name}")
    return value


def study_command(args, python: str, base: Path, output: Path) -> list[str]:
    command = [python, "-B", "scripts/run_budget_arms.py", "--run", str(base), "--output", str(output),
               "--device", args.device, "--threads", str(args.threads), "--trust-local-checkpoints",
               "--epochs", os.environ.get("CM_ARM_EPOCHS") or "100",
               "--patience", os.environ.get("CM_ARM_PATIENCE") or "15",
               "--seed", os.environ.get("CM_ARM_SEED") or "17"]
    for variable, flag in (("CM_ARM_BUDGET_RATIO", "--budget-ratio"), ("CM_ARM_HARD_CAP_RATIO", "--hard-cap-ratio"),
                           ("CM_ARM_COST_BOUNDS", "--cost-bounds")):
        if os.environ.get(variable):
            command += [flag, os.environ[variable]]
    if os.environ.get("CM_ALLOW_POSTHOC") == "1":
        command.append("--allow-posthoc")      # recorded; it never restores pre-admission validity
    if args.allow_fixture:
        command.append("--allow-fixture")
    return command


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--allow-fixture", action="store_true",
                        help="parents planned with --allow-fixture (a software check or the CPU execution)")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    source = Path(setting("CM_SOURCE_ROOT")).resolve()
    base = Path(setting("CM_BASE_RUN")).resolve()
    output = Path(setting("CM_ARM_OUTPUT")).absolute()
    if os.environ.get("CM_TRUST_CHECKPOINTS") != "1":
        print("Explicit checkpoint trust required", file=sys.stderr)
        return 2
    if not source.is_dir() or not base.is_dir():
        print("Missing source/base run", file=sys.stderr)
        return 2
    if output.exists():
        print("Experiment output exists; never overwrite original records", file=sys.stderr)
        return 2
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job = f"local-{stamp}-{os.getpid()}"
    root = output.with_name(f"{output.name}.job-{job}")
    root.mkdir(mode=0o700, parents=True)
    (root / "job_id.txt").write_text(job + "\n", encoding="utf-8")
    (root / "scheduler.json").write_text(json.dumps({
        "scheduler": "none", "job_id": job, "host": platform.node(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(), "device": args.device, "threads": args.threads, "started_utc": utc_now(),
        "accounting": "none: no scheduler allocation; the campaign export records the study as missing accounting"},
        indent=1) + "\n", encoding="utf-8")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(args.threads),
                       OPENBLAS_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    subprocess.run(["bash", "environment/arrhenius/capture_environment.sh", args.python, str(root / "environment")],
                   cwd=source, env=environment, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with (root / "pip-freeze.txt").open("w", encoding="utf-8") as handle:
        subprocess.run([args.python, "-m", "pip", "freeze", "--all"], cwd=source, env=environment, check=False,
                       stdout=handle, stderr=subprocess.DEVNULL)
    command = study_command(args, args.python, base, output)
    (root / "command.txt").write_text(" ".join(shlex.quote(c) for c in command) + "\n", encoding="utf-8")
    with (root / "stdout.log").open("w", encoding="utf-8") as out, (root / "stderr.log").open("w", encoding="utf-8") as err:
        code = subprocess.run(command, cwd=source, env=environment, stdout=out, stderr=err, check=False).returncode
    (root / "process_exit_code.txt").write_text(f"{code}\n", encoding="utf-8")
    if code == 0:
        replay = subprocess.run([args.python, "-B", "scripts/run_budget_arms.py", "--verify", str(output),
                                 "--report", str(root / "numerical_replay.json")],
                                cwd=source, env=environment, check=False).returncode
        if replay != 0:
            print(f"numerical replay of {output} failed (exit {replay})", file=sys.stderr)
            return replay
    print(f"{output}: exit {code}; job evidence {root}")
    return code


if __name__ == "__main__":
    sys.exit(main())
