"""Timing protocol E17 in the layout of the baselines record set (`timing.csv`).

A session times one method against the nominal execution (the frozen plant at full depth on
the same input). Every repeat calls the nominal execution and the method once each, in alternating
order, synchronising the accelerator before and after each call when PyTorch with CUDA is loaded,
and writes one row:

    dataset, method, seed, split, input_id, repeat, warmup, seconds, nominal_seconds,
    work, nominal_work, job_key, log

``warmup`` is 1 for the warm-up repeats, which the reducer ignores; the post-warm-up rows give the
wall ratio seconds/nominal_seconds and the work ratio work/nominal_work. ``work`` is the method's
own operation count for that execution (multiply-adds or the ledger the manuscript uses) and is
supplied by the caller, because only the executing code can count it.

Use from the evaluation script of a method, inside the scheduler job whose key the rows carry:

    from cm_campaigns.timing import TimingSession, write_rows
    session = TimingSession(dataset="ogbn-arxiv", method="AMP", seed=1, split="official",
                            job_key="arrhenius|7130412|2026-02-09T08:41:17|2026-02-09T08:41:20",
                            log="logs/cm.ogbn-arxiv.AMP.timing.s01-7130412.out")
    rows = session.measure(input_id="official-test", nominal=lambda: plant(graph), method=lambda: model(graph),
                           nominal_work=plant_work, method_work=model_work, warmup=50, repetitions=200)
    write_rows(rows, "timing.csv")

The job key is the one `cm_campaigns.accounting.job_key` derives from the sacct export
(Cluster|JobIDRaw|Submit|Start); the reducer requires it to name a completed GPU allocation.
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

COLUMNS = ("dataset", "method", "seed", "split", "input_id", "repeat", "warmup", "seconds", "nominal_seconds",
           "work", "nominal_work", "job_key", "log")


def _synchronize() -> None:
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(call: Callable[[], object]) -> float:
    _synchronize()
    start = time.perf_counter()
    call()
    _synchronize()
    return time.perf_counter() - start


@dataclass(frozen=True)
class TimingSession:
    dataset: str
    method: str
    seed: int
    split: str
    job_key: str
    log: str

    def measure(self, *, input_id: str, nominal: Callable[[], object], method: Callable[[], object],
                nominal_work: float, method_work: float, warmup: int = 50, repetitions: int = 200) -> list[dict]:
        if warmup < 1 or repetitions < 3:
            raise ValueError("protocol E17 needs at least one warm-up repeat and three measured repeats")
        if nominal_work <= 0 or method_work < 0:
            raise ValueError("work counts must be positive (nominal) and non-negative (method)")
        rows = []
        for index in range(warmup + repetitions):
            if index % 2 == 0:
                nominal_seconds, seconds = _timed(nominal), _timed(method)
            else:
                seconds, nominal_seconds = _timed(method), _timed(nominal)
            rows.append({"dataset": self.dataset, "method": self.method, "seed": self.seed, "split": self.split,
                         "input_id": input_id, "repeat": index, "warmup": int(index < warmup),
                         "seconds": f"{seconds:.9f}", "nominal_seconds": f"{nominal_seconds:.9f}",
                         "work": method_work, "nominal_work": nominal_work, "job_key": self.job_key, "log": self.log})
        return rows


def write_rows(rows: list[dict], path: str | Path) -> Path:
    """Append rows to timing.csv, writing the header when the file is new."""
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in COLUMNS})
    return path


def ratios(rows: list[dict]) -> dict:
    """What the reducer computes from the post-warm-up rows: mean wall and work ratios."""
    measured = [r for r in rows if int(r["warmup"]) == 0]
    if len(measured) < 3:
        raise ValueError("at least three measured repeats are required")
    wall = [float(r["seconds"]) / float(r["nominal_seconds"]) for r in measured]
    work = [float(r["work"]) / float(r["nominal_work"]) for r in measured]
    return {"wall_ratio": sum(wall) / len(wall), "work_ratio": sum(work) / len(work), "observations": len(measured)}
