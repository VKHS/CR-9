#!/usr/bin/env python3
"""Check that docs/REVIEWER_FAQ.md names only files and make targets that exist.

    python -B scripts/check_reviewer_faq.py            # exit 1 on a dangling reference

Every question must carry a ``Check:`` line with a command; every backticked
repository path, every relative Markdown link and every ``make <target>`` is
resolved against the checkout. Standard library only.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAQ = ROOT / "docs" / "REVIEWER_FAQ.md"
MAKEFILE = ROOT / "Makefile"
EXTENSIONS = (".py", ".md", ".json", ".sh", ".txt", ".cff", ".lock", ".yml", ".def", ".example")
BARE_FILES = {"LICENSE", "NOTICE", "Dockerfile", "Makefile"}
EXAMPLE_RUN = "examples/runs/peptides_func"
QUESTION = re.compile(r"^\*\*([A-F]\d+)\. (.+)\*\*$")
TOKEN = re.compile(r"`([^`]+)`")
LINK = re.compile(r"\]\(([^)]+)\)")


def make_targets(text: str) -> set[str]:
    targets = set()
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z][A-Za-z0-9_ -]*):(?!=)", line)
        if match:
            targets.update(match.group(1).split())
    return targets


def path_like(token: str) -> bool:
    if token in BARE_FILES:
        return True
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", token) or token.startswith("/") or "<" in token:
        return False
    return "/" in token or token.endswith(EXTENSIONS)


def check(faq: Path = FAQ, makefile: Path = MAKEFILE, root: Path = ROOT) -> list[str]:
    text = faq.read_text(encoding="utf-8")
    targets = make_targets(makefile.read_text(encoding="utf-8"))
    problems: list[str] = []
    questions = 0
    current = None
    has_check = {}
    for line in text.splitlines():
        match = QUESTION.match(line)
        if match:
            current = match.group(1)
            questions += 1
            has_check[current] = False
            continue
        if current and line.startswith("Check:"):
            command = line[len("Check:"):].strip()
            if not command.startswith("`") or not command.endswith("`"):
                problems.append(f"{current}: the Check line must hold one backticked command")
            has_check[current] = True
    for question, ok in has_check.items():
        if not ok:
            problems.append(f"{question}: no Check line")
    if questions < 20:
        problems.append(f"only {questions} questions found; the FAQ is expected to hold at least 20")
    for token in TOKEN.findall(text):
        for word in re.findall(r"make ([A-Za-z0-9_-]+)", token):
            if word not in targets:
                problems.append(f"unknown make target in `{token}`: {word}")
        inner = token.split()[0] if token.split() else token
        # Run-relative artifacts (config.json, confirm/decision.json, ...) are checked against the sealed example run.
        if path_like(inner) and not (root / inner).exists() and not (root / EXAMPLE_RUN / inner).exists():
            problems.append(f"missing path `{inner}`")
    for link in LINK.findall(text):
        target = link.split("#")[0]
        if not target or target.startswith(("http://", "https://")):
            continue
        if not (faq.parent / target).exists():
            problems.append(f"dangling link ({link})")
    return problems


def main() -> int:
    problems = check()
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    text = FAQ.read_text(encoding="utf-8")
    print(f"{sum(1 for line in text.splitlines() if QUESTION.match(line))} reviewer questions; every referenced path, link and make target exists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
