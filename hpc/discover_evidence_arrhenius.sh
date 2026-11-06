#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=cm_discover_evidence
# Inspect one explicit export root with the existing complete-run discovery tool.
# No training, repair, registration or network operation is performed.
#
# Create CM_EVIDENCE_DIR/logs with umask 077 before submitting. Keep the working
# directory and both Slurm logs outside the checkout and the source export tree:
#   sbatch --account="$CM_EVIDENCE_ACCOUNT" --partition="$CM_EVIDENCE_PARTITION" \
#     --chdir="$CM_EVIDENCE_DIR" \
#     --output="$CM_EVIDENCE_DIR/logs/discovery-%j.out" \
#     --error="$CM_EVIDENCE_DIR/logs/discovery-%j.err" --export=ALL \
#     "$CM_SOURCE_ROOT/hpc/discover_evidence_arrhenius.sh"
# Select a site allocation/partition compatible with CM_PYTHON or CM_VENV_DIR.
# This job requests CPU/storage work; it does not predict a completion time.
#
# Required: CM_SOURCE_ROOT, CM_ENV_FILE (trusted Bash profile), CM_EVIDENCE_ROOT,
# CM_EVIDENCE_DIR. Optional: CM_EXPECT_SOURCE_SHA256, CM_DISCOVERY_MAX_DEPTH (6),
# CM_DISCOVERY_MAX_DIRECTORIES (10000). The profile supplies the interpreter.
# Files: discovery-<job>.json/.md and plan-<job>.json, never overwritten.
# Exit 0: complete scan, at least one unique candidate, no ambiguous dataset.
# Exit 1: empty, ambiguous or incomplete scan; diagnostic reports are retained.
# Exit 2: invalid setup/inputs/output paths. Other process failures retain their
# original status. None of these statuses is a scientific-readiness verdict.
set -Eeuo pipefail
umask 077

for name in CM_SOURCE_ROOT CM_ENV_FILE CM_EVIDENCE_ROOT CM_EVIDENCE_DIR; do
  if [[ -z "${!name:-}" ]]; then
    printf 'Set %s before running discovery.\n' "$name" >&2
    exit 2
  fi
done
if [[ "$CM_ENV_FILE" != /* || ! -f "$CM_ENV_FILE" ]]; then
  echo 'CM_ENV_FILE must name an existing trusted profile by absolute path.' >&2
  exit 2
fi
# Profiles are trusted code, as in the existing launchers; never source an
# unreviewed file or put credentials in the public repository.
source "$CM_ENV_FILE"
if [[ -n "${CM_BUILD_MODULE:-}${CM_PYTHON_MODULE:-}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    echo 'The profile requests modules, but the module command is unavailable.' >&2
    exit 2
  fi
  # shellcheck disable=SC2086
  if ! module purge || ! module load ${CM_BUILD_MODULE:-} ${CM_PYTHON_MODULE:-}; then
    echo 'Could not load the configured modules.' >&2
    exit 2
  fi
fi
if [[ -n "${CM_PYTHON:-}" ]]; then
  PYTHON="$CM_PYTHON"
elif [[ -n "${CM_VENV_DIR:-}" ]]; then
  PYTHON="$CM_VENV_DIR/bin/python"
else
  echo 'Set CM_PYTHON or CM_VENV_DIR in the trusted profile.' >&2
  exit 2
fi
if [[ "$PYTHON" != /* || ! -x "$PYTHON" ]]; then
  echo 'The configured interpreter must be an executable absolute path.' >&2
  exit 2
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  if [[ ! "$SLURM_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo 'SLURM_JOB_ID must be numeric.' >&2
    exit 2
  fi
  stamp="$SLURM_JOB_ID"
else
  stamp="local-$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi

# Check every destination before mkdir. The discovery CLI repeats the source
# and output checks when it writes its private, exclusive files.
if ! "$PYTHON" - "$CM_SOURCE_ROOT" "$CM_EVIDENCE_ROOT" "$CM_EVIDENCE_DIR" "$stamp" \
    "${CM_DISCOVERY_MAX_DEPTH:-6}" "${CM_DISCOVERY_MAX_DIRECTORIES:-10000}" <<'PY'
from pathlib import Path
import stat
import sys

try:
    paths = []
    for label, value in zip(("CM_SOURCE_ROOT", "CM_EVIDENCE_ROOT", "CM_EVIDENCE_DIR"), sys.argv[1:4]):
        path = Path(value)
        if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError(f"{label} must be an absolute path without symlinks")
        paths.append(path.resolve())
    source, exports, evidence = paths
    if not source.is_dir() or not exports.is_dir():
        raise ValueError("the source checkout and export root must already be directories")
    if not (source / "scripts/evidence_pipeline.py").is_file() or not (source / "scripts/source_fingerprint.py").is_file():
        raise ValueError("CM_SOURCE_ROOT lacks the evidence pipeline or source fingerprint command")
    if evidence.is_relative_to(source) or evidence.is_relative_to(exports):
        raise ValueError("CM_EVIDENCE_DIR must be outside both the checkout and the export root")
    if evidence.exists() and (not evidence.is_dir() or stat.S_IMODE(evidence.stat().st_mode) & 0o077):
        raise ValueError("use an owner-private evidence directory (mode 0700)")
    for filename in (f"discovery-{sys.argv[4]}.json", f"discovery-{sys.argv[4]}.md", f"plan-{sys.argv[4]}.json"):
        path = evidence / filename
        if path.exists() or path.is_symlink():
            raise ValueError("a report or plan for this job already exists; use a new job, never overwrite it")
    depth, directories = int(sys.argv[5]), int(sys.argv[6])
    if depth < 0 or directories < 1:
        raise ValueError("discovery limits require depth >= 0 and directories >= 1")
except (OSError, ValueError) as exc:
    print(f"Invalid discovery setup: {exc}", file=sys.stderr)
    raise SystemExit(2)
PY
then
  exit 2
fi

cd "$CM_SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
fingerprint_args=(-B scripts/source_fingerprint.py)
if [[ -n "${CM_EXPECT_SOURCE_SHA256:-}" ]]; then
  fingerprint_args+=(--expect "$CM_EXPECT_SOURCE_SHA256")
fi
if ! source_sha=$("$PYTHON" "${fingerprint_args[@]}"); then
  echo 'Source identity could not be verified; discovery was not started.' >&2
  exit 2
fi
if [[ ! "$source_sha" =~ ^[[:xdigit:]]{64}$ ]]; then
  echo 'The source fingerprint command did not return one SHA-256 identity.' >&2
  exit 2
fi
mkdir -p -- "$CM_EVIDENCE_DIR"
report="$CM_EVIDENCE_DIR/discovery-$stamp.json"
plan="$CM_EVIDENCE_DIR/plan-$stamp.json"
printf 'Discovery job %s; runtime source identity %s.\n' "$stamp" "$source_sha"

if "$PYTHON" -B scripts/evidence_pipeline.py discover \
    --root "$CM_EVIDENCE_ROOT" --output "$report" \
    --plan-output "$plan" --deposits-dir "$CM_EVIDENCE_DIR/deposits" \
    --max-depth "${CM_DISCOVERY_MAX_DEPTH:-6}" \
    --max-directories "${CM_DISCOVERY_MAX_DIRECTORIES:-10000}"; then
  status=0
else
  status=$?
fi
case "$status" in
  0) echo 'Discovery found unambiguous candidates. Review every dataset and run full preflight; missing datasets may remain.' ;;
  1) echo 'Discovery requires review: the scan was empty, ambiguous or incomplete. Read the retained JSON/Markdown diagnostics.' ;;
  2) echo 'Discovery rejected its inputs or output paths. No readiness claim is established.' >&2 ;;
  *) printf 'Discovery did not complete (process exit %s). Inspect the private job log.\n' "$status" >&2 ;;
esac
exit "$status"
