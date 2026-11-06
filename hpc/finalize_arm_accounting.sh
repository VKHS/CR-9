#!/usr/bin/env bash
# Export final accounting from the real scheduler after the allocation is terminal.
set -Eeuo pipefail
[[ "$#" -eq 2 ]] || { echo 'Usage: finalize_arm_accounting.sh JOB_ID JOB_EVIDENCE_DIRECTORY' >&2; exit 2; }
JOB_ID="$1"; TARGET="$2"
[[ "$JOB_ID" =~ ^[0-9]+(_[0-9]+)?$ ]] || { echo 'Invalid allocation/array task identifier' >&2; exit 2; }
[[ -d "$TARGET" && ! -L "$TARGET" && ! -e "$TARGET/sacct.tsv" ]] || { echo 'Need an existing evidence directory without sacct.tsv' >&2; exit 2; }
[[ "$(cat "$TARGET/job_id.txt")" == "$JOB_ID" ]] || { echo 'Job identity differs from original capture' >&2; exit 2; }
tmp="$(mktemp "$TARGET/.sacct.XXXXXX")"
trap 'rm -f "$tmp"' EXIT
# Header retained; .tsv uses pipe separators as declared by --parsable2.
sacct --allocations --parsable2 -j "$JOB_ID" \
  --format=Cluster,JobIDRaw,JobID,Submit,Start,End,State,ExitCode,ElapsedRaw,AllocTRES%1000,JobName%200 > "$tmp"
[[ -s "$tmp" ]] || { echo 'No scheduler record returned' >&2; exit 2; }
awk -F '|' -v job="$JOB_ID" '
BEGIN{ok=1;n=0}
NR==1{if($1!="Cluster" || $2!="JobIDRaw" || $3!="JobID" || $7!="State")ok=0;next}
{n++;s=$7;sub(/ .*/,"",s);if($2!=job && $3!=job)ok=0;
 if(s !~ /^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)$/)ok=0}
END{exit !(ok && n>0)}' "$tmp" || { echo 'No matching terminal allocation record; export after completion' >&2; exit 2; }
# No-clobber promotion preserves the exact scheduler-returned bytes.
ln "$tmp" "$TARGET/sacct.tsv"
(cd "$TARGET" && sha256sum sacct.tsv > sacct.sha256)
printf '%s\n' 'Source: actual sacct --allocations --parsable2 export, with original header. Failed attempts retained; steps not double-counted. Separator: pipe.' > "$TARGET/accounting_scope.txt"
