#!/usr/bin/env bash
# Capture real Slurm accounting from Arrhenius. No emulation or synthesized job IDs.
# The export omits Account and NodeList: allocation identifiers and host names do not
# belong in a public deposit, and the reducers do not read them (JobName is kept for the job map).
# Usage: bash hpc/capture_arrhenius_accounting.sh "12345,12346" /absolute/evidence/path
set -Eeuo pipefail
if (( $# != 2 )); then echo 'Usage: script JOBID1,JOBID2 /new/output/dir' >&2; exit 2; fi
JOBS="$1"; DEST="$2"
[[ "${JOBS}" =~ ^[0-9_,]+$ ]] || { echo 'Job IDs must be comma-separated numeric Slurm IDs (arrays: 123_4)' >&2; exit 2; }
command -v sacct >/dev/null 2>&1 || { echo 'sacct is unavailable; run on an Arrhenius login node' >&2; exit 2; }
command -v scontrol >/dev/null 2>&1 || { echo 'scontrol unavailable' >&2; exit 2; }
[[ ! -e "${DEST}" ]] || { echo "Destination already exists: ${DEST}" >&2; exit 2; }
mkdir -p "$(dirname "${DEST}")"
TMP="$(mktemp -d "$(dirname "${DEST}")/.cm-arrhenius.XXXXXXXX")"
cleanup() { [[ -d "${TMP}" ]] && rm -rf "${TMP}"; }
trap cleanup EXIT
printf '%s\n' "${JOBS}" > "${TMP}/requested_job_ids.txt"
sacct --jobs "${JOBS}" --duplicates --array --parsable2 --format=Cluster,JobIDRaw,JobID,Submit,Start,End,State,ExitCode,ElapsedRaw,AllocTRES%1000,JobName%200,Partition%100,MaxRSS,MaxVMSize \
  > "${TMP}/sacct_original.psv"
if [[ $(wc -l < "${TMP}/sacct_original.psv") -lt 2 ]]; then
  echo 'No Slurm accounting rows available for the specified job IDs.' >&2; exit 3
fi
scontrol show config > "${TMP}/slurm_configuration.txt" 2>&1 || true
sinfo -h -o '%P|%a|%l|%D|%G' > "${TMP}/partition_snapshot.psv" 2>&1 || true
module list > "${TMP}/module_list.txt" 2>&1 || true
python3 - "${TMP}" <<'PY'
from pathlib import Path
import hashlib,json,sys
p=Path(sys.argv[1]);files=[]
for f in sorted(p.iterdir()):
 if f.is_file():files.append({'path':f.name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(p/'ACCOUNTING_MANIFEST.json').write_text(json.dumps({'schema':'cross-morse-actual-arrhenius-accounting-1',
 'origin':'sacct command executed on the host by the author; generated records are not simulated',
 'accounting_files':files,'identity_scope':'integrity relative to the captured bytes; no independent cluster attestation'},indent=2)+'\n')
PY
mv "${TMP}" "${DEST}"
trap - EXIT
printf 'Captured original sacct output: %s\n' "${DEST}"
