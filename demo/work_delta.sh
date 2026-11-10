#!/usr/bin/env bash
# Carry a production work directory between workflow jobs at the same absolute path.
#   bash demo/work_delta.sh mark MARKFILE                  before a job writes
#   bash demo/work_delta.sh save WORK MARKFILE OUT.tar.gz   everything written under WORK since the mark
# "Written" is the inode change time: a copy that keeps its source's modification time (shutil.copy2, as the
# E3-E6 runner's code snapshot) is still a file this job wrote.
#   bash demo/work_delta.sh restore A.tar.gz [B.tar.gz ...] unpack in order (later files win)
set -euo pipefail
case "${1:?mark|save|restore}" in
  mark)
    touch "${2:?mark file}"; sleep 1 ;;
  save)
    work="${2:?work directory}"; mark="${3:?mark file}"; out="${4:?output archive}"
    mkdir -p "$(dirname "${out}")"
    ( cd / && find "${work#/}" -cnewer "${mark}" \( -type f -o -type l \) -print0 | tar --null -czf "${out}" -T - )
    echo "saved $(tar -tzf "${out}" | wc -l) file(s), $(du -h "${out}" | cut -f1): ${out}" ;;
  restore)
    shift
    for archive in "$@"; do tar -xzf "${archive}" -C /; echo "restored ${archive}"; done ;;
  *)
    echo "usage: $0 mark|save|restore ..." >&2; exit 2 ;;
esac
