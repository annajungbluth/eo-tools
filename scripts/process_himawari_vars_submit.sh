#!/bin/bash
set -euo pipefail

# Submit process_himawari_vars.batch for multiple OFFSET values.
# Usage:
#   ./submit_process_himawari_vars_offsets.sh 0 10000 20000
# Or edit the default OFFSETS list below and run with no args.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_SCRIPT="${SCRIPT_DIR}/process_himawari_vars.batch"

if [[ ! -f "${BATCH_SCRIPT}" ]]; then
    echo "Batch script not found: ${BATCH_SCRIPT}" >&2
    exit 1
fi

if [[ $# -gt 0 ]]; then
    OFFSETS=("$@")
else
    OFFSETS=(0 10000 20000 30000 40000 50000 60000 70000 80000 90000) # HIMAWARI
fi

echo "Submitting ${#OFFSETS[@]} batch job(s) using ${BATCH_SCRIPT}"
for offset in "${OFFSETS[@]}"; do
    job_name="himawari_o${offset}"
    echo "Submitting OFFSET=${offset} as ${job_name}"
    sbatch --job-name="${job_name}" --export=ALL,OFFSET="${offset}" "${BATCH_SCRIPT}"
done

echo "All submissions sent."
