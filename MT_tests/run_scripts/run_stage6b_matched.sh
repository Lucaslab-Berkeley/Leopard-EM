#!/bin/bash
# Stage 6b calibration on the 823-matched known-domain synthetic, all three starts.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/jdickerson/miniconda3/envs/leopard-em/bin/python
TRUTH=Frames/synthetic_bundle_lattice_domains_matched.truth.json
for START in neutral found lattice; do
    echo "[$(date +%T)] $START"
    $PY run_scripts/stage6b_local_lattice.py \
        --table results_bundle/output_correlation_table_bundle_matched_ring.h5 \
        --start $START --truth $TRUTH --sizes 4 6 8 12 16 24 32 \
        > logs/stage6b_${START}_bundle_matched_ring.log 2>&1 || echo "  $START failed"
done
echo "[$(date +%T)] MATCHED 6B DONE"
