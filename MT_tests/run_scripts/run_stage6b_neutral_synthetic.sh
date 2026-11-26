#!/bin/bash
# Stage 6b neutral-start checks on both known-domain synthetic site sets:
# found sites at a neutral pose, and every lattice index at a neutral pose.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/jdickerson/miniconda3/envs/leopard-em/bin/python
TRUTH=Frames/synthetic_bundle_lattice_domains.truth.json
for START in neutral lattice; do
    for TAG in bundle_domains_ring bundle_domains_ringE; do
        echo "[$(date +%T)] $TAG $START"
        $PY run_scripts/stage6b_local_lattice.py \
            --table results_bundle/output_correlation_table_$TAG.h5 --start $START \
            --truth $TRUTH > logs/stage6b_${START}_$TAG.log 2>&1 || echo "  $TAG $START failed"
    done
done
echo "[$(date +%T)] NEUTRAL SYNTHETIC DONE"
