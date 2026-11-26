#!/bin/bash
# Stages 3-6 for in situ micrograph set1_823, run as soon as the Stage 2 search lands.
#
# Stage 2's gate comes first (PIPELINE.md): if no peak clears the analytic threshold,
# stop -- nothing downstream means anything without detections. Each stage's output
# goes to logs/in_situ_823_stage<N>.log; the chain stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/jdickerson/miniconda3/envs/leopard-em/bin/python
TAG=in_situ_823_13pf
RES=results_in_situ_823
TABLE=$RES/output_correlation_table_$TAG.h5
SEARCH_LOG=logs/match_in_situ_823_13pf.log

echo "[$(date +%T)] waiting for the Stage 2 search"
until grep -q "^EXIT" "$SEARCH_LOG"; do sleep 60; done
if ! grep -q "^EXIT 0" "$SEARCH_LOG"; then
    echo "[$(date +%T)] search FAILED:"; tail -5 "$SEARCH_LOG"; exit 1
fi

# Gate: the peak CSV holds peaks above the constrained analytic threshold.
PEAKS=$($PY -c "import pandas as pd; print(len(pd.read_csv('$RES/results_$TAG.csv')))")
echo "[$(date +%T)] search done: $PEAKS peak(s) above the analytic threshold"
if [ "$PEAKS" -eq 0 ]; then
    echo "GATE: zero peaks -- stop and return to Stage 0 (PIPELINE.md Stage 2)"; exit 2
fi

echo "[$(date +%T)] Stage 3: membership, polarity, PF harmonic"
$PY run_scripts/stage3_read_table.py --table "$TABLE" \
    --constraint configs/filament_constraint_in_situ_823.h5 \
    --micrograph results_in_situ/set1_823_Aug09_10.05.20_DW.mrc \
    > logs/in_situ_823_stage3.log 2>&1

echo "[$(date +%T)] Stage 4: continuous sites and rise per filament"
$PY run_scripts/stage4_sites.py --table "$TABLE" > logs/in_situ_823_stage4.log 2>&1

echo "[$(date +%T)] Stage 5: refine the sites, re-measure"
$PY run_scripts/stage5_refine.py --table "$TABLE" > logs/in_situ_823_stage5.log 2>&1

echo "[$(date +%T)] Stage 6: rival rings at the refined sites"
$PY run_scripts/compete_sites_curved.py --tag "$TAG" --results "$RES" \
    > logs/in_situ_823_stage6.log 2>&1

echo "[$(date +%T)] STAGES 3-6 DONE"
