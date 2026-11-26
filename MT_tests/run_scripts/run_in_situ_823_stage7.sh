#!/bin/bash
# Stage 7 for set1_823: patch membership, protofilament-resolved sites, paired register.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/jdickerson/miniconda3/envs/leopard-em/bin/python
TAG=in_situ_823_13pf_patch
RES=results_in_situ_823
TABLE=$RES/output_correlation_table_$TAG.h5
PATCH=6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB
SEARCH_LOG=logs/match_in_situ_823_13pf_patch.log

echo "[$(date +%T)] waiting for the patch search"
until grep -q "^EXIT" "$SEARCH_LOG"; do sleep 60; done
grep -q "^EXIT 0" "$SEARCH_LOG" || { echo "search FAILED"; tail -5 "$SEARCH_LOG"; exit 1; }
PEAKS=$($PY -c "import pandas as pd; print(len(pd.read_csv('$RES/results_$TAG.csv')))")
echo "[$(date +%T)] patch search done: $PEAKS peak(s) above the analytic threshold"
[ "$PEAKS" -gt 0 ] || { echo "GATE: zero peaks"; exit 2; }

echo "[$(date +%T)] Stage 3 (patch): membership from each detection's own axis point"
$PY run_scripts/stage3_read_table.py --table "$TABLE" --stem $PATCH \
    --constraint configs/filament_constraint_in_situ_823_refit.h5 \
    --micrograph results_in_situ/set1_823_Aug09_10.05.20_DW.mrc \
    > logs/in_situ_823_stage7_membership.log 2>&1
echo "[$(date +%T)] Stage 4 (patch): protofilament-resolved sites"
$PY run_scripts/stage4_sites.py --table "$TABLE" --stem $PATCH --resolve-protofilaments \
    > logs/in_situ_823_stage7_sites.log 2>&1
echo "[$(date +%T)] Stage 7: paired register"
$PY run_scripts/stage7_register.py --table "$TABLE" > logs/in_situ_823_stage7.log 2>&1
echo "[$(date +%T)] STAGE 7 DONE"
