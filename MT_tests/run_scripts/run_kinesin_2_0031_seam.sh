#!/bin/bash
# Stage 7 for kinesin_2_0031 (GMPCPP): patch search, patch membership, protofilament-
# resolved sites, paired in/out-of-register test. GPU 3 only.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/jdickerson/miniconda3/envs/leopard-em/bin/python
TAG=kinesin_2_0031_14pf_patch
TABLE=results_kinesin_2_0031/output_correlation_table_$TAG.h5
PATCH=6dpu_14pf_patch5_2rings_flatB
step() { echo "[$(date +%T)] $1"; }
step "patch search (GPU 3, ~7 h)"
$PY -u run_scripts/run_match_curved.py --prefix kinesin_ --tag 2_0031_14pf_patch \
    --results results_kinesin_2_0031 --constraint configs/filament_constraint_kinesin_2_0031_refit.yaml \
    --gpus 3 > logs/match_kinesin_2_0031_14pf_patch.log 2>&1 || { step "patch search FAILED"; exit 1; }
PEAKS=$($PY -c "import pandas as pd; print(len(pd.read_csv('results_kinesin_2_0031/results_$TAG.csv')))")
step "patch search done: $PEAKS peaks"
[ "$PEAKS" -gt 0 ] || { step "GATE: zero patch peaks -- stopping"; exit 2; }
step "Stage 3 (patch membership)"
$PY run_scripts/stage3_read_table.py --table $TABLE --stem $PATCH \
    --constraint configs/filament_constraint_kinesin_2_0031_refit.h5 \
    --micrograph results_kinesin/20260916_lysates_2_0031_DW.mrc \
    > logs/kinesin_2_0031_patch_stage3.log 2>&1 || { step "Stage 3 FAILED"; exit 1; }
step "Stage 4 (protofilament-resolved sites)"
$PY run_scripts/stage4_sites.py --table $TABLE --stem $PATCH --resolve-protofilaments \
    > logs/kinesin_2_0031_patch_stage4.log 2>&1 || { step "Stage 4 FAILED"; exit 1; }
step "Stage 7 (paired register, GPU 3)"
$PY run_scripts/stage7_register.py --table $TABLE --stem $PATCH --gpus 3 \
    > logs/kinesin_2_0031_stage7.log 2>&1 || { step "Stage 7 FAILED"; exit 1; }
step "SEAM CHAIN DONE"
