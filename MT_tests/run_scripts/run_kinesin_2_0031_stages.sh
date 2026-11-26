#!/bin/bash
# Stages 3-6b for kinesin_2_0031 (GMPCPP, 14-PF expanded reference), GPU 3 only.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/jdickerson/miniconda3/envs/leopard-em/bin/python
TAG=kinesin_2_0031_14pf
TABLE=results_kinesin_2_0031/output_correlation_table_$TAG.h5
RING=6dpu_14pf_2rings_flatB
step() { echo "[$(date +%T)] $1"; }
step "Stage 3: membership, polarity, PF harmonic"
$PY run_scripts/stage3_read_table.py --table $TABLE --stem $RING \
    --constraint configs/filament_constraint_kinesin_2_0031.h5 \
    --micrograph results_kinesin/20260916_lysates_2_0031_DW.mrc \
    > logs/kinesin_2_0031_stage3.log 2>&1 || { step "Stage 3 FAILED"; exit 1; }
step "Stage 4: continuous sites, rise"
$PY run_scripts/stage4_sites.py --table $TABLE --stem $RING \
    > logs/kinesin_2_0031_stage4.log 2>&1 || { step "Stage 4 FAILED"; exit 1; }
step "Stage 5: refine (GPU 3)"
$PY run_scripts/stage5_refine.py --table $TABLE --stem $RING --gpus 3 \
    > logs/kinesin_2_0031_stage5.log 2>&1 || { step "Stage 5 FAILED"; exit 1; }
step "Stage 6: PF number and lattice competition (GPU 3)"
$PY run_scripts/compete_sites_curved.py --tag $TAG --results results_kinesin_2_0031 \
    --rivals expanded14 --gpus 3 > logs/kinesin_2_0031_stage6.log 2>&1 || { step "Stage 6 FAILED"; exit 1; }
step "Stage 6b: lattice per segment, neutral pose (GPU 3)"
$PY run_scripts/stage6b_local_lattice.py --table $TABLE --rivals expanded14 --gpus 3 \
    > logs/kinesin_2_0031_stage6b.log 2>&1 || { step "Stage 6b FAILED"; exit 1; }
step "STAGES 3-6b DONE"
