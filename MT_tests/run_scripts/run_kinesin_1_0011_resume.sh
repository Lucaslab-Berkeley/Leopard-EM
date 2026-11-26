#!/bin/bash
# kinesin_1_0011 (lysate 1, 3.86 um underfocus), five curated tubes: the whole pipeline on
# GPU 3 -- ring search, Stages 3-6b (neutral poses, then the smooth roll re-fit), patch
# search on the ring-refitted constraint, Stage 7, then kinesin: a synthetic calibration
# at this frame's optics and the kinesin templates scored at the neutral poses.
set -uo pipefail
# RESUME after Stage 6b (neutral), which exits non-zero when no tube can be segmented.
cd "$(dirname "$0")/.."
PY=/home/jdickerson/miniconda3/envs/leopard-em/bin/python
MIC=results_kinesin/20260916_lysates_1_0011_DW.mrc
RES=results_kinesin_1_0011
TAG=kinesin_1_0011_14pf
TABLE=$RES/output_correlation_table_$TAG.h5
RING=6dpu_14pf_2rings_flatB
PATCH=6dpu_14pf_patch5_2rings_flatB
step() { echo "[$(date +%T)] $1"; }

step "Stage 6b with the smooth roll re-fit (GPU 3)"
$PY run_scripts/stage6b_local_lattice.py --table $TABLE --rivals expanded14 --gpus 3 \
    --refit-roll > logs/kinesin_1_0011_stage6b_rollfit.log 2>&1 || step "Stage 6b roll re-fit exited non-zero (no segmentable tube) -- stack written, continuing"

step "constraint re-fitted from the ring, for the patch"
$PY run_scripts/refit_constraint_from_ring.py --tag $TAG --results $RES \
    --source configs/filament_constraint_kinesin_1_0011.h5 \
    --paths configs/filament_constraint_kinesin_1_0011.paths.json \
    --out filament_constraint_kinesin_1_0011_refit --width-px 460 --micrograph $MIC \
    --psi-step 2.5 --theta-step 3.5 --restrict-polarity \
    > logs/kinesin_1_0011_refit.log 2>&1 || { step "refit FAILED"; exit 1; }

step "patch search (GPU 3, ~7 h)"
$PY -u run_scripts/run_match_curved.py --prefix kinesin_ --tag 1_0011_14pf_patch --results $RES \
    --constraint configs/filament_constraint_kinesin_1_0011_refit.yaml --gpus 3 \
    > logs/match_kinesin_1_0011_14pf_patch.log 2>&1 || { step "patch search FAILED"; exit 1; }
PEAKS=$($PY -c "import pandas as pd; print(len(pd.read_csv('$RES/results_${TAG}_patch.csv')))")
step "patch search done: $PEAKS peaks"
if [ "$PEAKS" -gt 0 ]; then
    step "Stage 3 (patch membership)"
    $PY run_scripts/stage3_read_table.py --table $RES/output_correlation_table_${TAG}_patch.h5 \
        --stem $PATCH --constraint configs/filament_constraint_kinesin_1_0011_refit.h5 \
        --micrograph $MIC > logs/kinesin_1_0011_patch_stage3.log 2>&1 || step "patch Stage 3 FAILED"
    step "Stage 4 (protofilament-resolved sites)"
    $PY run_scripts/stage4_sites.py --table $RES/output_correlation_table_${TAG}_patch.h5 \
        --stem $PATCH --resolve-protofilaments \
        > logs/kinesin_1_0011_patch_stage4.log 2>&1 || step "patch Stage 4 FAILED"
    step "Stage 7 (paired register, GPU 3)"
    $PY run_scripts/stage7_register.py --table $RES/output_correlation_table_${TAG}_patch.h5 \
        --stem $PATCH --gpus 3 > logs/kinesin_1_0011_stage7.log 2>&1 || step "Stage 7 FAILED"
else
    step "GATE: zero patch peaks -- seam skipped, kinesin still scored"
fi

step "kinesin: synthetic calibration at this frame's optics (GPU 3)"
$PY run_scripts/simulate_kinesin_occupancy.py --gpu 3 --no-mrc \
    --optics results_ctf_kinesin/20260916_lysates_1_0011_DW_optics.yaml \
    --noise-source $MIC --outdir results_kinesin_synthetic_1_0011 \
    > logs/kinesin_1_0011_synthetic.log 2>&1 || { step "synthetic calibration FAILED"; exit 1; }
for SFX in neutral neutral_rollfit; do
    step "kinesin: templates at the $SFX poses (GPU 3)"
    $PY run_scripts/score_kinesin_sites.py --tag $TAG --results $RES --stack-suffix $SFX \
        --paths configs/filament_constraint_kinesin_1_0011.paths.json \
        --background-paths results_autotrace/kinesin_1_0011_t2p5.paths.json \
        --background-min-row 1100 --clear-nm 40 --max-x-fraction 1.0 \
        --calibration results_kinesin_synthetic_1_0011/occupancy_summary.csv \
        > logs/kinesin_1_0011_kinesin_$SFX.log 2>&1 || step "kinesin $SFX FAILED"
done
step "1_0011 CHAIN DONE"
