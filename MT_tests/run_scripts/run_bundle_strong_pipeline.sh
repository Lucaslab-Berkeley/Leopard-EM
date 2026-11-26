#!/bin/bash
# Known-domain synthetic at 823's REAL strength. The "matched" one (target z 12.5)
# undershot 823 by ~1 z on both templates (refined ring 8.10 vs 9.03, patch 7.07 vs
# 7.99), which for the patch crosses the detection threshold (occupancy 11-62% vs
# ~75%). Same seed, layout, rolls and domains; only the signal is scaled.
# Patch route first (Stage 6b --patch calibration), then the ring route.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/jdickerson/miniconda3/envs/leopard-em/bin/python
TRUTH=Frames/synthetic_bundle_lattice_domains_strong.truth.json
PATCH=6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB
step() { echo "[$(date +%T)] $1"; }
step "simulate"
$PY run_scripts/simulate_microtubule_bundle.py --domains lattice --rolls 11 47 113 200 \
    --size 4000 --target-z 14.5 --tag lattice_domains_strong > logs/bundle_strong_simulate.log 2>&1 || exit 1
$PY ../programs/constrained_search/make_filament_constraint.py \
    Frames/synthetic_bundle_lattice_domains_strong.paths.json \
    -o configs/filament_constraint_bundle_domains_strong.yaml --cone-half-angle-deg 20 \
    --psi-step 2.5 --theta-step 3.5 --pixel-size-angstrom 0.9194 \
    > logs/bundle_strong_constraint.log 2>&1 || exit 1

TAG=bundle_strong_patch
TABLE=results_bundle/output_correlation_table_$TAG.h5
step "patch search"
$PY run_scripts/run_match_curved.py --prefix bundle_ --tag strong_patch --results results_bundle \
    --constraint configs/filament_constraint_bundle_domains_strong.yaml \
    > logs/${TAG}_search.log 2>&1 || exit 1
step "patch Stages 3, 4, 7"
$PY run_scripts/stage3_read_table.py --table $TABLE --stem $PATCH \
    --constraint configs/filament_constraint_bundle_domains_strong.h5 > logs/${TAG}_stage3.log 2>&1
$PY run_scripts/stage4_sites.py --table $TABLE --stem $PATCH --resolve-protofilaments \
    > logs/${TAG}_stage4.log 2>&1
$PY run_scripts/stage7_register.py --table $TABLE > logs/${TAG}_stage7.log 2>&1
step "patch Stage 6b"
$PY run_scripts/stage6b_local_lattice.py --table $TABLE --patch --truth $TRUTH \
    > logs/stage6b_${TAG}.log 2>&1
step "PATCH ROUTE DONE"

TAG=bundle_strong_ring
TABLE=results_bundle/output_correlation_table_$TAG.h5
step "ring search"
$PY run_scripts/run_match_curved.py --prefix bundle_ --tag strong_ring --results results_bundle \
    --constraint configs/filament_constraint_bundle_domains_strong.yaml \
    > logs/${TAG}_search.log 2>&1 || exit 1
step "ring Stages 3-5"
$PY run_scripts/stage3_read_table.py --table $TABLE \
    --constraint configs/filament_constraint_bundle_domains_strong.h5 > logs/${TAG}_stage3.log 2>&1
$PY run_scripts/stage4_sites.py --table $TABLE > logs/${TAG}_stage4.log 2>&1
$PY run_scripts/stage5_refine.py --table $TABLE > logs/${TAG}_stage5.log 2>&1
step "ring Stage 6b"
$PY run_scripts/stage6b_local_lattice.py --table $TABLE --truth $TRUTH > logs/stage6b_${TAG}.log 2>&1
step "RING ROUTE DONE"
