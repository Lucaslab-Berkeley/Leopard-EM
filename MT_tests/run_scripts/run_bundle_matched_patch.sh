#!/bin/bash
# The 823-matched known-domain synthetic, PATCH route: search, protofilament-resolved
# sites, and the in-register refine Stage 6b's patch readout uses.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/jdickerson/miniconda3/envs/leopard-em/bin/python
TAG=bundle_matched_patch
TABLE=results_bundle/output_correlation_table_$TAG.h5
PATCH=6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB
echo "[$(date +%T)] patch search"
$PY run_scripts/run_match_curved.py --prefix bundle_ --tag matched_patch \
    --results results_bundle --constraint configs/filament_constraint_bundle_domains_matched.yaml \
    > logs/bundle_matched_patch_search.log 2>&1
echo "[$(date +%T)] Stage 3 (patch)"
$PY run_scripts/stage3_read_table.py --table $TABLE --stem $PATCH \
    --constraint configs/filament_constraint_bundle_domains_matched.h5 \
    > logs/bundle_matched_patch_stage3.log 2>&1
echo "[$(date +%T)] Stage 4 (patch, protofilament-resolved)"
$PY run_scripts/stage4_sites.py --table $TABLE --stem $PATCH --resolve-protofilaments \
    > logs/bundle_matched_patch_stage4.log 2>&1
echo "[$(date +%T)] Stage 7 (in/out-register refine)"
$PY run_scripts/stage7_register.py --table $TABLE > logs/bundle_matched_patch_stage7.log 2>&1
echo "[$(date +%T)] MATCHED PATCH DONE"
