#!/bin/bash
# Calibration run for per-segment lattice state: the known-domain synthetic, end to end.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/jdickerson/miniconda3/envs/leopard-em/bin/python
TAG=bundle_domains_ring
TABLE=results_bundle/output_correlation_table_$TAG.h5
echo "[$(date +%T)] Stage 2 search"
$PY run_scripts/run_match_curved.py --prefix bundle_ --tag domains_ring \
    --results results_bundle --constraint configs/filament_constraint_bundle_domains.yaml \
    > logs/bundle_domains_search.log 2>&1
echo "[$(date +%T)] Stage 3"
$PY run_scripts/stage3_read_table.py --table $TABLE \
    --constraint configs/filament_constraint_bundle_domains.h5 \
    --micrograph Frames/synthetic_bundle_lattice_domains.mrc > logs/bundle_domains_stage3.log 2>&1
echo "[$(date +%T)] Stage 4"
$PY run_scripts/stage4_sites.py --table $TABLE > logs/bundle_domains_stage4.log 2>&1
echo "[$(date +%T)] Stage 5"
$PY run_scripts/stage5_refine.py --table $TABLE > logs/bundle_domains_stage5.log 2>&1
echo "[$(date +%T)] Stage 6 (compacted vs expanded only)"
$PY run_scripts/compete_sites_curved.py --tag $TAG --results results_bundle \
    --only 13pf 13pf_exp > logs/bundle_domains_stage6.log 2>&1
echo "[$(date +%T)] DOMAINS PIPELINE DONE"
