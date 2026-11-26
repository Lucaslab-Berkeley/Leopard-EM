#!/bin/bash
# Stage 6b calibration on the 823-matched known-domain synthetic, end to end.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/jdickerson/miniconda3/envs/leopard-em/bin/python
TAG=bundle_matched_ring
TABLE=results_bundle/output_correlation_table_$TAG.h5
until grep -q "EXPANDED RUN DONE" logs/bundle_domainsE_chain.log; do sleep 30; done
echo "[$(date +%T)] search"
$PY run_scripts/run_match_curved.py --prefix bundle_ --tag matched_ring \
    --results results_bundle --constraint configs/filament_constraint_bundle_domains_matched.yaml \
    > logs/bundle_matched_search.log 2>&1
echo "[$(date +%T)] Stages 3-5"
$PY run_scripts/stage3_read_table.py --table $TABLE \
    --constraint configs/filament_constraint_bundle_domains_matched.h5 > logs/bundle_matched_stage3.log 2>&1
$PY run_scripts/stage4_sites.py --table $TABLE > logs/bundle_matched_stage4.log 2>&1
$PY run_scripts/stage5_refine.py --table $TABLE > logs/bundle_matched_stage5.log 2>&1
echo "[$(date +%T)] Stage 6"
$PY run_scripts/compete_sites_curved.py --tag $TAG --results results_bundle \
    --only 13pf 13pf_exp > logs/bundle_matched_stage6.log 2>&1
echo "[$(date +%T)] MATCHED RUN DONE"
