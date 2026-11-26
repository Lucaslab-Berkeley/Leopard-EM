#!/bin/bash
# The same known-domain synthetic, searched with the EXPANDED ring: does the lattice
# competition's offset follow the template that found the sites?
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/jdickerson/miniconda3/envs/leopard-em/bin/python
TAG=bundle_domains_ringE
TABLE=results_bundle/output_correlation_table_$TAG.h5
STEM=6dpu_13pf_2rings_flatB
echo "[$(date +%T)] search (expanded ring)"
$PY run_scripts/run_match_curved.py --prefix bundle_ --tag domains_ringE \
    --results results_bundle --constraint configs/filament_constraint_bundle_domains.yaml \
    > logs/bundle_domainsE_search.log 2>&1
echo "[$(date +%T)] Stages 3-5"
$PY run_scripts/stage3_read_table.py --table $TABLE --stem $STEM \
    --constraint configs/filament_constraint_bundle_domains.h5 > logs/bundle_domainsE_stage3.log 2>&1
$PY run_scripts/stage4_sites.py --table $TABLE --stem $STEM > logs/bundle_domainsE_stage4.log 2>&1
$PY run_scripts/stage5_refine.py --table $TABLE --stem $STEM > logs/bundle_domainsE_stage5.log 2>&1
echo "[$(date +%T)] Stage 6 (compacted vs expanded)"
$PY run_scripts/compete_sites_curved.py --tag $TAG --results results_bundle \
    --only 13pf 13pf_exp > logs/bundle_domainsE_stage6.log 2>&1
echo "[$(date +%T)] EXPANDED RUN DONE"
