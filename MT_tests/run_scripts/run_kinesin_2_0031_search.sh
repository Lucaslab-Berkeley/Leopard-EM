#!/bin/bash
# Stage 2 for kinesin_2_0031: the constrained ring search on GPU 3 only.
cd "$(dirname "$0")/.."
echo "[$(date +%T)] search start"
/home/jdickerson/miniconda3/envs/leopard-em/bin/python -u run_scripts/run_match_curved.py \
    --prefix kinesin_ --tag 2_0031_14pf --results results_kinesin_2_0031 \
    --constraint configs/filament_constraint_kinesin_2_0031.yaml --gpus 3 \
    > logs/match_kinesin_2_0031_14pf.log 2>&1
echo "[$(date +%T)] SEARCH DONE (exit $?)"
