#!/bin/bash
# Patch pixel-size fit on the known-domain synthetic (truth 0.9194), after the 823 one.
cd "$(dirname "$0")/.."
while kill -0 2665813 2>/dev/null; do sleep 20; done
echo "[$(date +%T)] synthetic patch fit"
/home/jdickerson/miniconda3/envs/leopard-em/bin/python -u run_scripts/fit_pixel_size.py \
    --table results_bundle/output_correlation_table_bundle_strong_patch.h5 --patch \
    --steps -3 -2 -1 0 1 2 3 --truth-pixel-size 0.9194 > logs/pixel_fit_bundle_strong_patch.log 2>&1
echo "[$(date +%T)] SYNTHETIC PATCH FIT DONE (exit $?)"
