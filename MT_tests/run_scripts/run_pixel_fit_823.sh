#!/bin/bash
# Pixel-size fit: the synthetic check (known 0.9194) first, then set1_823.
cd "$(dirname "$0")/.."
while kill -0 2612284 2>/dev/null; do sleep 20; done
echo "[$(date +%T)] synthetic check done; 823 fit"
/home/jdickerson/miniconda3/envs/leopard-em/bin/python -u run_scripts/fit_pixel_size.py \
    --table results_in_situ_823/output_correlation_table_in_situ_823_13pf.h5 \
    --steps -3 -2 -1 0 1 2 3 4 5 > logs/pixel_fit_in_situ_823.log 2>&1
echo "[$(date +%T)] 823 PIXEL FIT DONE (exit $?)"
