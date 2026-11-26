#!/bin/bash
# Coarse auto-trace of every 16Sep2026 kinesin sum (GPU 3), overlays to curate from.
# 1_0040 skipped: its CTF fit failed (232 A defocus), so the trace's low-pass would be wrong.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/jdickerson/miniconda3/envs/leopard-em/bin/python
OUT=results_autotrace/kinesin_all
mkdir -p $OUT
step() { echo "[$(date +%T)] $1"; }
for mic in results_kinesin/*_DW.mrc; do
    stem=$(basename $mic _DW.mrc); short=${stem#20260916_lysates_}
    [ "$short" = "1_0040" ] && { step "$short skipped (failed CTF)"; continue; }
    [ -f $OUT/kinesin_${short}_overlay.png ] && { step "$short exists"; continue; }
    $PY ../programs/constrained_search/auto_trace_filaments.py --micrograph $mic \
        --template maps/6dpu_14pf_2rings_flatB_1.13_bscale0.5.mrc \
        --optics results_ctf_kinesin/${stem}_DW_optics.yaml \
        --outdir $OUT --prefix kinesin_$short --gpu 3 > $OUT/kinesin_${short}.log 2>&1 \
        && step "$short: $(grep -o '[0-9]* after merging' $OUT/kinesin_${short}.log | head -1) tracks" \
        || step "$short FAILED"
done
step "AUTOTRACE ALL DONE"
