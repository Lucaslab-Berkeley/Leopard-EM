#!/bin/bash
# 16Sep2026 kinesin + GMPCPP: the remaining 57 movies (batch 1 = every 5th, done), on GPU 3,
# with batch 1's exact settings -- RIPPLE (skips movies already summed), PICASSO on all 72
# (batch 1's CTF results kept first in results_ctf_kinesin_batch1: the 2_0031 and 1_0011
# searches used those values), then previews and a contact sheet for drawing.
set -uo pipefail
cd "$(dirname "$0")/.."
RIPPLE=/home/jdickerson/git/LucasLab/RIPPLE/.venv/bin/python
PICASSO=/home/jdickerson/git/LucasLab/PICASSO/.venv/bin/python
FRAMES=Frames/kinesin_GMPCPP
step() { echo "[$(date +%T)] $1"; }

for d in 2_lysates_1 3_lysates_2; do
    step "RIPPLE $d (all movies; existing sums skipped)"
    $RIPPLE run_scripts/run_ripple_in_situ.py --config configs/align_kinesin.yaml \
        --frames $FRAMES/$d --gain $(ls $FRAMES/$d/*.dm4) --outdir results_kinesin \
        --gpu 3 --shard 0/1 --fluence-per-frame 1.0 --time-points 10 --no-xc-prepass \
        > logs/ripple_kinesin_all_$d.log 2>&1 || step "RIPPLE $d FAILED (continuing)"
    step "RIPPLE $d: $(ls results_kinesin/*_DW.mrc | wc -l) sums in total"
done
[ -d results_ctf_kinesin_batch1 ] || cp -r results_ctf_kinesin results_ctf_kinesin_batch1
step "PICASSO CTF, all sums"
$PICASSO run_scripts/run_picasso_in_situ.py --config configs/ctf_kinesin.yaml \
    --frames $FRAMES/2_lysates_1 $FRAMES/3_lysates_2 > logs/picasso_kinesin_all.log 2>&1 \
    || step "PICASSO FAILED (previews still rendered)"
step "previews"
$RIPPLE run_scripts/render_dw_previews.py --indir results_kinesin \
    --outdir results_kinesin/previews --contact-sheet > logs/previews_kinesin_all.log 2>&1
step "KINESIN PREPROCESS (ALL) DONE"
