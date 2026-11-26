#!/bin/bash
# 16Sep2026 kinesin + GMPCPP: first batch (every 5th movie per directory) on GPU 3 --
# RIPPLE motion correction, PICASSO CTF, then binned PNG previews for drawing.
set -uo pipefail
cd "$(dirname "$0")/.."
RIPPLE=/home/jdickerson/git/LucasLab/RIPPLE/.venv/bin/python
PICASSO=/home/jdickerson/git/LucasLab/PICASSO/.venv/bin/python
PY=$RIPPLE   # has scikit-image, which the leopard-em env lacks
FRAMES=Frames/kinesin_GMPCPP
step() { echo "[$(date +%T)] $1"; }

for d in 2_lysates_1 3_lysates_2; do
    step "RIPPLE $d (every 5th movie)"
    $RIPPLE run_scripts/run_ripple_in_situ.py --config configs/align_kinesin.yaml \
        --frames $FRAMES/$d --gain $(ls $FRAMES/$d/*.dm4) --outdir results_kinesin \
        --gpu 3 --shard 0/5 --fluence-per-frame 1.0 --time-points 10 --no-xc-prepass \
        --redo > logs/ripple_kinesin_$d.log 2>&1 \
        || { step "RIPPLE $d FAILED"; exit 1; }
done
step "PICASSO CTF"
$PICASSO run_scripts/run_picasso_in_situ.py --config configs/ctf_kinesin.yaml \
    --frames $FRAMES/2_lysates_1 $FRAMES/3_lysates_2 > logs/picasso_kinesin.log 2>&1 \
    || step "PICASSO FAILED (previews still rendered)"
step "previews"
$PY run_scripts/render_dw_previews.py --indir results_kinesin \
    --outdir results_kinesin/previews --contact-sheet > logs/previews_kinesin.log 2>&1
step "KINESIN PREPROCESS DONE"
