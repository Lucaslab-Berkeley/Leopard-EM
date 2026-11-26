#!/usr/bin/env bash
# SVD autotrace of one micrograph: Panther-EM ring maps -> ridge tracer -> paths.json.
#
# usage: run_scripts/run_svd_autotrace.sh <prefix> <micrograph> <ring template> <optics yaml> [gpu] [extra tracer args...]
#   e.g. run_scripts/run_svd_autotrace.sh gdp_curved Frames/GDP_curved_mgraph.mrc \
#            maps/6dpu_atoms_6dpv_lattice_13pf_2rings_flatB_0.9194_bscale0.5.mrc \
#            results_ctf/GDP_curved_mgraph_optics.yaml 1 --reference configs/gdp_curved.paths.json
# Run from MT_tests/. Writes results_autotrace_svd/<prefix>{_svd_maps.npz,.paths.json,_overlay.png,...}.
# Two environments: the maps need Panther-EM's venv, the tracer Leopard-EM's (conda leopard-em).
set -euo pipefail
PREFIX=$1; MIC=$2; TEMPLATE=$3; OPTICS=$4; GPU=${5:-1}; shift $(( $# < 5 ? $# : 5 ))
OUT=${SVD_AUTOTRACE_OUT:-results_autotrace_svd}
PANTHER_ENV=${PANTHER_ENV:-$(dirname "$(realpath "$0")")/panther_env.sh}
LEOPARD_PY=${LEOPARD_PY:-$HOME/miniconda3/envs/leopard-em/bin/python}
[ -x "$LEOPARD_PY" ] || LEOPARD_PY=python
TRACE_THRESHOLD_SIGMA=${TRACE_THRESHOLD_SIGMA:-5}   # a true 2DTM z needs a stricter ridge cut than the cylinder's 4
mkdir -p "$OUT"
( source "$PANTHER_ENV"
  CUDA_VISIBLE_DEVICES=$GPU python ../programs/constrained_search/svd_trace_maps.py \
      --micrograph "$MIC" --template "$TEMPLATE" --optics "$OPTICS" \
      --outdir "$OUT" --prefix "$PREFIX" ${SVD_MAPS_ARGS:-} ) 2>&1 \
  | grep -v -i -E "warn|elem = |it/s\]|proj/s\]|from_numpy" > "$OUT/${PREFIX}_svd_maps.log"
tail -n 3 "$OUT/${PREFIX}_svd_maps.log"
"$LEOPARD_PY" ../programs/constrained_search/auto_trace_filaments.py \
    --micrograph "$MIC" --template "$TEMPLATE" --optics "$OPTICS" \
    --maps "$OUT/${PREFIX}_svd_maps.npz" --threshold-sigma "$TRACE_THRESHOLD_SIGMA" \
    --outdir "$OUT" --prefix "$PREFIX" "$@" 2>&1 | grep -v -i warn > "$OUT/${PREFIX}_trace.log"
grep -E "threshold|track\(s\) traced" "$OUT/${PREFIX}_trace.log"
