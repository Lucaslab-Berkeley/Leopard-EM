#!/usr/bin/env bash
# Run every constrained search on the real curved GDP micrograph, in sequence.
#
# The order is not arbitrary. Protofilament number comes first because it decides which
# of the later pairs is worth running at all -- the lattice question has to be asked at
# the winning N (competing a 13-PF expanded model against a 14-PF compacted one would
# confound the two), and so does the register question. Run the first stage, look at
# it, then run the second.
#
# Each search is skipped if its correlation table already exists, so this is safe to
# re-run after an interruption; delete a table to redo that one.
#
# Budget about an hour per search on four GPUs -- the two earlier full-frame runs on
# this box finished 53 minutes apart. The constraint does NOT make this cheaper: it
# gates which (pixel, orientation) pairs may win the MIP, but all 485,856 orientations
# are still correlated over the whole frame. So stage 1 is roughly 4 hours and the
# whole set roughly 8.
#
# Usage:
#   run_scripts/run_gdp_curved.sh              # stage 1 only: the four PF templates
#   run_scripts/run_gdp_curved.sh lattice13    # then, once N is known
#   run_scripts/run_gdp_curved.sh 13pf_patch   # or name any tags explicitly
#   GPUS="0 1" run_scripts/run_gdp_curved.sh   # fewer GPUs

set -euo pipefail
cd "$(dirname "$0")/.."

PREFIX=gdp_curved
RESULTS=results_gdp_curved
CONSTRAINT=configs/filament_constraint_${PREFIX}.yaml
GPUS=${GPUS:-}
LOGDIR=${RESULTS}/logs

STAGE1="12pf 13pf 14pf 15pf"

case "${1:-}" in
    ""|stage1|pf)   TAGS="$STAGE1" ;;
    lattice13)      TAGS="13pf_expanded" ;;
    lattice14)      TAGS="14pf_expanded" ;;
    all)            TAGS="$STAGE1 13pf_expanded 14pf_expanded 13pf_patch 14pf_patch" ;;
    *)              TAGS="$*" ;;
esac

mkdir -p "$LOGDIR"
echo "constraint $CONSTRAINT"
echo "results    $RESULTS/"
echo "tags       $TAGS"
echo "estimate   about 1 h each on 4 GPUs"
echo

for tag in $TAGS; do
    full="${PREFIX}_${tag}"
    table="${RESULTS}/output_correlation_table_${full}.h5"
    if [ -f "$table" ]; then
        echo "== ${full}: table exists, skipping (delete it to redo)"
        continue
    fi
    if [ ! -f "configs/match_tm_${full}.yaml" ]; then
        echo "== ${full}: no config -- run run_scripts/setup_gdp_curved.py" >&2
        exit 1
    fi
    echo "== ${full}: starting $(date '+%H:%M:%S')"
    args=(--prefix '' --tag "$full" --constraint "$CONSTRAINT" --results "$RESULTS")
    if [ -n "$GPUS" ]; then
        # shellcheck disable=SC2206
        args+=(--gpus $GPUS)
    fi
    python run_scripts/run_match_curved.py "${args[@]}" 2>&1 \
        | tee "${LOGDIR}/${full}.log"
    echo "== ${full}: done $(date '+%H:%M:%S')"
    echo
done

echo "Compare them with:"
echo "  python run_scripts/compare_gdp_curved.py"
echo "Then read one out with:"
echo "  python run_scripts/analyse_gdp_curved.py --tag ${PREFIX}_13pf"
