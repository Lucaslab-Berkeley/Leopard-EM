#!/usr/bin/env bash
# The four microtubule parameters for one micrograph on the SVD route (SVD_PROCEDURE.md):
# SVD finds (Panther-EM ring and patch searches), refine_template measures, and the register
# and patch lattice state are scored at EVERY lattice site; the register is then read per
# protofilament, which locates the seam and gives every site its own register
# (stage7_seam.py: stage7_seam[_sites]_<tag>_patch.csv and .png).
#
# usage (from MT_tests/):
#   run_scripts/run_svd_pipeline.sh <tag> <ring config> <patch config> <constraint base>
#     <tag>              output name, e.g. pipe_in_situ_823_13pf (the patch gets <tag>_patch)
#     <ring config>      Leopard-EM match_tm yaml of the RING (micrograph, optics, defocus)
#     <patch config>     the same for the 5-PF PATCH
#     <constraint base>  configs/<name> with <name>.yaml, <name>.h5 and <name>.paths.json
#   e.g.
#   run_scripts/run_svd_pipeline.sh pipe_in_situ_823_13pf configs/match_tm_in_situ_823_13pf.yaml \
#       configs/match_tm_in_situ_823_13pf_patch.yaml configs/filament_constraint_in_situ_823
#
# environment (defaults in brackets):
#   RESULTS [results_svd_pipeline]   GPUS ["0 1 2 3"] refine; SEARCH_GPU [0] SVD searches
#   RIVALS [compacted13]  PATCH_STEM [6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB]
#   RING_STEM [6dpu_atoms_6dpv_lattice_13pf_2rings_flatB] models/<stem>.pdb of the ring the
#     search used (Stages 3-5 read its lattice geometry). It must be the rival set's own
#     reference (built by models/build_mt_templates.py), since later stages score the
#     rivals, lattices and patch at the ring's poses. GMPCPP: RIVALS=expanded14
#     PATCH_STEM=6dpu_14pf_patch5_2rings_flatB RING_STEM=6dpu_14pf_2rings_flatB; a GMPCPP
#     tube the PF competition calls 13: RIVALS=expanded13
#     PATCH_STEM=6dpu_13pf_patch5_2rings_flatB RING_STEM=6dpu_13pf_2rings_flatB
#   WIDTH [the paths file's width_px] band width of the ring search, for the refit
#   AVOID_PATHS [""] extra *.paths.json the decoys must avoid (e.g. an autotrace's)
#   KEEP_WORK [0] keep the SVD searches' *_svd_work folders (~2 GB each; deleted otherwise,
#     also when a search fails)
#   PATCH_Z_CUT [""] z cut of the PATCH search's table (svd_match_filament.py --z-cut,
#     default 6.5; the ring search keeps 6.5). At 200 kV / 1.14 A/px the patch's
#     orientation peaks are broad: 330-370 entries per pixel at 6.5 (114 M and 230 M
#     entries) against 10-150 on 823/GDP/GMPCPP, and patch_stage3 time grows with
#     entries x filaments (94 min for 114 M on 2 tubes). 8.0 keeps 9-18 M there.
#   MIN_FREE_GB [50] a step does not start with less free on the results disk; the run
#     stops there (resumable) -- the disk filled once from old SVD work folders
#     (run_scripts/svd_work_dirs.py lists them). The SVD tools also refuse to write their
#     ~2 GB work folder below SVD_MIN_FREE_GB [30] + what they need.
#   PF_SWITCH [1] right after the PF competition (Stage 6, SVD), run_scripts/pf_switch.py
#     settles each filament's N: another N replaces the ring's only with paired t >=
#     PF_SWITCH_T [3] and scores not monotonic in N (template mass) -- first for all
#     filaments pooled, then for each filament against that. A filament where another N
#     leads below that t is probed first: run at that N up to its own pf_check (logs/
#     <probe tag>_probe.log), and moved there if that N then wins with t >= PF_SWITCH_T
#     (a wrong-N ring's sites hold the right N back: t 0.2-1.7 -> 23-33 on the mixed
#     synthetic). A confirmed probe is the run at that N, so it is reused. Everything after Stage 6 --
#     lattice, register, the poses they are scored at -- is asked at the ring's N, so:
#       all at the ring's N        carry on
#       all at one other N         stop, and restart the whole run at that N (exec)
#       at different N             stop, and run once per N over only its filaments
#                                  (their drawn paths: configs/<constraint>_fil<labels>)
#     with that N's rival set (same lattice), ring and patch (built and simulated if
#     missing, after checking the settings reproduce the current ring's map),
#     configs/match_tm_<new tag>[_patch].yaml, and the tag's _<N>pf replaced
#     (pipe_gmpcpp_14pf -> pipe_gmpcpp_13pf; split: <base>_fil3-4_14pf). A switched or
#     split run never switches again: if its own competition disagrees it stops.
#     PF_SWITCH=0 reports the decision and carries on at the ring's N.
#   STOP_AFTER [""] stop after this step (e.g. pf_check: polarity, PF number and rise
#     only); passed on to switched and split runs
#   REFINE_STAGE6 [0] also run Stage 6 on refine_template (compete_sites_curved.py, all
#     rivals x 6-8 roll seeds; ~13 min on 4 GPUs for 823) -- by default the PF competition
#     runs on the SVD scorer (stage6_svd.py: same ranking on 823, GDP and the synthetics,
#     ~5 min), and the lattice is left to Stage 6b, which refine scores at neutral poses
#
# Every step is timed, with the free disk after it (RESULTS/timings_<tag>.tsv), logged
# (logs/<tag>_<step>.log) and skipped on a re-run once done (RESULTS/.done_<tag>/<step>);
# delete a marker to redo it. One micrograph's outputs come to ~5 GB (823: 4.6 GB).
set -euo pipefail
SELF=$(realpath "$0")
cd "$(dirname "$0")/.."
[ $# -eq 4 ] || { sed -n '2,40p' "$0"; exit 2; }
TAG=$1; RING_CONFIG=$2; PATCH_CONFIG=$3; CONSTRAINT=$4
RESULTS=${RESULTS:-results_svd_pipeline}
GPUS=${GPUS:-"0 1 2 3"}
SEARCH_GPU=${SEARCH_GPU:-0}
RIVALS=${RIVALS:-compacted13}
PATCH_STEM=${PATCH_STEM:-6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB}
RING_STEM=${RING_STEM:-6dpu_atoms_6dpv_lattice_13pf_2rings_flatB}
MIN_FREE_GB=${MIN_FREE_GB:-50}
AVOID_PATHS=${AVOID_PATHS:-}
KEEP_WORK=${KEEP_WORK:-0}
PATCH_Z_CUT=${PATCH_Z_CUT:-}
REFINE_STAGE6=${REFINE_STAGE6:-0}
PF_SWITCH=${PF_SWITCH:-1}
PF_SWITCH_T=${PF_SWITCH_T:-3}
PF_SWITCHED_FROM=${PF_SWITCHED_FROM:-}
STOP_AFTER=${STOP_AFTER:-}
PASSED_STOP=0
PANTHER_ENV=${PANTHER_ENV:-$(dirname "$SELF")/panther_env.sh}
PY=${LEOPARD_PY:-$HOME/miniconda3/envs/leopard-em/bin/python}
[ -x "$PY" ] || PY=python
S=../programs/constrained_search
PTAG=${TAG}_patch
REFIT=filament_constraint_${TAG}_refit
T=$RESULTS/output_correlation_table_$TAG.h5
P=$RESULTS/output_correlation_table_$PTAG.h5

for f in "$RING_CONFIG" "$PATCH_CONFIG" "$CONSTRAINT.yaml" "$CONSTRAINT.h5" "$CONSTRAINT.paths.json" \
         "models/$RING_STEM.pdb" "models/$PATCH_STEM.pdb" "models/${PATCH_STEM}_shift1.pdb"; do
    [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
yget() { $PY -c "import yaml,sys; d=yaml.safe_load(open('$1')); print(eval('d'+sys.argv[1]))" "$2"; }
MIC=$(yget "$RING_CONFIG" "['micrograph_path']")
WIDTH=${WIDTH:-$($PY -c "import json; print(json.load(open('$CONSTRAINT.paths.json'))['paths'][0]['width_px'])")}
PSI_STEP=$(yget "$PATCH_CONFIG" "['orientation_search_config']['psi_step']")
THETA_STEP=$(yget "$PATCH_CONFIG" "['orientation_search_config']['theta_step']")

mkdir -p "$RESULTS" "$RESULTS/.done_$TAG" logs
TIMINGS=$RESULTS/timings_$TAG.tsv
[ -f "$TIMINGS" ] || printf "step\tstart\twall_s\tfree_gb_after\n" > "$TIMINGS"
free_gb() { df -P -B1G "$RESULTS" | awk 'NR == 2 { print $4 }'; }

# later stages read the search config as configs/match_tm_<table tag>.yaml
for pair in "$TAG:$RING_CONFIG" "$PTAG:$PATCH_CONFIG"; do
    dst=configs/match_tm_${pair%%:*}.yaml; src=${pair#*:}
    if [ -f "$dst" ]; then
        cmp -s "$dst" "$src" || { echo "$dst exists and differs from $src -- pick another tag"; exit 1; }
    else
        cp "$src" "$dst"
    fi
done

step() {  # step <name> <command...>
    local name=$1; shift
    if [ "$PASSED_STOP" = 1 ]; then echo "== $TAG stopped after $STOP_AFTER (STOP_AFTER)"; exit 0; fi
    [ "$name" = "$STOP_AFTER" ] && PASSED_STOP=1
    if [ -f "$RESULTS/.done_$TAG/$name" ]; then echo "[skip] $name (done)"; return; fi
    local log=logs/${TAG}_$name.log t0 free
    free=$(free_gb)
    if [ "$free" -lt "$MIN_FREE_GB" ]; then
        echo "  STOPPED before $name: $free GB free on $RESULTS's disk, below MIN_FREE_GB=$MIN_FREE_GB."
        echo "  Free space (python run_scripts/svd_work_dirs.py lists old SVD work folders), then"
        echo "  re-run the same command: it resumes here."
        exit 1
    fi
    t0=$(date +%s)
    echo "[$(date +%T)] $name -> $log ($free GB free)"
    if ! "$@" > "$log" 2>&1; then
        echo "  FAILED: $name (tail of $log):"; tail -n 15 "$log"; exit 1
    fi
    printf "%s\t%s\t%d\t%s\n" "$name" "$(date -Iseconds -d @"$t0")" $(( $(date +%s) - t0 )) \
        "$(free_gb)" >> "$TIMINGS"
    touch "$RESULTS/.done_$TAG/$name"
}
svd_search() {  # svd_search <config> <constraint yaml> <out prefix> [z cut]
    local keep=""
    [ "$KEEP_WORK" = 1 ] && keep=--keep-work
    ( source "$PANTHER_ENV"; CUDA_VISIBLE_DEVICES=$SEARCH_GPU python -u $S/svd_match_filament.py \
        --config "$1" --constraint "$2" --out "$3" $keep ${4:+--z-cut "$4"} )
}
svd_table() {  # svd_table <out prefix> <tag>
    $PY $S/svd_entries_to_table.py "$1_svd.h5" --tag "$2"
    [ "$KEEP_WORK" = 1 ] || rm -rf "$1_svd_work"
}

echo "== $TAG: $MIC; ring $RING_CONFIG, patch $PATCH_CONFIG, constraint $CONSTRAINT"
echo "   results $RESULTS, refine GPUs $GPUS, search GPU $SEARCH_GPU, rivals $RIVALS"
echo "   ring $RING_STEM, patch $PATCH_STEM; $(free_gb) GB free, stops below $MIN_FREE_GB"
[ -n "$PF_SWITCHED_FROM" ] && echo "   switched from $PF_SWITCHED_FROM by its PF competition"
[ -n "$PATCH_Z_CUT" ] && echo "   patch table z cut $PATCH_Z_CUT (ring 6.5)"
# --- ring: polarity, PF number, lattice spacing ---------------------------------------
step ring_search  svd_search "$RING_CONFIG" "$CONSTRAINT.yaml" "$RESULTS/$TAG"
step ring_table   svd_table "$RESULTS/$TAG" "$TAG"
step stage3       $PY run_scripts/stage3_read_table.py --table "$T" --stem "$RING_STEM" --constraint "$CONSTRAINT.h5" --micrograph "$MIC"
step stage4       $PY run_scripts/stage4_sites.py --table "$T" --stem "$RING_STEM"
step stage5       $PY run_scripts/stage5_refine.py --table "$T" --stem "$RING_STEM" --gpus $GPUS
step stage6_pf    $PY run_scripts/stage6_svd.py --table "$T" --rivals "$RIVALS" --gpus $GPUS
# --- the PF number: keep the ring's, or restart at the winner's ---------------------------
step pf_check     $PY run_scripts/pf_switch.py --tag "$TAG" --results "$RESULTS" --rivals "$RIVALS" \
                      --ring-config "$RING_CONFIG" --patch-config "$PATCH_CONFIG" \
                      --ring-stem "$RING_STEM" --patch-stem "$PATCH_STEM" --constraint "$CONSTRAINT" \
                      --min-t "$PF_SWITCH_T" \
                      --device "cuda:$SEARCH_GPU" ${PF_SWITCHED_FROM:+--switched-from "$PF_SWITCHED_FROM"} \
                      $([ "$PF_SWITCH" = 1 ] || echo --no-switch)
grep -a "^PF number" logs/${TAG}_pf_check.log | sed 's/^/   /' || true
SWITCH=0
[ -f "$RESULTS/pf_switch_$TAG.env" ] && source "$RESULTS/pf_switch_$TAG.env"
if [ "$SWITCH" = 1 ] && [ "$PF_SWITCH" = 1 ]; then
    echo "== $TAG stops here: restarting as $NEW_TAG ($NEW_RIVALS; ring $NEW_RING_STEM, patch $NEW_PATCH_STEM)"
    exec env RIVALS="$NEW_RIVALS" RING_STEM="$NEW_RING_STEM" PATCH_STEM="$NEW_PATCH_STEM" \
        PF_SWITCHED_FROM="$TAG" bash "$SELF" "$NEW_TAG" "$NEW_RING_CONFIG" "$NEW_PATCH_CONFIG" "$CONSTRAINT"
elif [ "$SWITCH" = split ] && [ "$PF_SWITCH" = 1 ]; then
    child() { local v="CHILD_${i}_$1"; echo "${!v}"; }
    echo "== $TAG stops here: its filaments differ in PF number -> $N_CHILDREN runs"
    failed=""
    for i in $(seq 1 "$N_CHILDREN"); do
        echo; echo "== [$i/$N_CHILDREN] filament(s) $(child FILAMENTS) of $TAG -> $(child TAG)"
        env RIVALS="$(child RIVALS)" RING_STEM="$(child RING_STEM)" PATCH_STEM="$(child PATCH_STEM)" \
            PF_SWITCHED_FROM="$TAG" bash "$SELF" "$(child TAG)" "$(child RING_CONFIG)" \
            "$(child PATCH_CONFIG)" "$(child CONSTRAINT)" || failed="$failed $(child TAG)"
    done
    [ -z "$failed" ] || { echo "== FAILED:$failed"; exit 1; }
    exit 0
fi
if [ "$REFINE_STAGE6" = 1 ]; then
    step stage6   $PY run_scripts/compete_sites_curved.py --tag "$TAG" --results "$RESULTS" --rivals "$RIVALS" --gpus $GPUS
fi
step stage6b      $PY run_scripts/stage6b_local_lattice.py --table "$T" --rivals "$RIVALS" --gpus $GPUS
# --- patch: seam / register, and the lattice state per protofilament-resolved site ----
step refit        $PY run_scripts/refit_constraint_from_ring.py --tag "$TAG" --results "$RESULTS" \
                      --source "$CONSTRAINT.h5" --paths "$CONSTRAINT.paths.json" --out "$REFIT" \
                      --width-px "$WIDTH" --micrograph "$MIC" --psi-step "$PSI_STEP" \
                      --theta-step "$THETA_STEP" --restrict-polarity
step patch_search svd_search "$PATCH_CONFIG" "configs/$REFIT.yaml" "$RESULTS/$PTAG" "$PATCH_Z_CUT"
step patch_table  svd_table "$RESULTS/$PTAG" "$PTAG"
step patch_stage3 $PY run_scripts/stage3_read_table.py --table "$P" --stem "$PATCH_STEM" --constraint "configs/$REFIT.h5" --micrograph "$MIC"
step patch_stage4 $PY run_scripts/stage4_sites.py --table "$P" --stem "$PATCH_STEM" --resolve-protofilaments
step stage7       $PY run_scripts/stage7_register.py --table "$P" --stem "$PATCH_STEM" --gpus $GPUS
step patch_6b     $PY run_scripts/stage6b_local_lattice.py --table "$P" --patch --start lattice --rivals "$RIVALS" --gpus $GPUS
step every_sites  $PY run_scripts/lattice_fill.py sites --table "$P" --patch --rivals "$RIVALS" \
                      --constraint "configs/$REFIT.h5" ${AVOID_PATHS:+--avoid-paths $AVOID_PATHS}
step stage7_every $PY run_scripts/stage7_every_site.py --table "$P" --stem "$PATCH_STEM" --gpus $GPUS
step seam         $PY run_scripts/stage7_seam.py --table "$P"

# --- summary ----------------------------------------------------------------------------
L=logs/${TAG}
echo; echo "== $TAG summary"
echo "-- polarity and PF harmonic (Stage 3):"; grep -aE "^\s*[0-9]+\s+[0-9,]+\s+[0-9]+nm" ${L}_stage3.log || true
echo "-- refined rise (Stage 5):"; sed -n '/^filament   sites/,/^$/p' ${L}_stage5.log || true
echo "-- PF number (Stage 6, SVD):"; grep -a "^  ranking:" ${L}_stage6_pf.log | tail -1 || true
grep -a "^PF number" ${L}_pf_check.log | sed 's/^/  /' || true
[ -n "$PF_SWITCHED_FROM" ] && echo "   (this run was switched from $PF_SWITCHED_FROM: logs/${PF_SWITCHED_FROM}_pf_check.log)"
[ -f ${L}_stage6.log ] && { echo "-- PF / lattice ranking (Stage 6, refine):"; grep -a "ranking:" ${L}_stage6.log | tail -1; }
echo "-- lattice per segment, ring (Stage 6b):"; grep -aE "^  filament [0-9]+:|^    (call|competition) " ${L}_stage6b.log || true
echo "-- lattice per segment, patch at every site (Stage 6b):"; grep -aE "^  filament [0-9]+:|^    (call|competition) " ${L}_patch_6b.log || true
echo "-- register, found sites (Stage 7):"; sed -n '/^filament  *sites  *even/,/^$/p' ${L}_stage7.log || true
echo "-- register, every site (Stage 7):"; awk '/^REFINE every site +1 /{on=1; print; next} on && /^REFINE/{exit} on' ${L}_stage7_every.log || true
echo "-- register per protofilament and the seam (every site; pooled t is the line above):"
sed -n '/^filament  *sites  *pooled t/,/^$/p' ${L}_seam.log 2>/dev/null || true
echo "-- wall times (s) and free disk after each step (GB):"; column -t "$TIMINGS"
awk -F'\t' 'NR > 1 { s += $3 } END { printf "   total %.1f min\n", s / 60 }' "$TIMINGS"
echo "-- disk: $(du -sch "$RESULTS"/*"$TAG"* 2>/dev/null | tail -1 | cut -f1) in $RESULTS for $TAG; $(free_gb) GB free"
