#!/usr/bin/env bash
# Progress of any match_template run started from run_scripts/.
#
#   ./check_progress.sh [logfile]
#
# Works with or without a log: without one it reports the running process, its
# elapsed time and the GPUs it is on; with one it also reports the tqdm percentage
# and which templates have already finished.
#
# Supersedes check_full_runs.sh, which was hardcoded to one run and one log path.

cd "$(dirname "$0")" || exit 1
LOG="${1:-}"

# Anchor on the python invocation: a bare script name also matches the wrapper
# shell and this script's own command line.
pid=$(pgrep -f "^/.*bin/python run_scripts/run_" | head -1)

if [ -n "$pid" ]; then
    script=$(ps -o args= -p "$pid" | grep -oE "run_scripts/run_[a-z0-9_]+\.py")
    et=$(ps -o etimes= -p "$pid" | tr -d ' ')
    printf "RUNNING  pid %s  %s  (%d min %02d s elapsed)\n" \
        "$pid" "$script" "$((et / 60))" "$((et % 60))"
else
    echo "NOT RUNNING"
fi

if [ -n "$LOG" ] && [ -f "$LOG" ]; then
    # tqdm redraws with \r, so split on it to reach the most recent bar.
    bar=$(tr '\r' '\n' < "$LOG" | grep -oE "2DTM progress: +[0-9]+%.*" | tail -1)
    [ -n "$bar" ] && echo "  $bar"

    echo
    echo "finished this session:"
    awk '/^=== /{t=$2} /wall time/{w=$3} /^  peaks/{printf "  %-42s %s  %s peaks\n", t, w, $2}' \
        "$LOG" | grep . || echo "  (none yet)"
elif [ -n "$LOG" ]; then
    echo "  (no log at $LOG)"
fi

echo
echo "peak tables on disk:"
ls -t results_cropped/results_*_crop.csv results_full/*.csv 2>/dev/null | head -8 |
    while read -r f; do
        printf "  %-62s %5s rows  %s\n" "$f" "$(($(wc -l < "$f") - 1))" \
            "$(date -r "$f" +%H:%M)"
    done
[ -z "$(ls results_cropped/results_*_crop.csv 2>/dev/null)" ] && echo "  (none yet)"

echo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader |
    awk -F', ' '{printf "GPU %s: %-10s %s\n", $1, $2, $3}'
