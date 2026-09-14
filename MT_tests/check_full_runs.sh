#!/usr/bin/env bash
# Progress of the full-micrograph match_template runs (ring, then patch).
LOG=/tmp/claude-2002/-home-jdickerson-git-LucasLab-Leopard-EM/4ef68f68-efb0-483e-978f-b9e47c263b75/scratchpad/full_runs.log
EST=170   # minutes per template, from the measured 1.31 s/1000 orientations x 16x area

cd "$(dirname "$0")" || exit 1
# Anchor on the python invocation: a bare "run_match_full_" also matches the
# wrapper shell and this script's own command line.
pid=$(pgrep -f "^/.*bin/python run_scripts/run_match_full_" | head -1)

if [ -n "$pid" ]; then
    tmpl=$(ps -o args= -p "$pid" | grep -oE "run_match_full_[a-z0-9]+" | sed 's/run_match_full_//')
    et=$(ps -o etimes= -p "$pid" | tr -d ' ')
    mins=$((et / 60))
    pct=$((mins * 100 / EST))
    [ "$pct" -gt 99 ] && pct=99
    echo "RUNNING: $tmpl   ${mins} min elapsed of ~${EST} min  (~${pct}%, ~$((EST - mins)) min left)"
else
    echo "NOT RUNNING"
fi

echo
echo "completed stages:"
grep -E "^=== (START|END)" "$LOG" 2>/dev/null | sed 's/^/  /' || echo "  (none yet)"

echo
echo "outputs:"
if ls results_full/*.mrc >/dev/null 2>&1; then
    ls -la results_full/ | awk 'NR>3 {printf "  %-52s %s\n", $9, $5}'
else
    echo "  none yet (written only when a stage finishes)"
fi

echo
echo "GPUs: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr '\n' ' ')"
