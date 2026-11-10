#!/bin/bash
# Stops the running gait session cleanly (CSV and recordings are closed properly).
cd "$(dirname "$0")/.." || exit 1
pkill -INT -f 04_run_tracker.py && sleep 2
D=$(cat runs/LAST_SESSION 2>/dev/null)
[ -n "$D" ] && [ -f "$D/angles.csv" ] && python3 scripts/analyze_session.py "$D/angles.csv" > "$D/summary.txt" 2>&1
notify-send "Gait 3D" "Session stopped - saved in $D" 2>/dev/null
echo "stopped: $D"
