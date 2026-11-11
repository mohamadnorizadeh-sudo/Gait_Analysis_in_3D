#!/bin/bash
# Stops the running gait session (v2.8.2).
# INT first (clean CSV close); waits up to 8 s; then TERM, then KILL if the
# tracker is stuck (seen 2026-11-11 after a power dip: Hailo call hung, INT ignored).
cd "$(dirname "$0")/.." || exit 1
stop_wait() { for _ in $(seq 1 "$1"); do pgrep -f 04_run_tracker.py >/dev/null || return 0; sleep 0.5; done; return 1; }
if pgrep -f 04_run_tracker.py >/dev/null; then
  pkill -INT -f 04_run_tracker.py; stop_wait 16 || {
    pkill -TERM -f 04_run_tracker.py; stop_wait 8 || { pkill -KILL -f 04_run_tracker.py; stop_wait 4; }
    echo "warning: tracker had to be force-stopped (last rows may be missing)"; }
fi
D=$(cat runs/LAST_SESSION 2>/dev/null)
[ -n "$D" ] && [ -f "$D/angles.csv" ] && python3 scripts/analyze_session.py "$D/angles.csv" > "$D/summary.txt" 2>&1
notify-send "Gait 3D" "Session stopped - saved in $D" 2>/dev/null
echo "stopped: $D"
