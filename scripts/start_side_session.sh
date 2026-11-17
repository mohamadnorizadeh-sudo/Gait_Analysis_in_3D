#!/bin/bash
# v3.0 side-camera session: Pi screen + web dashboard + recording.  Settings: config/side.env
cd "$(dirname "$0")/.." || exit 1
[ -f config/side.env ] && { set -a; source config/side.env; set +a; }
pkill -f side_view.py 2>/dev/null
for _ in $(seq 1 20); do pgrep -f "04_run_tracker.py|07_side_session.py" >/dev/null || break; sleep 0.5; done
if pgrep -f "04_run_tracker.py|07_side_session.py" >/dev/null; then echo "already running - scripts/stop_side_session.sh first"; exit 0; fi
D=runs/side_session_$(date +%Y%m%d_%H%M%S); mkdir -p "$D"
export DISPLAY=${DISPLAY:-:0}
ARGS=(--out "$D" --source "${SOURCE:-L=cam0,R=cam1}" --seconds "${MAX_SECONDS:-3600}" --web-port "${WEB_PORT:-8080}")
[ "${MARKERS:-0}" = "1" ] && ARGS+=(--markers)
[ -n "${SHANK_SEED:-}" ] && ARGS+=(--shank-seed "$SHANK_SEED")
nohup python3 -u scripts/07_side_session.py "${ARGS[@]}" > "$D/run.log" 2>&1 &
echo "$D" > runs/LAST_SESSION
echo "started: $D  (web: http://$(hostname -I | awk '{print $1}'):${WEB_PORT:-8080})"
