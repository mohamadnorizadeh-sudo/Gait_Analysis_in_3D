#!/bin/bash
# One-click gait session (v2.8.2): Pi-screen display + web dashboard + keypoint recording.
# Settings live in config/session.env (edit there, not here).
cd "$(dirname "$0")/.." || exit 1
set -a; source config/session.env; set +a
# A previous session that is still shutting down gets up to 10 s to finish
# (fixes the restart race: stop + start right after each other made no new session).
for _ in $(seq 1 20); do pgrep -f 04_run_tracker.py >/dev/null || break; sleep 0.5; done
if pgrep -f 04_run_tracker.py >/dev/null; then
  notify-send "Gait 3D" "A session is already running - press Stop first" 2>/dev/null
  echo "already running - run scripts/stop_session.sh first"; exit 0
fi
D=runs/session_$(date +%Y%m%d_%H%M%S); mkdir -p "$D"
export DISPLAY=${DISPLAY:-:0}
nohup python3 -u scripts/04_run_tracker.py --source shm --rot-left "$ROT_LEFT" --rot-right "$ROT_RIGHT" \
  --calib "$CALIB" --model "$MODEL" --thigh-mm "$THIGH_MM" --shin-mm "$SHIN_MM" \
  --seconds "$MAX_SECONDS" --web-port "$WEB_PORT" --csv "$D/angles.csv" --record-kp "$D/keypoints.jsonl" \
  > "$D/run.log" 2>&1 &
echo "$D" > runs/LAST_SESSION
notify-send "Gait 3D" "Session started - web: http://$(hostname -I | awk '{print $1}'):$WEB_PORT" 2>/dev/null
echo "started: $D"
