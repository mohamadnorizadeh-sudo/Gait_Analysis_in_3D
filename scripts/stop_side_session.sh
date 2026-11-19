#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
w() { for _ in $(seq 1 "$1"); do pgrep -f 07_side_session.py >/dev/null || return 0; sleep 0.5; done; return 1; }
pkill -TERM -f 07_side_session.py && { w 16 || { pkill -KILL -f 07_side_session.py; echo "force-stopped"; }; }
echo "stopped: $(cat runs/LAST_SESSION 2>/dev/null)"
