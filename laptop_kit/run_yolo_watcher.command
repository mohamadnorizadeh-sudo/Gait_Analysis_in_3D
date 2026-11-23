#!/bin/zsh
# Double-click once and leave open: every 20 s, runs YOLOv8x (30 frames per second) on any .MOV in Downloads/gait_check
# that has no kp_<name>_yolov8x.jsonl yet. Only writes new files. Close the window to stop.
source ~/.zshrc >/dev/null 2>&1
G="$HOME/Downloads/gait_check"; K="$HOME/Downloads/Gait_Train_Kit"; P="$G/yolo_watch_progress.txt"
PY=""
for c in "$HOME/yes/bin/python3" "$HOME/anaconda3/bin/python3" "$HOME/miniconda3/bin/python3" python3; do
  command -v "$c" >/dev/null 2>&1 && "$c" -c "import ultralytics" >/dev/null 2>&1 && PY="$c" && break
done
[ -z "$PY" ] && { echo "no python with ultralytics found" | tee -a "$P"; exit 1; }
cd "$K" || exit 1
[ -f yolov8x-pose.pt ] || cp "$HOME/Desktop/Gait_YOLO8x/yolov8x-pose.pt" . 2>/dev/null
echo "watcher started $(date)" | tee -a "$P"
while true; do
  for f in $(find "$G" -maxdepth 1 -iname "*.mov" | sort); do
    [ -f "$f" ] || continue
    v=$(basename "${f%.*}")
    [ -s "$G/kp_${v}_yolov8x.jsonl" ] && continue
    s1=$(stat -f %z "$f"); sleep 5; s2=$(stat -f %z "$f"); [ "$s1" = "$s2" ] || continue   # still copying
    echo "== $v $(date +%T)" | tee -a "$P"
    "$PY" -u scripts/yolo_big_keypoints.py "$f" "$G/kp_${v}_yolov8x.part" --split 1 --fps-out 30 2>&1 | tee -a "$P"
    [ -s "$G/kp_${v}_yolov8x.part" ] && mv "$G/kp_${v}_yolov8x.part" "$G/kp_${v}_yolov8x.jsonl" && echo "DONE $v $(date +%T)" | tee -a "$P"
  done
  echo "idle $(date +%T)" > "$G/yolo_watch_heartbeat.txt"
  sleep 20
done
