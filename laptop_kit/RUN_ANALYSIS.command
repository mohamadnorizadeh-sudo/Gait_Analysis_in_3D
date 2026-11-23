#!/bin/zsh
# Double-click: builds YOLOv8x references from the iPad videos, trains and tests the knee-angle model.
# Works on ~/Desktop/Gait_Train (walk1..walk4 + test). Only adds new files; never deletes or changes your videos.
source ~/.zshrc >/dev/null 2>&1
KIT="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HOME/Desktop/Gait_Train"
mkdir -p "$ROOT"/{walk1,walk2,walk3,walk4,test}
PY=""
for c in "$HOME/yes/bin/python3" "$HOME/anaconda3/bin/python3" "$HOME/miniconda3/bin/python3" /opt/anaconda3/bin/python3 /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import ultralytics" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  for c in "$HOME/yes/bin/python3" "$HOME/anaconda3/bin/python3" "$HOME/miniconda3/bin/python3" python3; do command -v "$c" >/dev/null 2>&1 && PY="$c" && break; done
  echo "installing ultralytics into $PY (once)..."; "$PY" -m pip install -q ultralytics
fi
"$PY" -c "import sklearn, scipy" >/dev/null 2>&1 || { echo "installing scikit-learn (once)..."; "$PY" -m pip install -q scikit-learn scipy; }
echo "python: $PY"
[ -f "$KIT/yolov8x-pose.pt" ] || [ ! -f "$HOME/Desktop/Gait_YOLO8x/yolov8x-pose.pt" ] || cp "$HOME/Desktop/Gait_YOLO8x/yolov8x-pose.pt" "$KIT/"
cd "$KIT" || exit 1
"$PY" -u scripts/offline_pipeline.py "$ROOT" "$@"
echo
echo "Report: $ROOT/report.txt   Model: $ROOT/model/"
open "$ROOT/report.txt" 2>/dev/null
