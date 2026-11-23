#!/usr/bin/env python3
"""Optional - export YOLOv8n-Pose to NCNN, the fastest CPU backend on the Pi 5.

PyTorch on the Pi 5 CPU runs yolov8n-pose at roughly 5-8 fps for one 640 image;
v2 runs TWO images (left + right) per frame, so use NCNN and a smaller imgsz.

Usage:  python3 scripts/05_export_ncnn.py --imgsz 416
        then: python3 scripts/04_run_tracker.py --model yolov8n-pose_ncnn_model --imgsz 416
"""

import argparse

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="yolov8n-pose.pt")
ap.add_argument("--imgsz", type=int, default=416)
a = ap.parse_args()
path = YOLO(a.weights).export(format="ncnn", imgsz=a.imgsz)
print(f"exported -> {path}")
