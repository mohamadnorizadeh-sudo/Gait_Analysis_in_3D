#!/usr/bin/env python3
"""Which rotation / camera order gives the best leg keypoints right now?
Run with a person standing in the walker. Read-only shm source, AI HAT+ 2 model."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.camera import ShmStereoSource
from gait3d.hailo_pose import HailoPoseEstimator
from gait3d.pose import best_person
hp = HailoPoseEstimator('/usr/share/hailo-models/yolov8m_pose_h10.hef')
frames = []
with ShmStereoSource() as s:
    for _ in range(8):
        frames.append(s.read()[1:3]); time.sleep(0.2)
import cv2
rot = {0: lambda im: im, 180: lambda im: cv2.rotate(im, cv2.ROTATE_180)}
for name, idx in (('cam0', 0), ('cam1', 1)):
    for r in (0, 180):
        c = []
        for f in frames:
            p = best_person(hp([rot[r](f[idx])])[0])
            c.append(np.mean(p.conf[11:17]) if p is not None else 0.0)
        print(f'{name} extra rotation {r:3d}: mean leg confidence {np.mean(c):.2f}')
hp.close()
