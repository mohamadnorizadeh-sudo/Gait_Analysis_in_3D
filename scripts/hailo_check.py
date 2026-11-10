#!/usr/bin/env python3
"""Validate the Hailo-10H pose decoder against the CPU model on live frames (read-only shm)."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.camera import ShmStereoSource
from gait3d.hailo_pose import HailoPoseEstimator, DEFAULT_HEF, letterbox
from gait3d.pose import PoseEstimator, people_from_result, best_person
import cv2

hef = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HEF
hp = HailoPoseEstimator(hef)
with ShmStereoSource() as s:
    t, L, R, dt = s.read()
lb, *_ = letterbox(cv2.cvtColor(L, cv2.COLOR_BGR2RGB), hp.size)
raw = hp._infer(lb)
for n, a in raw.items():
    print(f'{n:28s} {str(a.shape):16s} min {a.min():8.3f} max {a.max():8.3f}')
t0 = time.time(); n = 20
for _ in range(n):
    res_h = hp([L, R])
dt_h = (time.time() - t0) / n
cpu = PoseEstimator(os.path.expanduser('~/yolov8n-pose.pt'), 640)
res_c = cpu([L, R])
names = ['Lhip', 'Rhip', 'Lknee', 'Rknee', 'Lank', 'Rank']
for cam, h, c in (('cam0', res_h[0], res_c[0]), ('cam1', res_h[1], res_c[1])):
    a, b = best_person(h), best_person(people_from_result(c))
    print(cam, 'hailo persons', len(h), '| cpu persons', len(people_from_result(c)))
    if a is not None and b is not None:
        for nm, k in zip(names, range(11, 17)):
            print(f'  {nm:6s} hailo ({a.xy[k,0]:6.1f},{a.xy[k,1]:6.1f}) c={a.conf[k]:.2f} | cpu ({b.xy[k,0]:6.1f},{b.xy[k,1]:6.1f}) c={b.conf[k]:.2f} | diff {np.linalg.norm(a.xy[k]-b.xy[k]):5.1f}px')
print(f'Hailo {os.path.basename(hef)}: {dt_h*1000:.0f} ms per L+R pair -> {1/dt_h:.1f} stereo fps')
hp.close()
