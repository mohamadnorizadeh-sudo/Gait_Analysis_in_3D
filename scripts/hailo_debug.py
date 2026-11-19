#!/usr/bin/env python3
"""Hailo pose debug: input colour order / scaling vs person score, and raw timings."""
import os, sys, time
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.camera import ShmStereoSource
from gait3d.hailo_pose import HailoPoseEstimator, DEFAULT_HEF, letterbox, decode
hef = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HEF
hp = HailoPoseEstimator(hef)
print('input', hp.model.input().shape, 'outputs', hp.shapes)
with ShmStereoSource(rot=(0, 180)) as s:
    t, L, R, dt = s.read()
for name, img in (('RGB letterbox', cv2.cvtColor(L, cv2.COLOR_BGR2RGB)), ('BGR letterbox', L)):
    lb, sc, left, top = letterbox(img, hp.size)
    out = hp._infer(lb)
    cls = {n: float(a.max()) for n, a in out.items() if a.shape[-1] == 1}
    print(name, 'max cls per scale', {k.split('/')[-1]: round(v, 4) for k, v in cls.items()})
    # stretch (no letterbox)
    st = cv2.resize(img, (hp.size, hp.size))
    out = hp._infer(st)
    print(name.split()[0], 'stretched  max cls', round(max(float(a.max()) for a in out.values() if a.shape[-1] == 1), 4))
lb, *_ = letterbox(cv2.cvtColor(L, cv2.COLOR_BGR2RGB), hp.size)
t0 = time.time()
for _ in range(20):
    out = hp._infer(lb)
t_inf = (time.time() - t0) / 20
t0 = time.time()
for _ in range(20):
    decode(out, 0.25, size=hp.size)
t_dec = (time.time() - t0) / 20
print(f'infer {t_inf*1000:.1f} ms/frame, decode {t_dec*1000:.1f} ms/frame')
hp.close()
