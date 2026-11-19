#!/usr/bin/env python3
"""Live comparison of pose back-ends on the stereo rig (read-only shm):
per model -> speed, leg-joint confidences, ray gap between the two views,
and the knee-ankle distance triangulated directly (vs the tape length)."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.camera import ShmStereoSource
from gait3d.stereo_geometry import load_geometry
from gait3d.pose import PoseEstimator, people_from_result, best_person
from gait3d.hailo_pose import HailoPoseEstimator

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
geo = load_geometry('stereo_calibration.npz')
models = {
    'cpu yolov8n@320': PoseEstimator(os.path.expanduser('~/yolov8n-pose.pt'), 320),
    'hailo yolov8m@640': HailoPoseEstimator('/usr/share/hailo-models/yolov8m_pose_h10.hef'),
}
frames = []
with ShmStereoSource(rot=(0, 180)) as s:
    for _ in range(N):
        frames.append(s.read()[1:3])
        time.sleep(0.15)
for name, m in models.items():
    t0 = time.time(); stats = {'conf': [], 'gap': [], 'shin': [], 'thigh': []}
    for L, R in frames:
        a, b = (best_person(people_from_result(r)) for r in m([L, R]))
        if a is None or b is None:
            continue
        for hip, knee, ank in ((11, 13, 15), (12, 14, 16)):
            stats['conf'] += [min(a.conf[k], b.conf[k]) for k in (hip, knee, ank)]
            if min(a.conf[knee], b.conf[knee], a.conf[ank], b.conf[ank]) < 0.5:
                continue
            P = {k: geo.triangulate(*a.xy[k], *b.xy[k]) for k in (hip, knee, ank)}
            stats['gap'] += [geo.ray_gap_mm(*a.xy[k], *b.xy[k]) for k in (knee, ank)]
            stats['shin'].append(np.linalg.norm(P[knee] - P[ank]))
            if min(a.conf[hip], b.conf[hip]) >= 0.5:
                stats['thigh'].append(np.linalg.norm(P[hip] - P[knee]))
    dt = (time.time() - t0) / len(frames)
    f = lambda v: f'{np.median(v):6.1f} (IQR {np.percentile(v,25):.0f}-{np.percentile(v,75):.0f})' if v else '   n/a'
    print(f'{name:18s} {dt*1000:5.0f} ms/pair | min-conf median {np.median(stats["conf"]) if stats["conf"] else 0:.2f} | '
          f'ray gap {f(stats["gap"])} mm | shin {f(stats["shin"])} mm | thigh {f(stats["thigh"])} mm | legs {len(stats["shin"])}')
