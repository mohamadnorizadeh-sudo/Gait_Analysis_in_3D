#!/usr/bin/env python3
"""Live check window on the Pi screen: both cameras side by side with the leg
keypoints of a pose model, per-joint confidence, and the stereo ray gap
(how well the two views agree; should be < ~40 mm).  Read-only shm source.

  DISPLAY=:0 python3 scripts/live_view.py --rot-right 180 --model hailo
"""
import argparse, os, sys, time
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.camera import ShmStereoSource
from gait3d.stereo_geometry import load_geometry
from gait3d.pose import PoseEstimator, people_from_result, best_person

ap = argparse.ArgumentParser()
ap.add_argument('--rot-left', type=int, default=0); ap.add_argument('--rot-right', type=int, default=0)
ap.add_argument('--model', default='hailo', help="'hailo', 'cpu', or a weights path")
ap.add_argument('--calib', default='stereo_calibration.npz')
ap.add_argument('--seconds', type=float, default=0)
ap.add_argument('--log', default='')
a = ap.parse_args()
if a.model == 'hailo':
    from gait3d.hailo_pose import HailoPoseEstimator
    pose = HailoPoseEstimator('/usr/share/hailo-models/yolov8m_pose_h10.hef')
elif a.model == 'cpu':
    pose = PoseEstimator(os.path.expanduser('~/yolov8n-pose.pt'), 320)
else:
    pose = PoseEstimator(a.model, 480)
geo = load_geometry(a.calib)
NAMES = {11: 'Lhip', 12: 'Rhip', 13: 'Lkn', 14: 'Rkn', 15: 'Lank', 16: 'Rank'}
COL = {11: (255, 200, 0), 13: (255, 200, 0), 15: (255, 200, 0), 12: (0, 200, 255), 14: (0, 200, 255), 16: (0, 200, 255)}
log = open(a.log, 'w') if a.log else None
t0 = time.time()
with ShmStereoSource(rot=(a.rot_left, a.rot_right)) as s:
    while True:
        t, L, R, dt = s.read()
        pl, pr = (best_person(people_from_result(r)) for r in pose([L, R]))
        vis = [L.copy(), R.copy()]
        for v, p in zip(vis, (pl, pr)):
            if p is None:
                cv2.putText(v, 'no person', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                continue
            for a_, b_ in ((11, 13), (13, 15), (12, 14), (14, 16)):
                if min(p.conf[a_], p.conf[b_]) > 0.3:
                    cv2.line(v, tuple(map(int, p.xy[a_])), tuple(map(int, p.xy[b_])), COL[a_], 2)
            for k in NAMES:
                c = (0, 220, 0) if p.conf[k] >= 0.5 else (0, 0, 255)
                cv2.circle(v, tuple(map(int, p.xy[k])), 5, c, -1)
                cv2.putText(v, f'{NAMES[k]} {p.conf[k]:.2f}', (int(p.xy[k][0]) + 6, int(p.xy[k][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
        line = []
        if pl is not None and pr is not None:
            for k in NAMES:
                if min(pl.conf[k], pr.conf[k]) >= 0.5:
                    line.append(f'{NAMES[k]} {geo.ray_gap_mm(*pl.xy[k], *pr.xy[k]):.0f}')
        txt = 'ray gap mm: ' + ('  '.join(line) if line else '-')
        canvas = np.hstack(vis)
        cv2.rectangle(canvas, (0, canvas.shape[0] - 28), (canvas.shape[1], canvas.shape[0]), (30, 30, 30), -1)
        cv2.putText(canvas, txt, (6, canvas.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(canvas, 'cam0 (left)', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, 'cam1 (right)', (L.shape[1] + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('Gait v2 live check', canvas)
        if log:
            log.write(f'{time.time()-t0:.2f} {txt}\n'); log.flush()
        if (cv2.waitKey(1) & 0xFF) == ord('q') or (a.seconds and time.time() - t0 > a.seconds):
            break
cv2.destroyAllWindows()
