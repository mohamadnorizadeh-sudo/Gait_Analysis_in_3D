#!/usr/bin/env python3
"""Capture per-camera chessboard views from shared memory (v2.2).
Each camera is saved independently whenever IT sees a sharp board that moved,
so the intrinsics can be calibrated over the camera's WHOLE field of view
(pairs only cover the small region both converging cameras share)."""
import argparse, os, sys, time, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.camera import ShmStereoSource
ap = argparse.ArgumentParser()
ap.add_argument('--out', default='runs/calib_shm/mono'); ap.add_argument('--count', type=int, default=35)
ap.add_argument('--rot-left', type=int, default=0); ap.add_argument('--rot-right', type=int, default=0)
ap.add_argument('--show', action='store_true', help='preview window (needs a desktop, e.g. DISPLAY=:0)')
ap.add_argument('--cols', type=int, default=9); ap.add_argument('--rows', type=int, default=6)
a = ap.parse_args(); CB = (a.cols, a.rows)
F = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
st = {k: dict(n=0, last=None, cov=None) for k in ('cam0', 'cam1')}
for k in st: os.makedirs(f'{a.out}/{k}', exist_ok=True)
with ShmStereoSource(rot=(a.rot_left, a.rot_right)) as s:
    while min(v['n'] for v in st.values()) < a.count:
        t, im0, im1, dt = s.read()
        vis = []
        for k, im in (('cam0', im0), ('cam1', im1)):
            v = st[k]
            if v['cov'] is None: v['cov'] = np.zeros(im.shape[:2], np.uint8)
            shown = im.copy()
            if a.show:
                shown = cv2.addWeighted(shown, 0.8, cv2.cvtColor(v['cov'], cv2.COLOR_GRAY2BGR), 0.2, 0)
                cv2.putText(shown, f"{k} {v['n']}/{a.count} cov {100*v['cov'].mean()/255:.0f}%", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                vis.append(shown)
            if v['n'] >= a.count: continue
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ok, c = cv2.findChessboardCorners(g, CB, F)
            if not ok: continue
            if a.show: cv2.drawChessboardCorners(vis[-1], CB, c, True)
            x, y, w, h = cv2.boundingRect(c.astype(np.float32))
            sh = cv2.Laplacian(g[max(0,y):y+h, max(0,x):x+w], cv2.CV_64F).var()
            if sh < 60 or (v['last'] is not None and np.mean(np.linalg.norm(c - v['last'], axis=2)) < 30): continue
            cv2.imwrite(f"{a.out}/{k}/img_{v['n']:02d}.png", im)
            cv2.fillConvexPoly(v['cov'], cv2.convexHull(c.astype(np.int32)), 255)
            v['last'] = c; v['n'] += 1
            if a.show: vis[-1][:] = 255 - vis[-1]
        print(' | '.join(f"{k}: {v['n']}/{a.count} coverage {0 if v['cov'] is None else 100*v['cov'].mean()/255:.0f}%" for k, v in st.items()), flush=True)
        if a.show:
            cv2.imshow('single-camera lens calibration (cover the WHOLE view of each camera)', np.hstack(vis))
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        else:
            time.sleep(0.05)
print('DONE')
