#!/usr/bin/env python3
"""Per stereo pair: triangulate the board corners with a calibration and report the 3D square size,
board flatness and ray gap - finds the pairs that spoil a calibration.
  python3 scripts/calib_pairs_check.py runs/calib_v6 runs/calib_v6/stereo_calibration_v6.npz [--square-mm 25]"""
import argparse, glob, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.stereo_geometry import load_geometry
ap = argparse.ArgumentParser(); ap.add_argument('data'); ap.add_argument('calib')
ap.add_argument('--square-mm', type=float, default=25.0); ap.add_argument('--cols', type=int, default=9); ap.add_argument('--rows', type=int, default=6)
a = ap.parse_args()
geo = load_geometry(a.calib)
CB, F = (a.cols, a.rows), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
out = []
for pl in sorted(glob.glob(os.path.join(a.data, 'left', '*.png'))):
    pr = os.path.join(a.data, 'right', os.path.basename(pl))
    gl, gr = cv2.imread(pl, 0), cv2.imread(pr, 0)
    if gl is None or gr is None: continue
    okl, cl = cv2.findChessboardCorners(gl, CB, F); okr, cr = cv2.findChessboardCorners(gr, CB, F)
    if not (okl and okr): continue
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    cl = cv2.cornerSubPix(gl, cl, (5, 5), (-1, -1), crit).reshape(-1, 2)
    cr = cv2.cornerSubPix(gr, cr, (5, 5), (-1, -1), crit).reshape(-1, 2)
    if np.dot(cl[-1] - cl[0], cr[-1] - cr[0]) < 0:  # v2.17.2: same corner order in both views
        cr = cr[::-1].copy()
    P = np.array([geo.triangulate(*l, *r) for l, r in zip(cl, cr)])
    gaps = [geo.ray_gap_mm(*l, *r) for l, r in zip(cl, cr)]
    G = P.reshape(a.rows, a.cols, 3)
    d = np.concatenate([np.linalg.norm(np.diff(G, axis=1), axis=2).ravel(), np.linalg.norm(np.diff(G, axis=0), axis=2).ravel()])
    c = P - P.mean(0); n = np.linalg.svd(c)[2][-1]; flat = np.sqrt(np.mean((c @ n) ** 2))
    out.append((os.path.basename(pl), d.mean(), d.std(), flat, np.median(gaps), P[:, 2].mean(), cl[:, 1].mean() / gl.shape[0]))
print('pair          square_mm  sd   flat  gap   z_mm  img_height')
for r in sorted(out, key=lambda r: -abs(r[1] - a.square_mm)):
    flag = '  <-- bad' if abs(r[1] - a.square_mm) > 0.5 or r[2] > 1.0 else ''
    print(f'{r[0]:12s} {r[1]:7.2f} {r[2]:5.2f} {r[3]:5.2f} {r[4]:5.1f} {r[5]:6.0f}  {r[6]:4.2f}{flag}')
good = [r for r in out if abs(r[1] - a.square_mm) <= 0.5 and r[2] <= 1.0]
print(f'{len(good)} of {len(out)} pairs within 0.5 mm; bad: {" ".join(r[0] for r in out if r not in good)}')
