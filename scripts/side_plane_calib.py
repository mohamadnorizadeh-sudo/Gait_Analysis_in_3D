#!/usr/bin/env python3
"""v3.0 - map a camera's view onto a leg's sagittal plane with a checkerboard.

Hold the 9x6 checkerboard (25 mm squares) FLAT against the OUTSIDE of the leg at knee
height: upright, its long edge along the walking direction, still for ~2 s.
Run once per leg; both cameras are tried, each camera that sees the board gets a
homography for that leg ("cam0:L" ...), saved in config/side_planes.json.

  python3 scripts/side_plane_calib.py --leg L
  python3 scripts/side_plane_calib.py --leg R
  python3 scripts/side_plane_calib.py --check          # print what is stored

Also prints how slanted each camera is to the board (0 deg = looking straight at it),
which tells you whether to turn a camera.
"""
import argparse, os, sys, time
import cv2
import numpy as np
cv2.setNumThreads(1)   # v3.0.1: full-CPU corner search dropped the Pi (UPS) off the network twice
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.side2d import PlaneMap

ap = argparse.ArgumentParser()
ap.add_argument('--leg', choices=['L', 'R'])
ap.add_argument('--out', default='config/side_planes.json')
ap.add_argument('--calib', default='stereo_calibration.npz', help='old fisheye intrinsics (K_l=cam0, K_r=cam1)')
ap.add_argument('--calib-rot', type=int, default=180, help='rotation the intrinsics were captured with vs now')
ap.add_argument('--square-mm', type=float, default=25.0)
ap.add_argument('--frames', type=int, default=8)
ap.add_argument('--check', action='store_true')
ap.add_argument('--images', nargs=2, metavar=('CAM0_IMG', 'CAM1_IMG'), help='offline: use two saved images instead of the cameras')
a = ap.parse_args()

pm = PlaneMap.load(a.out)
W, H = 360, 640
if not pm.cams and os.path.exists(a.calib):
    c = np.load(a.calib)
    for cam, (Kk, Dk) in {'cam0': ('K_l', 'D_l'), 'cam1': ('K_r', 'D_r')}.items():
        K = c[Kk].astype(float).copy()
        if a.calib_rot % 360 == 180:
            K[0, 2], K[1, 2] = W - 1 - K[0, 2], H - 1 - K[1, 2]
        pm.cams[cam] = {'K': K, 'D': c[Dk].astype(float).reshape(4, 1)}
if a.check:
    for k, v in pm.planes.items():
        print(k, {kk: vv for kk, vv in v.items() if kk != 'H'})
    sys.exit(0)
if not a.leg:
    sys.exit('--leg L or --leg R')


def find(gray):
    for size in ((9, 6), (6, 9)):
        ok, cn = cv2.findChessboardCorners(gray, size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
        if ok:
            cn = cv2.cornerSubPix(gray, cn, (3, 3), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            cols, rows = size
            g = np.array([[x, y] for y in range(rows) for x in range(cols)], float) * a.square_mm / 10.0
            return cn.reshape(-1, 2).astype(float), g
    return None, None


frames = []
if a.images:
    frames = [(cv2.imread(a.images[0]), cv2.imread(a.images[1]))]
else:
    from gait3d.camera import ShmStereoSource
    with ShmStereoSource() as src:
        for _ in range(a.frames):
            _, i0, i1, _ = src.read()
            frames.append((i0, i1))
            time.sleep(0.25)
os.makedirs('runs/side_planes', exist_ok=True)
for ci, cam in enumerate(('cam0', 'cam1')):
    Hs, res, slant = [], [], []
    for fr in frames:
        img = fr[ci]
        cn, g = find(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if cn is None:
            continue
        u = pm.undistort(cam, cn)
        Hm, _ = cv2.findHomography(u, g)
        q = np.hstack([u, np.ones((len(u), 1))]) @ Hm.T
        res.append(float(np.sqrt(np.mean(np.sum((q[:, :2] / q[:, 2:3] - g) ** 2, 1)))))
        Hs.append(Hm / Hm[2, 2])
        if cam in pm.cams:
            ok, rv, tv = cv2.solvePnP(np.column_stack([g * 10, np.zeros(len(g))]), u, pm.cams[cam]['K'], None)
            n = cv2.Rodrigues(rv)[0][:, 2]
            slant.append((float(np.degrees(np.arccos(abs(n[2])))), float(np.degrees(np.arctan2(n[0], n[2]))),
                          float(np.degrees(np.arctan2(n[1], n[2]))), float(tv[2])))
        cv2.imwrite(f'runs/side_planes/{cam}_{a.leg}.png', img)
    if not Hs:
        print(f'{cam}: board not found for leg {a.leg}')
        continue
    i = int(np.argmin(res))
    pm.planes[f'{cam}:{a.leg}'] = {'H': Hs[i], 'rms_cm': round(res[i], 3), 'frames': len(Hs),
                                   'created': time.strftime('%Y-%m-%d %H:%M')}
    msg = f'{cam}:{a.leg} saved  frames {len(Hs)}/{len(frames)}  fit {res[i]:.2f} cm'
    if slant:
        s = np.median(np.array(slant), axis=0)
        msg += f'  slant {s[0]:.0f} deg (left/right {s[1]:+.0f}, up/down {s[2]:+.0f})  board {s[3]/10:.0f} cm away'
    print(msg)
pm.save(a.out)
print('saved', a.out)
