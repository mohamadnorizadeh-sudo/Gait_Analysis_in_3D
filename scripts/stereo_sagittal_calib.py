#!/usr/bin/env python3
"""v3.2.1 - sagittal planes from ONE board seen by BOTH cameras at the same time.

Why (22 Nov): the walker cameras see the legs front-diagonally. A board held flat on the
outside of the leg is then too slanted to detect at 360x640, and a board turned toward the
camera (slant 11-22 deg) is not the plane the knee bends in, so walking angles read about
half of the reference (11:15 walking trial).

How: hold the 9x6 board upright in front of the knees, facing straight ahead, so that both
cameras see it. solvePnP in each camera gives both camera positions in board coordinates.
Two estimates of the walker's left-right axis (= the normal of the sagittal plane) follow:
  baseline - the line between the two cameras (left and right rear posts),
  board    - the board's own left-right edge (if it faced straight ahead).
Their angle is printed as a cross-check; the plane normal is their mean by default.
Each camera then gets the homography of a plane with that normal beside each leg. Any
plane parallel to the knee's plane gives the same angles (central projection between
parallel planes is a uniform scaling), so the board does not have to touch the leg.

  python3 scripts/stereo_sagittal_calib.py            # capture ~3 s, report, save nothing
  python3 scripts/stereo_sagittal_calib.py --save     # also write cam0:L cam0:R cam1:L cam1:R
  python3 scripts/stereo_sagittal_calib.py --images runs/stereo_calib/<stamp>/f*.png [--save]

Frames are kept in runs/stereo_calib/<stamp>/ (they show the user: never commit them).
The camera poses go to config/stereo_board.json (projection matrices for 3D triangulation).
"""
import argparse, json, os, shutil, sys, time
import cv2
import numpy as np
cv2.setNumThreads(1)   # see side_plane_calib.py v3.0.1: full-CPU corner search dropped the Pi off the network
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.side2d import PlaneMap

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='config/side_planes.json')
ap.add_argument('--stereo-out', default='config/stereo_board.json')
ap.add_argument('--square-mm', type=float, default=25.0)
ap.add_argument('--frames', type=int, default=12)
ap.add_argument('--interval', type=float, default=0.25)
ap.add_argument('--normal', choices=['mean', 'baseline', 'board'], default='mean')
ap.add_argument('--leg-offset-cm', type=float, default=10.0, help='plane offset from the board centre toward each leg (sets the cm scale only)')
ap.add_argument('--save', action='store_true')
ap.add_argument('--images', nargs='+', help='offline: saved cam0|cam1 frames (side by side)')
a = ap.parse_args()

pm = PlaneMap.load(a.out)
if not all(c in pm.cams for c in ('cam0', 'cam1')):
    sys.exit(f'{a.out} has no intrinsics for cam0/cam1 (run side_plane_calib.py once first)')
W = int(pm.image_size[0])
CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
stamp = time.strftime('%Y%m%d_%H%M%S')

# ------------------------------------------------------------------ frames
if a.images:
    frames = [cv2.imread(p) for p in sorted(a.images)]
else:
    from gait3d.camera import ShmStereoSource
    frames = []
    with ShmStereoSource() as src:
        for _ in range(a.frames):
            _, i0, i1, _ = src.read()
            frames.append(np.hstack([i0, i1]))
            time.sleep(a.interval)
    d = f'runs/stereo_calib/{stamp}'
    os.makedirs(d, exist_ok=True)
    for k, f in enumerate(frames):
        cv2.imwrite(f'{d}/f{k:02d}.png', f)
    print(f'captured {len(frames)} frames -> {d}  (you can move now)', flush=True)


CLAHE = cv2.createCLAHE(3.0, (6, 9))
CLASSIC = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


def find(gray):
    """(corners Nx2 in grid order, grid Nx2 in cm, method) or (None, None, None).

    v3.2.1: the board faces both cameras at ~45-50 deg, so its squares are only ~6-11 px wide and the right
    camera saw it in shadow against a bright window. Plain detection found it in 1 of 24 views (12:52); a 2x
    upscale + CLAHE contrast found it in 11/11 right-camera frames. Order: plain -> 2x CLAHE classic ->
    2x CLAHE sector-based (SB) -> 3x CLAHE SB."""
    size = cn = None
    for sz in ((9, 6), (6, 9)):
        ok, c = cv2.findChessboardCorners(gray, sz, CLASSIC | cv2.CALIB_CB_FAST_CHECK)
        if ok:
            size, cn, how = sz, cv2.cornerSubPix(gray, c, (3, 3), (-1, -1), CRIT), 'plain'
            break
    for s, sb in ((2, False), (2, True), (3, True)):
        if cn is not None:
            break
        u = CLAHE.apply(cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC))
        for sz in ((9, 6), (6, 9)):
            if sb:
                ok, c = cv2.findChessboardCornersSB(u, sz, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY)
            else:
                ok, c = cv2.findChessboardCorners(u, sz, CLASSIC)
                if ok:
                    c = cv2.cornerSubPix(u, c, (5, 5), (-1, -1), CRIT)
            if ok:
                # pixel centres: x_orig = (x_up + 0.5) / s - 0.5
                size, cn, how = sz, (c.astype(float) + 0.5) / s - 0.5, f'{s}x{" SB" if sb else ""}'
                break
    if cn is None:
        return None, None, None
    cols, rows = size
    g = np.array([[x, y] for y in range(rows) for x in range(cols)], float) * a.square_mm / 10.0
    return cn.reshape(-1, 2).astype(float), g, how


def pose(cam, cn, g):
    """Board pose in a canonical frame that does not depend on the detector's corner order:
    origin = pattern centre, Z = normal toward the camera, Y = board edge closest to image-up,
    X = Y x Z.  Returns (R board->camera, t cm, reprojection rms px, distance cm, slant deg)."""
    K = pm.cams[cam]['K']
    u = pm.undistort(cam, cn)
    obj = np.column_stack([g, np.zeros(len(g))])
    ok, rv, tv = cv2.solvePnP(obj, u, K, None, flags=cv2.SOLVEPNP_IPPE)
    ok, rv, tv = cv2.solvePnP(obj, u, K, None, rv, tv, True, cv2.SOLVEPNP_ITERATIVE)
    R, t = cv2.Rodrigues(rv)[0], tv.ravel()
    pr = cv2.projectPoints(obj, rv, tv, K, None)[0].reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((pr - u) ** 2, 1))))
    tc = R @ obj.mean(0) + t
    Z = R[:, 2] if R[:, 2] @ -tc > 0 else -R[:, 2]
    Y = max([R[:, 0], -R[:, 0], R[:, 1], -R[:, 1]], key=lambda v: -v[1])   # image up = -y
    Y = Y - (Y @ Z) * Z
    Y /= np.linalg.norm(Y)
    Rc = np.column_stack([np.cross(Y, Z), Y, Z])
    slant = float(np.degrees(np.arccos(min(1.0, abs(Z @ tc) / np.linalg.norm(tc)))))
    return Rc, tc, rms, float(np.linalg.norm(tc)), slant


def ang(u, v):
    """angle between two LINES (deg)"""
    return float(np.degrees(np.arccos(min(1.0, abs(u @ v) / np.linalg.norm(u) / np.linalg.norm(v)))))


both = []
for k, f in enumerate(frames):
    r, msg = {}, []
    for ci, cam in enumerate(('cam0', 'cam1')):
        cn, g, how = find(cv2.cvtColor(f[:, ci * W:(ci + 1) * W], cv2.COLOR_BGR2GRAY))
        if cn is None:
            msg.append(f'{cam} --')
            continue
        r[cam] = pose(cam, cn, g)
        msg.append(f'{cam} ok {how} (slant {r[cam][4]:.0f}, {r[cam][3]:.0f} cm, rms {r[cam][2]:.2f} px)')
    print(f'f{k:02d}  ' + '   '.join(msg), flush=True)
    if len(r) == 2:
        both.append(r)

print(f'board seen by both cameras in {len(both)}/{len(frames)} frames')
if len(both) < 3:
    sys.exit('not enough: hold the board upright in front of the knees, facing straight ahead, fingers only on the '
             'white border, so that BOTH camera views show the whole pattern; keep still ~3 s')

# ------------------------------------------------------------------ geometry (board frame, cm)
C0 = np.array([-r['cam0'][0].T @ r['cam0'][1] for r in both])
C1 = np.array([-r['cam1'][0].T @ r['cam1'][1] for r in both])
B = C1 - C0                                     # cam0 (left) -> cam1 (right)
lb = B.mean(0) / np.linalg.norm(B.mean(0))
spread = max(ang(b, lb) for b in B)
d = B.mean(0)
X = np.array([1.0, 0, 0])
lx = X if X @ lb > 0 else -X                    # board left-right edge, oriented like the baseline
cross = ang(lb, lx)
n = {'baseline': lb, 'board': lx, 'mean': (lb + lx) / np.linalg.norm(lb + lx)}[a.normal]
print(f'camera baseline {np.linalg.norm(d):.1f} cm (frame-to-frame spread {spread:.1f} deg)')
print(f'board left-right edge vs camera baseline: {cross:.1f} deg  (small = board faced straight ahead and '
      f'cameras mounted symmetrically){"   <-- check: large" if cross > 8 else ""}')
print(f'sagittal normal = {a.normal}')

up = np.array([0, 1.0, 0])
planes = {}
for cam, k in (('cam0', 0), ('cam1', 1)):
    Rs = [r[cam][0] for r in both]
    ts = np.array([r[cam][1] for r in both])
    # average pose: mean rotation via SVD of the summed matrices
    U, _, Vt = np.linalg.svd(sum(Rs))
    Rm = U @ Vt
    tm = ts.mean(0)
    K = pm.cams[cam]['K']
    o = Rm.T @ np.array([0, 0, 1.0])            # optical axis in board frame
    view = ang(o, n)                            # 0 = looking straight along the knee axis (true side view)
    print(f'{cam}: views the knee plane {view:.0f} deg off a true side view')
    for leg, sgn in (('L', -1), ('R', +1)):
        p0 = sgn * a.leg_offset_cm * n
        e2 = -(up - (up @ n) * n)
        e2 /= np.linalg.norm(e2)                # down, in the plane
        e1 = np.cross(e2, n)
        G = K @ np.column_stack([Rm @ e1, Rm @ e2, Rm @ p0 + tm])
        Hm = np.linalg.inv(G)
        planes[f'{cam}:{leg}'] = {'H': Hm / Hm[2, 2], 'method': f'stereo-{a.normal}', 'frames': len(both),
                                  'view_deg': round(view, 1), 'board_vs_baseline_deg': round(cross, 1),
                                  'created': time.strftime('%Y-%m-%d %H:%M')}
    planes[cam] = (Rm, tm)

if not a.save:
    print('not saved (add --save to write the planes)')
    sys.exit(0)
bk = a.out.replace('.json', f'_{stamp}_pre.json')
if os.path.exists(a.out):
    shutil.copy(a.out, bk)
for key in [k for k in planes if ':' in k]:
    pm.planes[key] = planes[key]
pm.save(a.out)
st = {'created': time.strftime('%Y-%m-%d %H:%M'), 'units': 'cm, board frame: origin pattern centre, '
      'Y up the board, Z toward the cameras', 'normal': a.normal, 'n': n.tolist(),
      'baseline_dir': lb.tolist(), 'board_x': lx.tolist(), 'frames': len(both),
      'cams': {cam: {'R': planes[cam][0].tolist(), 't': planes[cam][1].tolist()} for cam in ('cam0', 'cam1')}}
with open(a.stereo_out, 'w') as fh:
    json.dump(st, fh, indent=1)
print(f'saved cam0:L cam0:R cam1:L cam1:R -> {a.out} (backup {bk}); camera poses -> {a.stereo_out}')
