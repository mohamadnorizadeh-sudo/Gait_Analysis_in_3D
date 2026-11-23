"""v3.2.0 synthetic check of scripts/stereo_sagittal_calib.py.

Renders a 9x6 board seen by two fisheye cameras placed front-diagonally (like the walker cameras),
runs the calibration offline, then projects known 3D knee angles in each leg's sagittal plane and
compares the plane-mapped 2D flexion with the true angle.

  python3 scripts/check_stereo_sagittal_synth.py [board_yaw_deg] [right_camera_forward_cm]
"""
import json, os, subprocess, sys, tempfile
import cv2
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = tempfile.mkdtemp(prefix='stereo_synth_')
yaw_board = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0      # board turned this much off straight ahead
asym_fwd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0       # right camera this much further forward (cm)
K = np.array([[260.0, 0, 180], [0, 260.0, 320], [0, 0, 1]])
D = np.array([0.03, -0.01, 0.004, -0.001]).reshape(4, 1)
W, H = 360, 640
UP = np.array([0, 1.0, 0])                                         # walker frame: X user's left, Y up, Z forward


def look(C, target):
    z = target - C
    z /= np.linalg.norm(z)
    y = -UP - (-UP @ z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.vstack([x, y, z])                                       # world -> camera
    return R, -R @ C


def rot_y(a):
    a = np.radians(a)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def rot_x(a):
    a = np.radians(a)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


cams = {'cam0': look(np.array([32.0, 65, 45]), np.array([6.0, 50, 0])),
        'cam1': look(np.array([-32.0, 65, 45 + asym_fwd]), np.array([-6.0, 50, 0]))}
Rb = rot_y(yaw_board) @ rot_x(-8)                                  # board axes (columns) in world
Pb = np.array([0, 52.0, 14])                                       # board centre: knee height, in front of knees

uu, vv = np.meshgrid(np.arange(W), np.arange(H))
pix = np.stack([uu.ravel(), vv.ravel()], 1).astype(float).reshape(-1, 1, 2)


def render(cam, rng):
    R, t = cams[cam]
    nrm = cv2.fisheye.undistortPoints(pix, K, D).reshape(-1, 2)
    d = np.column_stack([nrm, np.ones(len(nrm))]) @ R              # rays in world (R^T d)
    C = -R.T @ t
    o = Rb.T @ (C - Pb)
    dd = d @ Rb
    s = -o[2] / dd[:, 2]
    q = o[None, :2] + s[:, None] * dd[:, :2]
    img = np.full(len(q), 110.0)
    hit = (s > 0) & (np.abs(q[:, 0]) < 8.75 + 2.5) & (np.abs(q[:, 1]) < 12.5 + 2.5)
    pat = hit & (np.abs(q[:, 0]) < 8.75) & (np.abs(q[:, 1]) < 12.5)
    img[hit] = 235
    i = np.floor((q[pat, 0] + 8.75) / 2.5).astype(int)
    j = np.floor((q[pat, 1] + 12.5) / 2.5).astype(int)
    img[pat] = np.where((i + j) % 2 == 0, 25, 235)
    img = img.reshape(H, W)
    img = cv2.GaussianBlur(img, (0, 0), 0.8) + rng.normal(0, 3, img.shape)
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


rng = np.random.default_rng(1)
os.makedirs(f'{OUT}/frames', exist_ok=True)
paths = []
for k in range(4):
    f = np.hstack([render('cam0', rng), render('cam1', rng)])
    p = f'{OUT}/frames/f{k:02d}.png'
    cv2.imwrite(p, f)
    paths.append(p)
cfg = f'{OUT}/sp.json'
json.dump({'image_size': [W, H], 'cams': {c: {'K': K.tolist(), 'D': D.ravel().tolist()} for c in cams}, 'planes': {}},
          open(cfg, 'w'))
for normal in ('baseline', 'board', 'mean'):
    r = subprocess.run([sys.executable, f'{REPO}/scripts/stereo_sagittal_calib.py', '--images', *paths, '--out', cfg,
                        '--stereo-out', f'{OUT}/st.json', '--normal', normal, '--save'], capture_output=True, text=True, cwd=REPO)
    print(f'===== normal={normal}  (board yaw {yaw_board}, right cam +{asym_fwd} cm forward)')
    print('\n'.join(l for l in r.stdout.splitlines() if not l.startswith('f0')), r.stderr[-500:])
    sp = json.load(open(cfg))

    def flex2d(Pp):
        a, b = Pp[0] - Pp[1], Pp[2] - Pp[1]
        return 180 - np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))

    for cam, leg, x in (('cam0', 'L', 10.0), ('cam1', 'R', -10.0), ('cam1', 'L', 10.0), ('cam0', 'R', -10.0)):
        Hm = np.array(sp['planes'][f'{cam}:{leg}']['H'])
        R, t = cams[cam]
        errs = []
        for true in (0, 20, 40, 60):
            for thigh in (-20, 0, 25):                              # thigh angle from vertical (+ = forward)
                hip = np.array([x, 90.0, 0])
                knee = hip + 42 * np.array([0, -np.cos(np.radians(thigh)), np.sin(np.radians(thigh))])
                sh = thigh - true
                ank = knee + 40 * np.array([0, -np.cos(np.radians(sh)), np.sin(np.radians(sh))])
                P3 = np.array([hip, knee, ank])
                rv = cv2.Rodrigues(R)[0]
                px = cv2.fisheye.projectPoints(P3.reshape(-1, 1, 3), rv, t, K, D)[0].reshape(-1, 2)
                u = cv2.fisheye.undistortPoints(px.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
                q = np.column_stack([u, np.ones(3)]) @ Hm.T
                errs.append(flex2d(q[:, :2] / q[:, 2:3]) - true)
        print(f'  {cam}:{leg}  flexion error over 0-60 deg: max |err| {np.max(np.abs(errs)):.2f} deg')
