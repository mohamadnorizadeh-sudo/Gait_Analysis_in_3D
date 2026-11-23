"""Synthetic fisheye stereo calibration: known K, D, baseline -> recovered."""

import cv2
import numpy as np

from gait3d.calibration import object_points, stereo_calibrate
from gait3d.stereo_geometry import StereoGeometry


def test_synthetic_stereo_calibration():
    rng = np.random.default_rng(0)
    size = (640, 480)
    K = np.array([[290.0, 0, 322], [0, 290.0, 238], [0, 0, 1]])
    D = np.array([[0.05], [-0.02], [0.01], [-0.003]])
    R_lr = cv2.Rodrigues(np.array([0.01, -0.02, 0.005]))[0]
    T_lr = np.array([[-62.0], [0.5], [0.3]])
    objp = object_points(9, 6, 25.0)
    centre = objp[0].mean(axis=0)

    obj, il, ir = [], [], []
    while len(obj) < 30:
        rvec = rng.normal(0, 0.4, 3)
        tvec = np.array([rng.uniform(-150, 150), rng.uniform(-100, 100), rng.uniform(300, 700)])
        Rb = cv2.Rodrigues(rvec)[0]
        tb = tvec - Rb @ centre
        pts = objp.reshape(-1, 1, 3)
        pl, _ = cv2.fisheye.projectPoints(pts, cv2.Rodrigues(Rb)[0], tb.reshape(3, 1), K, D)
        Rr, tr = R_lr @ Rb, R_lr @ tb.reshape(3, 1) + T_lr
        pr, _ = cv2.fisheye.projectPoints(pts, cv2.Rodrigues(Rr)[0], tr, K, D)
        pl, pr = pl.reshape(-1, 2), pr.reshape(-1, 2)
        inside = lambda p: np.all((p > 5) & (p < np.array(size) - 5))
        if not (inside(pl) and inside(pr)):
            continue
        obj.append(objp.copy())
        il.append((pl + rng.normal(0, 0.1, pl.shape)).reshape(1, -1, 2))
        ir.append((pr + rng.normal(0, 0.1, pr.shape)).reshape(1, -1, 2))

    data, rep = stereo_calibrate(obj, il, ir, size)
    assert abs(rep["baseline_mm"] - np.linalg.norm(T_lr)) < 1.0
    assert rep["epipolar_mean_px"] < 0.3
    geo = StereoGeometry(data["K_l"], data["D_l"], data["R1"], data["P1"], data["K_r"], data["D_r"],
                         data["R2"], data["P2"], data["Q"], size)
    assert abs(geo.doffs) < 1e-6          # CALIB_ZERO_DISPARITY
    assert abs(geo.baseline_mm - np.linalg.norm(T_lr)) < 1.0
