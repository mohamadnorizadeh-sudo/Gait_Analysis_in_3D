"""Fisheye stereo calibration (v2).

Issues fixed versus notebook cell 4 (v1)
----------------------------------------
* v1 ran ``cv2.fisheye.stereoCalibrate`` directly with zero-initialised
  intrinsics.  v2 first calibrates each camera on its OWN detections
  (more views per camera, better K/D), automatically removes views that make
  CALIB_CHECK_COND fail, then runs the stereo solve with CALIB_FIX_INTRINSIC
  so only R/T are estimated.
* A low RMS on few views is not proof of accuracy.  v2 reports
  - per-camera RMS, stereo RMS, number of views used
  - baseline |T| in mm  -> compare with a ruler measurement of the rig!
  - rectified epipolar error (mean |y_left - y_right| of matched corners);
    should be < 0.5 px.  This is what actually matters for depth.
  - the closest depth the chosen SGBM disparity range can measure.
* ``stereoRectify`` uses CALIB_ZERO_DISPARITY so both rectified principal
  points coincide (v1 passed flags=0).  ``balance`` / ``fov_scale`` are
  exposed: balance=0 crops to valid pixels, 1 keeps the full 120 deg FOV.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

SUBPIX = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6)


def _flag(name):
    """OpenCV 4.x keeps fisheye flags in cv2.fisheye (Pi OS); OpenCV 5 moved them to cv2."""
    return getattr(cv2.fisheye, name, None) if hasattr(cv2.fisheye, name) else getattr(cv2, name)


MONO_FLAGS = _flag("CALIB_RECOMPUTE_EXTRINSIC") | _flag("CALIB_CHECK_COND") | _flag("CALIB_FIX_SKEW")
FIX_INTRINSIC = _flag("CALIB_FIX_INTRINSIC")


def object_points(cols, rows, square_mm):
    objp = np.zeros((1, cols * rows, 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return objp


def _mono(obj, img, size, label, max_view_px=2.0):
    """fisheye.calibrate with automatic removal of ill-conditioned AND outlier views.

    v2.2: one bad detection (e.g. a board whose corner order flipped) gave the
    right camera RMS 8.5 px on the walker; views whose own reprojection error is
    above max(max_view_px, 3x median) are now dropped and the camera re-solved.
    """
    keep = list(range(len(obj)))
    for _round in range(4):
        while True:
            if len(keep) < 6:
                raise RuntimeError(f"[{label}] fewer than 6 usable views - capture more calibration views")
            K, D = np.zeros((3, 3)), np.zeros((4, 1))
            try:
                rms, K, D, rv, tv = cv2.fisheye.calibrate(
                    [obj[i] for i in keep], [img[i] for i in keep], size, K, D,
                    flags=MONO_FLAGS, criteria=SUBPIX)
                break
            except cv2.error as e:
                m = re.search(r"input array (\d+)", str(e))
                if not m:
                    raise
                bad = keep[int(m.group(1))]
                print(f"[{label}] dropping ill-conditioned view {bad}")
                keep.remove(bad)
        errs = []
        for i, r, t in zip(keep, rv, tv):
            p, _ = cv2.fisheye.projectPoints(obj[i].reshape(-1, 1, 3), r, t, K, D)
            errs.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - img[i].reshape(-1, 2)) ** 2, 1)))))
        errs = np.array(errs)
        lim = max(max_view_px, 3.0 * float(np.median(errs)))
        bad = [keep[j] for j in np.where(errs > lim)[0]]
        if not bad:
            return rms, K, D, keep
        print(f"[{label}] dropping {len(bad)} outlier view(s) {bad} (error > {lim:.2f} px)")
        keep = [i for i in keep if i not in bad]
    return rms, K, D, keep


def stereo_calibrate(obj, img_l, img_r, size, balance=0.0, fov_scale=1.0,
                     mono_l=None, mono_r=None, square_mm=None, max_rectify_deg=20.0):
    """obj/img_*: lists of (1,N,3)/(1,N,2) float64 arrays (views seen by BOTH cameras).

    mono_l / mono_r (v2.2): optional (obj_list, img_list) of views seen by ONE
    camera.  On a converging rig the shared region is small, so intrinsics are
    calibrated on these whole-field views plus the pair views; the stereo solve
    then only estimates R, T (CALIB_FIX_INTRINSIC).
    """
    ol = list(obj) + (list(mono_l[0]) if mono_l else [])
    pl = list(img_l) + (list(mono_l[1]) if mono_l else [])
    orr = list(obj) + (list(mono_r[0]) if mono_r else [])
    pr = list(img_r) + (list(mono_r[1]) if mono_r else [])
    rms_l, K_l, D_l, keep_l = _mono(ol, pl, size, "left")
    rms_r, K_r, D_r, keep_r = _mono(orr, pr, size, "right")
    n = len(obj)
    keep = sorted(set(i for i in keep_l if i < n) & set(i for i in keep_r if i < n))
    o = [obj[i] for i in keep]
    il = [img_l[i] for i in keep]
    ir = [img_r[i] for i in keep]
    res = cv2.fisheye.stereoCalibrate(
        o, il, ir, K_l, D_l, K_r, D_r, size,
        flags=FIX_INTRINSIC, criteria=SUBPIX)
    rms, R, T = res[0], res[5], res[6]          # works for OpenCV 4.x signatures
    conv = float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))

    report = {
        "views_pairs": n, "views_pairs_used": len(keep),
        "views_mono_left": len(ol) - n, "views_mono_right": len(orr) - n,
        "rms_left": float(rms_l), "rms_right": float(rms_r), "rms_stereo": float(rms),
        "baseline_mm": float(np.linalg.norm(T)), "convergence_deg": conv,
        "fx_fy_left": [float(K_l[0, 0]), float(K_l[1, 1])], "fx_fy_right": [float(K_r[0, 0]), float(K_r[1, 1])],
    }
    data = dict(K_l=K_l, D_l=D_l, K_r=K_r, D_r=D_r, R=R, T=T, image_size=np.array(size))

    # Rectification only makes sense for a (nearly) parallel rig.
    if conv <= max_rectify_deg:
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            K_l, D_l, K_r, D_r, size, R, T, cv2.CALIB_ZERO_DISPARITY,
            newImageSize=size, balance=balance, fov_scale=fov_scale)
        dy = []
        for a, b in zip(il, ir):
            ul = cv2.fisheye.undistortPoints(a.reshape(-1, 1, 2), K_l, D_l, R=R1, P=P1)
            ur = cv2.fisheye.undistortPoints(b.reshape(-1, 1, 2), K_r, D_r, R=R2, P=P2)
            dy.append(np.abs(ul[:, 0, 1] - ur[:, 0, 1]))
        dy = np.concatenate(dy)
        report.update(epipolar_mean_px=float(dy.mean()), epipolar_p95_px=float(np.percentile(dy, 95)),
                      rectified_f_px=float(P1[0, 0]), mode="rectified")
        data.update(R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)
    else:
        report["mode"] = "raw (converging rig, no rectification)"
        # keep the v1 keys present so old loaders do not crash
        eye = np.eye(3)
        data.update(R1=eye, R2=eye, P1=np.zeros((3, 4)), P2=np.zeros((3, 4)), Q=np.zeros((4, 4)))

    # 3D validation: triangulate every shared view and measure the board.
    from .stereo_geometry import RawStereoGeometry
    geo = RawStereoGeometry(K_l, D_l, K_r, D_r, R, T, size)
    sq, gaps, flat = [], [], []
    cols = None
    for ob, a, b in zip(o, il, ir):
        P = np.array([geo.triangulate(*pa, *pb) for pa, pb in zip(a.reshape(-1, 2), b.reshape(-1, 2))])
        gaps += [geo.ray_gap_mm(*pa, *pb) for pa, pb in zip(a.reshape(-1, 2)[::7], b.reshape(-1, 2)[::7])]
        if cols is None:
            xs = ob.reshape(-1, 3)[:, 0]
            cols = int(np.sum(np.isclose(ob.reshape(-1, 3)[:, 1], ob.reshape(-1, 3)[0, 1])))
        G = P.reshape(-1, cols, 3)
        sq += list(np.linalg.norm(np.diff(G, axis=1), axis=2).ravel())
        sq += list(np.linalg.norm(np.diff(G, axis=0), axis=2).ravel())
        c = P.mean(axis=0)
        _, sv, _ = np.linalg.svd(P - c)
        flat.append(float(sv[-1] / np.sqrt(len(P))))
    sq = np.array(sq)
    sq = sq[np.isfinite(sq)]
    report.update(board_square_3d_mean_mm=float(sq.mean()), board_square_3d_std_mm=float(sq.std()),
                  board_flatness_rms_mm=float(np.median(flat)), ray_gap_median_mm=float(np.median(gaps)))
    if square_mm:
        report["board_scale_error_pct"] = float(100 * (sq.mean() / square_mm - 1))
    return data, report
