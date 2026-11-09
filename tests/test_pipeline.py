"""End-to-end smoke test of scripts/04_run_tracker.py with a simulated rig.

The Picamera2 cameras and YOLO are replaced by fakes driven by a synthetic
subject: 2 s standing still (bone calibration), then a walk where the leg
twists +/-35 deg while the knee angle follows a known curve.  Halfway through
the right leg is occluded and YOLO "ghosts" it onto the left leg.
"""

import csv
import os
import runpy
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from test_rigid_leg import L_SHIN, L_THIGH, make_geo, rot  # noqa: E402

FPS = 30.0


class _T:                       # mimics a torch tensor (.cpu().numpy())
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def __len__(self):
        return len(self.a)


class _Res:
    def __init__(self, xy, conf):
        self.keypoints = type("K", (), {"xy": _T(xy[None]), "conf": _T(conf[None])})()
        self.boxes = type("B", (), {"conf": _T([0.9])})()


def knee_truth(t):
    return 180.0 if t < 2.5 else 180.0 - (35 + 25 * np.sin(2 * np.pi * 0.8 * t))


def twist(t):
    return 0.0 if t < 2.5 else 35.0 * np.sin(2 * np.pi * 0.3 * t)


def leg(hip, ang, R):
    flex = np.radians(180 - ang)
    knee = hip + R @ np.array([0, L_THIGH, 0.0])
    ankle = knee + R @ (np.array([0, np.cos(flex), -np.sin(flex)]) * L_SHIN)
    return hip, knee, ankle


class Sim:
    geo = make_geo()
    rng = np.random.default_rng(0)
    frame = 0

    @classmethod
    def keypoints(cls, t):
        xy_l, xy_r, c = np.zeros((17, 2)), np.zeros((17, 2)), np.zeros(17)
        ang, R = knee_truth(t), rot((0, 1, 0), twist(t)) @ rot((1, 0, 0), 10)
        occluded = 6.0 < t < 7.0
        for side, hip, idx in (("L", np.array([-90.0, -380, 800]), (11, 13, 15)),
                               ("R", np.array([90.0, -380, 800]), (12, 14, 16))):
            for P, k in zip(leg(hip, ang if side == "L" else 175.0, R if side == "L" else np.eye(3)), idx):
                p = cls.geo.project_left(P) + cls.rng.normal(0, 0.7, 2)
                d = cls.geo.f * cls.geo.baseline_mm / P[2]
                xy_l[k], xy_r[k] = p, (p[0] - d + cls.rng.normal(0, 0.7), p[1] + cls.rng.normal(0, 0.5))
                c[k] = 0.9
        if occluded:      # right leg hidden -> YOLO puts the R labels on the left limb (ghost)
            for a, b in ((11, 12), (13, 14), (15, 16)):
                xy_l[b] = xy_l[a] + (6, 4)
                xy_r[b] = xy_r[a] + (6, 4)
                c[b] = 0.6
        return xy_l, xy_r, c


class FakeCam:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def read(self):
        Sim.frame += 1
        img = np.zeros((480, 640, 3), np.uint8)
        return Sim.frame / FPS, img, img, 1.0


class FakePose:
    def __init__(self, *a, **k):
        pass

    def __call__(self, images):
        xy_l, xy_r, c = Sim.keypoints(Sim.frame / FPS)
        return [_Res(xy_l, c), _Res(xy_r, c)]


def test_end_to_end(tmp_path, monkeypatch):
    geo = make_geo()
    calib = tmp_path / "calib.npz"
    np.savez(calib, K_l=geo.K_l, D_l=geo.D_l, R1=geo.R1, P1=geo.P1, K_r=geo.K_r, D_r=geo.D_r,
             R2=geo.R2, P2=geo.P2, Q=geo.Q, image_size=np.array([640, 480]))
    import gait3d.camera
    import gait3d.pose
    monkeypatch.setattr(gait3d.camera, "StereoCamera", FakeCam)
    monkeypatch.setattr(gait3d.pose, "PoseEstimator", FakePose)
    out_csv = tmp_path / "walk.csv"
    prof = tmp_path / "subject.json"
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "04_run_tracker.py")
    monkeypatch.setattr(sys, "argv", [script, "--calib", str(calib), "--headless", "--seconds", "1e9",
                                      "--csv", str(out_csv), "--profile-out", str(prof)])
    Sim.frame = 0
    # stop after 10 simulated seconds
    orig_read = FakeCam.read

    def limited(self):
        if Sim.frame >= int(10 * FPS):
            raise KeyboardInterrupt
        return orig_read(self)
    monkeypatch.setattr(FakeCam, "read", limited)
    try:
        runpy.run_path(script, run_name="__main__")
    except KeyboardInterrupt:
        pass

    assert prof.exists()
    import json
    p = json.load(open(prof))
    assert abs(p["L_thigh_L"] - L_THIGH) < 15 and abs(p["L_shin_L"] - L_SHIN) < 15

    rows = list(csv.DictReader(open(out_csv)))
    left = [r for r in rows if r["side"] == "L"]
    t = np.array([float(r["t"]) for r in left])
    est = np.array([float(r["knee_angle_deg"]) for r in left])
    filt = 180 - np.array([float(r["flexion_filtered_deg"]) for r in left])
    truth = np.array([knee_truth(x) for x in t])
    err = np.abs(est - truth)
    assert np.median(err) < 3.0, np.median(err)
    assert np.median(np.abs(filt - truth)) < 4.0
    # bone lengths are constants
    assert np.allclose([float(r["thigh_mm"]) for r in left], p["L_thigh_L"], atol=1.0)
    # during the ghost window only ONE leg is reported
    ghost = [r for r in rows if 6.05 < float(r["t"]) < 6.95]
    ts = {}
    for r in ghost:
        ts.setdefault(r["t"], []).append(r["side"])
    assert ts and all(len(v) == 1 for v in ts.values())
    print(f"\nend-to-end: {len(left)} left frames, knee angle error median {np.median(err):.2f} deg, "
          f"p95 {np.percentile(err, 95):.2f} deg (twist +/-35 deg)")
