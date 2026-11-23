"""v2.2: converging (toed-in) rig like the walker: ~0.8 m baseline, cameras turned
~83 deg toward the legs.  No rectification; keypoints undistorted + triangulated."""

import cv2
import numpy as np

from gait3d.rigid_leg import RigidLegSolver, joint_angle_deg
from gait3d.stereo_geometry import RawStereoGeometry, load_geometry

L_THIGH, L_SHIN = 420.0, 400.0


def make_raw():
    K = np.array([[240.0, 0, 180], [0, 240.0, 320], [0, 0, 1]])
    D = np.array([0.05, -0.01, 0.005, -0.001])
    # left camera at origin looking +z; right camera 800 mm to the right, turned 83 deg toward it
    ang = np.radians(83)
    R_wr = cv2.Rodrigues(np.array([0, ang, 0]))[0]        # right camera orientation (world->cam rotation)
    C_r = np.array([800.0, 0, 0])
    T = (-R_wr @ C_r).reshape(3, 1)
    return RawStereoGeometry(K, D, K, D, R_wr, T, (360, 640))


def rot(axis, deg):
    return cv2.Rodrigues(np.asarray(axis, float) / np.linalg.norm(axis) * np.radians(deg))[0]


def leg(angle, R, hip=np.array([250.0, -350, 600])):
    flex = np.radians(180 - angle)
    knee = hip + R @ np.array([0, L_THIGH, 0.0])
    ankle = knee + R @ (np.array([0, np.cos(flex), -np.sin(flex)]) * L_SHIN)
    return hip, knee, ankle


def test_geometry_basics():
    g = make_raw()
    assert abs(g.baseline_mm - 800) < 1e-6 and abs(g.convergence_deg - 83) < 1e-6
    X = np.array([300.0, 0, 550])
    pl, pr = g.project_left(X), g.project_right(X)
    assert np.allclose(g.triangulate(*pl, *pr), X, atol=1e-3)
    assert g.ray_gap_mm(*pl, *pr) < 1e-3
    assert g.ray_gap_mm(*pl, *(pr + [0, 40])) > 20          # wrong match -> large gap


def test_raw_rotation_invariance_with_noise():
    g, rng = make_raw(), np.random.default_rng(0)
    errs = []
    for deg in range(-35, 36, 5):
        joints = leg(130.0, rot((0, 1, 0), deg) @ rot((1, 0, 0), 10))
        rays, zs = {}, {}
        for n, X in zip(("hip", "knee", "ankle"), joints):
            pl = g.project_left(X) + rng.normal(0, 0.7, 2)
            pr = g.project_right(X) + rng.normal(0, 0.7, 2)
            rays[n] = g.ray(*pl)
            zs[n] = g.triangulate(*pl, *pr)[2]
        res = RigidLegSolver(L_THIGH, L_SHIN, depth_sigma_fn=g.depth_sigma).solve(rays, zs)
        assert res is not None and abs(res.thigh_mm - L_THIGH) < 2
        errs.append(abs(res.angle_deg - 130.0))
    assert np.median(errs) < 2.5 and max(errs) < 10, errs   # single frames, before One-Euro filtering


def test_load_geometry_picks_raw(tmp_path):
    g = make_raw()
    p = tmp_path / "c.npz"
    np.savez(p, K_l=g.K_l, D_l=g.D_l, K_r=g.K_r, D_r=g.D_r, R=g.R, T=g.T, image_size=np.array([360, 640]),
             R1=np.eye(3), R2=np.eye(3), P1=np.zeros((3, 4)), P2=np.zeros((3, 4)), Q=np.zeros((4, 4)))
    assert isinstance(load_geometry(str(p)), RawStereoGeometry)


def test_hip_seen_only_by_right_camera():
    """v2.3: on the walker cam0 barely sees the hips; the hip ray comes from cam1
    and its depth from the thigh length -- still a full 3D angle, never 2D."""
    g, rng = make_raw(), np.random.default_rng(1)
    errs = []
    for deg in range(-30, 31, 10):
        joints = leg(125.0, rot((0, 1, 0), deg))
        H, K, A = joints
        rays, zs = {}, {}
        hr = g.project_right(H) + rng.normal(0, 0.7, 2)
        rays["hip"] = (g.C_r, g.ray_right_in_left(*hr))
        zs["hip"] = None
        hint = {"hip": g.ray(*(g.project_left(H) + rng.normal(0, 15, 2)))}   # cam0's low-confidence guess
        for n, X in (("knee", K), ("ankle", A)):
            pl = g.project_left(X) + rng.normal(0, 0.7, 2)
            pr = g.project_right(X) + rng.normal(0, 0.7, 2)
            rays[n] = g.ray(*pl)
            zs[n] = g.triangulate(*pl, *pr)[2]
        res = RigidLegSolver(L_THIGH, L_SHIN, depth_sigma_fn=g.depth_sigma).solve(rays, zs, hint)
        assert res is not None and res.source["hip"] == "constrained"
        assert abs(res.thigh_mm - L_THIGH) < 2 and np.linalg.norm(res.hip - H) < 60
        errs.append(abs(res.angle_deg - 125.0))
    assert np.median(errs) < 3.0, errs
