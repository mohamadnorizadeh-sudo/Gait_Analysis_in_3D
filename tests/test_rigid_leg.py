"""Synthetic tests: a rigid leg with a FIXED knee angle is rotated/twisted in
front of a simulated rectified stereo pair (same geometry as the Pi rig).
The measured angle must stay constant -- the v1 failure mode.

Run:  python -m pytest -q tests
"""

import numpy as np
import pytest

from gait3d.pose import LegObs, Person, legs_from_people, suppress_ghost_legs
from gait3d.profile import BoneLengthCalibrator
from gait3d.rigid_leg import (RigidLegSolver, joint_angle_deg, naive_rescale,
                              ray_sphere_depths)
from gait3d.stereo_geometry import StereoGeometry, sample_sgbm_point

F, B, CX, CY = 300.0, 60.0, 320.0, 240.0
L_THIGH, L_SHIN = 420.0, 400.0


def make_geo():
    P1 = np.array([[F, 0, CX, 0], [0, F, CY, 0], [0, 0, 1, 0]], float)
    P2 = np.array([[F, 0, CX, -F * B], [0, F, CY, 0], [0, 0, 1, 0]], float)
    Tx = -B
    Q = np.array([[1, 0, 0, -CX], [0, 1, 0, -CY], [0, 0, 0, F], [0, 0, -1 / Tx, 0]], float)
    I = np.eye(3)
    return StereoGeometry(I, np.zeros(4), I, P1, I, np.zeros(4), I, P2, Q, (640, 480))


def rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def leg_pose(knee_angle_deg, R, hip=np.array([0.0, -250.0, 700.0])):
    """Leg hanging down (+y) with the given interior knee angle, rigidly rotated by R about the hip."""
    thigh = np.array([0.0, 1.0, 0.0]) * L_THIGH
    flex = np.radians(180 - knee_angle_deg)
    shin = np.array([0.0, np.cos(flex), -np.sin(flex)]) * L_SHIN   # knee bends toward camera
    knee = hip + R @ thigh
    ankle = knee + R @ shin
    return hip, knee, ankle


def observe(geo, P, rng, px_noise):
    xl = geo.project_left(P) + rng.normal(0, px_noise, 2)
    d = F * B / P[2]
    xr = xl[0] - d + rng.normal(0, px_noise)
    return xl, xr


def run_solver(geo, joints, rng, px_noise=0.7, drop=(), corrupt=None, solver=None):
    solver = solver or RigidLegSolver(L_THIGH, L_SHIN, depth_sigma_fn=lambda z: geo.depth_sigma(z, 1.5))
    rays, zs = {}, {}
    for name, P in zip(("hip", "knee", "ankle"), joints):
        xl, xr = observe(geo, P, rng, px_noise)
        rays[name] = geo.ray(*xl)
        z = geo.triangulate(xl[0], xl[1], xr)[2]
        if name in drop:
            z = None
        if corrupt and name in corrupt:
            z = z + corrupt[name]
        zs[name] = z
    return solver.solve(rays, zs)


def test_naive_rescale_does_not_change_angle():
    """The hand-off formula H = K + L*u/|u| cannot fix a wrong angle."""
    H, K, A = np.array([10., -400, 600]), np.array([0., 0, 500]), np.array([30., 380, 650])
    before = joint_angle_deg(H, K, A)
    after = joint_angle_deg(naive_rescale(K, H, L_THIGH), K, naive_rescale(K, A, L_SHIN))
    assert abs(before - after) < 1e-9


def test_ray_sphere_exact():
    K = np.array([20.0, 100.0, 600.0])
    ray = np.array([0.05, -0.4, 1.0])
    roots, hit = ray_sphere_depths(ray, K, 420.0)
    assert hit
    for z in roots:
        assert abs(np.linalg.norm(z * ray - K) - 420.0) < 1e-6


@pytest.mark.parametrize("axis", [(0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 0)])
def test_rotation_invariance_noise_free(axis):
    """Fixed 135 deg knee rotated -40..40 deg about several axes: angle must not move."""
    geo, rng = make_geo(), np.random.default_rng(0)
    for deg in range(-40, 41, 5):
        res = run_solver(geo, leg_pose(135.0, rot(axis, deg)), rng, px_noise=0.0)
        assert res is not None
        assert abs(res.angle_deg - 135.0) < 0.05
        assert abs(res.thigh_mm - L_THIGH) < 1e-3 and abs(res.shin_mm - L_SHIN) < 1e-3


def _random_trials(geo, missing=(), n=300, seed=0, px_noise=0.7):
    rng = np.random.default_rng(seed)
    errs, raw_errs = [], []
    for _ in range(n):
        R = rot(rng.normal(size=3), rng.uniform(-40, 40))
        ang = rng.uniform(100, 178)
        solver = RigidLegSolver(L_THIGH, L_SHIN, depth_sigma_fn=lambda z: geo.depth_sigma(z, 1.5))
        # previous frame: slightly different pose (normal gait motion) -> temporal prior
        run_solver(geo, leg_pose(ang + rng.normal(0, 3), R @ rot(rng.normal(size=3), 2)), rng, solver=solver)
        joints = leg_pose(ang, R)
        res = run_solver(geo, joints, rng, px_noise=px_noise, drop=missing, solver=solver)
        assert res is not None
        assert abs(res.thigh_mm - L_THIGH) < 2.0 and abs(res.shin_mm - L_SHIN) < 2.0
        errs.append(abs(res.angle_deg - ang))
        P = []
        for X in joints:
            xl, xr = observe(geo, X, rng, px_noise)
            P.append(geo.triangulate(xl[0], xl[1], xr))
        raw_errs.append(abs(joint_angle_deg(*P) - ang))
    return np.array(errs), np.array(raw_errs)


def test_rigid_solver_beats_raw_triangulation():
    """With 0.7 px keypoint noise and random 3D rotations the rigid model is more accurate."""
    geo = make_geo()
    errs, raw = _random_trials(geo)
    assert np.median(errs) < 0.6 * np.median(raw)
    assert np.median(errs) < 2.5 and np.percentile(errs, 95) < 8.0


@pytest.mark.parametrize("missing", ["hip", "ankle", "knee"])
def test_missing_depth_recovered_not_2d(missing):
    """A joint with NO depth is reconstructed in 3D from the bone length (never 2D)."""
    geo = make_geo()
    errs, _ = _random_trials(geo, missing=(missing,), n=200, seed=1)
    assert np.median(errs) < 3.0 and np.percentile(errs, 95) < 10.0


def test_outlier_depth_rejected():
    """A depth that is 15 cm wrong (disparity hole / background) is rejected."""
    geo, rng = make_geo(), np.random.default_rng(2)
    joints = leg_pose(150.0, rot((0, 1, 0), 25))
    res = run_solver(geo, joints, rng, corrupt={"ankle": 150.0})
    assert res.source["ankle"] == "constrained"
    assert abs(res.angle_deg - 150.0) < 6.0


def test_no_measurement_no_prior_gives_none():
    geo, rng = make_geo(), np.random.default_rng(3)
    res = run_solver(geo, leg_pose(150, np.eye(3)), rng, drop=("hip", "knee", "ankle"))
    assert res is None      # no 2D fallback, just "no lock"


def test_sgbm_saturation_rejected():
    """v1 bug: pixels stuck at the max disparity produced constant Z readings."""
    d_max = 66.0
    disp = np.full((50, 50), d_max - 0.5, np.float32)
    pts = np.dstack([np.zeros((50, 50)), np.zeros((50, 50)), np.full((50, 50), 240.0)]).astype(np.float32)
    assert sample_sgbm_point(pts, disp, 25, 25, d_max) is None


def test_depth_range_helpers():
    geo = make_geo()
    z_min = geo.min_measurable_depth(2, 64)
    assert abs(z_min - F * B / 65) < 1e-6           # ~277 mm on this synthetic rig
    nd = geo.num_disparities_for(150.0)
    assert nd % 16 == 0 and geo.min_measurable_depth(0, nd) <= 150.0


def _person(xy_leg, conf=0.9):
    xy = np.zeros((17, 2))
    c = np.zeros(17)
    for k, v in xy_leg.items():
        xy[k], c[k] = v, conf
    return Person(xy, c, 0.9)


def test_confidence_gate_hides_occluded_leg():
    p = _person({11: (300, 100), 13: (300, 250), 15: (300, 400),
                 12: (380, 100), 14: (380, 250), 16: (380, 400)})
    p.conf[16] = 0.3                                # right ankle occluded
    legs = legs_from_people(p, None, thr=0.5)
    assert set(legs) == {"L"}


def test_ghost_leg_suppressed():
    p = _person({11: (300, 100), 13: (300, 250), 15: (300, 400),
                 12: (330, 100), 14: (305, 255), 16: (302, 395)})
    p.conf[[12, 14, 16]] = 0.6
    legs = suppress_ghost_legs(legs_from_people(p, None, thr=0.5))
    assert set(legs) == {"L"}


def test_bone_calibration_median():
    geo, rng = make_geo(), np.random.default_rng(4)
    cal = BoneLengthCalibrator(duration_s=2.0, min_samples=10)
    t = 0.0
    while not cal.done(t):
        for side in "LR":
            pts = []
            for P in leg_pose(178.0, np.eye(3)):
                xl, xr = observe(geo, P, rng, 0.7)
                pts.append(geo.triangulate(xl[0], xl[1], xr))
            cal.add(side, *pts, t)
        t += 1 / 30
        assert t < 5
    prof = cal.result()
    assert abs(prof.L_thigh_L - L_THIGH) < 15 and abs(prof.L_shin_R - L_SHIN) < 15


def test_prior_alone_does_not_keep_a_leg_alive():
    """v2.6: live a prior-only solution re-fed itself and froze a leg at 160 deg."""
    geo, rng = make_geo(), np.random.default_rng(5)
    s = RigidLegSolver(L_THIGH, L_SHIN)
    assert run_solver(geo, leg_pose(140, np.eye(3)), rng, solver=s) is not None
    assert run_solver(geo, leg_pose(140, np.eye(3)), rng, solver=s, drop=("hip", "knee", "ankle")) is None


def test_ankle_on_lens_root_rejected():
    """v2.11: knee 540 mm away, shin 541 mm -> the second ankle root sits at the lens (Z~0).
    With min_depth_mm the solver must never return that root."""
    sol = RigidLegSolver(419.0, 541.0)
    sol.min_depth_mm = 200.0
    H, K, A = np.array([0., -380., 560.]), np.array([0., 0., 540.]), np.array([0., 530., 650.])
    rays = {n: P / P[2] for n, P in (("hip", H), ("knee", K), ("ankle", A))}
    res = sol.solve(rays, {"hip": 560.0, "knee": 540.0, "ankle": None})
    assert res is None or res.ankle[2] >= 200.0
