"""v3.0 side-view knee angle: homography from a board in the sagittal plane removes camera slant."""
import math

import numpy as np
import pytest

from gait3d.side2d import PlaneMap, SideConfig, SideTracker, flexion_2d

F, CX, CY = 300.0, 180.0, 320.0


def cam_pose(yaw_deg, pitch_deg, dist=400.0):
    """Camera looking at the sagittal plane (x forward, y down, z=0 plane) from the side, slanted."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    R = Rx @ Ry
    t = np.array([0.0, 0.0, dist])
    return R, t


def project(R, t, P):
    X = (R @ np.asarray(P, float).T).T + t
    return np.column_stack([F * X[:, 0] / X[:, 2] + CX, F * X[:, 1] / X[:, 2] + CY])


def leg(flex_deg, thigh_tilt_deg=10.0, Lt=420.0, Ls=400.0):
    """Hip at origin, thigh tilted forward, shank flexed backward, all in the z=0 plane (mm)."""
    a = math.radians(thigh_tilt_deg)
    hip = np.array([0.0, -300.0, 0.0])
    knee = hip + Lt * np.array([math.sin(a), math.cos(a), 0.0])
    b = a - math.radians(flex_deg)
    ankle = knee + Ls * np.array([math.sin(b), math.cos(b), 0.0])
    return np.array([hip, knee, ankle])


def board_homography(R, t):
    g = np.array([[x, y, 0.0] for x in np.linspace(-150, 150, 9) for y in np.linspace(-200, 200, 6)])
    import cv2
    H, _ = cv2.findHomography(project(R, t, g), g[:, :2] / 10.0)      # -> plane cm
    return H


@pytest.mark.parametrize("yaw,pitch", [(30, 8), (-25, 10), (0, 0)])
@pytest.mark.parametrize("flex", [0, 30, 60, 90])
def test_homography_recovers_true_flexion(yaw, pitch, flex):
    R, t = cam_pose(yaw, pitch)
    px = project(R, t, leg(flex))
    pm = PlaneMap(planes={"cam0:L": {"H": board_homography(R, t)}})
    f = flexion_2d(*pm.to_plane("cam0", "L", px))
    assert abs(f - flex) < 0.5


def test_slant_without_homography_is_wrong():
    R, t = cam_pose(40, 10)
    raw = flexion_2d(*project(R, t, leg(60)))
    assert abs(raw - 60) > 3          # documents why the calibration is needed


def _people(px, conf=0.95):
    xy = np.zeros((17, 2))
    c = np.zeros(17)
    for j, p in zip((11, 13, 15), px):
        xy[j], c[j] = p, conf
    for j, p in zip((12, 14, 16), px):
        xy[j], c[j] = p, conf
    return [{"xy": xy.tolist(), "c": c.tolist(), "s": 0.9}]


def test_tracker_offset_and_bend():
    R, t = cam_pose(20, 5)
    H = board_homography(R, t)
    pm = PlaneMap(planes={"cam0:L": {"H": H}, "cam1:R": {"H": H}})
    tr = SideTracker(pm, SideConfig(filter_min_cutoff=50.0, filter_beta=0.0))
    T = 0.0
    for _ in range(40):                     # 4 s quiet stand at 5 deg (keypoint offset)
        p = _people(project(R, t, leg(5)))
        tr.process(T, {"cam0": p, "cam1": p})
        T += 0.1
    assert tr.offset is not None and abs(tr.offset["L"] - 5) < 0.5
    for f in range(5, 66, 5):               # slow bend to 65
        p = _people(project(R, t, leg(f)))
        out = tr.process(T, {"cam0": p, "cam1": p})
        T += 0.1
    assert abs(out["L"].flexion - 60) < 2.0 and abs(out["R"].flexion - 60) < 2.0


def test_tracker_gates_low_confidence_and_jumps():
    R, t = cam_pose(0, 0)
    tr = SideTracker(PlaneMap(), SideConfig())
    p = _people(project(R, t, leg(10)), conf=0.3)
    out = tr.process(0.0, {"cam0": p, "cam1": p})
    assert out["L"].flexion is None and out["L"].status == "low confidence"
    tr.process(1.0, {"cam0": _people(project(R, t, leg(10))), "cam1": []})
    out = tr.process(1.1, {"cam0": _people(project(R, t, leg(80))), "cam1": []})
    assert out["L"].status == "jump gate"


def test_marker_snapping_moves_keypoints_onto_tape():
    import cv2
    img = np.full((640, 360, 3), 30, np.uint8)                 # dark trousers
    true = [(150, 120), (170, 330), (230, 520)]
    for x, y in true:
        cv2.rectangle(img, (x - 6, y - 6), (x + 6, y + 6), (255, 255, 255), -1)
    kp = np.array(true, float) + np.array([[25, -15], [-20, 18], [10, 30]])   # pose model is off by 2-3 cm
    from gait3d.side2d import snap_to_markers
    pts, n = snap_to_markers(img, kp)
    assert n == 3 and np.abs(pts - np.array(true)).max() < 1.0


def test_bystander_in_background_is_ignored():
    from gait3d.side2d import leg_keypoints
    def person(scale, score):
        xy = np.zeros((17, 2)); c = np.full(17, 0.95)
        for j, y in zip((11, 13, 15), (100, 100 + 150 * scale, 100 + 290 * scale)):
            xy[j] = (180, y)
        return {"xy": xy.tolist(), "c": c.tolist(), "s": score}
    far, near = person(0.3, 0.99), person(1.0, 0.80)
    assert leg_keypoints([far], "L", 170.0) is None
    kp = leg_keypoints([far, near], "L", 170.0)
    assert abs(kp[0][2][1] - 390) < 1


def test_snap_keeps_leg_order_with_wide_radius():
    """v3.0.8: knee keypoint closer to the shin tape must not swap knee and shin (read 170 deg)."""
    import numpy as np
    from gait3d.side2d import snap_to_markers
    blobs = [(300, 137, 600), (312, 317, 400), (335, 386, 400)]          # hip, knee, shin tapes
    seed = np.array([[225, 90], [330, 360], [338, 400]], float)          # knee kp sits near the shin tape
    fl = []
    out, n = snap_to_markers(None, seed, 100, blobs=blobs, flags=fl)
    assert n == 3 and fl == [True, True, True]
    assert tuple(out[1]) == (312, 317) and tuple(out[2]) == (335, 386)


def test_tan_markers_found_and_white_wall_ignored():
    """v3.0.9: brown cardboard squares on navy trousers are found; a white wall patch is not."""
    import cv2, numpy as np
    from gait3d.side2d import SideConfig, marker_blobs
    img = np.full((300, 200, 3), (235, 235, 235), np.uint8)             # white wall
    img[:, 40:160] = (60, 30, 25)                                       # navy trousers (BGR)
    tan = cv2.cvtColor(np.uint8([[[20, 110, 180]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    img[100:120, 90:110] = tan
    b = marker_blobs(img, SideConfig(marker_color="tan"))
    assert len(b) == 1 and abs(b[0][0] - 99.5) < 1 and abs(b[0][1] - 109.5) < 1
