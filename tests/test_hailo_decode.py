import numpy as np
from gait3d.hailo_pose import decode, letterbox


def test_decode_single_person():
    outs = {}
    for hw in (80, 40, 20):
        outs[f"b{hw}"] = np.zeros((hw, hw, 64), np.float32)
        outs[f"c{hw}"] = np.full((hw, hw, 1), -10.0, np.float32)
        outs[f"k{hw}"] = np.zeros((hw, hw, 51), np.float32)
    # one detection on the stride-16 grid (40x40) at cell (x=10, y=20)
    b = np.full((4, 16), -10.0, np.float32)
    b[:, 3] = 10.0                                   # each side distance = 3 cells
    outs["b40"][20, 10] = b.ravel()
    outs["c40"][20, 10, 0] = 5.0
    k = np.zeros((17, 3), np.float32)
    k[13] = [0.5, -0.25, 4.0]                        # left knee
    outs["k40"][20, 10] = k.ravel()
    boxes, scores, kpts = decode(outs)
    assert len(scores) == 1 and scores[0] > 0.99
    assert np.allclose(boxes[0], [(10.5 - 3) * 16, (20.5 - 3) * 16, (10.5 + 3) * 16, (20.5 + 3) * 16], atol=0.5)
    assert np.allclose(kpts[0, 13, :2], [(0.5 * 2 + 10) * 16, (-0.25 * 2 + 20) * 16])
    assert kpts[0, 13, 2] > 0.98


def test_letterbox_portrait():
    img = np.zeros((640, 360, 3), np.uint8)
    lb, s, left, top = letterbox(img, 640)
    assert lb.shape == (640, 640, 3) and s == 1.0 and left == 140 and top == 0
