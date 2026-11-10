import numpy as np
from gait3d.identity import LegIdentity
from gait3d.pose import LegObs


def obs(side, pts, conf=0.9):
    pts = np.array(pts, float)
    return LegObs(side, pts, np.full(3, conf), np.full(3, np.nan), np.full(3, np.nan), np.full(3, conf), pts.copy())


def test_shin_label_swap_is_undone():
    idn = LegIdentity()
    idn.update("L", 0.0, [[100, 100], [100, 250], [100, 400]])
    idn.update("R", 0.0, [[200, 100], [200, 250], [200, 400]])
    # YOLO swapped the ankles while the leg twisted
    legs = {"L": obs("L", [[102, 100], [101, 252], [198, 402]]), "R": obs("R", [[201, 101], [199, 249], [101, 398]])}
    out = idn.fix(legs, 0.2)
    assert np.allclose(out["L"].xy[2], [101, 398]) and np.allclose(out["R"].xy[2], [198, 402])


def test_duplicate_ankle_drops_other_leg():
    idn = LegIdentity()
    idn.update("L", 0.0, [[100, 100], [100, 250], [100, 400]])
    idn.update("R", 0.0, [[200, 100], [200, 250], [200, 400]])
    legs = {"L": obs("L", [[100, 100], [100, 250], [101, 401]]), "R": obs("R", [[200, 100], [190, 260], [105, 398]])}
    out = idn.fix(legs, 0.2)
    assert set(out) == {"L"}


def test_single_leg_relabelled():
    idn = LegIdentity()
    idn.update("L", 0.0, [[100, 100], [100, 250], [100, 400]])
    idn.update("R", 0.0, [[200, 100], [200, 250], [200, 400]])
    out = idn.fix({"L": obs("L", [[199, 99], [201, 251], [200, 399]])}, 0.2)
    assert set(out) == {"R"}
