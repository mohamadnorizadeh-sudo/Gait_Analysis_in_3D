"""Left/right leg identity across frames (v2.5).

Problem seen live on the walker: while one leg twists and is partly hidden
behind the other, YOLO gives the hidden leg's shin the OTHER leg's label (or
puts both labels on one shin).  The angle of the wrong leg then jumps by up
to 140 deg.  YOLO labels are decided per image, without memory, so we add it:

1. per joint, compare YOLO's L/R assignment with where each leg was in the
   previous frame (its solved 3D joints projected into the image) and swap
   that joint between legs if the swapped assignment fits much better;
2. if a joint of both legs lands on the same spot (duplicate / ghost), keep it
   only for the leg that was there before; the other leg is incomplete this
   frame and disappears (requirement: never draw guessed legs);
3. a single detected leg is relabelled if it sits where the other leg was.
"""

from __future__ import annotations

import numpy as np

FIELDS = ("xy", "conf", "xr", "yr", "conf_r", "xy_r")


def _get(obs, f, j):
    a = getattr(obs, f)
    return None if a is None else np.array(a[j], copy=True)


def _swap_joint(a, b, j):
    for f in FIELDS:
        va, vb = getattr(a, f), getattr(b, f)
        if va is None or vb is None:
            continue
        tmp = np.array(va[j], copy=True)
        va[j] = vb[j]
        vb[j] = tmp


class LegIdentity:
    def __init__(self, max_age_s=0.7, swap_ratio=0.6, dup_px=18.0):
        self.prev = {}            # side -> (t, (3,2) left-image projections of hip, knee, ankle)
        self.max_age_s, self.swap_ratio, self.dup_px = max_age_s, swap_ratio, dup_px
        self.swaps = 0

    def _recent(self, side, t):
        p = self.prev.get(side)
        return p[1] if p is not None and t - p[0] <= self.max_age_s else None

    def update(self, side, t, proj_left):
        self.prev[side] = (t, np.asarray(proj_left, float))

    def forget(self, side):
        self.prev.pop(side, None)

    def fix(self, legs: dict, t: float) -> dict:
        pL, pR = self._recent("L", t), self._recent("R", t)

        # 3. single leg: relabel if it sits where the other leg was
        if len(legs) == 1:
            (side, obs), = legs.items()
            other = "R" if side == "L" else "L"
            p_same, p_other = (pL, pR) if side == "L" else (pR, pL)
            if p_other is not None:
                d_other = np.median(np.linalg.norm(obs.xy[1:] - p_other[1:], axis=1))
                d_same = np.median(np.linalg.norm(obs.xy[1:] - p_same[1:], axis=1)) if p_same is not None else np.inf
                if d_other < self.swap_ratio * d_same:
                    obs.side = other
                    self.swaps += 1
                    return {other: obs}
            return legs

        if "L" not in legs or "R" not in legs:
            return legs
        L, R = legs["L"], legs["R"]

        # 1. per-joint swap against the previous frame
        if pL is not None and pR is not None:
            for j in range(3):
                keep = np.linalg.norm(L.xy[j] - pL[j]) + np.linalg.norm(R.xy[j] - pR[j])
                swap = np.linalg.norm(L.xy[j] - pR[j]) + np.linalg.norm(R.xy[j] - pL[j])
                if swap < self.swap_ratio * keep:
                    _swap_joint(L, R, j)
                    self.swaps += 1

        # 2. duplicated knee/ankle (both labels on one physical joint)
        drop = set()
        for j in (1, 2):
            if np.linalg.norm(L.xy[j] - R.xy[j]) < self.dup_px:
                if pL is not None and pR is not None:
                    owner = "L" if np.linalg.norm(L.xy[j] - pL[j]) <= np.linalg.norm(R.xy[j] - pR[j]) else "R"
                else:
                    owner = "L" if L.conf[j] >= R.conf[j] else "R"
                drop.add("R" if owner == "L" else "L")
        return {s: o for s, o in legs.items() if s not in drop}
