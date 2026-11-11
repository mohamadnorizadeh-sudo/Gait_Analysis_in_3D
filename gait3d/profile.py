"""Standing calibration of the subject's bone lengths (L_thigh, L_shin).

The user stands still with straight legs for ~2 s.  We collect thigh / shin
lengths only from frames where hip, knee and ankle were all triangulated
directly (no constraints, no priors), the leg is nearly straight, and the
lengths are anatomically plausible.  The robust median then becomes a
physical constant for the session.

Sanity limits: adult femur ~ 350-550 mm, tibia ~ 300-500 mm (joint-centre to
joint-centre).  v1 measured 20-210 mm because the stereo depth was saturated;
if this calibration fails with very short bones, run scripts/03_check_depth.py.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

THIGH_RANGE = (250.0, 650.0)
SHIN_RANGE = (220.0, 600.0)


@dataclass
class SubjectProfile:
    L_thigh_L: float
    L_shin_L: float
    L_thigh_R: float
    L_shin_R: float
    n_samples: int = 0
    created: str = ""
    method: str = "standing"

    def lengths(self, side):
        return (self.L_thigh_L, self.L_shin_L) if side == "L" else (self.L_thigh_R, self.L_shin_R)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    @classmethod
    def manual(cls, thigh_mm, shin_mm):
        return cls(thigh_mm, shin_mm, thigh_mm, shin_mm, 0, time.strftime("%Y-%m-%d %H:%M"), "manual")


def robust_median(values):
    v = np.asarray(values, float)
    if v.size == 0:
        return np.nan, 0
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-6
    keep = np.abs(v - med) < 3.0 * 1.4826 * mad
    return float(np.median(v[keep])), int(keep.sum())


class BoneLengthCalibrator:
    def __init__(self, duration_s=2.0, min_samples=12, min_straight_deg=160.0):
        self.duration_s, self.min_samples, self.min_straight = duration_s, min_samples, min_straight_deg
        self.samples = {"L": {"thigh": [], "shin": []}, "R": {"thigh": [], "shin": []}}
        self.t_start = None
        self.rejected = 0

    def add(self, side, hip, knee, ankle, t):
        """Add one frame of fully measured joints. Returns True when finished."""
        from .rigid_leg import joint_angle_deg

        if self.t_start is None:
            self.t_start = t
        th = float(np.linalg.norm(hip - knee))
        sh = float(np.linalg.norm(ankle - knee))
        ang = joint_angle_deg(hip, knee, ankle) or 0.0
        if (THIGH_RANGE[0] <= th <= THIGH_RANGE[1] and SHIN_RANGE[0] <= sh <= SHIN_RANGE[1]
                and ang >= self.min_straight):
            self.samples[side]["thigh"].append(th)
            self.samples[side]["shin"].append(sh)
        else:
            self.rejected += 1
        return self.done(t)

    def elapsed(self, t):
        return 0.0 if self.t_start is None else t - self.t_start

    def done(self, t):
        n = min(len(self.samples[s]["thigh"]) for s in "LR" if self.samples[s]["thigh"]) \
            if any(self.samples[s]["thigh"] for s in "LR") else 0
        return self.elapsed(t) >= self.duration_s and n >= self.min_samples

    def result(self) -> SubjectProfile:
        out, n_tot = {}, 0
        for s in "LR":
            th, n1 = robust_median(self.samples[s]["thigh"])
            sh, n2 = robust_median(self.samples[s]["shin"])
            out[s] = (th, sh)
            n_tot += min(n1, n2)
        # one leg missing during calibration -> assume symmetric bones
        for s, o in (("L", "R"), ("R", "L")):
            if not np.isfinite(out[s][0]) and np.isfinite(out[o][0]):
                out[s] = out[o]
        if not all(np.isfinite(out[s][0]) and np.isfinite(out[s][1]) for s in "LR"):
            raise RuntimeError(
                f"Bone calibration failed: not enough valid standing frames "
                f"(rejected {self.rejected}). Stand straight, both hips/knees/ankles visible, "
                f"or pass --thigh-mm/--shin-mm manually.")
        return SubjectProfile(out["L"][0], out["L"][1], out["R"][0], out["R"][1], n_tot,
                              time.strftime("%Y-%m-%d %H:%M"), "standing")


def load_or_none(path):
    return SubjectProfile.load(path) if path and os.path.exists(path) else None
