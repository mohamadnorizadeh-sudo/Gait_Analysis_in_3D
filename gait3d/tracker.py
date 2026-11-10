"""Frame-level gait pipeline (camera-independent, so it can be unit-tested).

    rectified L/R images -> YOLO (batched) -> confidence gate -> ghost-leg check
      -> stereo keypoint triangulation (+ optional SGBM patch fallback)
      -> rigid bone-length solver (never 2D) -> One-Euro filter -> overlay / CSV
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np

from .filters import OneEuroFilter
from .identity import LegIdentity
from .pose import (LegObs, best_person, legs_from_people, match_right,
                   people_from_result, suppress_ghost_legs)
from .profile import BoneLengthCalibrator, SubjectProfile
from .rigid_leg import JOINTS, LegResult, RigidLegSolver
from .stereo_geometry import StereoGeometry, sample_sgbm_point

COLORS = {"L": (255, 200, 0), "R": (0, 200, 255)}           # BGR
SRC_COLORS = {"measured": (0, 220, 0), "constrained": (0, 165, 255), "prior": (0, 0, 255)}


@dataclass
class TrackerConfig:
    kp_conf: float = 0.50             # per-joint confidence gate (requirement 4)
    epi_tol_px: float = 4.0           # rectified rig: max |y_left - y_right| (px)
    ray_gap_mm: float = 80.0          # converging rig: max distance between viewing rays
                                      # (v2.3: 40 -> 80 mm; live gaps were 4-68 mm at 97 deg convergence)
    z_range_mm: tuple = (120.0, 2000.0)
    ghost_ratio: float = 0.35
    ghost_3d_mm: float = 60.0
    filter_min_cutoff: float = 1.2
    filter_beta: float = 0.03


@dataclass
class FrameOutput:
    legs: Dict[str, LegResult] = field(default_factory=dict)
    proj: Dict[str, np.ndarray] = field(default_factory=dict)   # solved joints projected into the left image
    shin_measured: Dict[str, float] = field(default_factory=dict)  # raw stereo knee-ankle distance (validation)
    obs: Dict[str, LegObs] = field(default_factory=dict)
    angle_filtered: Dict[str, float] = field(default_factory=dict)
    flexion_corrected: Dict[str, float] = field(default_factory=dict)   # filtered flexion minus standing offset
    status: str = ""


class NeutralOffset:
    """v2.6 standing ('neutral') trial: pose keypoints sit on the skin/surface,
    not at the joint centres, so a straight leg reads ~10-20 deg of flexion
    (seen live: 13-20 deg while standing). Like a gait-lab static trial, the
    median flexion during quiet standing is recorded per leg and subtracted."""

    def __init__(self, seconds=3.0, max_spread=8.0):
        self.seconds, self.max_spread = seconds, max_spread
        self.samples = {"L": [], "R": []}
        self.t0 = None
        self.offset = None

    def add(self, t, flex):
        if self.t0 is None:
            self.t0 = t
        for s, f in flex.items():
            self.samples[s].append(f)
        if t - self.t0 >= self.seconds and all(len(v) >= 8 for v in self.samples.values()):
            off = {}
            for s, v in self.samples.items():
                v = np.asarray(v)
                med = float(np.median(v))
                if np.percentile(v, 90) - np.percentile(v, 10) > self.max_spread or med > 35:
                    self.t0, self.samples = None, {"L": [], "R": []}     # not still/straight: retry
                    return False
                off[s] = med
            self.offset = off
            return True
        return False


class GaitTracker:
    def __init__(self, geo: StereoGeometry, profile: Optional[SubjectProfile] = None,
                 cfg: TrackerConfig = TrackerConfig(), calib_seconds: float = 2.0,
                 neutral_seconds: float = 3.0):
        self.geo, self.cfg = geo, cfg
        self.profile = profile
        self.calibrator = None if profile else BoneLengthCalibrator(calib_seconds)
        self.solvers: Dict[str, RigidLegSolver] = {}
        self.filters = {s: OneEuroFilter(cfg.filter_min_cutoff, cfg.filter_beta) for s in "LR"}
        self.identity = LegIdentity()
        self.neutral = NeutralOffset(neutral_seconds) if neutral_seconds > 0 else None
        if profile:
            self._make_solvers()

    def _make_solvers(self):
        sig = lambda z: self.geo.depth_sigma(z, 1.5)
        self.solvers = {s: RigidLegSolver(*self.profile.lengths(s), depth_sigma_fn=sig) for s in "LR"}

    def recalibrate(self, seconds=2.0):
        self.profile, self.solvers = None, {}
        self.calibrator = BoneLengthCalibrator(seconds)
        for f in self.filters.values():
            f.reset()

    # ------------------------------------------------------------------ #
    def _measure(self, obs: LegObs, sgbm=None):
        """Per-joint depth: stereo triangulation first, SGBM patch second."""
        zs, pts = {}, {}
        for j, name in enumerate(JOINTS):
            x, y = obs.xy[j]
            P = None
            if obs.conf[j] < self.cfg.kp_conf:
                zs[name] = None             # left keypoint is a guess: never triangulate it
                continue
            if np.isfinite(obs.xr[j]):
                yr = obs.yr[j] if obs.yr is not None else None
                P = self.geo.triangulate(x, y, obs.xr[j], yr)
            if (P is None or not np.isfinite(P[2])) and sgbm is not None:
                pts3d, disp, d_max = sgbm
                P = sample_sgbm_point(pts3d, disp, x, y, d_max, z_range=self.cfg.z_range_mm)
            if P is not None and np.isfinite(P[2]) and self.cfg.z_range_mm[0] < P[2] < self.cfg.z_range_mm[1]:
                zs[name], pts[name] = float(P[2]), P
            else:
                zs[name] = None
        return zs, pts

    def process(self, t, result_l, result_r, sgbm=None) -> FrameOutput:
        """sgbm: optional (points_3d, disparity, d_max) tuple for patch fallback."""
        out = FrameOutput()
        left = best_person(people_from_result(result_l))
        if left is None:
            out.status = "no person"
            return out
        rect = getattr(self.geo, "rectified", True)
        tol = self.cfg.epi_tol_px if rect else self.cfg.ray_gap_mm
        merr = None if rect else self.geo.match_error
        right = match_right(left, people_from_result(result_r), self.cfg.kp_conf, tol,
                            None if rect else merr) if result_r is not None else None
        if rect:
            d_lo = float(self.geo.disparity_from_depth(self.cfg.z_range_mm[1]))
            d_hi = float(self.geo.disparity_from_depth(self.cfg.z_range_mm[0]))
        else:
            d_lo, d_hi = -1e9, 1e9
        legs = legs_from_people(left, right, self.cfg.kp_conf, tol, (d_lo, d_hi), merr)
        legs = suppress_ghost_legs(legs, self.cfg.ghost_ratio)
        legs = self.identity.fix(legs, t)          # v2.5: L/R continuity across frames
        out.obs = legs

        # ---------------- bone-length calibration phase ----------------
        if self.calibrator is not None:
            for side, obs in legs.items():
                zs, pts = self._measure(obs, sgbm)
                if all(zs[n] is not None for n in JOINTS):
                    self.calibrator.add(side, pts["hip"], pts["knee"], pts["ankle"], t)
            el = self.calibrator.elapsed(t)
            n = {s: len(self.calibrator.samples[s]["thigh"]) for s in "LR"}
            out.status = f"CALIBRATING - stand straight  {el:.1f}s  samples L{n['L']} R{n['R']}"
            if self.calibrator.done(t):
                self.profile = self.calibrator.result()
                self.calibrator = None
                self._make_solvers()
                out.status = (f"calibrated: thigh L{self.profile.L_thigh_L:.0f} R{self.profile.L_thigh_R:.0f} mm, "
                              f"shin L{self.profile.L_shin_L:.0f} R{self.profile.L_shin_R:.0f} mm")
            return out

        # ---------------- tracking ----------------
        for side, obs in legs.items():
            zs, mp = self._measure(obs, sgbm)
            if "knee" in mp and "ankle" in mp:
                out.shin_measured[side] = float(np.linalg.norm(mp["knee"] - mp["ankle"]))
            rays, hints = {}, {}
            for j, n in enumerate(JOINTS):
                if obs.conf[j] >= self.cfg.kp_conf or obs.conf_r is None or not hasattr(self.geo, "C_r"):
                    rays[n] = self.geo.ray(*obs.xy[j])
                else:
                    # v2.3: joint only seen by the right camera -> its ray, depth from bone length
                    rays[n] = (self.geo.C_r, self.geo.ray_right_in_left(*obs.xy_r[j]))
                    hints[n] = self.geo.ray(*obs.xy[j])     # cam0's low-confidence guess
                    zs[n] = None
            res = self.solvers[side].solve(rays, zs, hints)
            if res is not None:
                out.legs[side] = res
                if hasattr(self.geo, "project_left"):
                    out.proj[side] = np.array([self.geo.project_left(P) for P in (res.hip, res.knee, res.ankle)])

        # 3D ghost check: two "legs" whose knees coincide in space are one limb
        if "L" in out.legs and "R" in out.legs:
            if np.linalg.norm(out.legs["L"].knee - out.legs["R"].knee) < self.cfg.ghost_3d_mm:
                drop = "L" if legs["L"].mean_conf < legs["R"].mean_conf else "R"
                out.legs.pop(drop)
                out.obs.pop(drop, None)

        for side, res in list(out.legs.items()):
            # v2.5 motion gate: a knee cannot teleport (> 300 mm between frames 0.2 s apart)
            prev = getattr(self, "_last_knee", {}).get(side)
            if prev is not None and t - prev[0] < 0.5 and np.linalg.norm(res.knee - prev[1]) > 300.0:
                out.legs.pop(side)
                out.proj.pop(side, None)
                self.solvers[side].reset()
                continue
            self.__dict__.setdefault("_last_knee", {})[side] = (t, res.knee.copy())
            if side in out.proj:
                self.identity.update(side, t, out.proj[side])
            out.angle_filtered[side] = self.filters[side](res.angle_deg, t)
        for side in "LR":
            if side not in out.legs:
                self.filters[side].reset()
        flex = {s: 180.0 - a for s, a in out.angle_filtered.items()}
        if self.neutral is not None and self.neutral.offset is None:
            if len(flex) == 2 and self.neutral.add(t, flex):
                pass
            out.status = "NEUTRAL: stand straight & still" if self.neutral.offset is None else "neutral offset saved"
        off = self.neutral.offset if (self.neutral is not None and self.neutral.offset) else {}
        out.flexion_corrected = {s: f - off.get(s, 0.0) for s, f in flex.items()}
        if not out.status:
            out.status = "3D rigid lock" if out.legs else "waiting for legs (conf < 0.5)"
        return out


# ---------------------------------------------------------------------- #
# drawing / logging
# ---------------------------------------------------------------------- #
def draw(frame, out: FrameOutput, fps=0.0, skew_ms=0.0):
    img = frame.copy()
    for side, obs in out.obs.items():
        res = out.legs.get(side)
        src_xy = out.proj.get(side, obs.xy)          # draw the solved 3D joints where available
        pts = [tuple(int(v) for v in p) for p in src_xy]
        col = COLORS[side]
        if res is None and out.legs:      # leg passed 2D gate but has no 3D solution: draw thin
            cv2.line(img, pts[0], pts[1], col, 1)
            cv2.line(img, pts[1], pts[2], col, 1)
            continue
        cv2.line(img, pts[0], pts[1], col, 3)
        cv2.line(img, pts[1], pts[2], col, 3)
        for j, n in enumerate(JOINTS):
            c = SRC_COLORS[res.source[n]] if res else (200, 200, 200)
            cv2.circle(img, pts[j], 6 if n == "knee" else 4, c, -1)
        if res is not None:
            fc = out.flexion_corrected.get(side, 180 - out.angle_filtered.get(side, res.angle_deg))
            cv2.putText(img, f"{side} {fc:.0f} flex", (pts[1][0] + 10, pts[1][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(img, (8, 8), (400, 96), (25, 25, 25), -1)
    cv2.putText(img, "GAIT 3D v2 - rigid bone model", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y = 48
    for side in "LR":
        if side in out.legs:
            r = out.legs[side]
            a = out.angle_filtered.get(side, r.angle_deg)
            fc = out.flexion_corrected.get(side, 180 - a)
            txt = f"{side}: flex {fc:5.1f}  (raw {180 - a:5.1f})  Z {r.knee[2] / 10:.0f}cm"
        else:
            txt = f"{side}: --"
        cv2.putText(img, txt, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[side], 1)
        y += 18
    cv2.putText(img, f"{out.status} | {fps:.1f} fps | skew {skew_ms:.1f} ms", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return img


class CsvLogger:
    FIELDS = ["t", "side", "knee_angle_deg", "flexion_deg", "flexion_filtered_deg",
              "hip_x", "hip_y", "hip_z", "knee_x", "knee_y", "knee_z", "ankle_x", "ankle_y", "ankle_z",
              "src_hip", "src_knee", "src_ankle", "thigh_mm", "shin_mm", "shin_stereo_mm",
              "flexion_corrected_deg"]

    def __init__(self, path):
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        self.w.writeheader()

    def log(self, t, out: FrameOutput):
        for side, r in out.legs.items():
            row = {"t": f"{t:.4f}", "side": side, "knee_angle_deg": f"{r.angle_deg:.2f}",
                   "flexion_deg": f"{r.flexion_deg:.2f}",
                   "flexion_filtered_deg": f"{180 - out.angle_filtered.get(side, r.angle_deg):.2f}",
                   "thigh_mm": f"{r.thigh_mm:.1f}", "shin_mm": f"{r.shin_mm:.1f}",
                   "shin_stereo_mm": f"{out.shin_measured[side]:.1f}" if side in out.shin_measured else "",
                   "flexion_corrected_deg": f"{out.flexion_corrected[side]:.2f}" if side in out.flexion_corrected else ""}
            for n, P in (("hip", r.hip), ("knee", r.knee), ("ankle", r.ankle)):
                row[f"{n}_x"], row[f"{n}_y"], row[f"{n}_z"] = (f"{v:.1f}" for v in P)
                row[f"src_{n}"] = r.source[n]
            self.w.writerow(row)

    def close(self):
        self.f.close()
