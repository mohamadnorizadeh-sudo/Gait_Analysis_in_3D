"""Side-view (sagittal) knee flexion from one lateral camera per leg (v3.0).

Why (see CHANGELOG 15 Nov): a front-facing stereo pair cannot measure sagittal knee
flexion once the knee bends.  Each leg now has its own lateral camera on the walker
(cam0 = left side, cam1 = right side).  The knee angle is computed in 2D, after
mapping the keypoints into the leg's sagittal plane with a homography measured from
a checkerboard held flat against that plane (removes camera slant/perspective).

Pipeline per leg and frame
  keypoints (hip, knee, ankle) of the chosen camera
  -> confidence gate (>= kp_conf)
  -> undistort (fisheye intrinsics, optional) + homography to plane centimetres
  -> segment-length gate (thigh / shank within +-len_tol of the standing lengths)
  -> flexion = 180 - angle(hip-knee, ankle-knee)
  -> jump gate (max_step_deg per frame), short hold (hold_s), One-Euro filter
  -> standing offset (median of a quiet stand in the first 10 s, as v2.6/v2.13)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .filters import OneEuroFilter

JOINTS = {"L": (11, 13, 15), "R": (12, 14, 16)}       # COCO hip, knee, ankle
DEFAULT_SOURCE = {"L": "cam0", "R": "cam1"}            # near camera of each leg


def flexion_2d(hip, knee, ankle) -> Optional[float]:
    """Clinical flexion (0 = straight) from three 2D points."""
    u = np.asarray(hip, float) - np.asarray(knee, float)
    v = np.asarray(ankle, float) - np.asarray(knee, float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return None
    c = float(np.clip(u @ v / (nu * nv), -1.0, 1.0))
    return 180.0 - math.degrees(math.acos(c))


# ---------------------------------------------------------------------- #
# camera -> sagittal plane mapping
# ---------------------------------------------------------------------- #
class PlaneMap:
    """Per (camera, leg): optional fisheye intrinsics + homography (undistorted px -> plane cm).

    JSON layout (config/side_planes.json):
      {"image_size": [360, 640],
       "cams": {"cam0": {"K": 3x3, "D": [4]}, ...},
       "planes": {"cam0:L": {"H": 3x3, "rms_cm": .., "created": ..}, ...}}
    Without an entry the mapping is the identity (raw pixels), so the tracker still runs.
    """

    def __init__(self, cams=None, planes=None, image_size=(360, 640)):
        self.cams = cams or {}
        self.planes = planes or {}
        self.image_size = tuple(image_size)

    @classmethod
    def load(cls, path):
        try:
            with open(path) as f:
                d = json.load(f)
        except FileNotFoundError:
            return cls()
        cams = {k: {"K": np.array(v["K"], float), "D": np.array(v["D"], float).reshape(4, 1)}
                for k, v in d.get("cams", {}).items()}
        planes = {k: {**v, "H": np.array(v["H"], float)} for k, v in d.get("planes", {}).items()}
        return cls(cams, planes, d.get("image_size", (360, 640)))

    def save(self, path):
        d = {"image_size": list(self.image_size),
             "cams": {k: {"K": v["K"].tolist(), "D": v["D"].ravel().tolist()} for k, v in self.cams.items()},
             "planes": {k: {**{kk: vv for kk, vv in v.items() if kk != "H"}, "H": np.asarray(v["H"]).tolist()}
                        for k, v in self.planes.items()}}
        with open(path, "w") as f:
            json.dump(d, f, indent=1)

    def undistort(self, cam, pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        c = self.cams.get(cam)
        if c is None:
            return pts
        import cv2
        return cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2), c["K"], c["D"], P=c["K"]).reshape(-1, 2)

    def has_plane(self, cam, leg):
        return f"{cam}:{leg}" in self.planes

    def to_plane(self, cam, leg, pts):
        p = self.undistort(cam, pts)
        e = self.planes.get(f"{cam}:{leg}")
        if e is None:
            return p
        Hm = e["H"]
        q = np.hstack([p, np.ones((len(p), 1))]) @ Hm.T
        return q[:, :2] / q[:, 2:3]


# ---------------------------------------------------------------------- #
# tape markers (keypoint-guided)
# ---------------------------------------------------------------------- #
def white_blobs(img, min_area=12, max_area=3000, v_min=165, s_max=70, dark_ring=True, ring_v_max=110,
                hue=None, s_min=0):
    """Small bright, unsaturated, compact blobs (2 cm white tape squares) -> list of (x, y, area).
    v3.0.2: with dark_ring, a blob must sit on dark trousers (median brightness of a ring around it
    <= ring_v_max) - the walls and floor gave many false white blobs (16 Nov)."""
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 2] >= v_min) & (hsv[..., 1] <= s_max) & (hsv[..., 1] >= s_min)
    if hue is not None:                       # v3.0.9: coloured markers (OpenCV hue 0-179)
        m &= (hsv[..., 0] >= hue[0]) & (hsv[..., 0] <= hue[1])
    m = m.astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (min_area <= a <= max_area):
            continue
        if max(w, h) > 2.5 * min(w, h) or a < 0.45 * w * h:       # tape is a compact square
            continue
        if dark_ring:
            # v3.0.5: thin ring (5 px) and "at least 45 % dark" - a wide ring around a 4x4 cm marker reached
            # the bright background beside the leg and rejected every hip/knee marker (16 Nov).
            r = 5
            H_, W_ = m.shape
            y0, y1, x0, x1 = max(0, y - r), min(H_, y + h + r), max(0, x - r), min(W_, x + w + r)
            patch = hsv[y0:y1, x0:x1, 2]
            ring = np.ones(patch.shape, bool)
            ring[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
            if not ring.any() or np.mean(patch[ring] <= ring_v_max) < 0.45:
                continue
        out.append((float(cen[i][0]), float(cen[i][1]), float(a)))
    return out


MARKER_COLORS = {
    # name: kwargs for white_blobs
    "white": lambda c: dict(v_min=c.marker_v_min, s_max=70),
    "tan": lambda c: dict(v_min=100, s_min=70, s_max=255, hue=(14, 25)),
}


def marker_blobs(img, cfg):
    return white_blobs(img, **MARKER_COLORS[cfg.marker_color](cfg))


def snap_to_markers(img, pts, radius_px=45.0, blobs=None, flags=None, prev=None, shank=None, shank_perp=0.3):
    """Move each keypoint (hip, knee, ankle/shin) onto a white tape blob within radius.
    Returns (points, n_snapped).  A joint without a blob keeps the keypoint (and counts as unsnapped).
    flags (list) receives True/False per joint.

    v3.0.8: joint assignment instead of greedy nearest-first.  With the 100 px radius (v3.0.7) the knee
    search reached the shin tape and the shin search the knee tape; the swap read 170 deg standing
    (17 Nov, side_session_20261117_104124) and continuity kept it swapped.  Now every combination of
    distinct blobs is scored by total distance, and tapes must keep the leg order hip -> knee -> shin
    along the hip-to-shin direction."""
    import itertools
    blobs = white_blobs(img) if blobs is None else blobs
    P = np.array(pts, float)
    out = P.copy()
    centres = []
    for j, p in enumerate(P):
        if prev is not None and prev[j] is not None:           # continuity: search around last tape
            p = 0.5 * (np.asarray(p) + np.asarray(prev[j]))
        centres.append(np.asarray(p, float))
    axis = P[-1] - P[0]
    nrm = float(np.linalg.norm(axis))
    axis = axis / nrm if nrm > 1e-6 else np.array([0.0, 1.0])
    def on_shank(q):
        # v3.0.10: the shin tape must lie ON the shank (pose knee -> ankle), not just near the seed.
        # Deep flexion (right leg ~87 deg, 17 Nov): the shin tape was not visible and the knee tape,
        # 70 px below the knee keypoint, was taken as the shin tape -> status "H-A", one 122 deg spike.
        if shank is None:
            return True
        k0, a0 = np.asarray(shank[0], float), np.asarray(shank[1], float)
        v = a0 - k0
        L2 = float(v @ v)
        if L2 < 1:
            return True
        w = np.asarray(q, float) - k0
        t = float(w @ v) / L2
        perp = abs(w[0] * v[1] - w[1] * v[0]) / math.sqrt(L2)
        return 0.05 < t < 1.0 and perp < shank_perp * math.sqrt(L2)

    cand = []
    for j, p in enumerate(centres):
        c = [(k, math.hypot(x - p[0], y - p[1])) for k, (x, y, _a) in enumerate(blobs)
             if j != 2 or on_shank((x, y))]
        cand.append([None] + [k for k, d in sorted(c, key=lambda t: t[1]) if d < radius_px][:6])
    best, best_key = None, None
    for combo in itertools.product(*cand):
        used = [k for k in combo if k is not None]
        if len(used) != len(set(used)):
            continue
        proj = [float(np.dot(np.array(blobs[k][:2]) - P[0], axis)) for k in combo if k is not None]
        if any(b <= a for a, b in zip(proj, proj[1:])):
            continue
        dist = sum(math.hypot(blobs[k][0] - centres[j][0], blobs[k][1] - centres[j][1])
                   for j, k in enumerate(combo) if k is not None)
        key = (-len(used), dist)                                  # most tapes first, then closest
        if best_key is None or key < best_key:
            best, best_key = combo, key
    n = 0
    if flags is not None:
        flags[:] = [False] * len(P)
    for j, k in enumerate(best or [None] * len(P)):
        if k is not None:
            out[j] = blobs[k][:2]
            n += 1
            if flags is not None:
                flags[j] = True
    return out, n


# ---------------------------------------------------------------------- #
# tracker
# ---------------------------------------------------------------------- #
@dataclass
class SideConfig:
    kp_conf: float = 0.5
    min_leg_px: float = 170.0        # hip-knee-ankle chain; the user's near leg is ~300 px, a bystander < 120
    len_tol: float = 0.5             # segment length gate vs standing lengths (v3.0.2: 0.25 dropped every bend)
    max_step_deg: float = 25.0       # per-frame jump gate (~10 fps)
    hold_s: float = 0.25             # keep the last good value this long, then "--"
    filter_min_cutoff: float = 1.5
    filter_beta: float = 0.05
    neutral_window_s: float = 10.0   # quiet stand only accepted in the first 10 s
    neutral_seconds: float = 2.0
    neutral_max_spread: float = 6.0
    source: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SOURCE))
    markers: bool = False            # snap hip/knee/ankle to white tape markers (needs images)
    marker_radius_px: float = 50.0
    # v3.0.7: tape brightness threshold.  At 165 the sheen along the trouser edge (17 Nov afternoon, re-aimed
    # cameras) joined the knee and shin tapes into one 73x331 px strip that failed the size test; at 200 all
    # three tapes on cam0 were found individually.
    marker_v_min: int = 200
    # v3.0.9: marker colour.  "white" = tape (fails in front of white walls, 17 Nov 14:05); "tan" = brown
    # cardboard squares, measured 17 Nov on both cameras: hue 18-21, saturation 77-143, brightness 108-211.
    marker_color: str = "white"
    # v3.0.10: if the hip and knee tapes are found but the shin tape is not (deep flexion hides it), take the
    # shank DIRECTION from the pose knee -> ankle keypoints, anchored at the knee tape.
    shin_fallback: bool = True
    markers_required: int = 3        # frames with fewer snapped joints are rejected in marker mode
    # v3.0.2: which joints must come from tape (others fall back to the pose keypoint).  The ankle
    # tape at the trouser hem is often hidden from the near camera; the ankle keypoint is reliable.
    marker_joints: str = "HKA"
    # v3.0.3 (Fable review): the third tape sits on the lateral SHANK, 10-15 cm above the hem (the hem
    # hides an ankle tape).  Its search starts this far from the knee towards the ankle keypoint.
    shank_seed: float = 0.7
    # standing offset: subtract the quiet-stand angle so standing reads 0.  None = automatic:
    # ON for pose keypoints (they sit on the surface, not the joint centre), OFF with tape markers
    # on the bony landmarks (the angle is anatomical; stroke patients may really stand flexed).
    use_offset: Optional[bool] = None


@dataclass
class LegOut:
    flexion: Optional[float] = None        # filtered, offset-corrected (what clinicians see)
    raw: Optional[float] = None            # this frame, before filter / offset
    status: str = ""
    pts_px: Optional[np.ndarray] = None    # hip, knee, ankle in the source image


def leg_keypoints(people: List, side: str, min_leg_px: float = 0.0):
    """Nearest person (longest hip-knee-ankle chain in pixels, weighted by confidence) ->
    (3x2 xy, 3 conf) for the COCO side, or None.  v3.0.5: legs shorter than min_leg_px are
    ignored, so a bystander in the background is not measured (seen 16 Nov, empty walker)."""
    best, bs = None, -1.0
    for p in people or []:
        xy = np.asarray(p["xy"] if isinstance(p, dict) else p.xy, float)
        c = np.asarray(p["c"] if isinstance(p, dict) else p.conf, float)
        ix = list(JOINTS[side])
        L = float(np.linalg.norm(xy[ix[0]] - xy[ix[1]]) + np.linalg.norm(xy[ix[2]] - xy[ix[1]]))
        if L < min_leg_px:
            continue
        q = L * float(np.mean(c[ix]))
        if q > bs:
            best, bs = (xy[ix], c[ix]), q
    return best


class SideTracker:
    def __init__(self, planes: Optional[PlaneMap] = None, cfg: Optional[SideConfig] = None):
        self.planes = planes or PlaneMap()
        self.cfg = cfg or SideConfig()
        self.reset()

    def reset(self):
        c = self.cfg
        self.filters = {s: OneEuroFilter(c.filter_min_cutoff, c.filter_beta) for s in "LR"}
        self.last = {s: None for s in "LR"}           # (t, raw) of last accepted value
        self.lengths = {s: None for s in "LR"}        # standing (thigh, shank) in plane units
        self._len_samples = {s: [] for s in "LR"}
        self._last_tape = {}
        self.offset = None
        self.neutral_failed = False
        self._t_first = None
        self._neutral = {s: [] for s in "LR"}
        self._neutral_t0 = None

    # -- standing reference (first window only) --------------------------
    def _neutral_step(self, t, raws, lens):
        c = self.cfg
        if self.offset is not None:
            return
        if self._t_first is None:
            self._t_first = t
        if t - self._t_first > c.neutral_window_s:
            self.offset, self.neutral_failed = {"L": 0.0, "R": 0.0}, True
            for s in "LR":                       # still learn lengths so the gate works
                if self.lengths[s] is None and len(self._len_samples[s]) >= 5:
                    self.lengths[s] = tuple(np.median(np.array(self._len_samples[s]), axis=0))
            return
        if not all(s in raws for s in "LR"):
            return
        if self._neutral_t0 is None:
            self._neutral_t0 = t
        for s in "LR":
            self._neutral[s].append(raws[s])
            self._len_samples[s].append(lens[s])
        if t - self._neutral_t0 >= c.neutral_seconds and all(len(v) >= 8 for v in self._neutral.values()):
            off = {}
            for s, v in self._neutral.items():
                v = np.asarray(v)
                if np.percentile(v, 90) - np.percentile(v, 10) > c.neutral_max_spread or np.median(v) > 30:
                    self._neutral, self._neutral_t0 = {"L": [], "R": []}, None
                    self._len_samples = {"L": [], "R": []}
                    return
                off[s] = float(np.median(v))
            self.offset = off
            for s in "LR":
                self.lengths[s] = tuple(np.median(np.array(self._len_samples[s]), axis=0))

    def process(self, t, people: Dict[str, List], images: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, LegOut]:
        """people: {"cam0": [...], "cam1": [...]} pose results (gait3d Person or recorded dicts).
        images: the same frames, needed only in marker mode."""
        c = self.cfg
        out, raws, lens = {}, {}, {}
        for s in "LR":
            cam = c.source[s]
            o = LegOut()
            kp = leg_keypoints(people.get(cam, []), s, c.min_leg_px)
            if kp is None:
                o.status = "no person"
            elif float(kp[1].min()) < c.kp_conf:
                o.status, o.pts_px = "low confidence", kp[0]
            else:
                pts = kp[0]
                if c.markers:
                    img = (images or {}).get(cam)
                    fl = [False, False, False]
                    seed = np.array(pts, float).copy()
                    seed[2] = seed[1] + c.shank_seed * (seed[2] - seed[1])
                    if img is not None:
                        pts, _ = snap_to_markers(img, seed, c.marker_radius_px, flags=fl,
                                                 blobs=marker_blobs(img, c), shank=(kp[0][1], kp[0][2]),
                                                 prev=self._last_tape.get(s))
                        self._last_tape[s] = [tuple(q) if f else None for q, f in zip(pts, fl)]
                    need = [j for j, ch in enumerate("HKA") if ch in c.marker_joints]
                    if c.shin_fallback and fl[0] and fl[1] and not fl[2]:
                        pts = np.array(pts, float)
                        pts[2] = pts[1] + c.shank_seed * (np.asarray(kp[0][2], float) - np.asarray(kp[0][1], float))
                        fl[2] = True
                        o.status = "ok (shin from pose)"
                    if not all(fl[j] for j in need):
                        o.status, o.pts_px = "markers " + "".join("HKA"[j] if fl[j] else "-" for j in range(3)), pts
                        out[s] = o
                        continue
                o.pts_px = pts
                P = self.planes.to_plane(cam, s, pts)
                th, sh = float(np.linalg.norm(P[0] - P[1])), float(np.linalg.norm(P[2] - P[1]))
                ref = self.lengths[s]
                if ref is not None and not (abs(th / ref[0] - 1) <= c.len_tol and (o.status == "ok (shin from pose)" or abs(sh / ref[1] - 1) <= c.len_tol)):
                    o.status = "length gate"
                else:
                    f = flexion_2d(*P)
                    if f is not None:
                        prev = self.last[s]
                        if prev is not None and t - prev[0] < 0.5 and abs(f - prev[1]) > c.max_step_deg:
                            o.status = "jump gate"
                        else:
                            o.raw = f
                            if o.status != "ok (shin from pose)":     # reference lengths: real tapes only
                                raws[s], lens[s] = f, (th, sh)
            out[s] = o
        self._neutral_step(t, raws, lens)
        for s, o in out.items():
            if o.raw is not None:
                self.last[s] = (t, o.raw)
                use = c.use_offset if c.use_offset is not None else not c.markers
                off = self.offset[s] if (self.offset and use) else 0.0
                o.flexion = float(self.filters[s](o.raw - off, t))
                o.status = o.status or ("calibrating: stand still" if self.offset is None else "ok")
            elif self.last[s] is not None and t - self.last[s][0] <= c.hold_s and self.filters[s]._t is not None:
                o.flexion = float(self.filters[s]._x.y)          # brief dropout: hold
            else:
                self.filters[s].reset()
        return out
