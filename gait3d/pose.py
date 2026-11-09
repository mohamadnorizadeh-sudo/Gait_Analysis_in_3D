"""YOLOv8-Pose wrapper with per-joint confidence gating and ghost-leg suppression.

Issues fixed versus v1
----------------------
* v1 tested ``hip[0] > 0`` to decide a joint is visible.  YOLO always outputs a
  coordinate (often a guess), so occluded legs were still drawn.  v2 gates on
  ``keypoints.conf`` -- hip, knee AND ankle must each be >= 0.50.
* v1 always took ``keypoints.xy[0]`` (first detection), not the best person.
* When only one leg is visible YOLO often places BOTH the L and R labels on the
  same limb ("ghost leg").  v2 detects two legs whose knees and ankles nearly
  coincide and keeps only the more confident one.
* v2 runs the detector on the left AND right rectified images in one batch and
  matches keypoints along epipolar lines, so each joint gets a direct stereo
  triangulation instead of relying on SGBM texture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# COCO keypoint indices
LEG_IDX = {"L": (11, 13, 15), "R": (12, 14, 16)}   # (hip, knee, ankle)


@dataclass
class Person:
    xy: np.ndarray      # (17, 2) pixels
    conf: np.ndarray    # (17,)
    score: float


@dataclass
class LegObs:
    side: str
    xy: np.ndarray                 # (3, 2) left rectified pixels: hip, knee, ankle
    conf: np.ndarray               # (3,)
    xr: np.ndarray                 # (3,) right-image x (nan if not matched)

    @property
    def mean_conf(self) -> float:
        return float(np.mean(self.conf))


def people_from_result(result) -> List[Person]:
    out = []
    kp = getattr(result, "keypoints", None)
    if kp is None or kp.xy is None or len(kp.xy) == 0:
        return out
    xy = kp.xy.cpu().numpy()
    conf = kp.conf.cpu().numpy() if kp.conf is not None else np.ones(xy.shape[:2])
    boxes = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(xy))
    for i in range(len(xy)):
        out.append(Person(xy[i], conf[i], float(boxes[i])))
    return out


def best_person(people: List[Person]) -> Optional[Person]:
    if not people:
        return None
    legs = [11, 12, 13, 14, 15, 16]
    return max(people, key=lambda p: float(np.mean(p.conf[legs])) * p.score)


def match_right(left: Person, rights: List[Person], thr: float, epi_tol_px: float) -> Optional[Person]:
    """Pick the right-image person whose leg joints sit on the same rectified rows."""
    legs = [11, 12, 13, 14, 15, 16]
    best, best_err = None, np.inf
    for p in rights:
        ok = (left.conf[legs] >= thr) & (p.conf[legs] >= thr)
        if ok.sum() < 2:
            continue
        idx = np.array(legs)[ok]
        dy = np.abs(left.xy[idx, 1] - p.xy[idx, 1])
        disp = left.xy[idx, 0] - p.xy[idx, 0]
        if np.median(disp) <= 0:
            continue
        err = float(np.median(dy))
        if err < best_err:
            best, best_err = p, err
    return best if best_err <= 3 * epi_tol_px else None


def legs_from_people(left: Person, right: Optional[Person], thr: float = 0.5,
                     epi_tol_px: float = 4.0, d_range=(1.0, 400.0)) -> Dict[str, LegObs]:
    """Return only legs whose hip, knee and ankle ALL pass the confidence gate."""
    legs = {}
    for side, idx in LEG_IDX.items():
        idx = list(idx)
        c = left.conf[idx]
        if np.any(c < thr):
            continue                       # occluded -> leg disappears entirely
        xr = np.full(3, np.nan)
        if right is not None:
            for j, k in enumerate(idx):
                if right.conf[k] < thr:
                    continue
                dy = abs(left.xy[k, 1] - right.xy[k, 1])
                d = left.xy[k, 0] - right.xy[k, 0]
                if dy <= epi_tol_px and d_range[0] <= d <= d_range[1]:
                    xr[j] = right.xy[k, 0]
        legs[side] = LegObs(side, left.xy[idx].astype(float), c.astype(float), xr)
    return legs


def suppress_ghost_legs(legs: Dict[str, LegObs], ratio: float = 0.35) -> Dict[str, LegObs]:
    """If L and R knees AND ankles almost coincide, they are the same limb."""
    if "L" not in legs or "R" not in legs:
        return legs
    L, R = legs["L"], legs["R"]
    shin_px = 0.5 * (np.linalg.norm(L.xy[1] - L.xy[2]) + np.linalg.norm(R.xy[1] - R.xy[2]))
    if shin_px < 1:
        return legs
    d_knee = np.linalg.norm(L.xy[1] - R.xy[1])
    d_ankle = np.linalg.norm(L.xy[2] - R.xy[2])
    if d_knee < ratio * shin_px and d_ankle < ratio * shin_px:
        keep = "L" if L.mean_conf >= R.mean_conf else "R"
        return {keep: legs[keep]}
    return legs


class PoseEstimator:
    """Thin wrapper so the rest of the code does not depend on ultralytics."""

    def __init__(self, weights: str = "yolov8n-pose.pt", imgsz: int = 480, det_conf: float = 0.25):
        from ultralytics import YOLO        # lazy import (slow on the Pi)
        self.model = YOLO(weights, task="pose")
        self.imgsz, self.det_conf = imgsz, det_conf
        # PyTorch weights batch L+R in one call; exported backends (NCNN, ONNX) are
        # exported with batch=1, so run them one image at a time.
        self.batched = str(weights).endswith(".pt")

    def __call__(self, images):
        kw = dict(imgsz=self.imgsz, conf=self.det_conf, verbose=False)
        if self.batched:
            return list(self.model(list(images), **kw))
        return [self.model(im, **kw)[0] for im in images]
