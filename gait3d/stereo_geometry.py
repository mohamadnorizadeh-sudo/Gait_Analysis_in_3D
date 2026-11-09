"""Stereo geometry helpers built on top of stereo_calibration.npz.

All 3D coordinates are in the *rectified left camera* frame, in millimetres,
exactly the same frame cv2.reprojectImageTo3D(disparity, Q) produces.

v2 notes (issues fixed versus the v1 notebook)
---------------------------------------------
* v1 always computed full-frame SGBM depth and sampled a patch at each joint.
  On plain clothing that gives holes, and when the leg is closer than
  f*B/(minDisparity+numDisparities) the matcher saturates -- that is why every
  joint in the v1 log read exactly 24.0 cm or 25.9 cm.  Here we expose
  ``min_measurable_depth`` and ``num_disparities_for`` so the range can be sized
  from the real rig geometry, and ``triangulate`` so YOLO keypoints from both
  cameras can be triangulated directly (no SGBM needed for joints).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class StereoGeometry:
    K_l: np.ndarray
    D_l: np.ndarray
    R1: np.ndarray
    P1: np.ndarray
    K_r: np.ndarray
    D_r: np.ndarray
    R2: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    image_size: tuple  # (w, h)

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str = "stereo_calibration.npz", image_size=(640, 480)) -> "StereoGeometry":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' not found. Run scripts/02_calibrate.py first.")
        c = np.load(path)
        size = tuple(int(v) for v in c["image_size"]) if "image_size" in c else tuple(image_size)
        return cls(c["K_l"], c["D_l"], c["R1"], c["P1"],
                   c["K_r"], c["D_r"], c["R2"], c["P2"], c["Q"], size)

    # ------------------------------------------------------------------ #
    # intrinsics of the rectified pair
    # ------------------------------------------------------------------ #
    @property
    def f(self) -> float:
        return float(self.P1[0, 0])

    @property
    def fy(self) -> float:
        return float(self.P1[1, 1])

    @property
    def cx(self) -> float:
        return float(self.P1[0, 2])

    @property
    def cy(self) -> float:
        return float(self.P1[1, 2])

    @property
    def cx_right(self) -> float:
        return float(self.P2[0, 2])

    @property
    def baseline_mm(self) -> float:
        # P2[0,3] = -f * B for a horizontal rig (sign depends on camera order)
        return abs(float(self.P2[0, 3]) / float(self.P2[0, 0]))

    @property
    def doffs(self) -> float:
        """Principal-point offset between the rectified cameras (0 with CALIB_ZERO_DISPARITY)."""
        return self.cx - self.cx_right

    def depth_from_disparity(self, d):
        return self.f * self.baseline_mm / (np.asarray(d, dtype=np.float64) - self.doffs)

    def disparity_from_depth(self, z_mm):
        return self.f * self.baseline_mm / np.asarray(z_mm, dtype=np.float64) + self.doffs

    def min_measurable_depth(self, min_disparity: int, num_disparities: int) -> float:
        """Closest depth SGBM can represent. Anything nearer saturates (v1 bug)."""
        return float(self.depth_from_disparity(min_disparity + num_disparities - 1))

    def num_disparities_for(self, z_min_mm: float, min_disparity: int = 0) -> int:
        """Smallest multiple of 16 that lets SGBM measure down to z_min_mm."""
        d_max = float(self.disparity_from_depth(z_min_mm)) - min_disparity + 1
        return int(np.ceil(max(d_max, 16) / 16.0) * 16)

    # ------------------------------------------------------------------ #
    # rectification
    # ------------------------------------------------------------------ #
    def rectify_maps(self):
        w, h = self.image_size
        ml = cv2.fisheye.initUndistortRectifyMap(self.K_l, self.D_l, self.R1, self.P1, (w, h), cv2.CV_16SC2)
        mr = cv2.fisheye.initUndistortRectifyMap(self.K_r, self.D_r, self.R2, self.P2, (w, h), cv2.CV_16SC2)
        return ml, mr

    # ------------------------------------------------------------------ #
    # 3D
    # ------------------------------------------------------------------ #
    def triangulate(self, xl, yl, xr) -> np.ndarray:
        """Rectified left pixel + right x-coordinate -> 3D point (mm) via Q.

        Uses the same Q as cv2.reprojectImageTo3D, so the result lives in the
        same frame as an SGBM point cloud.
        """
        d = float(xl) - float(xr)
        v = self.Q @ np.array([float(xl), float(yl), d, 1.0])
        if abs(v[3]) < 1e-12:
            return np.array([np.nan, np.nan, np.nan])
        return v[:3] / v[3]

    def ray(self, x, y) -> np.ndarray:
        """Back-projected ray of a rectified left pixel, scaled so ray[2] == 1.

        A point on the ray at depth Z is simply ``Z * ray``.
        """
        return np.array([(float(x) - self.cx) / self.f, (float(y) - self.cy) / self.fy, 1.0])

    def project_left(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.array([self.f * X[0] / X[2] + self.cx, self.fy * X[1] / X[2] + self.cy])

    def depth_sigma(self, z_mm: float, disparity_sigma_px: float = 1.0) -> float:
        """1-sigma depth uncertainty: sigma_Z = Z^2 / (f*B) * sigma_d."""
        return float(z_mm) ** 2 / (self.f * self.baseline_mm) * disparity_sigma_px


def make_sgbm(num_disparities: int, min_disparity: int = 0, block_size: int = 5):
    """SGBM tuned for close-range limbs.  num_disparities MUST be sized with
    StereoGeometry.num_disparities_for(z_min) -- the v1 value (64) could not
    see anything closer than ~25 cm on this rig."""
    num_disparities = int(np.ceil(num_disparities / 16.0) * 16)
    return cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size ** 2,          # single-channel (grayscale) penalties
        P2=32 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,                   # in disparity units (x16 internally)
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,  # multi-threaded, fastest on the Pi 5's 4 cores
    )


def sample_sgbm_point(points_3d, disparity, px, py, d_max, z_range=(80.0, 2500.0), window=9):
    """Median 3D point in a patch, ignoring holes AND saturated disparities.

    v1 accepted saturated pixels (disparity == max) as valid, which produced
    the constant 24.0 / 25.9 cm readings.
    """
    h, w = disparity.shape[:2]
    px, py = int(round(px)), int(round(py))
    half = window // 2
    y0, y1 = max(0, py - half), min(h, py + half + 1)
    x0, x1 = max(0, px - half), min(w, px + half + 1)
    if y0 >= y1 or x0 >= x1:
        return None
    d = disparity[y0:y1, x0:x1].reshape(-1)
    pts = points_3d[y0:y1, x0:x1].reshape(-1, 3)
    ok = (d > 0) & (d < d_max - 1.0) & np.isfinite(pts).all(axis=1)
    ok &= (pts[:, 2] > z_range[0]) & (pts[:, 2] < z_range[1])
    if ok.sum() < max(4, window):
        return None
    return np.median(pts[ok], axis=0)
