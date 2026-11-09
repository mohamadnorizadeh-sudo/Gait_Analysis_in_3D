#!/usr/bin/env python3
"""Step 1 - capture stereo chessboard pairs (v2).

Changes versus notebook cell 0 (v1)
-----------------------------------
* Saves a pair ONLY if the chessboard is found in BOTH images, is sharp, and
  has moved since the last capture (v1 saved every 4 s blindly, so blurred or
  half-visible boards silently reduced the usable set -- the notebook notes
  "run with 7 pic").
* Uses the synchronised, exposure-locked StereoCamera and the SAME
  LensPosition you will track with.  Focus changes the focal length, so a
  calibration done at infinity focus does not exactly fit a 30 cm focus.
* Default target is 40 pairs.  Cover the WHOLE field of view (corners and
  edges matter most for a 120 deg fisheye), tilt the board +/-45 deg, and
  hold it at the distances where the legs will be (25-70 cm).
* Picamera2 RGB888 is already BGR -> saved without colour conversion.

Usage:  python3 scripts/01_capture_calibration.py --lens 3.0 --count 40
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gait3d.camera import StereoCamera  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lens", type=float, default=3.0, help="LensPosition in dioptres (3.0 = ~33 cm)")
ap.add_argument("--count", type=int, default=40)
ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
ap.add_argument("--out", default="calib_data")
ap.add_argument("--min-move-px", type=float, default=25.0)
ap.add_argument("--min-sharpness", type=float, default=60.0)
args = ap.parse_args()

CB = (args.cols, args.rows)
os.makedirs(f"{args.out}/left", exist_ok=True)
os.makedirs(f"{args.out}/right", exist_ok=True)
FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK


def find(gray):
    ok, c = cv2.findChessboardCorners(gray, CB, FLAGS)
    return (c if ok else None)


def sharpness(gray, corners):
    x, y, w, h = cv2.boundingRect(corners.astype(np.float32))
    roi = gray[max(0, y):y + h, max(0, x):x + w]
    return cv2.Laplacian(roi, cv2.CV_64F).var() if roi.size else 0.0


count = len(os.listdir(f"{args.out}/left"))
last = None
coverage = np.zeros((480, 640), np.uint8)
print(f"Starting at pair {count}. Press q to quit.")
with StereoCamera(lens_position=args.lens) as cam:
    while count < args.count:
        t, L, R, skew = cam.read()
        gl, gr = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
        cl, cr = find(gl), find(gr)
        vis_l, vis_r = L.copy(), R.copy()
        msg = "board not in both views"
        if cl is not None and cr is not None:
            cv2.drawChessboardCorners(vis_l, CB, cl, True)
            cv2.drawChessboardCorners(vis_r, CB, cr, True)
            sh = min(sharpness(gl, cl), sharpness(gr, cr))
            moved = last is None or np.mean(np.linalg.norm(cl - last, axis=2)) > args.min_move_px
            if sh < args.min_sharpness:
                msg = f"blurry ({sh:.0f}) - hold still"
            elif not moved:
                msg = "move the board to a new place / angle"
            elif skew > 8.0:
                msg = f"frames not in sync ({skew:.1f} ms)"
            else:
                cv2.imwrite(f"{args.out}/left/img_{count:02d}.png", L)
                cv2.imwrite(f"{args.out}/right/img_{count:02d}.png", R)
                cv2.fillConvexPoly(coverage, cv2.convexHull(cl.astype(np.int32)), 255)
                last, count = cl, count + 1
                msg = f"SAVED {count}/{args.count}"
                print(msg)
                vis_l[:] = 255 - vis_l
        cov = cv2.cvtColor(coverage, cv2.COLOR_GRAY2BGR)
        vis_l = cv2.addWeighted(vis_l, 0.85, cov, 0.15, 0)
        prev = np.hstack([vis_l, vis_r])
        cv2.putText(prev, f"{count}/{args.count}  {msg}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(prev, f"coverage {100 * coverage.mean() / 255:.0f}% (aim > 80%)", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("stereo calibration capture (v2)", prev)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cv2.destroyAllWindows()
print(f"Done: {count} pairs in {args.out}/  -> next: python3 scripts/02_calibrate.py")
