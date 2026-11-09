#!/usr/bin/env python3
"""Step 2 - fisheye stereo calibration -> stereo_calibration.npz (v2).

Keeps the same keys as v1 (K_l, D_l, R1, P1, K_r, D_r, R2, P2, Q) and adds
R, T and image_size.  Writes calibration_report.txt with the quality checks
explained in gait3d/calibration.py.

Usage:  python3 scripts/02_calibrate.py --square-mm 25 --balance 0.0
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gait3d.calibration import SUBPIX, object_points, stereo_calibrate  # noqa: E402
from gait3d.stereo_geometry import StereoGeometry  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="calib_data")
ap.add_argument("--cols", type=int, default=9)
ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--square-mm", type=float, default=25.0, help="MEASURE your printed square with a ruler")
ap.add_argument("--balance", type=float, default=0.0)
ap.add_argument("--fov-scale", type=float, default=1.0)
ap.add_argument("--out", default="stereo_calibration.npz")
ap.add_argument("--z-min-mm", type=float, default=150.0, help="closest leg distance you need to measure")
args = ap.parse_args()

CB = (args.cols, args.rows)
objp = object_points(args.cols, args.rows, args.square_mm)
pl = sorted(glob.glob(f"{args.data}/left/*.png"))
pr = sorted(glob.glob(f"{args.data}/right/*.png"))
assert len(pl) == len(pr) and pl, "no / mismatched calibration images"

flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
obj, il, ir, size = [], [], [], None
for a, b in zip(pl, pr):
    ga, gb = cv2.imread(a, cv2.IMREAD_GRAYSCALE), cv2.imread(b, cv2.IMREAD_GRAYSCALE)
    size = ga.shape[::-1]
    oka, ca = cv2.findChessboardCorners(ga, CB, flags)
    okb, cb = cv2.findChessboardCorners(gb, CB, flags)
    if not (oka and okb):
        print(f"skip {os.path.basename(a)} (board not found in both)")
        continue
    # 5x5 search window (v1 used 3x3, too small to converge on blurred corners)
    ca = cv2.cornerSubPix(ga, ca, (5, 5), (-1, -1), SUBPIX)
    cb = cv2.cornerSubPix(gb, cb, (5, 5), (-1, -1), SUBPIX)
    obj.append(objp.copy())
    il.append(ca.reshape(1, -1, 2).astype(np.float64))
    ir.append(cb.reshape(1, -1, 2).astype(np.float64))

print(f"{len(obj)}/{len(pl)} pairs with corners in both views")
if len(obj) < 15:
    print("WARNING: < 15 pairs. Aim for 30-40 covering the whole field of view.")

data, rep = stereo_calibrate(obj, il, ir, size, args.balance, args.fov_scale)
np.savez(args.out, **data)

geo = StereoGeometry(data["K_l"], data["D_l"], data["R1"], data["P1"], data["K_r"], data["D_r"],
                     data["R2"], data["P2"], data["Q"], tuple(size))
rep["sgbm_num_disparities_for_z_min"] = geo.num_disparities_for(args.z_min_mm)
rep["z_min_mm_used"] = args.z_min_mm
rep["z_min_with_v1_sgbm(2,64)_mm"] = geo.min_measurable_depth(2, 64)

lines = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in rep.items()]
verdict = []
if rep["epipolar_mean_px"] > 0.5:
    verdict.append("! epipolar error > 0.5 px - recapture with sharper, better spread images")
if rep["views_used"] < 15:
    verdict.append("! few views - intrinsics may be over-fitted even if RMS looks low")
verdict.append(f"-> check: does the measured baseline ({rep['baseline_mm']:.1f} mm) match a ruler "
               f"measurement between the two lens centres? If not, --square-mm is wrong.")
txt = "\n".join(lines + [""] + verdict)
print("\n" + txt)
with open("calibration_report.txt", "w") as f:
    f.write(txt + "\n")
with open("calibration_report.json", "w") as f:
    json.dump(rep, f, indent=2)
print(f"\nsaved {args.out}, calibration_report.txt")
