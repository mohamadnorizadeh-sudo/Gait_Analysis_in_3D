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
from gait3d.stereo_geometry import StereoGeometry  # noqa: E402,F401

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="calib_data")
ap.add_argument("--cols", type=int, default=9)
ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--square-mm", type=float, default=25.0, help="MEASURE your printed square with a ruler")
ap.add_argument("--balance", type=float, default=0.0)
ap.add_argument("--fov-scale", type=float, default=1.0)
ap.add_argument("--out", default="stereo_calibration.npz")
ap.add_argument("--z-min-mm", type=float, default=150.0, help="closest leg distance you need to measure")
ap.add_argument("--mono-dir", default="", help="v2.2: folder with cam0/ and cam1/ single-camera views "
                "(scripts/cap_mono.py) used for the intrinsics of a converging rig")
ap.add_argument("--ruler-baseline-mm", type=float, default=0.0, help="lens-to-lens distance measured with a ruler")
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


def load_mono(folder):
    o, im = [], []
    for f in sorted(glob.glob(f"{folder}/*.png")):
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        ok, c = cv2.findChessboardCorners(g, CB, flags)
        if not ok:
            continue
        c = cv2.cornerSubPix(g, c, (5, 5), (-1, -1), SUBPIX)
        o.append(objp.copy())
        im.append(c.reshape(1, -1, 2).astype(np.float64))
    print(f"{folder}: {len(o)} single-camera views")
    return o, im


mono_l = load_mono(f"{args.mono_dir}/cam0") if args.mono_dir else None
mono_r = load_mono(f"{args.mono_dir}/cam1") if args.mono_dir else None
data, rep = stereo_calibrate(obj, il, ir, size, args.balance, args.fov_scale,
                             mono_l=mono_l, mono_r=mono_r, square_mm=args.square_mm)
np.savez(args.out, **data)

if rep["mode"] == "rectified":
    geo = StereoGeometry(data["K_l"], data["D_l"], data["R1"], data["P1"], data["K_r"], data["D_r"],
                         data["R2"], data["P2"], data["Q"], tuple(size))
    rep["sgbm_num_disparities_for_z_min"] = geo.num_disparities_for(args.z_min_mm)
if args.ruler_baseline_mm:
    rep["ruler_baseline_mm"] = args.ruler_baseline_mm
    rep["baseline_vs_ruler_pct"] = 100 * (rep["baseline_mm"] / args.ruler_baseline_mm - 1)

lines = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in rep.items()]
verdict = []
if rep.get("epipolar_mean_px", 0) > 0.5:
    verdict.append("! epipolar error > 0.5 px - recapture with sharper, better spread images")
if max(rep["rms_left"], rep["rms_right"]) > 0.8:
    verdict.append("! single-camera RMS > 0.8 px - add more views covering the whole image")
if rep["board_square_3d_std_mm"] > 0.05 * args.square_mm:
    verdict.append("! reconstructed squares vary > 5 % - 3D precision is limited")
verdict.append(f"-> baseline {rep['baseline_mm']:.1f} mm; compare with a ruler between the lens centres. "
               f"A constant ratio means --square-mm is wrong (measure 9 squares / 9).")
txt = "\n".join(lines + [""] + verdict)
print("\n" + txt)
with open("calibration_report.txt", "w") as f:
    f.write(txt + "\n")
with open("calibration_report.json", "w") as f:
    json.dump(rep, f, indent=2)
print(f"\nsaved {args.out}, calibration_report.txt")
