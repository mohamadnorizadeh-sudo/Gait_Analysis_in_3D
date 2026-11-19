#!/usr/bin/env python3
"""Step 3 - depth diagnostic: is the stereo range right for your legs?

This reproduces the v1 failure on purpose.  It prints the closest depth each
SGBM setting can measure and, live, the fraction of SATURATED pixels
(disparity at the max of the search range) in the centre of the image.
If saturation is high, the legs are closer than the matcher can see and
every Z collapses to one value (the 24.0 / 25.9 cm readings in v1).

Usage:  python3 scripts/03_check_depth.py --lens 3.0 --z-min-mm 150
        add --headless to print stats without opening windows.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gait3d.camera import open_source  # noqa: E402
from gait3d.stereo_geometry import StereoGeometry, make_sgbm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--calib", default="stereo_calibration.npz")
ap.add_argument("--lens", type=float, default=3.0)
ap.add_argument("--z-min-mm", type=float, default=150.0)
ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = until q)")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--source", choices=["picamera", "shm"], default="picamera")
ap.add_argument("--rot-left", type=int, default=0, help="extra rotation of the LEFT image (0/90/180/270)")
ap.add_argument("--rot-right", type=int, default=0, help="extra rotation of the RIGHT image, e.g. 180 if mounted upside down")
ap.add_argument("--swap", action="store_true")
args = ap.parse_args()

geo = StereoGeometry.load(args.calib)
print(f"rectified focal f = {geo.f:.1f} px, baseline B = {geo.baseline_mm:.1f} mm, doffs = {geo.doffs:.2f}")
for md, nd in ((2, 64), (0, 80), (0, 128), (0, 160)):
    print(f"  SGBM min={md:3d} num={nd:3d} -> closest measurable Z = {geo.min_measurable_depth(md, nd) / 10:.1f} cm")
nd = geo.num_disparities_for(args.z_min_mm)
print(f"-> for Z_min = {args.z_min_mm / 10:.0f} cm use numDisparities = {nd}")

(ml, mr) = geo.rectify_maps()
sgbm = make_sgbm(nd)
n = 0
with open_source(args.source, lens_position=args.lens, swap=args.swap, rot=(args.rot_left, args.rot_right)) as cam:
    while True:
        t, L, R, skew = cam.read()
        rl = cv2.remap(L, *ml, cv2.INTER_LINEAR)
        rr = cv2.remap(R, *mr, cv2.INTER_LINEAR)
        gl, gr = cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY)
        disp = sgbm.compute(gl, gr).astype(np.float32) / 16.0
        h, w = disp.shape
        c = disp[h // 4:3 * h // 4, w // 4:3 * w // 4]
        valid = c > 0
        sat = (c >= nd - 1.5).mean()
        z = geo.depth_from_disparity(c[valid]) if valid.any() else np.array([np.nan])
        line = (f"skew {skew:4.1f} ms | valid {valid.mean() * 100:4.0f}% | saturated {sat * 100:4.1f}% | "
                f"centre Z median {np.nanmedian(z) / 10:5.1f} cm")
        print(line, end="\r")
        if not args.headless:
            vis = cv2.applyColorMap(np.clip(disp * 255.0 / nd, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            vis[disp <= 0] = 0
            vis[disp >= nd - 1.5] = (255, 255, 255)        # saturated pixels in white
            cv2.putText(vis, line, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cv2.imshow("rectified L | R", np.hstack([rl, rr]))
            cv2.imshow("disparity (white = saturated, black = hole)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1
        if args.frames and n >= args.frames:
            break
print()
cv2.destroyAllWindows()
