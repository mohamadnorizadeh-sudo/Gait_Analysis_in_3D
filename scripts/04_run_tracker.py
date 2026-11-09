#!/usr/bin/env python3
"""Step 4 - rigid, rotation-invariant 3D knee-angle tracker (v2).

On start the subject stands straight for 2 s (bone-length calibration),
unless --profile or --thigh-mm/--shin-mm is given.  Then:
  * hip, knee, ankle must each have confidence >= 0.50 or the leg is hidden
  * each joint is triangulated from the left+right keypoints (SGBM optional)
  * the rigid solver keeps L_thigh / L_shin exact and fills missing depth
  * no 2D fallback, ever: without 3D the HUD shows "--"

Keys:  q quit | c re-run standing calibration | s save subject profile

Usage examples
  python3 scripts/04_run_tracker.py                          # calibrate, then track
  python3 scripts/04_run_tracker.py --profile subject.json   # reuse bone lengths
  python3 scripts/04_run_tracker.py --thigh-mm 430 --shin-mm 410
  python3 scripts/04_run_tracker.py --model yolov8n-pose_ncnn_model --imgsz 416   # ~3x faster
  python3 scripts/04_run_tracker.py --headless --seconds 30 --csv walk1.csv
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gait3d.camera import StereoCamera  # noqa: E402
from gait3d.pose import PoseEstimator  # noqa: E402
from gait3d.profile import SubjectProfile, load_or_none  # noqa: E402
from gait3d.stereo_geometry import StereoGeometry, make_sgbm  # noqa: E402
from gait3d.tracker import CsvLogger, GaitTracker, TrackerConfig, draw  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--calib", default="stereo_calibration.npz")
ap.add_argument("--model", default="yolov8n-pose.pt")
ap.add_argument("--imgsz", type=int, default=480)
ap.add_argument("--lens", type=float, default=3.0, help="LensPosition (dioptres) - same as calibration!")
ap.add_argument("--profile", default=None, help="reuse bone lengths from a saved profile JSON")
ap.add_argument("--profile-out", default="subject_profile.json", help="where the standing calibration is saved")
ap.add_argument("--thigh-mm", type=float)
ap.add_argument("--shin-mm", type=float)
ap.add_argument("--kp-conf", type=float, default=0.50)
ap.add_argument("--sgbm", action="store_true", help="also use SGBM patches as a depth fallback (slower)")
ap.add_argument("--z-min-mm", type=float, default=150.0)
ap.add_argument("--csv", default=time.strftime("gait_%Y%m%d_%H%M%S.csv"))
ap.add_argument("--headless", action="store_true")
ap.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds (0 = until q)")
ap.add_argument("--save-video", default="")
args = ap.parse_args()

geo = StereoGeometry.load(args.calib)
(ml, mr) = geo.rectify_maps()
print(f"[stereo] f={geo.f:.1f}px  B={geo.baseline_mm:.1f}mm")

if args.thigh_mm and args.shin_mm:
    profile = SubjectProfile.manual(args.thigh_mm, args.shin_mm)
else:
    profile = load_or_none(args.profile)     # None -> 2 s standing calibration at start
if profile:
    print(f"[profile] thigh L/R {profile.L_thigh_L:.0f}/{profile.L_thigh_R:.0f} mm, "
          f"shin L/R {profile.L_shin_L:.0f}/{profile.L_shin_R:.0f} mm ({profile.method})")

tracker = GaitTracker(geo, profile, TrackerConfig(kp_conf=args.kp_conf))
pose = PoseEstimator(args.model, args.imgsz)
sgbm = None
if args.sgbm:
    nd = geo.num_disparities_for(args.z_min_mm)
    sgbm = make_sgbm(nd)
    print(f"[sgbm] numDisparities={nd} -> closest Z {geo.min_measurable_depth(0, nd) / 10:.1f} cm")

log = CsvLogger(args.csv)
writer = None
fps, t_prev, t0 = 0.0, None, time.monotonic()
saved_profile = profile
try:
    with StereoCamera(lens_position=args.lens) as cam:
        while True:
            t, L, R, skew = cam.read()
            rl = cv2.remap(L, *ml, cv2.INTER_LINEAR)
            rr = cv2.remap(R, *mr, cv2.INTER_LINEAR)
            res_l, res_r = pose([rl, rr])                      # one batched inference
            sg = None
            if sgbm is not None:
                gl, gr = cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY)
                disp = sgbm.compute(gl, gr).astype(np.float32) / 16.0
                sg = (cv2.reprojectImageTo3D(disp, geo.Q), disp, float(sgbm.getNumDisparities()))
            out = tracker.process(t, res_l, res_r, sg)
            log.log(t, out)

            now = time.monotonic()
            if t_prev is not None:
                fps = 0.9 * fps + 0.1 / max(1e-3, now - t_prev) if fps else 1.0 / max(1e-3, now - t_prev)
            t_prev = now
            vis = draw(rl, out, fps, skew)
            if args.save_video:
                if writer is None:
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                             vis.shape[1::-1])
                writer.write(vis)
            if args.headless:
                legs = "  ".join(f"{s}:{180 - out.angle_filtered[s]:5.1f}" for s in out.angle_filtered)
                print(f"{fps:4.1f} fps  skew {skew:4.1f}ms  {out.status:<40} {legs}", end="\r")
            else:
                cv2.imshow("Gait 3D v2", vis)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("c"):
                    tracker.recalibrate()
                if k == ord("s") and tracker.profile:
                    tracker.profile.save(args.profile_out)
                    print(f"\nsaved {args.profile_out}")
            if tracker.profile is not None and tracker.profile is not saved_profile:
                saved_profile = tracker.profile                    # new calibration finished
                if saved_profile.method == "standing":
                    saved_profile.save(args.profile_out)
                    print(f"\n[profile] {out.status} -> saved {args.profile_out}")
            if args.seconds and now - t0 > args.seconds:
                break
finally:
    log.close()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\nangles logged to {args.csv}")
