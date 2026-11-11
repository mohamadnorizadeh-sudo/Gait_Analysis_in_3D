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
from gait3d.camera import open_source  # noqa: E402
from gait3d.pose import PoseEstimator  # noqa: E402
from gait3d.profile import SubjectProfile, load_or_none  # noqa: E402
from gait3d.stereo_geometry import load_geometry, make_sgbm  # noqa: E402
from gait3d.tracker import CsvLogger, GaitTracker, TrackerConfig, draw  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--calib", default="stereo_calibration.npz")
ap.add_argument("--model", default="yolov8n-pose.pt",
                help="Ultralytics weights, or a Hailo .hef (e.g. /usr/share/hailo-models/yolov8m_pose_h10.hef)")
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
ap.add_argument("--web-port", type=int, default=8080, help="live web dashboard port (0 = off)")
ap.add_argument("--record-kp", default="", help="also save raw keypoints (JSON lines) for offline replay")
ap.add_argument("--source", choices=["picamera", "shm"], default="picamera",
                help="shm = read frames from the running camera_shm_writer.py (cameras stay with it)")
ap.add_argument("--rot-left", type=int, default=0, help="extra rotation of the LEFT image (0/90/180/270)")
ap.add_argument("--rot-right", type=int, default=0, help="extra rotation of the RIGHT image, e.g. 180 if mounted upside down")
ap.add_argument("--swap", action="store_true", help="shm source: use cam1 as the LEFT camera")
args = ap.parse_args()

geo = load_geometry(args.calib)
RECT = getattr(geo, "rectified", True)
if RECT:
    (ml, mr) = geo.rectify_maps()
    print(f"[stereo] rectified  f={geo.f:.1f}px  B={geo.baseline_mm:.1f}mm")
else:
    # v2.2: converging rig -> YOLO on raw images, keypoints undistorted + triangulated directly
    print(f"[stereo] raw (no rectification)  B={geo.baseline_mm:.1f}mm  convergence={geo.convergence_deg:.1f}deg")
    if args.sgbm:
        print("[sgbm] disabled: dense matching needs a rectified pair")
        args.sgbm = False

if args.thigh_mm and args.shin_mm:
    profile = SubjectProfile.manual(args.thigh_mm, args.shin_mm)
else:
    profile = load_or_none(args.profile)     # None -> 2 s standing calibration at start
if profile:
    print(f"[profile] thigh L/R {profile.L_thigh_L:.0f}/{profile.L_thigh_R:.0f} mm, "
          f"shin L/R {profile.L_shin_L:.0f}/{profile.L_shin_R:.0f} mm ({profile.method})")

tracker = GaitTracker(geo, profile, TrackerConfig(kp_conf=args.kp_conf))
if args.model.endswith(".hef"):
    from gait3d.hailo_pose import HailoPoseEstimator          # AI HAT+ 2 (Hailo-10H), v2.4
    try:
        pose = HailoPoseEstimator(args.model)
        print(f"[pose] Hailo {args.model}")
    except Exception as e:                                     # v2.8.1: keep working without the HAT
        fb = os.path.expanduser("~/yolov8n-pose.pt")
        print(f"[pose] WARNING: AI HAT+ 2 not available ({type(e).__name__}); "
              f"falling back to CPU {fb} - lower accuracy. Reboot the Pi to reset the Hailo driver.")
        pose = PoseEstimator(fb, 320)
else:
    pose = PoseEstimator(args.model, args.imgsz)
sgbm = None
if args.sgbm:
    nd = geo.num_disparities_for(args.z_min_mm)
    sgbm = make_sgbm(nd)
    print(f"[sgbm] numDisparities={nd} -> closest Z {geo.min_measurable_depth(0, nd) / 10:.1f} cm")

log = CsvLogger(args.csv)
from gait3d.dashboard import LiveData, WebDashboard, render_screen  # noqa: E402
live = LiveData()
from gait3d.recorder import KeypointRecorder  # noqa: E402
rec = KeypointRecorder(args.record_kp) if args.record_kp else None
live.csv_path = args.csv
web = WebDashboard(live, args.web_port).start() if args.web_port else None
if web:
    import socket
    try:                                   # primary LAN address (not 127.0.1.1)
        _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _s.connect(("10.255.255.255", 1))
        ip = _s.getsockname()[0]
        _s.close()
    except OSError:
        ip = "<pi-ip>"
    print(f"[web] live dashboard: http://{ip}:{args.web_port}  (or http://localhost:{args.web_port} on the Pi)")
writer = None
fps, t_prev, t0 = 0.0, None, time.monotonic()
saved_profile = profile
# v2.8.2: explicit stop flag. With the Qt window + HailoRT threads, a plain
# KeyboardInterrupt was sometimes swallowed and the process never exited.
import signal  # noqa: E402
STOP = {"flag": False}
def _stop(signum, frame):
    STOP["flag"] = True
signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)
try:
    with open_source(args.source, lens_position=args.lens, swap=args.swap, rot=(args.rot_left, args.rot_right)) as cam:
        while True:
            t, L, R, skew = cam.read()
            if RECT:
                rl = cv2.remap(L, *ml, cv2.INTER_LINEAR)
                rr = cv2.remap(R, *mr, cv2.INTER_LINEAR)
            else:
                rl, rr = L, R
            res_l, res_r = pose([rl, rr])                      # one batched inference
            sg = None
            if sgbm is not None:
                gl, gr = cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY)
                disp = sgbm.compute(gl, gr).astype(np.float32) / 16.0
                sg = (cv2.reprojectImageTo3D(disp, geo.Q), disp, float(sgbm.getNumDisparities()))
            out = tracker.process(t, res_l, res_r, sg)
            if rec:
                rec.write(t, res_l, res_r)
            log.log(t, out)
            log.f.flush()

            now = time.monotonic()
            if t_prev is not None:
                fps = 0.9 * fps + 0.1 / max(1e-3, now - t_prev) if fps else 1.0 / max(1e-3, now - t_prev)
            t_prev = now
            vis = draw(rl, out, fps, skew)
            try:                                   # v2.12: small live inset of the RIGHT camera (depth camera)
                h, w = vis.shape[:2]
                iw = w // 3
                ih = int(rr.shape[0] * iw / rr.shape[1])
                ins = cv2.resize(rr, (iw, ih))
                sx, sy = iw / rr.shape[1], ih / rr.shape[0]
                for side_, ob_ in out.obs.items():
                    if ob_.xy_r is None:
                        continue
                    col_ = (255, 128, 0) if side_ == "L" else (0, 128, 255)
                    for (x_, y_), c_ in zip(ob_.xy_r, ob_.conf_r):
                        if c_ >= 0.5:
                            cv2.circle(ins, (int(x_ * sx), int(y_ * sy)), 3, col_, -1)
                cv2.rectangle(ins, (0, 0), (iw - 1, ih - 1), (255, 255, 255), 1)
                cv2.putText(ins, "right cam (depth)", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                vis[h - ih - 6:h - 6, w - iw - 6:w - 6] = ins
            except Exception:  # noqa: BLE001  never let the display stop the measurement
                pass
            if tracker.profile is not None:
                live.subject = {s_: f"{tracker.profile.lengths(s_)[0]:.0f}/{tracker.profile.lengths(s_)[1]:.0f}" for s_ in "LR"}
            live.push(t, out.flexion_corrected, out.status, fps, vis)
            if args.save_video:
                if writer is None:
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                             vis.shape[1::-1])
                writer.write(vis)
            if args.headless:
                legs = "  ".join(f"{s}:{180 - out.angle_filtered[s]:5.1f}" for s in out.angle_filtered)
                print(f"{fps:4.1f} fps  skew {skew:4.1f}ms  {out.status:<40} {legs}", end="\r")
            else:
                cv2.imshow("Gait 3D v2 - knee flexion", render_screen(vis, live))
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
            if STOP["flag"]:
                print("\n[stop] signal received - closing")
                break
finally:
    log.close()
    if rec:
        rec.close()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    try:
        if hasattr(pose, "close"):
            pose.close()                       # release the AI HAT for the next run
    except Exception as e:  # noqa: BLE001
        print(f"[stop] pose close: {e}")
    print(f"\nangles logged to {args.csv}", flush=True)
    _exc = sys.exc_info()[0]
    if _exc is not None and _exc is not KeyboardInterrupt:
        import traceback
        traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    if not os.environ.get("GAIT3D_NO_HARD_EXIT"):   # tests run this script in-process
        os._exit(1 if _exc not in (None, KeyboardInterrupt) else 0)                                # do not wait on lingering library threads
