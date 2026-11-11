# Changelog

Every version records **what was wrong** and **what changed**. The original
notebook `calibrate_capture.ipynb` is kept unchanged as the v1 reference.

---

## v2.17 (2026-11-11) - learned fallback bias
**Issue (live v2.16.1, session 12:17):** while the right leg was bent, the straight left leg flickered between
~0 deg (both cameras saw it -> direct measurement) and 25-28 deg (partly hidden in the right camera -> fixed-length
solver, which over-reads).
**Fix:** on frames where all joints are triangulated the solver is also run and its excess flexion vs the direct
measurement is learned per leg (EMA 0.05); on fallback frames that bias is removed (`fallback_bias`, 0 = off).
**Replay of session 12:17:** straight left leg during the right hold 22 / 28 / 22 deg -> **8 / 13 / 8 deg**;
right-leg jumps in the hold 10 -> 4; all other phases unchanged (standing -2..0 deg, left hold 36-42 deg).
**Live v2.17 (session 12:20, deployed):** standing -3 / 0 deg; left hold 30-35 deg (spread 1.4-3.7);
right hold 39-46 deg (spread 3-7); straight right leg during the left hold -2..-3 deg; straight left leg during
the right hold -3..9 deg (was 22-28). Twist transitions still spread ~20 deg. No walker walking captured yet
(subject walked into the room, 1-2 m from the cameras).

## v2.16.1 (2026-11-11) - fixed solver lengths in the fallback
**Issue (live v2.16 test, session 12:12):** the straight left leg read 17-19 deg while the right leg was bent.
Replay A/B of that session (standing left leg during the right hold, 3 blocks): learning lengths always
18 / 17 / 19 deg, only from a near-straight leg 14 / 15 / 17, not at all **8 / 5 / 10 deg**. Other phases identical.
**Fix:** `adapt_lengths = 0` (option kept). Live v2.16 otherwise: left hold 43-45 deg smooth, right hold 55-61 deg
smooth, 0-deg reference saved when standing still.

## v2.16 (2026-11-11) - stereo-first joints
**Issue (user):** "left leg is more reliable than right". `diagnose_replay` now prints the solver angle next to the
angle straight from the triangulated joints: the fixed-length solver added 8-28 deg of false flexion
(deep right bend: solver 85 deg vs stereo 57 deg; left hold 62 vs 49). Triangulation itself was fine
(ray gaps 9-30 mm on the right leg). Note: an earlier "57 mm vs 8 mm" ray-gap claim came from wrong time windows;
an extra round of edge calibration views (calib v5b) changed nothing measurable and is not used.
**Fix:** `TrackerConfig.stereo_first=1`: when hip, knee and ankle are all triangulated (segments within +-35 % of the
subject's lengths) they are used directly; the rigid solver only fills in when a joint is hidden, with its bone
lengths following the measured ones (EMA 0.1, clamped +-20 %).
**Replay of session 11:58:** spread standing 4/6 deg (was 10/9), deep right bend 1 deg (was 5) at 52 deg (was 80),
left hold 1.5 deg (was 4) at 34 deg (was 43), jumps 5 (was 17). Known weak spot: the straight leg reads 13-15 deg
while the other leg is deeply bent (was ~5), because it is not fully visible in one camera then.
`--stereo-first 0` restores v2.15 behaviour.

## v2.15 (2026-11-11) - cameras moved, calibration v5, new leg lengths
**Issue:** video review (`frames_at.py`, `view_check.py`) showed the old mounting was the limit: a lifted knee came
within 28 cm of the main camera, and the right camera saw the legs from the side (one leg hiding the other).
**Change (hardware, user):** cameras re-aimed to look at the legs from the front: convergence 51 -> 36 deg.
**Calibration v5:** 35+35 single-camera views, 40 pairs; RMS 0.35 / 0.41 / stereo 0.31 px; board squares 25.06 +- 0.24 mm,
scale error 0.2 %, flatness 0.4 mm, baseline 382 mm. Capture tool got `--max-skew-ms` + a stillness check (frames were
8-9 ms apart after a reboot and every pair was refused).
**Leg lengths (rig-measured):** thigh 407 / 412, shin 442 / 423 mm (tape 420 / 410). Old rig gave a 541 mm left shin.
**Replay of session 11:52 with these lengths:** standing L 1-10 / R 0-4 deg (spread <= 6 / <= 2), left hold ~56 deg
with spread 1-9 deg while the standing right leg stays at 2-6 deg. Still open: during a deep right bend the standing
LEFT leg reads ~31 deg, and the deep right bend itself wobbles (spread 15-17 deg).

## v2.13 (2026-11-11) - standing reference only in the first 10 s
**Issue (user):** the 0-deg reference was taken from ANY 3 s of quiet standing. If the patient did not stand
cleanly at the start, a tired, bent-knee stance later in the session could become the new 0 deg.
**Fix:** `NeutralOffset(window_s=10)`: the reference can only be captured within 10 s after both legs are first
seen. If no clean stand happens in that window, raw angles are used (offset 0) and the status says so; restart the
session to redo. v2.12 (same day): live inset of the right (depth) camera in the Pi window.

## v2.11 (2026-11-11) - no joint on the camera lens  [deployed]
**Issue:** walking spikes of 85-112 deg. A per-frame probe (`diagnose_replay.py --probe`) showed the ankle solved
at Z = 0-30 mm: knee ~540 mm from the camera with a 541 mm shin puts the second ray-sphere root at the lens.
**Fix:** `TrackerConfig.min_joint_depth_mm = 200` - roots closer than 20 cm are discarded; if only an impossible
solution exists the leg is hidden for that frame (no false value). Anatomy rule (`anatomy=1`) tried, no effect, left off.
**Replay of session 05:48:** holds/standing identical to v2.9; walking spread L 29.6 -> 18.7 deg, R 23.5 -> 13.4 deg;
2 of 3 large spikes gone; identity swaps 32 -> 14. Live start/stop check OK on the Pi.

## v2.10 (WIP, off by default) - knee-speed gate
**Idea:** hide flexion changes faster than 350 deg/s (walking spikes to 85-110 deg).
**Try 1 (2e0931c):** spikes lasting >= 3 frames passed, because the allowed change grew with the time since the
last accepted value. **Try 2 (5809274):** allowance capped at 0.15 s. Replay of session 05:48: holds unchanged,
but walking got WORSE: after a gap the first (wrong) value was accepted and the correct values that followed were
hidden (R leg 76-125 deg at t 199-201 s). **Decision:** gate stays off (`max_rate_dps = 0`); the live system is v2.9.
Next: decide L/R and depth root from both cameras jointly instead of filtering the angle afterwards.

## v2.9 (2026-11-11) - camera-measured bone lengths
**Issue:** with tape lengths (thigh 420 / shin 410 mm) the solver rejected most stereo measurements as
outliers: only 10-45 % of joints were "measured", although keypoint confidence was 0.9-0.99 in both cameras.
The rig measures the shin from YOLO's knee/ankle points as longer (L 541, R 446 mm; thighs 419/426 match the tape),
so the two disagreed and the angle came from the bone-length guess. Holds read 52 / 39 deg vs ~40 on a phone goniometer.
**Fix:** `config/subject_stereo.json` with the rig's own lengths, used by `start_session.sh` through `PROFILE=`
in `config/session.env`. New tools: `diagnose_replay.py`, `probe_raygap.py`, `replay_lengths.py`.
**Replay of session 05:48 (same keypoints):** holds L 41.2 / R 42.9 deg (phone ~40), standing -1.0 / 2.8,
measured joints 45-61 %, spread during holds 3.8-5.2 deg, no jumps in holds.
**Still open:** twisting and walking (spread 20-30 deg, 32 identity swaps while walking).

## v2.8.2 (2026-11-11)
**Issue:** the session of 05:31 had almost no leg detections (knee conf 0.1-0.26, no person after 90 s)
and "restart" created no new session. The tracker had stopped responding after a power dip
(`throttled=0x50000`) and ignored the stop signal, so the restart found it still running and gave up.
**Checked:** `scripts/orientation_check.py` (new) confirms rotation 180/180 is right (leg conf 0.97 vs 0.71-0.75).
**Fix:** `stop_session.sh` escalates INT -> TERM -> KILL and waits for exit; `start_session.sh` waits up to 10 s
for a previous session to finish before starting. `04_run_tracker.py` now handles SIGINT/SIGTERM with a stop
flag, closes the AI HAT and always exits (session of 05:48 still needed a force-stop).

**Protocol test 05:48 (rotation 180/180, Hailo m):** standing L 3-5 deg / R 15-17 deg (spread 2-6);
left hold 47-60 deg (spread 2-7); right hold 44-48 deg (spread 1-8, best 1 deg). Twisting and walking
still noisy (spread 15-50 deg, 27 probable L/R swaps) -> next work item.

## v2.0.0 — rigid, rotation-invariant 3D tracker (2026-11-09)

### Diagnosis of v1 (from the saved notebook outputs)

| # | Symptom in v1 | Root cause | Fix in v2 |
|---|---|---|---|
| 1 | Hip/knee/ankle Z were almost always **exactly 24.0 cm or 25.9 cm** (cell 11), so the "3D" angle was really a 2D angle and changed when the leg twisted | SGBM disparity range (`minDisparity=2, numDisparities=64`) **saturated**: the legs were at or closer than the nearest depth the matcher can represent (`Z_min = f·B / (minDisp+numDisp−1)`). Saturated pixels were accepted as valid depth | Joints are now **triangulated directly from YOLO keypoints in both cameras** (`gait3d/stereo_geometry.py::triangulate`). Optional SGBM is sized from the real rig (`num_disparities_for(z_min)`) and saturated pixels are rejected (`sample_sgbm_point`). `scripts/03_check_depth.py` shows saturation live |
| 2 | Thigh 2–21 cm, shin 1–19 cm (cell 15); real bones ≈ 40 cm | Same depth saturation + independent per-frame points | Standing **bone-length calibration** (`gait3d/profile.py`) with anatomical sanity limits; lengths then become constants |
| 3 | Angle changes with leg rotation | 2D fallback (cell 15) + wrong depth | **No 2D fallback anywhere.** `gait3d/rigid_leg.py` solves the depth of each joint *along its pixel ray* with exact bone lengths |
| 4 | The hand-off formula `H_fixed = K + L·u/‖u‖` | It only rescales the bone, the **direction is unchanged, so the angle is unchanged** (proved in `tests/test_rigid_leg.py::test_naive_rescale_does_not_change_angle`) | Replaced by the correct closed form: **ray–sphere intersection** `‖Z·r − K‖ = L` (quadratic in Z) + a 3-unknown robust least-squares step that chooses the knee depth and rejects bad depth readings |
| 5 | Ghost legs: two skeletons on one limb | Visibility test was `x > 0` (YOLO always outputs a guess); first detection used | Per-joint **confidence gate ≥ 0.50** for hip, knee AND ankle; 2D + 3D **ghost-leg suppression**; best person selected |
| 6 | Noisy depth while walking | Left/right frames captured one after another (**not synchronised**, up to 33 ms apart) | Threaded capture, `SensorTimestamp` pairing, identical `FrameDurationLimits`, optional libcamera software sync (`SyncMode`) |
| 7 | Low texture / blur at 30 cm | `LensPosition = 0.0` focuses at **infinity** | Default `--lens 3.0` dioptres (≈ 33 cm). **Recalibrate at the same lens position** |
| 8 | L/R brightness mismatch | Independent auto-exposure/AWB per camera | Meter once, cap exposure (motion blur), copy identical exposure/gain/colour gains to both |
| 9 | Colours swapped | Picamera2 `RGB888` is already BGR; v1 converted RGB→BGR | No conversion; `BGR2GRAY` |
| 10 | Calibration accuracy unknown ("run with 7 pic"); RMS 0.30 on few views may be over-fitting | Blind 4-s timer capture, no quality checks; stereo solve with zero-initialised intrinsics; `stereoRectify(flags=0)` | Capture only sharp, moved, both-view boards (40 pairs, coverage map); per-camera calibration with auto-removal of bad views, then stereo with `FIX_INTRINSIC`; `CALIB_ZERO_DISPARITY`; report with **baseline (compare with a ruler)** and **rectified epipolar error** |
| 11 | Jitter / lag | Fixed EMA (0.35) | One-Euro adaptive filter |
| 12 | Slow on the Pi | Full-frame SGBM + `reprojectImageTo3D` every frame | SGBM off by default; batched YOLO on L+R; NCNN export script; solver costs ~0.4 ms per leg |

### Verified (synthetic stereo rig, `python -m pytest tests`, 19 tests)

* Noise-free leg with a fixed 135° knee rotated ±40° about 4 axes → angle error < 0.05°, bone lengths exact.
* 0.7 px keypoint noise, 300 random 3D poses: knee-angle error **median 1.6° / p95 6.6°** vs raw triangulation **3.4° / 12.1°**.
* One joint with **no depth at all** (hip / knee / ankle): recovered in 3D, median 1.5–1.9°.
* End-to-end run of `scripts/04_run_tracker.py` with simulated cameras: 2 s standing calibration recovers bone lengths within 2 mm; walking with ±35° twist → median error 2.3°; ghost-leg window reports a single leg.
* Synthetic fisheye stereo calibration recovers baseline within 1 mm and epipolar error < 0.3 px.

### Still to verify on the real Pi 5

* Run `scripts/03_check_depth.py` → saturation should be ~0 %.
* Re-run calibration at `--lens 3.0` and check the reported baseline against a ruler.
* FPS with `yolov8n-pose_ncnn_model` at `--imgsz 416`.
* YOLOv8 (trained on full-body COCO) may still be weak on close-up legs. If hip confidence rarely passes 0.50, the next step is fine-tuning on a few hundred walker frames.

---

## v2.1.0 — runs side-by-side with the existing camera pipeline (2026-11-09, tested on the Pi)

### Found on the real Pi 5 (Walker_CAM, Debian 13, Python 3.13, OpenCV 5.0, Ultralytics 8.4)

| # | Finding | Consequence | Change |
|---|---|---|---|
| 13 | `Gait-Pose-Estimation/rpi_src/camera_shm_writer.py` runs from boot and **owns both cameras** (56 fps, 640×360, portrait-rotated, JPEG in shared memory `cam_frame_shm`) | v2 could not open the cameras with Picamera2 | New read-only `ShmStereoSource` (`--source shm` in scripts 01/03/04). SeqLock read, `track=False` so the segment is never unlinked. Tested live: **55.6 fps, L/R sync 0.73 ms**, writer untouched |
| 14 | Old `stereo_calibration.npz`: left distortion `[0.20, 2.15, −17.9, 48.7]`, rectified principal points at x = 778 and −118 in a 640-px image | Rectification was invalid → every depth wrong | Re-running v2's calibration on the **same** 20 old photos: 19/20 used, RMS 0.22/0.25 px, normal distortion, epipolar 0.48 px |
| 15 | Both old and new calibrations give a **baseline ≈ 0.77 m** | With legs at 0.3–0.6 m the disparity is hundreds of pixels; v1's SGBM (`numDisparities=64`) could only see objects **> 3.3 m** away | v2 triangulates keypoints directly (no disparity search limit). **To confirm with a ruler** |
| 16 | YOLOv8n-pose on the Pi CPU, L+R pair: 606 ms @640, 344 ms @480, 268 ms @416, 168 ms @320 | 1.7–6 stereo fps with PyTorch | NCNN export in progress; Hailo accelerator (`hailo1x_pci`) is present for a later version |
| 17 | `calib_data` photos are 640×480 unrotated, but the running writer delivers 640×360 portrait frames from a different sensor mode | A calibration is only valid for the exact capture format | Recalibrate from the source you will track with (`01_capture_calibration.py --source shm --headless`) |

All 19 tests pass on the Pi (7.2 s).

---

## v2.2 – v2.5 — live calibration and leg tests on the walker (2026-11-10)

| Version | Found live | Fix |
|---|---|---|
| 2.2.0 | Cameras are **~0.83 m apart and turned ~97° toward each other**. `fisheye.stereoRectify` degenerated (rectified f = 0.0006 px); shared-view pairs covered only 15 % of each image | `RawStereoGeometry`: no rectification, YOLO on raw images, undistorted keypoints triangulated with R,T; match test = distance between viewing rays. Intrinsics from **single-camera views** (`scripts/cap_mono.py`) + pairs, stereo with `FIX_INTRINSIC`; 3D validation in the report |
| 2.2.1 | Right camera RMS 8.5 px (one bad detection) | Per-view outlier removal → RMS 0.66 / 0.68 px, stereo 0.81 px, baseline 827.6 mm (ruler ≈ 850), reconstructed squares 25.13 mm, rays meet within 0.35 mm |
| 2.3.0 | cam0 sees the **hips with confidence 0.24–0.32** (cam1: 0.59–0.75); knee/ankle ray gaps 4–68 mm → no 3D lock | A joint needs ≥ 0.5 in **at least one** camera; a joint seen only by cam1 uses cam1's ray and the bone length (cam0's guess only picks the right root). Match gate 80 mm |
| 2.4.0 | AI HAT+ 2 is a **Hailo-10H** with YOLOv8s/m-pose HEFs pre-installed | `HailoPoseEstimator` (`--model *.hef`); class output is already a probability on-chip. Keypoints agree with the CPU model; person scores and speed still being tuned |
| 2.5.0 | While one leg twists behind the other, **YOLO gives its shin the other leg's label** (15 jumps > 150 mm in 949 frames, flexion spikes to 143°) | `LegIdentity`: per-joint L/R continuity against the previous frame, duplicated joints kept only by the leg that owned them, single legs relabelled; knee motion gate (> 300 mm per frame rejected) |

Thigh/shin tape lengths (trochanter → knee joint line → lateral malleolus): 42 / 41 cm. Direct stereo knee–ankle distance: 47–49 cm (keypoint placement differs between the two very different views; the bone constraint uses the tape values).

## v2.6.0 — AI HAT+ 2 live, neutral offset, no frozen legs (2026-11-10)

Live comparison on the re-mounted rig (cameras 455 mm apart, 22° convergence; full recalibration: RMS 0.46/0.46 px, stereo 0.57 px):

| | CPU YOLOv8n @320 | **Hailo-10H YOLOv8m @640** |
|---|---|---|
| Frames logged in ~2.5 min | 779 | **2264** (≈ 11–12 fps) |
| Ankle jumps > 150 mm | 23 (3.0 %) | **15 (0.7 %)** |
| Ankle motion p95 per frame | 110 mm | **59 mm** |
| Quiet standing, spread per 2 s | 8–75° | **2–8°** |
| Held bend, spread per 2 s | 3–22° | **3–9°** |

Found and fixed:
* **Frozen leg:** with no measurement at all, the previous frame's solution was accepted as a "prior" and kept re-feeding itself (left knee frozen at 160° for 20 s). A frame now needs at least one measured joint.
* **Straight leg reads 13–20°:** pose keypoints lie on the skin, not at joint centres. A 3-s **neutral (standing) trial** at start records each leg's offset, which is subtracted (`flexion_corrected_deg`), as in gait-lab static trials.
* Stereo knee–ankle distance still 9–22 % longer than the tape (bone constraint uses the tape values).

## v2.7.0 — smoother angles + doctors' display (2026-11-10)

Last live run (fixed mounting, 388 mm / 51°, calibration 0.24/0.32 px, 3D squares 25.015 ± 0.27 mm): left-knee hold 51–54° (±2°), but right-knee holds and quiet standing still had 8–34° spread per 2 s.

* **3D direction smoothing:** the thigh and shin unit vectors are One-Euro filtered in 3D and the angle is taken from the filtered vectors (rotation-invariant). Synthetic static leg with 2.5 px keypoint noise: angle SD 5.9° → 3.4°.
* **Confidence weighting:** each joint's depth measurement noise is scaled by 1 / min(cam0, cam1 confidence), so obliquely seen joints count less.
* **Doctors' display:**
  * Pi screen: camera overlay + live LEFT/RIGHT knee-flexion graphs (last 30 s) with big numbers.
  * Web page (`http://<pi-ip>:8080`): same live graphs, camera snapshot, Reset session, CSV download and a printable **session report** (mean, ROM p5–p95, peak, steps from swing peaks, L–R symmetry).
* CSV is flushed every frame (no data loss if the tracker is stopped).

## v2.8.0 — sessions without a terminal, offline replay (2026-11-10)

* `scripts/start_session.sh` / `stop_session.sh` + desktop icons (`launchers/install_desktop_icons.sh`); settings in `config/session.env`. Each session gets its own folder with `angles.csv`, `keypoints.jsonl`, `summary.txt`.
* **Keypoint recorder + `scripts/replay.py`:** the raw YOLO keypoints of both cameras are saved every frame, so filters and gates can be tuned offline on a real session instead of repeating walker sessions.
* `docs/PROTOCOL.md`: session protocol for clinicians and a goniometer validation sheet (0/30/60/90°, both legs, acceptance proposal ≤ 5° mean error).
* Web dashboard prints the Pi's real LAN address; neutral trial shows a countdown and the saved offsets.
