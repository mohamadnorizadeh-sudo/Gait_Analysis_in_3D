# Changelog

Every version records **what was wrong** and **what changed**. The original
notebook `calibrate_capture.ipynb` is kept unchanged as the v1 reference.

---

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
