# Gait_Analysis_in_3D

3D knee-angle tracking for stroke rehabilitation on a smart walker:
Raspberry Pi 5 (16 GB) + 2 × Camera Module 3 Wide (IMX708, 120°) stereo rig,
fisheye-rectified, YOLOv8n-Pose, rigid bone-length model.

See **[CHANGELOG.md](CHANGELOG.md)** for what was wrong in v1 and what each version fixes.

## Layout

```
calibrate_capture.ipynb       v1 notebook (kept unchanged for reference)
gait_v2.ipynb                 v2 notebook: runs the steps below from Jupyter
gait3d/                       v2 library
  camera.py                   synchronised Picamera2 pair, locked focus/exposure/AWB
  calibration.py              fisheye stereo calibration + quality report
  stereo_geometry.py          rectification, triangulation, rays, SGBM helpers
  pose.py                     YOLO wrapper, confidence gate, ghost-leg suppression
  rigid_leg.py                ray-sphere bone-length solver (no 2D fallback)
  profile.py                  standing bone-length calibration
  filters.py                  One-Euro filter
  tracker.py                  per-frame pipeline, overlay, CSV logging
scripts/
  01_capture_calibration.py   capture 40 good chessboard pairs
  02_calibrate.py             -> stereo_calibration.npz + calibration_report.txt
  03_check_depth.py           depth-range / saturation diagnostic
  04_run_tracker.py           the tracker
  05_export_ncnn.py           faster YOLO on the Pi CPU
tests/                        synthetic-rig tests (no camera needed)
```

## Run on the Pi 5 (Raspberry Pi OS Bookworm 64-bit)

```bash
git clone https://github.com/mohamadnorizadeh-sudo/Gait_Analysis_in_3D.git
cd Gait_Analysis_in_3D && git checkout v2-rigid-3d
sudo apt install -y python3-picamera2 python3-opencv
pip install -r requirements.txt --break-system-packages

python3 -m pytest -q tests                                  # 1. sanity check (no camera)
python3 scripts/01_capture_calibration.py --lens 3.0        # 2. 40 chessboard pairs
python3 scripts/02_calibrate.py --square-mm 25              # 3. check the report!
python3 scripts/03_check_depth.py --lens 3.0                # 4. saturation should be ~0 %
python3 scripts/05_export_ncnn.py --imgsz 416               # 5. optional speed-up
python3 scripts/04_run_tracker.py --model yolov8n-pose_ncnn_model --imgsz 416
```

At start the subject **stands straight for 2 s** (bone calibration; saved to
`subject_profile.json`, reuse with `--profile subject_profile.json`). Keys:
`q` quit, `c` recalibrate bones, `s` save profile. Angles are logged to CSV
(`flexion_deg`: 0 = straight knee).

Joint dot colours: green = measured depth, orange = depth reconstructed from
bone length, red = from temporal prior only.

## Doctors: one-click sessions (v2.8)

```bash
bash launchers/install_desktop_icons.sh     # once: Start / Stop icons on the Pi desktop
```
Settings (bone lengths, rotation, model, calibration) are in `config/session.env`.
Live graphs on the Pi screen and at `http://<pi-address>:8080`. See **[docs/PROTOCOL.md](docs/PROTOCOL.md)**
for the session protocol and the goniometer validation sheet. Every session also records the raw keypoints;
`scripts/replay.py` re-runs a session offline with different settings.

## Notes

* Calibrate and track with the **same** `--lens` value and resolution.
* If the full thigh is not visible from the walker (hip confidence < 0.5),
  enter bone lengths manually: `--thigh-mm 430 --shin-mm 410`.
* Frame sync: the HUD shows L/R skew. If it stays > 6 ms, your libcamera may
  lack software sync; frames are then paired by nearest timestamp.
