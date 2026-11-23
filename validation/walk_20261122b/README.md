# Walking trial 22 Nov 2026 13:01 (after the two-camera calibration, v3.2.1)

- Pi session: `runs/side_session_20261122_100105` (279 s, 2813 frames, ~9.4 fps), markerless, planes from
  `stereo_sagittal_calib.py` (both cameras 46-52 deg off a true side view).
- Reference: iPad `IMG_0018.MOV` (1080p60, 274.5 s), ~5 m to the side; reduced to 960x540 at 30 fps and run
  through `scripts/ref_movenet_track.py` -> `ref_ipad_movenet.csv` (t = seconds from the video start).
- Protocol: stand still, 2 knee bends, 8 passes past the iPad (4 each way), stand still.
- Compare: `python3 scripts/walk_compare_channels.py <series.b64> validation/walk_20261122b/ref_ipad_movenet.csv
  --sync 3,30 --off-min 5 --off-max 25` (the .b64 is from `scripts/side_export_series.py` on the Pi).
