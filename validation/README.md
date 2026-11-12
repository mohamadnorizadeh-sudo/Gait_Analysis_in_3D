# Validation against a side-view marker reference (2026-11-12)

Method: three 2 cm white tape markers on the lateral hip (greater trochanter), lateral knee joint line and
lateral malleolus of the LEFT leg; phone video from the left side (1920x1080, 30 fps). Markers detected per
frame (`scripts/validate_markers_detect.py`, `scripts/validate_markers_track.py`), 2D knee angle from the
three points. Clocks aligned by two quick knee kicks (best offset 19.5 s, mean |diff| 9.0 deg over 79 bins).
System: v2.17, session `runs/session_20261112_064604`, left knee flexion (0-deg reference applied), 0.5 s bins
in `sys_L.txt`.

| Segment | Reference | System | Difference | System spread |
|---|---|---|---|---|
| Standing | 1.7 | -1.0 | -2.7 | 1.0 |
| Hold ~50 deg | 50.2 | 36.0 | **-14.2** | 0.7 |
| Hold ~74 deg | 73.8 | 53.0 | **-20.8** | 2.3 |
| Hold ~101 deg | 101.4 | 99.0 | -2.4 | 7.4 |
| Standing after | 1.7 | -2.0 | -3.7 | 0.4 |

Walking could not be validated: the markers were lost once the subject moved with the walker.
![left knee](validation_left_knee.png)
