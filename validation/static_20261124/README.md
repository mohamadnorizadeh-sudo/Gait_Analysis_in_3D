# 24 Nov 2026 static check: which iPad reference is right? (numbers only)

Phone zip-tied to the outer side of the shin of the leg facing the iPad (iPad IMG_0023, 4K 60 fps, loose trousers),
Measure app -> Level read from the 4K video at each held pose. Seated: leg extended, halfway, shin upright, foot
tucked; each leg twice. `shin_tilt_true_deg` = shin angle from vertical from the phone (landscape reading r ->
90-|r|; portrait -> |r|); `still_spread_deg` = YOLOv8x shin-angle range over +-0.5 s (> 4 = moving, excluded).
YOLOv8x ran on a 1080p copy, MoveNet Thunder on the 4K frames.

Result (8 still, unambiguous poses): shin error vs phone - MoveNet mean -1.4 deg (mean |err| 5.8), YOLOv8x mean
-5.5 (8.3); with the shin tilted > 45 deg: MoveNet -8.1, YOLOv8x -15.8. Both read the shin as more vertical than
it is (loose fabric hangs vertically); YOLOv8x much more so. Standing, the phone read 5-7 deg vs 1-3 deg from the
models: a mounting offset of about 4-5 deg, within the uncertainty of this check.
