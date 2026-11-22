# Changelog

Every version records **what was wrong** and **what changed**. The original
notebook `calibrate_capture.ipynb` is kept unchanged as the v1 reference.

---

## v3.2.2 - 2026-11-22 - export adds 3D-triangulated knee flexion from both cameras

**Why:** after the two-camera calibration both cameras see both legs, and `config/stereo_board.json` holds their
poses. A 3D angle from the two views does not depend on the knee-plane orientation at all, so it is the natural
check of the plane method (and a candidate for the product).

**What changed:** `side_export_series.py` adds, per keypoints frame:
- `k3`: 3D flexion of each leg, triangulated from cam0 and cam1 (fisheye-undistorted keypoints, linear
  triangulation in the calibration board frame);
- `k2x`: plane flexion of each leg from the other camera (cam1 for the left leg, cam0 for the right);
- `e3`: triangulation reprojection rms (px), and `l3`: 3D thigh and shank lengths (cm), to judge the quality of
  each frame.
The leg seen by the far camera uses its own length gate (`--min-leg-px-far`, 100 px). On synthetic two-camera
walking data, `k3` is within 0.15 deg of the true angle, with 0.2 px reprojection error and the exact segment
lengths.

---

## v3.2.1 - 2026-11-22 - board detection for the two-camera calibration: 2x upscale + local contrast

**What was wrong:** the first two-camera capture (12:52, `runs/stereo_calib/20261122_095158/` on the Pi) found the
board in only 1 of 24 views (left camera, slant 47 deg, 61 cm). Facing straight ahead, the board is seen by each
camera about 45-50 deg off face-on, so its squares are only ~6-11 px wide at 360x640. The right camera also saw it
in shadow against a bright window.

**What changed:** `find()` in `stereo_sagittal_calib.py` now tries, in order: plain detection, then 2x upscale +
CLAHE (classic, then sector-based), then 3x + CLAHE (sector-based). Corners are mapped back to the original
pixels. On the 12:52 frames the right camera went from 0/12 to 11/11 detected. On a synthetic board, the
upscaled corners agree with plain detection to 0.1-0.25 px on average.

---

## v3.2.0 - 2026-11-22 - knee planes from one board seen by both cameras (`stereo_sagittal_calib.py`)

**What was wrong:** after the cameras were re-aimed (about 12:20), the hand-held side board failed twice: "board
not found" in both cameras. A snapshot (kept on the Pi in `runs/board_try/`, not committed because it shows the
user) showed two causes:
- a fingertip over the top row of squares in 3 of 6 frames, which stops detection (the other 3 were detected);
- the board that was detected faced the camera (slant 11-14 deg, 50 cm), not the side. The cameras see the legs
  front-diagonally. A board flat on the outside of the leg is then too slanted to detect at 360x640, and a board
  turned toward the camera is not the plane the knee bends in. That mismatch is what made the 11:15 walking
  angles about half.

**What changed:** new `scripts/stereo_sagittal_calib.py`.
- One board, held upright in front of the knees and facing straight ahead, is seen by both cameras at once.
- solvePnP in each camera gives both camera positions. It uses a board frame that does not depend on the
  detector's corner order: pattern centre, the normal toward the cameras, and the edge nearest image-up.
- The walker's left-right axis (the knee plane's normal) is estimated twice: from the line between the two
  cameras (assumes symmetric mounting), and from the board's left-right edge (assumes it faced straight ahead).
  The angle between the two is printed as a cross-check. By default the normal is their mean.
- Each camera gets the homography of a plane with that normal beside each leg: `cam0:L`, `cam0:R`, `cam1:L`,
  `cam1:R`. Any plane parallel to the knee's plane gives the same angles, because central projection between
  parallel planes is a uniform scaling. So the board does not have to touch the leg.
- The camera poses are saved to `config/stereo_board.json`, so the two views can also be triangulated in 3D
  offline (`keypoints.jsonl` already keeps both legs in both cameras).
- Frames are kept in `runs/stereo_calib/<stamp>/`. The classic corner detector runs first, with the sector-based
  detector (`findChessboardCornersSB`) as a fallback.

**Synthetic check** (`scripts/check_stereo_sagittal_synth.py`: two fisheye cameras rendered 61 deg off a true side
view, like the walker's). With the exact normal, the plane-mapped flexion is within 0.5 deg of the true 3D angle
over 0-60 deg. A 5 deg error in the normal gives up to 4-6 deg of error at 60 deg flexion; the mean of the two
estimates halves that.

---

## Trial 2026-11-22 11:15 - walking vs a far-view iPad reference: the Pi reads about half the knee bend

**Why:** the 19 Nov sit-down showed the Pi under-reading with the hip flexed. The question for the product is
walking.

**Setup:** markerless (v3.1.7, `MARKERS=0`), cameras checked unchanged (`cam_check.py`: cam0 < 1 px, cam1 2-4 px /
0.3 deg). iPad Air `IMG_0017` (1080p60) on a stand ~5 m to the side at hip height, whole body in view. Protocol:
stand still 5 s, 2 slow knee bends, then 8 passes past the iPad (4 each way). Pi
`side_session_20261122_081509`, 266 s.

**Reference:** MoveNet Thunder with a person-following crop on the iPad video (new
`scripts/ref_movenet_track.py`). Clock offset from the knee bends: t_iPad = t_Pi + 15.98 s, r = 0.99. All 8
passes were found automatically. MoveNet's left/right labels matched the Pi's legs in every pass (r 0.56-0.83 for
the left leg).

**Findings** (`validation/walk_20261122/`, new `scripts/walk_compare_ref.py`, 103 swing peaks):

| Leg, side facing the iPad | Reference swing peak | Pi markerless | Ratio |
|---|---|---|---|
| left, near | 39 deg | 20 deg | 0.51 |
| left, far | 46 deg | 21 deg | 0.45 |
| right, near | 31 deg | 18 deg | 0.58 |
| right, far | 47 deg | 20 deg | 0.48 |

- The Pi's peaks line up in time with the reference's peaks but are about half as high (all strides: Pi = 0.46 x
  ref + 3; per frame r 0.67 left, 0.47 right). This matches the gait lab's ~50 deg against our ~20-25 deg.
- **The right camera's live output failed:** during the standing calibration, its pose model read the straight
  leg as about 63 deg. The resulting length reference made the gate reject 84 % of the right leg's walking frames
  (`length gate` 2287 of 2726). The left leg was fine (89 % ok).
- Pi frames at two swing peaks (`pi_frames_swing_L_t59.9_R_t61.3.jpg`): the camera sees the leg almost straight
  while the reference shows about 45 deg.

**Plane-stretch diagnostic** (`plane_stretch.txt`, `scripts/walk_plane_stretch.py`): the fore-aft axis of the
calibrated plane is multiplied by k, and each k is scored against the reference. If the side camera sees the
sagittal plane more obliquely than the board calibration assumed, walking angles shrink roughly by a constant
factor, while a seated 90 deg hardly changes. That fits both trials.
- Stretching by about 2-2.3 brings the swing-peak size to the reference (p95 ratio 0.48 -> 0.87-0.96 left,
  0.53 -> 0.98 right) and the mean bias to about 0.
- It does not fix frame-by-frame agreement: left r about 0.5, RMSE about 11 deg; right r about 0.2, RMSE 13-16 deg.
  So the plane orientation explains the size of the under-read, but not the scatter.

**Next options:** (1) redo the board calibration with the board flat on the outer thigh, parallel to the walking
direction, and repeat this walking test; (2) aim the cameras squarely at the side of the leg, whole leg in view;
(3) self-calibrate the plane from walking; (4) make the live standing calibration reject impossible poses (the
right camera's 63 deg "standing" reading).

## v3.1.9 - walking comparison tools

- `scripts/ref_movenet_track.py`: far-view reference with a person-following crop. Writes t, flex_L, flex_R,
  confidences and mid-hip x per frame.
- `scripts/walk_compare_ref.py`: clock offset from a sync window, walking passes, swing peaks reference vs Pi
  (ungated markerless and live).
- `scripts/walk_plane_stretch.py`: diagnostic that scales the fore-aft axis of the plane coordinates by k and
  scores each k against the reference.

---

## v3.1.8 - `scripts/side_export_series.py`: per-frame series over the remote shell

**What was wrong:** walking needs every frame (about 10 fps), not the 0.5 s bins that `side_dump*.py` print.
The Pi's only channel to the analysis machine is the remote-shell text.

**What changed:** new `scripts/side_export_series.py` writes one base64 line (zlib npz, delta-encoded) with, per
frame: the ungated plane-mapped markerless flexion and keypoint confidence of both legs (from
`keypoints.jsonl`), plus the live tracker output, raw angle and status (from `angles.csv`). A 2,700-frame
session is about 10-25 k characters. `decode()` in the same file turns it back into arrays.

---

## v3.1.7 - camera check script; markerless by default for the walking validation

**Why:** the next trial (22 Nov) validates markerless walking against the far-view iPad reference (MoveNet), so
no markers are worn. With `MARKERS=1` the live tracker would reject every frame for missing markers. The camera
check (has a side camera moved on the walker since the board calibration?) was done by hand before.

**What changed**
- `config/side.env`: `MARKERS=0` (markerless live angle with the standing offset). Recordings still keep
  `raw.mp4` + `keypoints.jsonl`, so markers can be replayed later if wanted.
- New `scripts/cam_check.py`: ORB match of the current `runs/side_view.jpg` against a frame of a reference
  recording. It reports, per camera, the dominant shift/rotation/scale and the median shift of the matches in
  the lower 40 % of the image (walker tubes and wheel, which move only if the camera moved on the walker).
  It also writes `runs/cam_check.jpg`.

---

## Trial 2026-11-19 13:23 - sit-down with pauses vs a marker-free far-view reference: the Pi reads far too low

**Why:** the user's gait lab measured a near-normal knee curve (~50 deg swing peak) for the same walker gait for
which we read ~20-30 deg. The 12:22 trial could not settle it, because its references were cardboard markers on
loose trousers.

**Setup:** lower chair, three stand/sit cycles with pauses on the way down and up. Markers were near the knee. The
left trouser leg was taped tight; the right was not. When seated, the user set the knee to ~90 deg. iPhone 13 Pro
`IMG_4964` filmed the left leg (close, frame cut at mid-thigh). iPad Air `IMG_0012` filmed the right side from
~2.5 m with the whole body in view. Both were 1080p60. Pi `side_session_20261119_102313`. Clock offsets from
creation times: iPad +18 s, iPhone +27 s. The iPad reference's stand/sit transitions line up with the Pi traces.

**Reference:** the phone videos' own cardboard markers were unusable: the iPhone's markers read 54-62 deg while
the user sat at ~90-100 deg. So the reference is a pose model on the far iPad view: MoveNet SinglePose Thunder
(ONNX), via the new `scripts/ref_movenet.py` (v3.1.6). It reads 1-12 deg standing and 93-109 deg seated,
consistent with the user's 90 deg setting.

**Findings** (40 plateaus, `validation/sts_20261119b/plateaus.txt`; Pi markerless = `side_dump_kp.py`, ungated):
- Pi markers (right leg) vs reference: -15 deg at 20 deg, -24 at 40, -33 at 60, and 40-56 deg when seated (~100).
- Pi markerless, left: -13 to -17 deg across 20-60 deg. Right: +2 at 20, -9 at 40, -19 at 60. Seated 39-76 for ~100.
- The right camera's pose failed on 2 of 4 standing plateaus: 60 deg for a straight leg (image-plane angle ~128 deg).
- Pi frames at the 70 deg pause (`pi_frames_L_top_R_bottom_t30_t72.jpg`): the thigh points toward the camera and
  looks foreshortened, and the pose hip sits at the top edge of the image. A 2D angle from this close, oblique view
  is not the sagittal knee angle. The plane homography, calibrated on a standing leg, cannot correct it.

**Conclusion:** the under-reading that the gait lab suggested is real. It comes from the measurement geometry, not
from dropped frames or filtering, and markers on clothing make it worse. A standing offset cannot fix it, because
the error grows with flexion.

**Saved for later:** the Mac-side video helpers used for the phone/iPad references on 18-19 Nov are now in
`validation/mac_tools/` (`blobs.py` tan-marker blobs, `peek.py`/`peek1.py` annotated frames, `grid.py` gridded
crops for manual reading, `track.py` seeded 3-marker tracking).

**Next:** validate walking against the same kind of far-view reference (iPad on a stand 3-4 m to the side, whole
body in view, Pi recording at the same time). Then decide on camera placement or a correction.

## v3.1.6 - `scripts/ref_movenet.py`: marker-free reference from a far side-view video

**What was wrong:** cardboard markers on clothing do not follow the bones, so phone videos tracked by their
markers are not a ground truth (19 Nov, both trials).

**What changed:** new `scripts/ref_movenet.py` runs MoveNet SinglePose Thunder v4 (ONNX, onnxruntime CPU) on a
video (via ffmpeg, optional crop) or a frame folder. It writes the image-plane knee flexion and confidences of
both legs as CSV, and optionally all 17 keypoints. The model is not stored in the repo; its source is in the
script header.

---

## v3.1.5 - `scripts/side_dump_kp.py`: ungated markerless dump

**What was wrong:** `side_dump_markerless.py` runs the full SideTracker. Its segment-length gate is referenced to
the first still period. The 19 Nov 13:23 sit-down session started seated, so that reference was a seated leg and
most later bins came out empty (e.g. pi t = 26-60 s: no markerless value at all, while the live marker tracker
read both legs).

**What changed:** new `scripts/side_dump_kp.py` applies no tracker gates. Per frame it takes each leg's pose
keypoints and prints 0.5 s medians of the plane-mapped flexion (`BINS_PL`), the undistorted-pixel flexion with no
homography (`BINS_UN`) and the min keypoint confidence x10 (`BINS_CF`). Frames below `--conf` (0.3) are skipped.

---

## Trial 2026-11-19 12:22 - sit-to-stand with two references (iPhone + iPad), NOT a clean validation

**Why:** the user reports that a gait lab measured a near-normal knee curve (~50 deg swing peak) for the same
walker gait for which we read ~20-30 deg, so our swing reading is probably too low. A swing check
(`side_gait_cycles.py --swing-report`, v3.1.4) on the 18 Nov walk found nothing missing: almost every swing
frame was valid, with peaks of 13-33 deg. So the low value comes from the measurement, not from dropped frames.

**Setup:** high lab stool behind the walker, 2-3 slow sit-to-stand repetitions with pauses. Markers moved closer
to the knee (thigh ~20 cm above, shin ~15-18 cm below the knee marker). Loose trousers (no tights), blue shirt.
iPhone 13 Pro (IMG_4962, right leg) and iPad Air (IMG_0011, left leg), 1080p60, low side views. Pi
`side_session_20261119_092259`. Reference angles come from the same three cardboard markers, tracked in the
videos on the user's Mac (seeded nearest-blob tracking with a segment-length check). Clock offsets by correlation
are iPhone 49.5 s and iPad 43.5 s, matching the files' creation times. r = 0.95 (iPhone vs Pi R markers) and
0.95 (iPad vs Pi L markerless). Plateau table: `validation/sts_20261119/plateaus.txt`.

**Findings**
- Right leg, Pi markers vs iPhone (same markers): the Pi reads lower, increasingly so with flexion: -1 to -7 deg
  below 20 deg, -8 to -12 at 25-40 deg, -16 to -22 at 44-52 deg.
- Left leg, Pi markerless vs iPad: -7 to -18 deg in mid-range (15-45 deg), -4 to -6 at the seated 66-72 deg.
- Pi left markers failed: the thigh marker sat ~170 px below the pose hip keypoint, outside the 140 px search
  radius ("markers -KA" on 780 frames). Pi right markerless was mostly gated out.
- One frame (`pi_frames_t160R_t170L.png`, Pi t = 160 s, right leg): the three cardboards lie almost on one line
  on the back half of the leg silhouette. The pose knee keypoint sits ~60 px in front of the knee cardboard.
  markers 35 deg, markerless 63 deg, iPhone (same cardboards) 52 deg. On loose trousers, with markers near the
  knee, the cardboards do not follow the bones, so marker angles (Pi AND phone) are not a ground truth here.

**Conclusion:** this test shows the Pi under-reading during hip-flexed postures, consistent with the gait-lab
report, but it cannot give the true angle. The next reference must not depend on cardboard placement: a
goniometer/inclinometer at held positions, or a simultaneous gait-lab recording.

## Validation 2026-11-18 16:30 - MARKERLESS vs phone on the held bends (both days)

Keypoint-only replay (`scripts/side_dump_markerless.py`, no standing offset) of the two hold recordings.
Phone references and clock offsets as before (18 Nov IMG_4933, 13.5 s; 17 Nov IMG_4909, 25.0 s). The 17 Nov
replay uses the 17 Nov planes (`config/side_planes_20261117.json`). Traces are saved in
`validation/markerless_bins_*.txt`.

| day | leg | phase | phone | markers | markerless | err markers | err markerless |
|---|---|---|---|---|---|---|---|
| 18 | L | stand | 0.2 | 8.0 | 1.0 | +7.8 | +0.8 |
| 18 | L | ~20 | 30.0 | 25.0 | 27.0 | -5.0 | -3.0 |
| 18 | L | ~40 | 46.0 | 42.0 | 43.0 | -4.0 | -3.0 |
| 18 | L | ~60 | 60.8 | 58.0 | 61.0 | -2.8 | +0.2 |
| 18 | R | stand | 4.4 | 11.0 | 2.5 | +6.6 | -1.9 |
| 18 | R | ~20 | 26.9 | 27.0 | 32.0 | +0.1 | +5.1 |
| 18 | R | ~40 | 38.6 | 39.0 | 46.0 | +0.4 | +7.4 |
| 18 | R | ~60 | 56.0 | 60.0 | 73.0 | +4.0 | +17.0 |
| 17 | L | stand | 4.3 | 14.0 | 7.0 | +9.7 | +2.7 |
| 17 | L | ~50 | 51.4 | 47.0 | 51.0 | -4.4 | -0.4 |
| 17 | L | ~60 | 60.4 | 57.0 | 59.0 | -3.4 | -1.4 |
| 17 | L | ~77 | 76.6 | 74.0 | 83.0 | -2.6 | +6.4 |
| 17 | R | ~50 | 51.3 | 55.0 | 50.0 | +3.7 | -1.3 |
| 17 | R | ~63 | 62.9 | 68.0 | 63.0 | +5.1 | +0.1 |

**Findings**
- **Standing:** markerless agrees with the phone (-1.9 to +2.7 deg). The markers' +7-10 deg standing offset comes
  from where the cardboard sits on loose trousers, not from the cameras.
- **Left leg (both days) and right leg on 17 Nov:** markerless within 3 deg up to ~60 deg (+6.4 at 77 deg).
- **Right leg on 18 Nov** (cam1 re-aimed after the screw, recalibrated 13:40): markerless over-reads more with
  flexion (+5, +7, +17 deg) while markers on the same frames are right (+0, +0, +4). Not an L/R label swap:
  the cam1 R-chain is the bending leg, conf 0.94-0.98; L-chain 10-16 deg. Likely cause: pose joint centres
  lie ~5-10 cm inside the calibrated lateral-surface plane at different depths (hip > knee > ankle). The
  homography then maps them with a geometry-dependent error, which changed when cam1 moved.
- **Consequence for the markerless product:** after each camera calibration, a short held-bend check
  (with markers or a goniometer) should confirm or correct the markerless reading per leg.

## Walking 2026-11-18 15:25 re-analysed with v3.1.2 - markers vs markerless vs WBDS

`side_session_20261118_122542`, `side_gait_cycles.py` (v3.1.2: no jump-gate lock-out, walking bouts only),
saved as `validation/gait_cycles_20261118_122542.json` and `validation/walk_20261118_vs_wbds_T01.png`.

| | L markers | L markerless | R markers | R markerless |
|---|---|---|---|---|
| strides | 13 | 14 | 7 | 12 |
| stride time (s) | 2.69 +- 0.58 | | 2.25 +- 0.34 | |
| swing peak, per-stride median (deg) | 22.1 | 26.9 | 18.9 | 18.3 |
| swing peak timing (mean curve) | 77 % | 69 % | 68 % | 74 % |
| stance minimum (mean curve) | 10.0 | 4.0 | 6.0 | 6.0 |
| r with WBDS 0.50 m/s mean curve | 0.95 | 0.91 | 0.63 | 0.56 |

- **Markerless vs markers, per stride:** the swing-peak difference has a median of +1.1 deg (L, 13 pairs) and
  +3.8 deg (R, 7 pairs); 10 of 20 pairs are within 5 deg. Outliers: one marker spike (92 deg) and two strides
  at -14 deg. Whole-curve bias: -5.7 (L) and -0.6 (R) deg. The left bias is in stance, where markers read
  ~10 deg and markerless ~4 deg. In the holds, standing markers read +8 deg over the phone, so markerless is
  closer there.
- **vs healthy walking (WBDS 0.50 m/s):** the timing matches (swing peak at 68-77 % vs 76 %), and the left
  shape correlates at r = 0.91-0.95. The amplitude is far smaller: swing peak ~16-27 vs 52 deg, ROM 10-16
  vs 49 deg, and the stride is slower (2.3-2.7 s vs ~1.6 s).
- The phone video (IMG_4941) shows why: short shuffling steps with the feet barely leaving the floor (phone
  swing peaks ~20-22 deg). So the small flexion is the walking style, not the measurement. A trial with
  normal steps is needed for a fair comparison with the dataset.
- The right leg had markers in only 7 of 12 strides (length gate on 14 % of walking frames, probably the
  brown shirt next to the hip marker).

## v3.1.2 - 2026-11-18 (afternoon) - walking fixes: jump-gate lock-out, walking-bout filter

**Issue:** diagnosing the 15:25 walking recording with the new `scripts/side_walk_diag.py`
(9.3 fps, 443 s): ~1700 frames per leg were "no person" (stepping out of view, turning). Walking bouts
added up to 89 s (L) and 69 s (R). In walking frames the marker tracker was valid 95 % (L) / 78 % (R); the
right leg lost 14 % of frames to the length gate (likely the brown shirt next to the hip marker). Raw
flexion while walking: L median 10.3, p95 20.8, max 46.9 deg; R median 8.5, p95 24.2, max 65.3 deg.
Two bugs flattened the gait curves:
1. **Jump-gate lock-out.** Each frame was compared with the last ACCEPTED value, so after one rejected
   jump the following frames kept being rejected for up to 0.5 s. There were 41 runs, 14 of them 3-4 frames
   long, enough to delete a whole swing peak.
2. **Standing counted as strides.** `side_gait_cycles.py` cut standing, turning and stepping in/out into
   "strides" (65-80 in 7 min).

**Changes**
- `SideConfig.max_rate_dps` (450): the jump gate allows max(25 deg, 450 deg/s x dt). Two consecutive
  frames that agree within 25 deg are accepted as real movement. Test: a 10->50->62->50->15 deg swing
  at 9.3 fps passes; a single 120 deg spike is still rejected.
- `side_gait_cycles.py`: walking-bout mask (ankle-hip fore-aft range > 25 % of the leg length within
  +-1.25 s). A stride must be >= 80 % inside a bout with its own excursion > 25 %, and within 0.6-1.6 x the
  median stride time. Per-stride swing peaks and the markerless-minus-marker peak difference are reported.
  Synthetic check (20 s walking at 2.0 s strides between 12 s + 18 s of noisy standing): 10 strides per
  leg, swing peak 51.7 deg (truth 52.3); standing is no longer counted.

## Walking trial 2026-11-18 15:25 - first gait-cycle run (preliminary, not validated)

Recording `side_session_20261118_122542` (~7 min, 4344 frames, ~10 fps; passes back and forth in the walker),
phone IMG_4941 on a stand beside the path. `side_gait_cycles.py` on the Pi:
- L: 65 "strides" (stride 2.21 +- 1.05 s). Marker mean curve: swing peak 14 deg, ROM 4.7. Markerless: swing
  peak 10.9 deg at 69 %, ROM 7.6. Markerless - marker: bias -4.9, RMSE 7.1 (5.1 after bias).
- R: 80 "strides" (1.86 +- 0.76 s). Marker swing peak 12.4 deg, ROM 4.1. Markerless 12.1 deg at 77 %, ROM 5.9.
  Markerless - marker: bias -0.9, RMSE 5.2 (5.1 after bias).

**Issues found (to fix before these curves mean anything):**
1. The curves are flattened. The 0.5 s marker trace shows swing peaks of ~50-70 deg only now and then, and many
   swing frames are missing (no value) while stance frames are steady at 8-12 deg. Suspects: the 25 deg/frame
   jump gate at ~10 fps, the 0.25 s hold, and marker blur in swing.
2. Stride detection also counts standing and turning periods (65-80 "strides", stride-time SD ~1 s). It needs a
   walking-bout filter (ankle-hip excursion amplitude) before averaging.
3. Phone reference: low placement, so hip markers are often out of frame; 0.5 s traces correlate poorly
   (r ~0.4). The phone is not usable for walking without a better placement.
4. The user wore a brown shirt (same hue as the cardboard markers) and loose trousers.

## Validation 2026-11-18 13:46 - small-angle holds (0-60 deg), markers vs phone

Recording `side_session_20261118_104622` (v3.1.1, cam1:R recalibrated 13:40) against phone IMG_4933
(1x, low side view). Phone angles come from the same cardboard markers (`validation/ref_phone_tan_20261118.py`,
saved to `validation/ref_20261118_IMG_4933.csv`). Clock offset by correlation is 13.5 s for both legs (r = 0.987 each).

| leg | phase | phone | Pi | error |
|---|---|---|---|---|
| L | stand | 0.2 | 8.0 | +7.8 |
| L | ~20 | 30.0 | 25.0 | -5.0 |
| L | ~40 | 46.0 | 42.0 | -4.0 |
| L | ~60 | 60.8 | 58.0 | -2.8 |
| R | stand | 4.4 | 11.0 | +6.6 |
| R | ~20 | 26.9 | 27.0 | +0.1 |
| R | ~40 | 38.6 | 39.0 | +0.4 |
| R | ~60 | 56.0 | 60.0 | +4.0 |

**Result:** all six held bends within 5.0 deg. Standing reads 7-8 deg above the phone on **both** legs
(17 Nov: left +9.7, right +2.7). Loose trousers move the cardboards. The phone is at floor level, so its
own standing value is uncertain too. Still needs a goniometer check at standing.

## Calibration 2026-11-18 13:40 - cam1:R redone after a mount screw was tightened

**Issue:** the camera check (walker-frame features vs the 17 Nov recording) found cam0 unchanged
(0-1 px, 0.4 deg). After the user tightened a screw, cam1 had moved 17-27 px sideways and 24-28 px down,
rotated 1.2-1.9 deg, scale 0.96-0.98, with small changes between three checks.
**Done:** board recaptured for cam1:R. Slant 22 deg (left/right +22, up/down -4; was 21 / -11), fit 0.02 cm,
40 cm. The previous planes are backed up as `config/side_planes_20261117.json` on the Pi.

## v3.1.0 - 2026-11-18 - walking analysis tools + normative reference (WBDS)

**Why:** after 17 Nov the focus moves to walking and smaller knee angles, and the final product will be
markerless. Every walking recording should give, from the same frames: the marker angle (reference), the
markerless angle (product), and a comparison with healthy walking from a public dataset.

**Added**
- `validation/normative/wbds_knee_flexion.csv` + README: knee flexion over the gait cycle (mean, SD;
  42 healthy adults) from the WBDS dataset (Fukuchi, Fukuchi & Duarte 2018, PeerJ 6:e4640, CC BY 4.0)
  at treadmill 0.50 / 0.68 / 0.87 / 1.24 m/s and overground slow/comfortable. At 0.50 m/s:
  loading-response peak 9.6 +- 7.3, stance minimum 2.4 +- 5.5, swing peak 52.7 +- 4.9 deg at 76 %.
- `scripts/side_gait_cycles.py`: replays a session through a marker tracker and a markerless tracker at
  once. Gait events come from pose keypoints with the Zeni (2008) rule (heel strike = ankle most anterior
  to the hip; the camera travels with the walker, as on a treadmill). Strides are resampled to 0-100 %, and
  it prints per leg the mean/SD curves and markerless-minus-marker bias/RMSE.
- `scripts/compare_normative.py`: r, RMSE, offset-free RMSE, % of cycle within +-2 SD, and key features
  against a WBDS condition, plus an optional PNG plot.
- Check on synthetic walking built from the WBDS 0.50 m/s mean curves (30 s, 1.6 s strides, both legs):
  18 strides per leg found, swing peak 52.0 deg at 75 % (truth 52.3 deg at 76 %).

## v3.0.10 - 2026-11-17 (afternoon) - deep flexion: shin tape check + shin direction from pose

**Issue:** in the 14:24 validation the right ~87 deg hold was lost and the left deep hold had ~125 deg spikes.
Frames from cam1 at 104-108 s (status "markers H-A") showed the shin cardboard not visible in deep flexion.
The knee cardboard, ~70 px below the pose knee keypoint, was taken as the "shin" tape because it was nearer
the shin search point, so no knee tape was left and no angle came out.

**Changes**
- `snap_to_markers(shank=(knee_kp, ankle_kp))`: a shin-tape candidate must lie on the pose shank
  (0.05 < t < 1 along knee->ankle, perpendicular distance < 0.3 x shank length).
- `SideConfig.shin_fallback` (on): if the hip and knee tapes are found but no shin tape, the shank direction
  comes from the pose knee -> ankle keypoints, anchored at the knee tape. Status "ok (shin from pose)".
  The shank length gate is skipped for these frames (the pose shank is foreshortened in deep flexion), and
  they are not used for the reference lengths.
- Tests: the real 104 s cam1 frame (knee tape not taken as shin) and the fallback angle (80-100 deg).

**Replay of side_session_20261117_112430 vs phone (IMG_4909), v3.0.9 -> v3.0.10**

| leg | phase | phone | v3.0.9 | v3.0.10 |
|---|---|---|---|---|
| L | stand  |  4.3 | 14.0 (+9.7) | 14.0 (+9.7) |
| L | hold 1 | 51.4 | 47.0 (-4.4) | 47.0 (-4.4) |
| L | hold 2 | 60.4 | 57.0 (-3.4) | 57.0 (-3.4) |
| L | hold 3 | 76.6 | 75.5 (-1.1), spikes to 126 | 74.0 (-2.6), range 66-79 |
| R | stand  | 15.8 | 18.5 (+2.7) | 18.0 (+2.2) |
| R | hold 1 | 51.3 | 54.5 (+3.2) | 55.0 (+3.7) |
| R | hold 2 | 62.9 | 68.0 (+5.1) | 68.0 (+5.1) |
| R | hold 3 | 87.3 | lost | 81.5 (-5.8), 100% valid |

Still open: left standing +9.7 deg (it shrinks with flexion, so it's not an offset). The left cam0 view shows
the three cardboards bent ~14 deg while the phone sees 4 deg. Suspected cause: off-plane cardboards at the
22 deg camera slant, and/or the phone at floor level. Needs a goniometer reading at standing to settle.

## Validation 2026-11-17 14:24 - first recording within ~5 deg (cardboard markers, re-aimed cameras)

Recording `side_session_20261117_112430` (v3.0.9, MARKER_COLOR=tan, MARKER_RADIUS=140) against phone video
IMG_4909 (1x, floor level, side view). Phone angles come from the same three cardboard markers
(`validation/ref_phone_tan.py`, seeded per leg, nearest-blob tracking), saved to
`validation/ref_20261117_IMG_4909.csv`. Clock offset by correlation is 25.0 s for both legs (left r = 0.985).

| leg | phase | phone | Pi | error | Pi valid |
|---|---|---|---|---|---|
| L | stand  |  4.3 | 14.0 | +9.7 | 100% |
| L | hold 1 | 51.4 | 47.0 | -4.4 | 100% |
| L | hold 2 | 60.4 | 57.0 | -3.4 | 100% |
| L | hold 3 | 76.6 | 75.5 | -1.1 |  83% |
| R | stand  | 15.8 | 18.5 | +2.7 | 100% |
| R | hold 1 | 51.3 | 54.5 | +3.2 | 100% |
| R | hold 2 | 62.9 | 68.0 | +5.1 | 100% |
| R | hold 3 | 87.3 |  -   |  -   |   0% |

**Result:** 6 of 7 measured phases within 5.1 deg. Open issues:
1. Left standing reads +9.7 deg. The error shrinks with flexion (+9.7, -4.4, -3.4, -1.1), so it is not a
   constant offset.
2. Right deep hold (~87 deg) was lost. Status "markers H-A": the knee marker wasn't found in deep flexion,
   and one 122 deg spike got through before the jump gate. The left deep hold also had two ~125 deg spikes
   at its start.
3. The phone was at floor level looking slightly up, so the reference itself is approximate (a few deg).

## v3.0.9 - 2026-11-17 (afternoon) - brown cardboard markers (colour detection)

**Issue:** the 13:56 recording (side_session_20261117_105608) failed. The walker stood close to a white wall, and
the white tape squares merged with the wall wherever they sat near the leg's edge in the image. The user replaced
them with 3x3 cm brown cardboard squares.

**Measured** (side_view frame, user standing): cardboard on cam0 hue 20-21, sat 77-94, brightness 194-211.
On cam1 hue 18-19, sat 114-143, brightness 108-121. All of them fully surrounded by dark trousers. Distractors:
shoe laces (hue 25, sat 74) and dark tan items (brightness < 90).

**Change:** `white_blobs(hue=, s_min=)`, `marker_blobs(img, cfg)`, and `SideConfig.marker_color`
("white" | "tan": hue 14-25, sat >= 70, brightness >= 100, dark-trouser ring kept).
`07_side_session.py --marker-color`, `MARKER_COLOR=tan` in `config/side.env`. New test: tan square found,
white wall ignored.
- MARKER_RADIUS 100 -> 140: the first live check read "L markers -KA". The left hip keypoint sits at the waist,
  about 125 px above the hip cardboard. After the change both legs read "ok" standing (L 21, R 17 deg).

## v3.0.8 - 2026-11-17 (afternoon) - knee/shin tape swap fixed

**Issue:** the left leg read 170 deg while the user stood straight (side_session_20261117_104124).
Replaying it offline showed the knee snapped to the shin tape (334,389) and the shin to the knee tape (312,319).
The 100 px radius (v3.0.7) let the knee search reach the shin tape, and greedy nearest-first assignment took it.
Continuity then kept the swap.

**Change:** `snap_to_markers` now scores every combination of distinct blobs within the radius. Most tapes
wins, then smallest total distance. Tapes must keep the leg order hip -> knee -> shin along the hip-to-shin
direction. New test `test_snap_keeps_leg_order_with_wide_radius`.

## v3.0.7 - 2026-11-17 (afternoon) - tape detection: brighter threshold, wider search

**Issue:** 20 s test with the re-aimed cameras: status "L markers ---", "R markers -K-", so no angle at all.
On a recorded frame (side_session_20261117_103818): at brightness >= 165 the trouser-edge sheen joined the
cam0 knee and shin tapes into one 73x331 px strip (rejected as too big). The cam0 hip tape was found but 87 px
from the hip keypoint (search radius 50). On cam1 the right leg stood at the left image edge, so the hip and
shin tapes merged with the bright background at the border.

**Changes**
- `SideConfig.marker_v_min` (default 200); the tracker passes it to `white_blobs`. At 200, cam0 found all
  three tapes individually: (299,137), (312,317), (335,386).
- `07_side_session.py --marker-radius --marker-vmin`; `config/side.env` MARKER_RADIUS=100, MARKER_VMIN=200.
- Setup: stand about 10 cm further back, so the right leg isn't at cam1's image edge.

## v3.0.6 - 2026-11-17 (afternoon) - cameras re-aimed, planes redone, shin-tape seed configurable

**Issue:** the morning test read the right leg 30-38 deg against a 60-90 deg phone reference. The cameras saw
each leg from ~45 deg in front, and the knee/shin tapes sat forward of the side seam (off the calibrated plane).
After re-aiming, the walker's diagonal blue tube hid the low shin tape, so the tape moved up to mid/upper
shin. The live tracker still searched for it 70% of the way down the shin (fixed 0.7), 50 px radius, so it would miss.

**Changes**
- Cameras turned inward. User stands ~20 cm further into the walker for calibration (no longer arms: vibration).
- Board recaptured: cam1:R slant 21 deg (was 47), fit 0.03 cm, 42 cm. cam0:L slant 22 deg (was 43), fit 0.03 cm, 40 cm.
- `07_side_session.py --shank-seed`, `SHANK_SEED` in `config/side.env` (now 0.4).
- Simulation (homography, 3 joints): moving the leg sideways/fore-aft parallel to the calibrated plane gives
  0 deg error at any slant (parallel planes map by pure scale), so the patient's distance from the frame
  does not matter while the leg stays in view. Slant amplifies leg turn-out and off-plane tapes: worst
  case for 10 deg turn-out or a 3 cm off-plane knee tape is ~1.5 deg at 15 deg slant, ~3 deg at 26, ~5 deg at 40.

## Validation 2026-11-17 (both legs, new board calibrations, markers moved to front-outer) - NOT validated
Board calibrations redone from the normal standing position: left cam0 slant 43 deg, 47 cm, fit 0.03 cm;
right cam1 slant 47 deg, 40 cm, fit 0.04 cm (yesterday's kept as config/side_planes_20261116.json).
Hip/knee markers moved toward the front-outer side -> markers found in most frames (L 1010/1381, R 1108/1381
valid; was 53 / 0). Session `runs/side_session_20261117_072553`, phone IMG_4900.
System (median per second): left standing 16 deg (should be ~5), left holds ~35 and ~54, the deepest hold
dropped out (42-52 s); right standing 6-10, right holds only ~30-38 although the phone shows ~60-90.
The phone reference could not be extracted automatically this time (front-outer markers sit on the edge of
the leg in the true side view).
Diagnosis: the homography is exact only for points in the calibrated plane (the side of the leg). Markers
moved 4-6 cm to the front leave that plane; at 40-47 cm range and 43-47 deg slant this parallax changes the
angle by tens of degrees. Root cause is the camera aim: each camera looks at its leg from the front-outside
at ~45 deg instead of from the side.
Next: turn each camera inward to face its leg perpendicularly (slant < 15 deg), markers back on the lateral
side, redo the two board captures, repeat holds.

## Analysis 2026-11-17 of the 16 Nov two-leg recording - NOT validated
`runs/side_session_20261116_120108` vs `validation/ref_20261116_IMG_4865.csv` (scripts/side_compare.py,
scripts/side_variants.py). Live tracker: only 53 of ~1500 frames with a person were valid for the left leg and
none for the right (status `markers --A` / `H-A` / `-KA`): with the 4x4 cm markers the shin marker is found but
the hip and knee markers are not - the hip marker is seen edge-on (9x9 px, 67 px from the hip keypoint) and the
knee marker is hidden or fails the shape test from the near camera.
Offline re-analysis, held bends, system minus phone:
* relaxed markers (80 px, fill 0.3): left -12..-23 deg (one +50), right -54..+13; mean abs ~20 deg.
* shin marker + pose hip/knee: mean abs 22 (left) / 28 (right) deg.
* pose keypoints only: right leg reads 3-8 deg during 54-89 deg holds (clock alignment for this part doubtful).
Clock alignment by correlation was weak (r 0.2-0.64), so the per-hold numbers are indicative only.
Best result so far remains the 16 Nov left-leg test with 2 cm tapes (within ~6 deg up to ~50 deg, +8..13 deg at
85-100 deg, 344 valid frames). The +-5 deg target is not met; right leg and walking are not validated.

## v3.0.4-v3.0.5 (2026-11-16, evening) - both legs calibrated; final recording NOT yet analysed
* Right-leg board: cam1 slant 46 deg, fit 0.03 cm, board only 25 cm away (user stood off-centre; left was 58 cm).
* 4x4 cm markers on both legs (hip, lateral knee, front-outer shin); blob area limit raised (v3.0.4).
* v3.0.5: a bystander behind the empty walker was measured (R 63 deg) -> minimum leg size, nearest person wins.
* Final recording `runs/side_session_20261116_120108` (both legs, holds + walking) is on the Pi. The phone
  reference (IMG_4865) is extracted: `validation/ref_20261116_IMG_4865.csv` (standing 4-5 deg; left holds
  ~45/46/46, 63/65/64, 74/85/88; right holds 55-60 ... 89). The Pi was unreachable afterwards (desktop bridge
  offline), so the comparison is still open:
  `python3 scripts/side_compare.py runs/side_session_20261116_120108 validation/ref_20261116_IMG_4865.csv --leg L --ref-window 0-75`
  (new script: aligns the clocks by correlation, compares every held bend).

## Validation 2026-11-16 (v3.0.3, left leg, cam0 + plane homography + 3 tapes) - first usable result
Session `runs/side_session_20261116_111052`, phone IMG_4862 (tripod, left side; reference = same three tapes
tracked in the phone video, shank tape instead of ankle). Held bends, median of each hold, phone -> system:
38->32, 44->38, 52->50, 86->97, 84->92, 81->74..83, ~100->113, ~101->108..117; standing 7-8 -> 10 (when all three
tapes are found). So within ~6 deg up to ~50 deg, over-reading by 8-13 deg at 85-100 deg. Pairing by order of
the holds (clocks ~14 s apart), not frame-exact.
Problems: the shank tape faces sideways and cam0 views the leg from the front-left, so it is often not seen
while standing (status `markers HK-` in 443 of ~1000 frames with a person; only 344 frames fully valid).
Walking not validated. Right leg not calibrated yet.

## v3.0.1-v3.0.3 (2026-11-16) - first side-camera test with tape markers (left leg)
* Board on the left leg: cam0 slant 36 deg (33 sideways, 19 down), 58 cm, fit 0.02 cm -> cam0 really views the
  leg from the front-left; the homography raised its bend readings by ~8 deg.
* Keypoints only (cam0 + homography) vs phone markers measured by hand (+-5 deg, time pairing uncertain):
  ~30/43/58 vs ~48/79/71 -> the near camera's pose keypoints still read far too low.
* Tracker dropped every bend frame: thigh/shank length gate +-25% too tight -> +-50% (v3.0.2).
* Tape markers: walls/floor gave many false white blobs; ankle tape at the hem hidden. v3.0.2: blob must sit on
  dark trousers; hip+knee from tape -> stand 3, bends 23/36/70 (peak 76), walking peaks 24-35.
* v3.0.3 (Fable review): third tape moved to the lateral SHANK 10-15 cm above the hem (search seeded 70 % from
  knee to the ankle keypoint), snap radius 50 px with continuity to the last tape position.
* v3.0.1: board search single-threaded (the remote shell dropped while it used all cores).

## v3.0 (2026-11-15) - side cameras, 2D sagittal knee angle (not yet validated)
**Why:** front stereo cannot measure a bent knee (validation 15 Nov). Cameras moved to 25 cm arms on the
rear posts, one per side, ~70 cm high, tilted down 5-10 deg (cameras 90 cm apart). Standing: hip/knee/ankle of
both legs found in both cameras, confidence 0.73-0.98.
**First recording** (`runs/side_20261115_122209`, held bends, no plane calibration yet): every camera sees both
legs, the L/R labels were right, but per leg the two cameras disagreed by 25-37 deg, the NEAR camera always lower
(L: cam0 14/37/57 vs cam1 40/71/90; R: cam1 26/31/45 vs cam0 65/69/82). Fisheye undistortion changed < 1 deg.
Simulation: camera slant changes angles both ways and too little to explain this. Likely cause (also Fable's
review): at ~35 cm the near camera's HIP keypoint slides forward with the knee (thigh stays "vertical").
**New code:**
* `gait3d/side2d.py` - per-camera homography to the leg's sagittal plane (from a checkerboard held flat on the
  leg), 2D flexion, confidence / segment-length / jump gates, short hold, One-Euro filter, standing offset
  (keypoint mode only), optional snapping of hip/knee/ankle to white 2 cm tape markers (keypoint-guided).
* `scripts/side_plane_calib.py` (board -> `config/side_planes.json`, prints camera slant),
  `scripts/07_side_session.py` + `start_side_session.sh`/`stop_side_session.sh` (`config/side.env`): live
  Pi screen + web dashboard + raw video/keypoints/angles recording; `scripts/side_replay.py` (offline, compares
  both cameras per leg); `scripts/side_view.py --record`.
* Tests: homography recovers the true angle for slanted cameras (+-0.5 deg); end-to-end synthetic recording
  with tape markers: 59 deg for a true 60 deg hold, 5 deg standing (uncalibrated slanted camera: 64 deg).

## Validation 2026-11-15 (v7c, high mounting) + v2.19 trial (not deployed, reverted)
Session `runs/session_20261115_094925`, phone IMG_4841 on the left. Clock: video = system + 2.25 s (left-ankle
height curves, r = 0.96). Walking left the phone's view, so only standing and one long left bend could be compared.
| Segment | Reference | System |
|---|---|---|
| Standing (4-12 s) | 6-7 deg | 6-10 deg (OK) |
| Bend, knee ~63 deg (21.5-25 s) | 60-65 | **6-27** (brief 36-45) |
| Knee lowering ~30-45 (25.5-26.5 s) | 29-37 | **55-63** |
**Why:** while the knee bends, the left/right knee keypoints disagree by 80-100 mm (cameras 37 cm apart, 38 deg
convergence, conf 0.86-0.98), above `ray_gap_mm` = 80 -> knee unmatched; hip (~400 mm) and ankle (~900 mm) are
measured. With the standing thigh (386 mm) the knee ray meets the hip sphere only at ~170 or ~690 mm, while the knee
is at ~545 mm in frames where it does match: in the bent pose the keypoints no longer fit the standing bone length,
so the fixed-length solver has no correct root and picks the straight-leg one (~10 deg).
**v2.19 trial:** multi-start LM from both knee roots (replay of this session): output unchanged - neither root is
right. Reverted; nothing changed on the live system.
**Conclusion:** same as 12 Nov - a front-facing stereo pair cannot measure sagittal knee flexion reliably once the
knee bends. Next: a side-view camera (decision with the user).

## Calibration v7c + v2.18 (2026-11-15) - camera moved; far-board corner bug
**Issue 1 - shins far too long** with v6d (stereo L/R 538/468 mm vs tape 410; old rig 442/423).
Ankles stand ~1.0-1.15 m from the cameras (feet behind the knees while leaning on the walker).
**Issue 2 - far boards bent (v2.18 fix):** in the 640x360 stream a board ~1 m away has 6-9 px squares; the fixed
11x11 px `cornerSubPix` window reached the neighbouring corners (boards triangulated bent by up to 39 mm). The
window is now 40 % of the smallest square (2-5 px half-size) in `02_calibrate.py` and `calib_pairs_check.py`.
(A first guess, a 180 deg corner-order flip between the views - v2.17.2 - changed no pair; kept only as a guard.)
**Issue 3 - a camera moved since 14 Nov:** today's near boards (0.54-0.69 m) read 26.0-26.8 mm against v6d, while
yesterday's boards at the same place read 24.9-25.2 mm. Adding today's far boards to the v6d set made the solve
inconsistent (stereo RMS 0.36 -> 0.72 px, baseline 371 -> 388 mm).
**Fix - v7c from today's boards only:** 37 pairs (far 0.86-1.12 m, middle, near, hip height), yesterday's mono
views for the intrinsics. Stereo RMS 0.30 px, RMS L/R 0.37/0.28, squares 25.09 +- 0.45 mm, scale error 0.35 %,
flatness 0.78 mm, all 37 pairs within 0.5 mm; baseline 373.1 mm, convergence 38.2 deg (was 36.5).
Hold-out check: 8 of 10 hip-height boards, not used by v7, were within 0.5 mm; the 2 highest have one corner that
lands behind a camera (returns NaN, which the tracker already treats as not measured). Active on the Pi; v6d kept.
**Leg lengths (profile):** L thigh/shin 386/462, R 362/426 mm (session_20261115_083639, first 6 s). A second
standing recording (feet ~10 cm further back) gave R shin ~500: its right-ankle ray gap was 50 mm (27 mm in the
first), i.e. the two cameras did not see the same ankle point. Keypoint ray gaps of 20-60 mm at hip/knee are normal
for this wide, converging rig (as in v2.3), so bone lengths from direct triangulation carry ~5-10 % noise.
**Lesson:** the camera mounts move easily - recalibrate (or run a board check) after any knock.

## Calibration v6 (2026-11-15) - cameras high, looking down from the front
Mounting: ~36 cm apart (rig: 367.5 mm), convergence 36.5 deg, rotation 180/180, PSU_MAX_CURRENT=5000 + I2C enabled
(UPS gauge MAX1704x at 0x36: battery 4.16 V, 96 %).
**Issue:** first solve was poor (stereo RMS 0.76 px, board squares 25.76 +- 4.05 mm, scale error 3 %).
`scripts/calib_pairs_check.py` (new, per-pair 3D check) showed 20/39 pairs off; viewed on screen: the board was held
low and almost FLAT (grazing view from above, ~1 m away). Sharpness and ray gaps were normal - geometry, not blur.
**Fix:** those 20 pairs moved to `runs/calib_v6/rejected/`; 15 extra low pairs with the board tilted towards the
cameras (13 good, 2 far ones rejected). Final v6d: 32 pairs, all within 0.5 mm; squares 25.04 +- 0.44 mm,
scale error 0.17 %, flatness 0.59 mm, RMS 0.51 / 0.50 / stereo 0.36 px. Active on the Pi; v5 kept as backup.
Note: board positions ~95 cm away in the lower image half stay unreliable; watch ankle depth there.
Extra single-camera views for the image bottom (coverage 0-4 %) could not be filled (walker tube) and did not change
the result.

## v2.17.1 (2026-11-15) - power load + robust frame reader
**Issue:** UPS supply sagged under load (EXT5V min 4.80 V, p10 4.90 V; throttled=0x50000) and the AI HAT
dropped out. `camera_shm_writer.py` (user pipeline) used 52 % CPU encoding both cameras at 56 fps while the tracker
uses ~11 fps.
**Try 1 - FrameRate 56 -> 30: reverted.** Two free-running cameras at 30 fps lock into a constant phase offset larger
than MAX_SYNC_ERROR_MS, so `get_synced_pair` never returned a pair and nothing was published.
**Fix:** capture stays at 56 fps (sync unchanged); only every 2nd synced pair is JPEG-encoded/published (~28 fps).
Original kept as `camera_shm_writer.py.bak_56fps_20261115`. Writer CPU 52 % -> 37 %; with live view + AI HAT running:
EXT5V min 4.80 -> **4.93 V**, p10 4.90 -> **4.98 V**, median 5.03 V.
`ShmStereoSource.read()` now skips unreadable (torn) snapshots instead of crashing.
**Left to the user (system settings, need sudo):** `PSU_MAX_CURRENT=5000` in the EEPROM config, enable I2C for the UPS
monitor, optionally `dtparam=pciex1_gen=2` if the AI HAT keeps disconnecting.

## Status 2026-11-12 (end of day) - next steps
* Validation (`validation/README.md`): front-facing cameras under-read knee flexion: left holds -14/-21/-2 deg,
  right holds -36/-20/-77 deg, walking swing peaks about half. Detection is fine (conf 0.9-1.0); the error is the
  depth geometry of a front view.
* Cameras re-mounted (not yet fixed): higher, looking down from the front, ~50 cm apart, rotation still 180/180.
  Live view: all leg keypoints 0.97-1.00 in both cameras. Calibration v5 is NOT valid for this mounting.
* AI HAT dropped out again ("Device disconnected", throttled=0x50000 under-voltage) -> reboot needed; power
  supply to be checked (27 W direct).
* Next: fix cameras -> reboot -> recalibrate (parts 1+2) -> leg lengths -> repeat marker validation (both legs)
  and compare with the 12 Nov numbers. If still poor: side-view camera.

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
