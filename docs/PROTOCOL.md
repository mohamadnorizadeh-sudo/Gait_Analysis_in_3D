# Gait 3D — session protocol and validation sheet

Research prototype for knee-flexion measurement on the smart walker
(2 × IMX708 Wide stereo, Raspberry Pi 5 + AI HAT+ 2). **Not a certified
medical device.** Validate with the sheet below before clinical use.

## Before the session

- [ ] Cameras untouched since the last calibration (no bumps, focus rings not turned).
  If either camera moved, recalibrate (5 min, paired views; see README).
- [ ] Subject's bone lengths are entered in `config/session.env`, measured with a tape on the
  outer leg while standing:
  - **Thigh** = greater trochanter → lateral knee joint line
  - **Shin** = lateral knee joint line → centre of the lateral malleolus
- [ ] Shorts or close-fitting trousers; hips, knees and ankles not covered by the walker frame.

## Running a session

1. Double-click **Start gait session** on the Pi desktop.
   A window with the camera view and live LEFT / RIGHT knee graphs opens.
   On a tablet/laptop on the same network open `http://<pi-address>:8080`.
2. **Standing (neutral) trial**: the subject stands straight and still, both legs visible,
   until the status changes from `STAND STRAIGHT & STILL x/3 s` to `neutral saved`.
   Standing then reads ≈ 0°.
3. Walk / exercise. Green joint dots = measured by both cameras; orange = placed using
   bone length; a leg that cannot be seen disappears instead of being guessed.
4. Web page → **Session report** (Print → Save as PDF) and **Download CSV**.
5. Double-click **Stop gait session**. Files: `~/gait_v2/runs/session_<date>/`
   (`angles.csv`, `keypoints.jsonl`, `summary.txt`).

## What the numbers mean

| Field | Meaning |
|---|---|
| Flexion (deg) | 0 = straight knee, corrected by the standing trial |
| ROM p5–p95 | range of motion, ignoring the 5 % most extreme samples at each end |
| Steps | swing-phase peaks above 25°, at least 0.6 s apart |
| Symmetry L–R | difference between left and right ROM / p95 flexion |

## Validation against a goniometer

Hold each angle still for 5 s with a clinician holding the goniometer on the lateral
knee (axis on the lateral femoral epicondyle, arms towards trochanter and malleolus).
Read the system value from the screen (or the CSV median over the 5 s).

| Leg | Target | Goniometer (deg) | System (deg) | Difference | Notes |
|---|---|---|---|---|---|
| Left | 0 (standing) | | | | |
| Left | ~30 | | | | |
| Left | ~60 | | | | |
| Left | ~90 | | | | |
| Right | 0 (standing) | | | | |
| Right | ~30 | | | | |
| Right | ~60 | | | | |
| Right | ~90 | | | | |

Acceptance target (proposal): mean absolute difference ≤ 5°, no single reading > 8°.
Repeat the same table with the leg rotated in / out to check rotation invariance.
