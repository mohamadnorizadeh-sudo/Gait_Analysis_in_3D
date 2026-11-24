# Closed loop: gait deviation -> doctor -> walker speed and projected footprints (design, v3.6.0)

Status: the analysis and recommendation step is implemented (`scripts/gait_report.py`); the doctor link, the walker
speed control and the floor projector are designs, not built. The rollator used so far has no motor or projector.

## Flow
1. **Measure (built).** Pi 5 + Hailo: two cameras -> YOLOv8m-pose -> learned model -> left/right knee angle at ~10 fps,
   saved per session to `~/gait_data/knee_angles/knee_angles_<Qatar time>.csv`.
2. **Analyse (built, prototype).** `gait_report.py` splits strides, measures swing flexion, stance minimum, range,
   stride time and variability per leg, matches walking speed by stride time to healthy adults at controlled speeds
   (Moreira et al. 2021), and flags: reduced swing flexion (stiff knee), excess stance flexion, irregular
   stepping, left/right asymmetry (v3.6.1: right knee and asymmetry only with --use-right; hyperextension off
   until the knee angle has a sign). Thresholds are defaults in `config/normative_knee.json` until the clinical
   team sets them.
3. **Recommend (built, draft text).** Report JSON + plain-text summary in `~/gait_data/reports/`. Suggestions are for a
   clinician to confirm: walker speed (reduce ~15 % when deviations appear, keep or raise up to 10 % when within range)
   and footprint targets (which leg, cadence).
4. **Doctor in the loop (design).** The Pi's web dashboard (port 8080) gets a "Session reports" page; the doctor sees
   the summary, the knee curves against the healthy band, and approves or edits each suggestion. Only approved
   settings go to the walker. Remote access through the same secure channel as Raspberry Pi Connect or a hospital
   server; no automatic changes without approval.
5. **Walker speed (design).** Needs a motorised or braked rollator: wheel encoders for speed, motor driver or
   electromagnetic brakes on the rear wheels, controller (e.g. microcontroller over serial/CAN) taking a speed limit
   from the Pi. Safety: hard speed ceiling, hand-brake and dead-man handle override, emergency stop, ramped changes.
6. **Projected footprints (design).** Small projector on the walker's front frame pointing at the floor ahead.
   Calibrate projector-to-floor with one homography (project a pattern, view it with a camera). Draw footprint
   targets at the approved step length and side, timed to the approved cadence (steps/min), moving with the walker
   (encoder speed). Step length needs a foot-position estimate (ankle keypoints + floor plane) not yet built.

## Before clinical use
- Fix the right-leg camera view (CHANGELOG v3.5.0), retrain, repeat the held-out test.
- Validate knee angles against a gait lab on several people, including patients.
- Clinical team sets thresholds; ethics approval for patient sessions.
