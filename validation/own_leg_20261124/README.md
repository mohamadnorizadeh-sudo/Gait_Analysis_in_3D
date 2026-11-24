# 24 Nov 2026: own-leg angle model (v3.7.0), numbers only

Why: the independent review (`../review_20261124/REVIEW.md`, finding C1) showed that with the version-2 features
each knee's model also uses the other leg, so in symmetric walking it can copy the other leg's movement; that would
fail for a stroke patient. v3.7.0 adds own-leg features (each knee's model sees only its own leg's points), trains
only on frames where both cameras see the leg, and adds strict scoring and a desync test to
`scripts/train_angle_model.py`.

Model choice was made on the three training walks only (pooled leave-one-walk-out, strict scoring); the held-out
walk was then scored once for each candidate (it had already been used in v3.4.1 and by the review, so it is no
longer a clean test).

Strict scoring: strides cut on the reference alone (between knee-flexion minima, 0.7-3.5 s, near-leg passes); each
signal's own swing peak and stance minimum in the stride; brackets = 95 % CI from resampling whole passes.

| Model | Pooled LOWO L: median / bias / within +-5 | Pooled LOWO R | Test L | Test R |
|---|---|---|---|---|
| both-leg gbm 800x0.1 (deployed until v3.7.0) | 4.2 / -2.9 / 61 % | 4.7 / -0.6 / 53 % | 3.7 / +4.2 / 55 % | 5.0 / +1.1 / 52 % |
| own-leg gbm 800x0.1 | 4.5 / -2.6 / 59 % | 4.8 / -0.6 / 54 % | 4.4 / +4.6 / 55 % | 4.2 / +1.9 / 61 % |
| **own-leg gbm regularised (chosen)** | **4.0 / -2.9 / 54 %** | **4.4 / -1.2 / 59 %** | **3.4 / +3.8 / 62 %** | **3.8 / +0.3 / 64 %** |
| own-leg ridge | 4.0 / -3.4 / 54 % | 4.9 / -2.0 / 51 % | 4.0 / +1.6 / 66 % | 3.9 / -0.6 / 58 % |
| both-leg ridge | 3.7 / -3.8 / 61 % | 4.6 / -1.5 / 53 % | 3.1 / +2.7 / 76 % | 5.3 / -0.3 / 48 % |

(The test numbers of the deployed model in this table come from a model retrained in this container, scikit-learn
1.9.1; with the exact v3.4.1 model the strict test result is L 4.2 / +4.1 / 59 %, R 3.8 / +1.1 / 67 %,
`both_leg_model_test.log`.)

Deployed: the same command run on the Pi (scikit-learn 1.4.2) on 24 Nov wrote `config/angle_model.pkl` (feature
version 3; the previous model is kept as `config/angle_model_v2_both.pkl`). Its strict results: pooled leave-one-walk-
out L 4.1 / -2.9 / 58 %, R 4.7 / -1.3 / 54 %; held-out walk L 3.7 / +3.5 / 60 % (limits -7.6..+14.6), R 4.1 / +0.5 /
64 % (limits -11.8..+12.7); desync change 0.00 deg; 9.5 ms per frame on one thread (the version-2 model: 23.6 ms).

Chosen: own-leg gbm (300 iterations, learning rate 0.05, at least 60 samples per leaf, L2 1.0, 15 leaves), the
lowest pooled median error of the own-leg candidates.

Desync test on the held-out walk (other leg's points replaced by points from other walking moments, 5 shifts): the
both-leg model's knee angles change by 4.5-4.8 deg (L) and 4.2-4.6 deg (R) on average; the own-leg model's by 0.00.

Deep knee bends (14 frames where the reference shows both knees above 70 deg, up to 108 / 93 deg): both-leg model
max L 64 / R 22, own-leg model max L 60 / R 27. The own-leg change does not fix deep bends: the training walks have
none, and cam1 does not see the right knee.

Bottom line: removing the cross-leg shortcut costs about 0-0.5 deg of median error, and each knee now depends only
on what the cameras see of that leg. Neither knee reaches the goal: about 55-65 % of strides are within +-5 deg
(goal 95 %), and the 95 % limits are about +-12 to +-20 deg.

Reproduce:

    for n in 1 2 3; do python3 scripts/pi_slim_decode.py validation/walk_20261123/walk${n}_pi_*.b64 /tmp/w$n; done
    python3 scripts/pi_slim_decode.py validation/test_20261124/test_pi_side_session_20261124_083023.b64 /tmp/test
    python3 scripts/train_angle_model.py \
      --session /tmp/w1 validation/walk_20261123/walk1_ref_ipad_movenet_clean.csv \
      --session /tmp/w2 validation/walk_20261123/walk2_ref_ipad_movenet_clean.csv \
      --session /tmp/w3 validation/walk_20261123/walk3_ref_ipad_movenet_clean.csv \
      --test /tmp/test validation/test_20261124/test_ref_ipad_movenet_clean.csv \
      --sync 3,30 --off-min -30 --off-max 60 --out /tmp/m.pkl

`scripts/` here holds the comparison prototypes (`prep.py`, `exp.py`, `pooled.py`; paths at the top need editing).
