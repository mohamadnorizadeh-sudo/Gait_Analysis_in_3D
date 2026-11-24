# 24 Nov 2026 held-out test walk (numbers only)

Recorded with the final camera position and the live model `config/angle_model.pkl` (trained on walks 1-3 of
23 Nov against the MoveNet reference). Never used for training. iPad IMG_0025 (1080p60), 8 passes; knee bends at
48-57 s of the Pi recording; iPad started 47.8 s before the Pi (walking match, per-pass offsets within 0.25 s).

Reproduce:

    python3 scripts/pi_slim_decode.py validation/test_20261124/test_pi_side_session_20261124_083023.b64 /tmp/test
    for n in 1 2 3; do python3 scripts/pi_slim_decode.py validation/walk_20261123/walk${n}_pi_*.b64 /tmp/w$n; done
    python3 scripts/train_angle_model.py \
      --session /tmp/w1 validation/walk_20261123/walk1_ref_ipad_movenet_clean.csv \
      --session /tmp/w2 validation/walk_20261123/walk2_ref_ipad_movenet_clean.csv \
      --session /tmp/w3 validation/walk_20261123/walk3_ref_ipad_movenet_clean.csv \
      --test /tmp/test validation/test_20261124/test_ref_ipad_movenet_clean.csv \
      --sync 3,30 --off-min -30 --off-max 60 --out /tmp/m.pkl

Result (gbm, swing peaks): left median |err| 2.2 deg, bias +2.9, 95% LoA -3.9..+9.6 (PASS);
right 6.2 / -1.7 / -16.8..+13.4 (FAIL on median and LoA). Raw 2D angles without the model: right bias -18.5.
