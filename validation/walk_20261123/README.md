# 23 Nov 2026 walks (new, closer camera position) - numbers only

- `walkN_pi_*.b64`: the Pi's leg keypoints (points 11-16 per camera), decode with
  `python3 scripts/pi_slim_decode.py walkN_pi_X.b64 OUT_DIR` -> `OUT_DIR/keypoints.jsonl`.
- `walk1_ref_ipad_yolov8x.csv`: YOLOv8x reference from the iPad video of walk 1.
- `walkN_ref_ipad_movenet_clean.csv`: MoveNet Thunder reference from the iPad videos (cleaned, see CHANGELOG v3.3.7).

Reproduce the leave-one-walk-out result:

    for n in 1 2 3; do python3 scripts/pi_slim_decode.py validation/walk_20261123/walk${n}_pi_*.b64 /tmp/w$n; done
    python3 scripts/train_angle_model.py --session /tmp/w1 validation/walk_20261123/walk1_ref_ipad_movenet_clean.csv \
      --session /tmp/w2 validation/walk_20261123/walk2_ref_ipad_movenet_clean.csv \
      --session /tmp/w3 validation/walk_20261123/walk3_ref_ipad_movenet_clean.csv \
      --sync 3,30 --off-min -30 --off-max 60 --out /tmp/m.pkl
