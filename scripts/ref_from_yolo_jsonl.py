#!/usr/bin/env python3
"""Reference CSV (same columns as ref_movenet_track.py) from yolo_big_keypoints.py output on the iPad video (v3.3.4).

  python3 scripts/ref_from_yolo_jsonl.py kp_ipad_yolov8x.jsonl ref_ipad_yolov8x.csv --fps 30

Why (23 Nov): YOLOv8x-pose on the iPad side video is more confident than MoveNet Thunder (mean leg confidence 0.95
vs 0.63) and, used as the training reference on the 13:01 walk, cut the learned model's swing-peak bias from about
-4 deg to -1..-2.5 deg and narrowed the limits of agreement. Frames with no person get conf 0 and interpolated angles.
"""
import argparse
import json

import numpy as np


def flex(h, k, a):
    u, v = h - k, a - k
    c = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    return 180.0 - np.degrees(np.arccos(np.clip(c, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jsonl')
    ap.add_argument('out_csv')
    ap.add_argument('--fps', type=float, default=30.0, help='frame rate of the video given to yolo_big_keypoints.py')
    a = ap.parse_args()
    rows = []
    for line in open(a.jsonl):
        r = json.loads(line)
        t = r['i'] / a.fps
        people = r.get('cam0') or []
        if not people:
            rows.append([t, np.nan, np.nan, 0.0, 0.0, np.nan])
            continue
        p = max(people, key=lambda q: q.get('s', 0))
        xy, c = np.array(p['xy'], float), np.array(p['c'], float)
        rows.append([t, flex(xy[11], xy[13], xy[15]), flex(xy[12], xy[14], xy[16]),
                     min(c[11], c[13], c[15]), min(c[12], c[14], c[16]), (xy[11, 0] + xy[12, 0]) / 2])
    R = np.array(rows)
    bad = ~np.isfinite(R[:, 1]) | ((R[:, 3] < 0.005) & (R[:, 4] < 0.005))   # no person, or no visible legs
    idx = np.arange(len(R))
    for j in (1, 2, 5):
        R[bad, j] = np.interp(idx[bad], idx[~bad], R[~bad, j])
    with open(a.out_csv, 'w') as f:
        f.write('t,flex_L,flex_R,conf_L,conf_R,hip_x\n')
        for r in R:
            f.write('%.3f,%.1f,%.1f,%.2f,%.2f,%.1f\n' % tuple(r))
    print('%d frames (%d without a person or visible legs) -> %s' % (len(R), bad.sum(), a.out_csv))


if __name__ == '__main__':
    main()
