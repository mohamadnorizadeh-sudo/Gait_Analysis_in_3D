#!/usr/bin/env python3
"""Decode a Pi recording packed by the remote-shell transfer (v3.3.7) back into keypoints.jsonl.

On the Pi, /tmp/slim.py packs t plus, per camera, the chosen person's six leg points (11-16: hips, knees, ankles) as
x*10, y*10, conf*1000 int16 (missing = -32768), gzip + base64. That is everything the learned angle model reads
(gait3d/angle_model.py uses points 11-16 only), at about 1/15 of the size, so a 4-minute walk fits in 4 terminal
screens and can be copied through Raspberry Pi Connect without scp.

  python3 scripts/pi_slim_decode.py pi_side_session_X.b64 OUT_DIR      # writes OUT_DIR/keypoints.jsonl
"""
import base64
import gzip
import json
import os
import sys

import numpy as np


def main():
    src, out = sys.argv[1], sys.argv[2]
    raw = gzip.decompress(base64.b64decode(open(src).read().strip()))
    n = len(raw) // (8 + 2 * 6 * 3 * 2)
    t = np.frombuffer(raw[:8 * n], '<f8')
    I = np.frombuffer(raw[8 * n:], '<i2').reshape(n, 2, 6, 3)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'keypoints.jsonl'), 'w') as f:
        for k in range(n):
            rec = {'t': float(t[k])}
            for c, cam in enumerate(('cam0', 'cam1')):
                q = I[k, c]
                if (q == -32768).all():
                    rec[cam] = []
                    continue
                xy = [[0.0, 0.0]] * 11 + [[round(q[i, 0] / 10, 1), round(q[i, 1] / 10, 1)] for i in range(6)]
                cf = [0.0] * 11 + [round(q[i, 2] / 1000, 3) for i in range(6)]
                rec[cam] = [{'xy': xy, 'c': cf, 's': 1.0}]
            f.write(json.dumps(rec) + '\n')
    print('%d frames, %.0f s -> %s' % (n, t[-1] - t[0], os.path.join(out, 'keypoints.jsonl')))


if __name__ == '__main__':
    main()
