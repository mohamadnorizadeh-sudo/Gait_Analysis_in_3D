#!/usr/bin/env python3
"""Pack a Pi recording's leg keypoints for copying through the Raspberry Pi Connect shell (v3.3.7).

  python3 scripts/pi_slim_pack.py runs/side_session_X      # writes /tmp/slim.b64, prints frames and size

Keeps t and, per camera, the chosen person's leg points 11-16 (x*10, y*10, conf*1000 as int16), gzip + base64:
about 200 kB of text for a 4-minute walk. Decode with scripts/pi_slim_decode.py; the learned model's features are
identical before and after (checked on the 23 Nov walk 1: max difference 0.0).
"""
import base64
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from gait3d.angle_model import frame_points  # noqa: E402

D = sys.argv[1]
T, Q = [], []
for line in open(os.path.join(D, 'keypoints.jsonl')):
    if not line.strip().endswith('}'):
        continue
    r = json.loads(line)
    T.append(r['t'])
    Q.append(frame_points(r)[:, 11:17, :])
P, t = np.array(Q), np.array(T)
I = np.where(np.isnan(P), -32768, np.round(P * np.array([10, 10, 1000]))).astype('<i2')
b = base64.b64encode(gzip.compress(t.astype('<f8').tobytes() + I.tobytes(), 9)).decode()
open('/tmp/slim.b64', 'w').write(b)
print('frames', len(t), 'b64', len(b))
