#!/usr/bin/env python3
"""Markerless replay of a side session, dumped compactly (v3.1.3).

Runs the keypoint-only tracker (no markers, no video needed) over keypoints.jsonl and prints the median
raw knee flexion per 0.5 s bin as "BINS L,R;L,R;..." so the trace can be copied out of the remote shell and
compared with a phone reference.

Usage: python3 scripts/side_dump_markerless.py runs/side_session_... [--planes config/side_planes.json]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.side2d import PlaneMap, SideConfig, SideTracker  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('rec')
ap.add_argument('--planes', default='config/side_planes.json')
ap.add_argument('--source', default='L=cam0,R=cam1')
a = ap.parse_args()
src = dict(kv.split('=') for kv in a.source.split(','))
tr = SideTracker(PlaneMap.load(a.planes), SideConfig(source=src, markers=False, use_offset=False))
rows = [json.loads(l) for l in open(os.path.join(a.rec, 'keypoints.jsonl'))]
t0 = rows[0]['t']
B = {}
for r in rows:
    t = r['t'] - t0
    o = tr.process(t, r)
    B.setdefault(int(t / 0.5), []).append(tuple(np.nan if o[s].raw is None else o[s].raw for s in 'LR'))
m = lambda v: '' if np.all(np.isnan(v)) else '%d' % round(float(np.nanmedian(v)))
print('BINS ' + ';'.join(m([p[0] for p in B.get(b, [(np.nan, np.nan)])]) + ',' +
                         m([p[1] for p in B.get(b, [(np.nan, np.nan)])]) for b in range(max(B) + 1)))
