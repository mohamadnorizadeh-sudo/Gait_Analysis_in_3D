#!/usr/bin/env python3
"""Ungated markerless dump of a side session (v3.1.5).

side_dump_markerless.py runs the full SideTracker, whose segment-length gate is referenced to the first
still period.  When a session starts seated (19 Nov sit-down trial) that reference is wrong for every other
posture and most bins come out empty.  This script applies NO tracker gates: per frame it takes the pose
keypoints of each leg (nearest person, as the tracker does) and computes the flexion three ways:

  PL  plane-mapped (undistort + per-leg homography, what the product reports, minus gates/offset/filter)
  UN  undistorted pixels, no homography (shows what the plane mapping changes)
  CF  min(hip, knee, ankle) keypoint confidence x10 (0-9), to see when the pose model is guessing

Bins of 0.5 s (median over frames whose min confidence >= --conf):
  BINS_PL L,R;L,R;...   BINS_UN ...   BINS_CF ...

Usage: python3 scripts/side_dump_kp.py runs/side_session_... [--planes config/side_planes.json] [--conf 0.3]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.side2d import PlaneMap, flexion_2d, leg_keypoints  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('rec')
ap.add_argument('--planes', default='config/side_planes.json')
ap.add_argument('--source', default='L=cam0,R=cam1')
ap.add_argument('--conf', type=float, default=0.3)
ap.add_argument('--min-leg-px', type=float, default=60.0)
a = ap.parse_args()
src = dict(kv.split('=') for kv in a.source.split(','))
pm = PlaneMap.load(a.planes)
rows = [json.loads(l) for l in open(os.path.join(a.rec, 'keypoints.jsonl'))]
t0 = rows[0]['t']
B = {}
for r in rows:
    b = int((r['t'] - t0) / 0.5)
    rec = []
    for s in 'LR':
        kp = leg_keypoints(r.get(src[s], []), s, a.min_leg_px)
        if kp is None:
            rec.append((np.nan, np.nan, np.nan))
            continue
        xy, c = np.asarray(kp[0], float), np.asarray(kp[1], float)
        cf = float(c.min())
        if cf < a.conf:
            rec.append((np.nan, np.nan, cf))
            continue
        P = pm.to_plane(src[s], s, xy)
        U = pm.undistort(src[s], xy)
        pl, un = flexion_2d(*P), flexion_2d(*U)
        rec.append((np.nan if pl is None else pl, np.nan if un is None else un, cf))
    B.setdefault(b, []).append(rec)


def med(v, scale=1.0):
    v = np.asarray(v, float)
    return '' if np.all(np.isnan(v)) else '%d' % round(float(np.nanmedian(v)) * scale)


for k, name, sc in ((0, 'PL', 1.0), (1, 'UN', 1.0), (2, 'CF', 10.0)):
    out = []
    for b in range(max(B) + 1):
        fr = B.get(b, [])
        out.append(','.join(med([f[i][k] for f in fr] or [np.nan], sc) for i in range(2)))
    print('BINS_%s ' % name + ';'.join(out))
