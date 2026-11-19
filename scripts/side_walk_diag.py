#!/usr/bin/env python3
"""Walking diagnostics for a recorded side session (v3.1.2).

No video decoding: uses the live angles.csv (statuses as recorded) and keypoints.jsonl.
Prints
  * frame rate and duration,
  * per leg: status counts over the whole recording and inside walking bouts,
  * walking bouts (from the pose ankle-minus-hip forward excursion, as in side_gait_cycles.py),
  * how many 'jump gate' rejections come in runs (a rejected jump keeps rejecting later frames, because they
    are compared with the last ACCEPTED value - the lock-out suspected on 18 Nov).

Usage: python3 scripts/side_walk_diag.py runs/side_session_...
"""
import collections
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.side2d import leg_keypoints  # noqa: E402

rec = sys.argv[1]
src = {'L': 'cam0', 'R': 'cam1'}
A = list(csv.DictReader(open(os.path.join(rec, 'angles.csv'))))
K = [json.loads(l) for l in open(os.path.join(rec, 'keypoints.jsonl'))]
t = np.array([float(x['t']) for x in A])
t = t - t[0]
print('frames %d  fps %.1f  duration %.0f s  (keypoint rows %d)' % (len(A), 1 / np.median(np.diff(t)), t[-1], len(K)))


def bouts_for(leg):
    """walking = the ankle swings fore-aft relative to the hip by > 25 % of the leg length within 2.5 s"""
    x = np.full(len(K), np.nan)
    L = []
    for i, r in enumerate(K):
        kp = leg_keypoints(r.get(src[leg], []), leg, 170)
        if kp is not None and min(kp[1]) >= 0.3:
            x[i] = kp[0][2][0] - kp[0][0][0]
            L.append(np.linalg.norm(np.asarray(kp[0][2]) - np.asarray(kp[0][0])))
    if not L:
        return np.zeros(len(K), bool), np.nan
    leg_px = float(np.median(L))
    n = len(K)
    tk = np.array([r['t'] for r in K]) - K[0]['t']
    walk = np.zeros(n, bool)
    for i in range(n):
        lo, hi = np.searchsorted(tk, tk[i] - 1.25), np.searchsorted(tk, tk[i] + 1.25)
        seg = x[lo:hi]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= 8 and (seg.max() - seg.min()) > 0.25 * leg_px:
            walk[i] = True
    return walk, leg_px


for leg in 'LR':
    st = [x[leg + '_status'] for x in A]
    walk, leg_px = bouts_for(leg)
    walk = walk[:len(st)]
    print('\n%s leg (pose leg length %.0f px)' % (leg, leg_px))
    print('  all frames   :', collections.Counter(st).most_common(8))
    print('  walking (%4.0f s):' % (walk.sum() / 9.9), collections.Counter(s for s, w in zip(st, walk) if w).most_common(8))
    # bouts
    edges = np.flatnonzero(np.diff(np.r_[0, walk.astype(int), 0]))
    bl = [(t[a], t[min(b, len(t)) - 1]) for a, b in zip(edges[::2], edges[1::2]) if t[min(b, len(t)) - 1] - t[a] > 3]
    print('  bouts > 3 s  :', ', '.join('%.0f-%.0f' % b for b in bl))
    # jump-gate runs
    runs, cur = [], 0
    for s in st:
        if s == 'jump gate':
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if runs:
        print('  jump-gate runs: %d runs, lengths %s' % (len(runs), dict(collections.Counter(runs).most_common(6))))
    # raw value range in walking frames
    raw = np.array([float(x[leg + '_raw']) if x[leg + '_raw'] not in ('', 'nan', 'None') else np.nan for x in A])
    w = raw[walk & ~np.isnan(raw)]
    if len(w):
        print('  raw while walking: valid %.0f %%  p5 %.1f  median %.1f  p95 %.1f  max %.1f' % (
            100 * len(w) / max(1, walk.sum()), *np.percentile(w, [5, 50, 95]), w.max()))
