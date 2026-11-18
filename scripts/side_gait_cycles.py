#!/usr/bin/env python3
"""Gait-cycle analysis of a side-camera session (v3.1.0).

Replays a recorded session (raw.mp4 + keypoints.jsonl) through two trackers at once:
  * marker  - cardboard markers (the reference on the walker), and
  * pose    - markerless (pose keypoints only), the mode the final product will use.
Gait events come from the pose keypoints with the Zeni et al. (2008) rule, adapted to a camera that
travels with the walker: heel strike = ankle at its most anterior position relative to the hip,
toe-off = most posterior.  The anterior direction is chosen per leg so that the swing-phase knee peak
lands in 55-90 % of the cycle.

Each stride (heel strike -> next heel strike) is resampled to 101 points (0-100 % gait cycle).
Output (stdout + <session>/gait_cycles.json): per leg the number of strides, stride time, and mean/SD
curves for both trackers, plus per-stride markerless-vs-marker error.

Usage: python3 scripts/side_gait_cycles.py runs/side_session_... [--t0 S --t1 S] [--compact]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.side2d import PlaneMap, SideConfig, SideTracker, leg_keypoints  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('rec')
ap.add_argument('--planes', default='config/side_planes.json')
ap.add_argument('--source', default='L=cam0,R=cam1')
ap.add_argument('--t0', type=float, default=0.0)
ap.add_argument('--t1', type=float, default=1e9)
ap.add_argument('--marker-radius', type=float, default=140)
ap.add_argument('--shank-seed', type=float, default=0.4)
ap.add_argument('--marker-color', default='tan')
ap.add_argument('--min-stride', type=float, default=0.7)
ap.add_argument('--max-stride', type=float, default=4.0)
ap.add_argument('--compact', action='store_true', help='print curves every 2 %% as integers')
a = ap.parse_args()

src = dict(kv.split('=') for kv in a.source.split(','))
pm = PlaneMap.load(a.planes)
mk = SideTracker(pm, SideConfig(source=src, markers=True, marker_radius_px=a.marker_radius,
                                shank_seed=a.shank_seed, marker_color=a.marker_color))
ps = SideTracker(pm, SideConfig(source=src, markers=False))

rows = [json.loads(l) for l in open(os.path.join(a.rec, 'keypoints.jsonl'))]
cap = cv2.VideoCapture(os.path.join(a.rec, 'raw.mp4'))
T0 = rows[0]['t']
rec = {s: {'t': [], 'mk': [], 'ps': [], 'ank': [], 'hip': []} for s in 'LR'}
for r in rows:
    ok, fr = cap.read()
    if not ok:
        break
    t = r['t'] - T0
    W = fr.shape[1] // 2
    imgs = {'cam0': fr[:, :W], 'cam1': fr[:, W:]}
    om = mk.process(t, r, imgs)
    op = ps.process(t, r, imgs)
    if not (a.t0 <= t <= a.t1):
        continue
    for s in 'LR':
        kp = leg_keypoints(r.get(src[s], []), s, 170)
        d = rec[s]
        d['t'].append(t)
        d['mk'].append(np.nan if om[s].raw is None else om[s].raw)
        d['ps'].append(np.nan if op[s].raw is None else op[s].raw)
        if kp is None or min(kp[1]) < 0.3:
            d['ank'].append((np.nan, np.nan)); d['hip'].append((np.nan, np.nan))
        else:
            d['ank'].append(tuple(kp[0][2])); d['hip'].append(tuple(kp[0][0]))


def fill(v):
    v = np.asarray(v, float).copy()
    ok = ~np.isnan(v)
    if ok.sum() < 2:
        return v
    i = np.arange(len(v))
    v[~ok] = np.interp(i[~ok], i[ok], v[ok])
    return v


def smooth(v, k=3):
    return np.convolve(fill(v), np.ones(k) / k, mode='same')


def peaks(v, t, min_gap, prom):
    """local maxima at least min_gap s apart and prom above the minimum on both sides"""
    idx = [i for i in range(1, len(v) - 1) if v[i] >= v[i - 1] and v[i] > v[i + 1]]
    keep = []
    for i in sorted(idx, key=lambda j: -v[j]):
        if all(abs(t[i] - t[j]) >= min_gap for j in keep):
            lo = max(0, np.searchsorted(t, t[i] - 1.5)); hi = min(len(v), np.searchsorted(t, t[i] + 1.5))
            if v[i] - min(v[lo:i + 1].min(), v[i:hi].min()) >= prom:
                keep.append(i)
    return sorted(keep)


def strides(d, sign):
    t = np.asarray(d['t'])
    ank = np.asarray(d['ank'], float); hip = np.asarray(d['hip'], float)
    s = smooth(sign * (ank[:, 0] - hip[:, 0]))
    leg = np.nanmedian(np.linalg.norm(ank - hip, axis=1))
    hs = peaks(s, t, a.min_stride * 0.8, 0.15 * leg)
    out = []
    for i, j in zip(hs[:-1], hs[1:]):
        dur = t[j] - t[i]
        if not (a.min_stride <= dur <= a.max_stride):
            continue
        seg = slice(i, j + 1)
        cur = {}
        for k in ('mk', 'ps'):
            v = np.asarray(d[k][seg], float)
            if np.mean(~np.isnan(v)) < 0.7:
                cur[k] = None
                continue
            tt = (t[seg] - t[i]) / dur * 100
            cur[k] = np.interp(np.arange(101), tt, fill(v))
        out.append(dict(t=float(t[i]), dur=float(dur), **cur))
    return out


res = {}
for s in 'LR':
    best = None
    for sign in (1, -1):
        st = strides(rec[s], sign)
        ref = [x['mk'] if x['mk'] is not None else x['ps'] for x in st]
        ref = [c for c in ref if c is not None]
        if not ref:
            continue
        pk = np.mean([np.argmax(c[40:95]) + 40 for c in ref])
        score = abs(pk - 74)
        if best is None or score < best[0]:
            best = (score, sign, st, pk)
    if best is None:
        res[s] = {'n': 0}
        print(s, 'no strides found')
        continue
    _, sign, st, pk = best
    r = {'n': len(st), 'sign': sign, 'stride_s': [round(x['dur'], 2) for x in st],
         'start_s': [round(x['t'], 1) for x in st]}
    for k in ('mk', 'ps'):
        C = np.array([x[k] for x in st if x[k] is not None])
        r[k + '_n'] = len(C)
        if len(C):
            r[k + '_mean'] = np.round(C.mean(0), 1).tolist()
            r[k + '_sd'] = np.round(C.std(0), 1).tolist()
    both = [x for x in st if x['mk'] is not None and x['ps'] is not None]
    if both:
        e = np.array([x['ps'] - x['mk'] for x in both])
        r['ps_minus_mk_rmse_per_stride'] = np.round(np.sqrt((e ** 2).mean(1)), 1).tolist()
        r['ps_minus_mk_bias'] = round(float(e.mean()), 1)
        r['ps_minus_mk_rmse'] = round(float(np.sqrt((e ** 2).mean())), 1)
        r['ps_minus_mk_rmse_after_bias'] = round(float(np.sqrt(((e - e.mean()) ** 2).mean())), 1)
    res[s] = r
    print('%s: %d strides (anterior sign %+d), stride %.2f +- %.2f s, swing peak at %.0f %%' % (
        s, len(st), sign, np.mean(r['stride_s']), np.std(r['stride_s']), pk))
    for k, name in (('mk', 'marker'), ('ps', 'markerless')):
        if r.get(k + '_n'):
            m = np.array(r[k + '_mean'])
            print('  %-10s n=%d  loading %.1f  stance-min %.1f  swing-peak %.1f @%d%%  ROM %.1f' % (
                name, r[k + '_n'], m[:20].max(), m[20:55].min(), m[55:90].max(), m[55:90].argmax() + 55,
                m.max() - m.min()))
            if a.compact:
                print('   ', k, ' '.join('%d' % round(v) for v in m[::2]))
    if both:
        print('  markerless - marker: bias %+.1f  RMSE %.1f  (after bias %.1f)  per stride %s' % (
            r['ps_minus_mk_bias'], r['ps_minus_mk_rmse'], r['ps_minus_mk_rmse_after_bias'],
            r['ps_minus_mk_rmse_per_stride']))
json.dump(res, open(os.path.join(a.rec, 'gait_cycles.json'), 'w'))
print('saved', os.path.join(a.rec, 'gait_cycles.json'))
