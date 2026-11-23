#!/usr/bin/env python3
"""Does a fore-aft stretch of the leg plane explain the walking under-read? (v3.1.9, diagnostic)

Hypothesis: the side cameras see the sagittal plane more obliquely than the board calibration assumed (e.g. the
board was turned toward the camera instead of lying flat on the outer thigh). Then the fore-aft (horizontal)
component of every segment is shortened by a constant factor and the knee angle in walking, where thigh and
shank stay near vertical, comes out roughly scaled down; a seated 90 deg would hardly change.

For each leg this replays keypoints.jsonl through the plane map, rotates the plane so the mean hip->ankle
direction during the walking passes is vertical, multiplies the horizontal coordinate by k and recomputes the
flexion. Each k is scored against the far-view reference (ref_ipad_movenet.csv: t_pi, flex_L, flex_R, conf_L,
conf_R) inside the walking passes: regression slope ref~Pi (1 = right scale), r, bias, RMSE and the median
ratio of Pi/ref swing peaks.

  python3 scripts/walk_plane_stretch.py runs/side_session_20261122_081509 \
      validation/walk_20261122/ref_ipad_movenet.csv validation/walk_20261122/passes.txt --offset 15.98
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.side2d import PlaneMap, leg_keypoints  # noqa: E402


def flex(h, k, a):
    u, v = h - k, a - k
    c = np.sum(u * v, -1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return 180.0 - np.degrees(np.arccos(np.clip(c, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rec')
    ap.add_argument('ref')
    ap.add_argument('passes')
    ap.add_argument('--offset', type=float, required=True, help='t_ref = t_pi + offset (passes are in ref time)')
    ap.add_argument('--planes', default='config/side_planes.json')
    ap.add_argument('--source', default='L=cam0,R=cam1')
    a = ap.parse_args()
    src = dict(kv.split('=') for kv in a.source.split(','))
    pm = PlaneMap.load(a.planes)
    import io
    ref = np.genfromtxt(io.StringIO(''.join(l for l in open(a.ref) if not l.startswith('#'))), delimiter=',', names=True)
    passes = np.loadtxt(a.passes, comments='#', ndmin=2)
    T, P = [], {'L': [], 'R': []}
    t0 = None
    for line in open(os.path.join(a.rec, 'keypoints.jsonl')):
        r = json.loads(line)
        t0 = r['t'] if t0 is None else t0
        T.append(r['t'] - t0)
        for s in 'LR':
            kp = leg_keypoints(r.get(src[s], []), s, 170.0)
            if kp is None or min(kp[1]) < 0.5:
                P[s].append(np.full((3, 2), np.nan))
            else:
                P[s].append(np.asarray(pm.to_plane(src[s], s, np.asarray(kp[0], float)), float))
    T = np.array(T)
    inwalk = np.zeros(len(T), bool)
    for p0, p1, _ in passes:
        inwalk |= (T + a.offset >= p0) & (T + a.offset <= p1)
    for s in 'LR':
        X = np.array(P[s])                                   # N x 3 x 2 (hip, knee, ankle)
        ok = inwalk & ~np.isnan(X[:, 0, 0])
        d = X[ok, 2] - X[ok, 0]
        v = np.mean(d / np.linalg.norm(d, axis=1, keepdims=True), axis=0)
        v /= np.linalg.norm(v)
        h = np.array([v[1], -v[0]])
        rf = np.interp(T[ok], ref['t_pi'], ref['flex_' + s])
        rc = np.interp(T[ok], ref['t_pi'], ref['conf_' + s])
        good = rc >= 0.3
        print('%s leg: %d walking frames, plane "down" axis (%.2f, %.2f)' % (s, good.sum(), v[0], v[1]))
        print('    k   slope(ref~Pi)   r    bias   RMSE  p95 Pi/ref')
        for k in np.arange(1.0, 3.01, 0.25):
            Y = np.stack([X[ok] @ h * k, X[ok] @ v], axis=-1)
            f = flex(Y[:, 0], Y[:, 1], Y[:, 2])
            x, y = f[good], rf[good]
            sl = np.polyfit(x, y, 1)[0]
            print('  %.2f    %5.2f        %.2f  %+5.1f  %5.1f   %.2f' % (
                k, sl, np.corrcoef(x, y)[0, 1], np.mean(x - y), np.sqrt(np.mean((x - y) ** 2)),
                np.percentile(x, 95) / np.percentile(y, 95)))


if __name__ == '__main__':
    main()
