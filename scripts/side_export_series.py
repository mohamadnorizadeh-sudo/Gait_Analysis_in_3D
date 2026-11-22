#!/usr/bin/env python3
"""Export a side session's per-frame angle series as ONE compact text line (v3.1.8).

The Pi's only channel to the analysis machine is the remote-shell text, so walking analyses (which need every
frame, not 0.5 s bins) get the series as base64(zlib(npz)), delta-encoded and stored column by column to keep
it short. Arrays in the npz:

  t0, an_t0  float64  time of the first keypoints.jsonl / angles.csv frame (epoch s)
  kp_dt      uint16   ms between keypoints.jsonl frames (first entry 0); an_dt the same for angles.csv
  kp         int16    (2, N) x10 deg, first-differenced along time: ungated plane-mapped flexion L, R (pose
                      keypoints, no tracker gates / offset / filter); -32768 = no leg (before differencing)
  kp_c       uint8    (2, N) x100: min(hip, knee, ankle) keypoint confidence L, R
  an         int16    (4, N) x10 deg, first-differenced: live L_flexion, R_flexion, L_raw, R_raw; -32768 = empty
  an_s       uint8    (2, N) status code L, R; codes = newline-separated status strings

  python3 scripts/side_export_series.py runs/<session> [--min-leg-px 170] > /tmp/series.b64
  python3 -c "from side_export_series import decode; d = decode(open('series.b64').read())"
     -> kp_t, kp (N,2 deg, NaN = none), kp_c (N,2), an_t, an (N,4 deg, NaN), an_s (N,2), codes
"""
import base64
import io
import os
import sys
import zlib

import numpy as np

NA = -32768


def _undiff(d):
    v = np.cumsum(d.astype(np.int64), axis=1).astype(np.int16).T.astype(float)
    v[v == NA] = np.nan
    return v / 10.0


def decode(text):
    d = np.load(io.BytesIO(zlib.decompress(base64.b64decode(text.strip()))))
    return {'kp_t': float(d['t0']) + np.cumsum(d['kp_dt'].astype(np.int64)) / 1000.0,
            'kp': _undiff(d['kp']), 'kp_c': d['kp_c'].T / 100.0,
            'an_t': float(d['an_t0']) + np.cumsum(d['an_dt'].astype(np.int64)) / 1000.0,
            'an': _undiff(d['an']), 'an_s': d['an_s'].T, 'codes': str(d['codes']).split('\n')}


def _diff(v):
    v = np.asarray(v, np.int64)
    return np.diff(np.vstack([np.zeros((1, v.shape[1]), np.int64), v]), axis=0).astype(np.int16).T.copy()


def main():
    import argparse
    import csv
    import json
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from gait3d.side2d import PlaneMap, flexion_2d, leg_keypoints

    ap = argparse.ArgumentParser()
    ap.add_argument('rec')
    ap.add_argument('--planes', default='config/side_planes.json')
    ap.add_argument('--source', default='L=cam0,R=cam1')
    ap.add_argument('--min-leg-px', type=float, default=170.0)
    a = ap.parse_args()
    src = dict(kv.split('=') for kv in a.source.split(','))
    pm = PlaneMap.load(a.planes)

    kt, kv, kc = [], [], []
    for line in open(os.path.join(a.rec, 'keypoints.jsonl')):
        r = json.loads(line)
        kt.append(r['t'])
        vals, confs = [], []
        for s in 'LR':
            kp = leg_keypoints(r.get(src[s], []), s, a.min_leg_px)
            if kp is None:
                vals.append(NA)
                confs.append(0)
                continue
            xy, c = np.asarray(kp[0], float), np.asarray(kp[1], float)
            f = flexion_2d(*pm.to_plane(src[s], s, xy))
            vals.append(NA if f is None else int(round(f * 10)))
            confs.append(int(round(100 * float(c.min()))))
        kv.append(vals)
        kc.append(confs)

    at, av, ast, codes = [], [], [], []
    for r in csv.DictReader(open(os.path.join(a.rec, 'angles.csv'))):
        at.append(float(r['t']))
        av.append([NA if r[k] == '' else int(round(float(r[k]) * 10))
                   for k in ('L_flexion', 'R_flexion', 'L_raw', 'R_raw')])
        row = []
        for s in 'LR':
            st = r[s + '_status']
            if st not in codes:
                codes.append(st)
            row.append(codes.index(st))
        ast.append(row)

    kt, at = np.array(kt), np.array(at)
    buf = io.BytesIO()
    np.savez(buf, t0=np.float64(kt[0]), kp_dt=np.r_[0, np.round(np.diff(kt) * 1000)].astype(np.uint16),
             kp=_diff(kv), kp_c=np.array(kc, np.uint8).T.copy(),
             an_t0=np.float64(at[0]), an_dt=np.r_[0, np.round(np.diff(at) * 1000)].astype(np.uint16),
             an=_diff(av), an_s=np.array(ast, np.uint8).T.copy(), codes=np.array('\n'.join(codes)))
    sys.stdout.write(base64.b64encode(zlib.compress(buf.getvalue(), 9)).decode() + '\n')


if __name__ == '__main__':
    main()
