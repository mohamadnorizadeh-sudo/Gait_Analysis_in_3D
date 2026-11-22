#!/usr/bin/env python3
"""Export a side session's per-frame angle series as ONE compact text line (v3.1.8, 3D in v3.2.2).

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
  v3.2.2, when config/stereo_board.json exists (stereo_sagittal_calib.py --save), per keypoints frame:
  k3         int16    (2, N) x10 deg, first-differenced: 3D flexion L, R triangulated from BOTH cameras
  k2x        int16    (2, N) x10 deg, first-differenced: plane flexion of each leg seen by the OTHER camera
  e3         uint16   (2, N) x10 px: triangulation reprojection rms (hip, knee, ankle, both views); 0 = none
  l3         uint16   (4, N) x10 cm: 3D thigh L, shank L, thigh R, shank R; 0 = none

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
    out = {'kp_t': float(d['t0']) + np.cumsum(d['kp_dt'].astype(np.int64)) / 1000.0,
           'kp': _undiff(d['kp']), 'kp_c': d['kp_c'].T / 100.0,
           'an_t': float(d['an_t0']) + np.cumsum(d['an_dt'].astype(np.int64)) / 1000.0,
           'an': _undiff(d['an']), 'an_s': d['an_s'].T, 'codes': str(d['codes']).split('\n')}
    if 'k3' in d:
        out['k3'], out['k2x'] = _undiff(d['k3']), _undiff(d['k2x'])
        for k in ('e3', 'l3'):
            v = d[k].T.astype(float) / 10.0
            v[v == 0] = np.nan
            out[k] = v
    return out


def triangulate(st, pm, xy0, xy1):
    """3D hip, knee, ankle (cm, board frame of stereo_sagittal_calib) from cam0 and cam1 pixels.
    Returns (3x3 points, reprojection rms px)."""
    import cv2
    from gait3d.side2d import fisheye_normalize
    ns, Ps, fs = [], [], []
    for cam, xy in (('cam0', xy0), ('cam1', xy1)):
        c = pm.cams[cam]
        ns.append(fisheye_normalize(xy, c['K'], c['D']))       # v3.2.3: robust near the image edge
        Ps.append(np.hstack([np.array(st['cams'][cam]['R'], float), np.array(st['cams'][cam]['t'], float).reshape(3, 1)]))
        fs.append(float(c['K'][0, 0]))
    X = cv2.triangulatePoints(Ps[0], Ps[1], ns[0].T, ns[1].T)
    X = (X[:3] / X[3]).T
    err = []
    for P, n, f in zip(Ps, ns, fs):
        q = P @ np.vstack([X.T, np.ones(len(X))])
        err.append(np.sum((q[:2] / q[2] - n.T) ** 2, 0) * f * f)
    return X, float(np.sqrt(np.mean(np.concatenate(err))))


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
    ap.add_argument('--min-leg-px-far', type=float, default=100.0, help='leg seen by the other camera (v3.2.2)')
    ap.add_argument('--stereo', default='config/stereo_board.json', help='camera poses for 3D (v3.2.2); "" = skip')
    a = ap.parse_args()
    src = dict(kv.split('=') for kv in a.source.split(','))
    oth = {s: ({'cam0', 'cam1'} - {src[s]}).pop() for s in 'LR'}
    pm = PlaneMap.load(a.planes)
    st = json.load(open(a.stereo)) if a.stereo and os.path.exists(a.stereo) else None

    kt, kv, kc, k3, k2x, e3, l3 = [], [], [], [], [], [], []
    for line in open(os.path.join(a.rec, 'keypoints.jsonl')):
        r = json.loads(line)
        kt.append(r['t'])
        vals, confs, v3, vx, ee, ll = [], [], [], [], [], []
        for s in 'LR':
            kp = leg_keypoints(r.get(src[s], []), s, a.min_leg_px)
            kf = leg_keypoints(r.get(oth[s], []), s, a.min_leg_px_far) if st else None
            f = None
            if kp is not None:
                f = flexion_2d(*pm.to_plane(src[s], s, np.asarray(kp[0], float)))
            vals.append(NA if f is None else int(round(f * 10)))
            confs.append(0 if kp is None else int(round(100 * float(np.min(kp[1])))))
            if st is None:
                continue
            g = None if kf is None else flexion_2d(*pm.to_plane(oth[s], s, np.asarray(kf[0], float)))
            vx.append(NA if g is None else int(round(g * 10)))
            h, e, L2 = None, 0, (0, 0)
            if kp is not None and kf is not None:
                xy = {src[s]: kp[0], oth[s]: kf[0]}
                X, rms = triangulate(st, pm, xy['cam0'], xy['cam1'])
                h = flexion_2d(*X)
                e = min(65535, int(round(rms * 10)))
                L2 = tuple(min(65535, int(round(10 * float(np.linalg.norm(X[j] - X[1]))))) for j in (0, 2))
            v3.append(NA if h is None else int(round(h * 10)))
            ee.append(e)
            ll.extend(L2)
        kv.append(vals)
        kc.append(confs)
        if st is not None:
            k3.append(v3)
            k2x.append(vx)
            e3.append(ee)
            l3.append(ll)

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
    extra = {}
    if st is not None:
        extra = dict(k3=_diff(k3), k2x=_diff(k2x), e3=np.array(e3, np.uint16).T.copy(), l3=np.array(l3, np.uint16).T.copy())
    np.savez(buf, t0=np.float64(kt[0]), kp_dt=np.r_[0, np.round(np.diff(kt) * 1000)].astype(np.uint16),
             kp=_diff(kv), kp_c=np.array(kc, np.uint8).T.copy(),
             an_t0=np.float64(at[0]), an_dt=np.r_[0, np.round(np.diff(at) * 1000)].astype(np.uint16),
             an=_diff(av), an_s=np.array(ast, np.uint8).T.copy(), codes=np.array('\n'.join(codes)), **extra)
    sys.stdout.write(base64.b64encode(zlib.compress(buf.getvalue(), 9)).decode() + '\n')


if __name__ == '__main__':
    main()
