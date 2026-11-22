#!/usr/bin/env python3
"""Learned knee angle from both cameras' leg keypoints (v3.3.0).

Why (22 Nov): the walker cameras see the legs 46-52 deg off a true side view, so the flat 2D angle loses about
half the swing bend. Two pose models (YOLOv8 on the Pi, MoveNet on the laptop) agree on that small image angle, so
the geometry is the limit, not the model. The positions of all six leg points in BOTH views still change with the
bend. On the 13:01 walk, a model trained on 7 passes predicted the 8th within 4.3-8.0 deg RMSE, reaching 94 % of
the reference swing peak (laptop test with MoveNet keypoints).

Inputs per session: the Pi recording folder (keypoints.jsonl from 07_side_session.py: every person, 17 keypoints,
both cameras) and the far-view reference CSV from ref_movenet_track.py. The clocks are aligned automatically:
offset from the sync knee bends, then a per-pass drift fit.

Features per frame: for each camera, hips/knees/ankles of both legs relative to the hip midpoint / leg length,
their confidences, and both raw 2D knee angles; plus the same from the previous and next frame.
Models: gradient boosting (scikit-learn) per leg, and a ridge model saved as plain numbers (runs anywhere).
Validation: leave one session out (2+ sessions) or leave one pass out (1 session).

  python3 scripts/train_angle_model.py --session runs/side_session_A refA.csv --session runs/side_session_B refB.csv \
      --out config/angle_model.pkl
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
from scipy.signal import find_peaks, medfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from walk_compare_channels import passes_from_ref  # noqa: E402

LEGS = ((11, 13, 15), (12, 14, 16))
FEATURE_VERSION = 1


def best_person(people):
    best, bs = None, -1.0
    for p in people or []:
        xy, c = np.asarray(p['xy'], float), np.asarray(p['c'], float)
        L = sum(np.linalg.norm(xy[i] - xy[k]) + np.linalg.norm(xy[a] - xy[k]) for i, k, a in LEGS)
        q = L * float(np.mean(c[11:17]))
        if q > bs:
            best, bs = (xy, c), q
    return best


def load_session(rec):
    t, P = [], []
    for line in open(os.path.join(rec, 'keypoints.jsonl')):
        r = json.loads(line)
        t.append(r['t'])
        fr = np.full((2, 17, 3), np.nan)
        for ci, cam in enumerate(('cam0', 'cam1')):
            b = best_person(r.get(cam))
            if b is not None:
                fr[ci, :, :2], fr[ci, :, 2] = b[0], b[1]
        P.append(fr)
    return np.array(t), np.array(P)          # (N,), (N, 2 cams, 17, 3)


def flex(xy, idx):
    u, v = xy[:, idx[0]] - xy[:, idx[1]], xy[:, idx[2]] - xy[:, idx[1]]
    c = np.sum(u * v, 1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return 180 - np.degrees(np.arccos(np.clip(c, -1, 1)))


def features(P):
    out = []
    for ci in range(2):
        xy, c = P[:, ci, :, :2], P[:, ci, :, 2]
        mid = (xy[:, 11] + xy[:, 12]) / 2
        leg = np.linalg.norm(xy[:, 11] - xy[:, 13], axis=1) + np.linalg.norm(xy[:, 13] - xy[:, 15], axis=1) + 1e-6
        out += [((xy[:, 11:17] - mid[:, None]) / leg[:, None, None]).reshape(len(P), -1), c[:, 11:17]]
        out += [flex(xy, idx)[:, None] for idx in LEGS]
    X = np.hstack(out)
    X = np.hstack([X, np.roll(X, 1, 0), np.roll(X, -1, 0)])
    return np.nan_to_num(X, nan=0.0)


def align(tau, P, ref, sync, off_min, off_max, speed):
    """Pi time -> reference time: offset from the knee bends, then per-pass drift (as walk_compare_channels)."""
    t = ref['t']
    F = np.column_stack([medfilt(ref['flex_L'], 5), medfilt(ref['flex_R'], 5)])
    raw = np.column_stack([flex(P[:, 0, :, :2], LEGS[0]), flex(P[:, 1, :, :2], LEGS[1])])
    s0, s1 = sync
    sel = (tau > s0) & (tau < s1)
    best = (-2, 0.0)
    for j in range(2):
        q = raw[sel, j]
        ok = ~np.isnan(q)
        if ok.sum() < 20:
            continue
        for o in np.arange(off_min, off_max, 0.02):
            c = np.corrcoef(np.interp(tau[sel][ok] + o, t, np.maximum(F[:, 0], F[:, 1])), q[ok])[0, 1]
            if c > best[0]:
                best = (c, o)
    off = best[1]
    passes = passes_from_ref(t, ref['hip_x'], speed)
    mids, offs = [], []
    for p0, p1, _ in passes:
        m = (tau + off >= p0) & (tau + off <= p1)
        bb = (-2, off)
        for j in range(2):
            y = raw[m, j]
            ok = ~np.isnan(y) & (y < 75)
            if ok.sum() < 30:
                continue
            for o in np.arange(off - 1.0, off + 1.0, 0.02):
                c = np.corrcoef(np.interp(tau[m][ok] + o, t, F[:, j]), y[ok])[0, 1]
                if c > bb[0]:
                    bb = (c, o)
        if bb[0] > 0.4:
            mids.append(np.mean(tau[m]))
            offs.append(bb[1])
    drift, b = np.polyfit(mids, offs, 1) if len(mids) >= 3 else (0.0, off)
    tr = tau + b + drift * tau
    print('  clock: sync r %.2f, t_ref = t_pi + %.2f + %.5f t_pi, %d passes' % (best[0], b, drift, len(passes)))
    return tr, F, t, passes


def report(name, pred, Y, inp, t, F, tr, passes):
    for j, s in ((0, 'L'), (1, 'R')):
        e = pred[inp, j] - Y[inp, j]
        pk = []
        for p0, p1, _ in passes:
            ii = np.where((t >= p0) & (t <= p1))[0]
            for p in find_peaks(F[ii, j], prominence=12, distance=24)[0]:
                k = ii[p]
                w = (tr >= t[k] - 0.2) & (tr <= t[k] + 0.2)
                if w.any():
                    pk.append((F[k, j], pred[w, j].max()))
        pk = np.array(pk) if pk else np.zeros((0, 2))
        print('  %-6s %s  r %.2f  RMSE %.1f  bias %+.1f  swing peak %.1f vs ref %.1f, |peak err| median %.1f deg (n %d)' % (
            name, s, np.corrcoef(pred[inp, j], Y[inp, j])[0, 1], np.sqrt(np.mean(e ** 2)), e.mean(),
            np.median(pk[:, 1]) if len(pk) else np.nan, np.median(pk[:, 0]) if len(pk) else np.nan,
            np.median(np.abs(pk[:, 1] - pk[:, 0])) if len(pk) else np.nan, len(pk)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', nargs=2, action='append', metavar=('REC_DIR', 'REF_CSV'), required=True)
    ap.add_argument('--sync', default='3,30')
    ap.add_argument('--off-min', type=float, default=0.0)
    ap.add_argument('--off-max', type=float, default=40.0)
    ap.add_argument('--speed', type=float, default=12.0)
    ap.add_argument('--out', default='config/angle_model.pkl')
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    S = []
    for rec, refp in a.session:
        print(rec)
        tp, P = load_session(rec)
        tau = tp - tp[0]
        ref = np.genfromtxt(refp, delimiter=',', names=True, comments='#')
        tr, F, t, passes = align(tau, P, ref, tuple(map(float, a.sync.split(','))), a.off_min, a.off_max, a.speed)
        X = features(P)
        Y = np.column_stack([np.interp(tr, t, F[:, j]) for j in range(2)])
        grp = -np.ones(len(tau), int)
        for i, (p0, p1, _) in enumerate(passes):
            grp[(tr >= p0) & (tr <= p1)] = i
        S.append(dict(X=X, Y=Y, grp=grp, tr=tr, F=F, t=t, passes=passes))

    mk = {'gbm': lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05), 'ridge': lambda: Ridge(1.0)}
    for name in mk:
        print('validation (%s):' % ('leave one session out' if len(S) > 1 else 'leave one pass out'))
        for si, s in enumerate(S):
            pred = np.full_like(s['Y'], np.nan)
            folds = [si] if len(S) > 1 else sorted(set(s['grp'][s['grp'] >= 0]))
            for f in folds:
                if len(S) > 1:
                    Xtr = np.vstack([q['X'][q['grp'] >= 0] for k, q in enumerate(S) if k != si])
                    Ytr = np.vstack([q['Y'][q['grp'] >= 0] for k, q in enumerate(S) if k != si])
                    te = np.ones(len(s['X']), bool)
                else:
                    tr_ = (s['grp'] >= 0) & (s['grp'] != f)
                    Xtr, Ytr, te = s['X'][tr_], s['Y'][tr_], s['grp'] == f
                for j in range(2):
                    pred[te, j] = mk[name]().fit(Xtr, Ytr[:, j]).predict(s['X'][te])
            report(name, pred, s['Y'], s['grp'] >= 0, s['t'], s['F'], s['tr'], s['passes'])

    Xa = np.vstack([q['X'][q['grp'] >= 0] for q in S])
    Ya = np.vstack([q['Y'][q['grp'] >= 0] for q in S])
    models = {name: [mk[name]().fit(Xa, Ya[:, j]) for j in range(2)] for name in mk}
    ridge = {'coef': [m.coef_.tolist() for m in models['ridge']], 'intercept': [float(m.intercept_) for m in models['ridge']]}
    with open(a.out, 'wb') as fh:
        pickle.dump({'feature_version': FEATURE_VERSION, 'gbm': models['gbm'], 'ridge': ridge,
                     'sessions': [s_[0] for s_ in a.session]}, fh)
    with open(a.out.replace('.pkl', '_ridge.json'), 'w') as fh:
        json.dump({'feature_version': FEATURE_VERSION, **ridge}, fh)
    print('saved', a.out, 'and', a.out.replace('.pkl', '_ridge.json'), '(trained on %d frames)' % len(Xa))


if __name__ == '__main__':
    main()
