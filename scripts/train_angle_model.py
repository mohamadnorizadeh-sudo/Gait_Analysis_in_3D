#!/usr/bin/env python3
"""Learned knee angle from both cameras' leg keypoints: train, validate, test (v3.3.0; test mode v3.3.1).

Why (22 Nov): the walker cameras see the legs 46-52 deg off a true side view, so the flat 2D angle loses about
half the swing bend. Two pose models (YOLOv8 on the Pi, MoveNet on the laptop) agree on that small image angle, so
the geometry is the limit, not the model. The positions of all six leg points in BOTH views still change with the
bend. On the 13:01 walk, a model trained on 7 passes predicted the 8th within 4.3-8.0 deg RMSE (laptop test).

Inputs per session: the Pi recording folder (keypoints.jsonl from 07_side_session.py) and the far-view reference
CSV from ref_movenet_track.py. Clocks are aligned automatically (knee-bend offset + per-pass drift).
Features: gait3d/angle_model.py (shared with the live session, so training and live use the same code).

  # train (+ cross-validation), save the model
  python3 scripts/train_angle_model.py --session runs/A refA.csv --session runs/B refB.csv --out config/angle_model.pkl
  # held-out test walk, never used for training; --styles labels the passes in walking order
  python3 scripts/train_angle_model.py --session runs/A refA.csv --session runs/B refB.csv \
      --test runs/T refT.csv --styles normal,normal,shuffle,shuffle,stiffL,stiffL,stiffR,stiffR
  # test a frozen model without retraining
  python3 scripts/train_angle_model.py --load config/angle_model.pkl --test runs/T refT.csv --styles ...

Pass gates (fixed before testing): median |swing-peak error| <= 5 deg, |bias| <= 3 deg, 95 % limits of
agreement within +-10 deg, per leg on the test walk.
"""
import argparse
import os
import pickle
import sys

import numpy as np
from scipy.signal import find_peaks, medfilt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gait3d.angle_model import FEATURE_VERSION, LEGS, AngleModel, features, flex, frame_points, valid  # noqa: E402
from walk_compare_channels import passes_from_ref  # noqa: E402


def load_session(rec):
    import json
    t, P = [], []
    for line in open(os.path.join(rec, 'keypoints.jsonl')):
        r = json.loads(line)
        t.append(r['t'])
        P.append(frame_points(r))
    return np.array(t), np.array(P)          # (N,), (N, 2 cams, 17, 3)


def align(tau, P, ref, sync, off_min, off_max, speed):
    """Pi time -> reference time: offset from the knee bends, then a per-pass drift fit.

    v3.3.2: every camera/leg pair is tried (the camera on the opposite side sees a leg's bend best), a pass counts
    only when its best pair correlates at r > 0.6, and the drift line is refitted without passes more than 0.15 s
    off it. v3.3.1 used only cam0-L / cam1-R and found a drift of 0.00015 on the 13:01 walk where the plane angles
    gave 0.0017 (0.4 s apart by the last pass, enough to miss +-0.2 s swing peaks)."""
    t = ref['t']
    F = np.column_stack([medfilt(ref['flex_L'], 5), medfilt(ref['flex_R'], 5)])
    raw = {(c, j): flex(P[:, c, :, :2], LEGS[j]) for c in range(2) for j in range(2)}
    s0, s1 = sync
    sel = (tau > s0) & (tau < s1)
    best = (-2, 0.0)
    for (c, j), q0 in raw.items():
        q = q0[sel]
        ok = ~np.isnan(q)
        if ok.sum() < 20:
            continue
        for o in np.arange(off_min, off_max, 0.02):
            r_ = np.corrcoef(np.interp(tau[sel][ok] + o, t, np.maximum(F[:, 0], F[:, 1])), q[ok])[0, 1]
            if r_ > best[0]:
                best = (r_, o)
    off = best[1]
    passes = passes_from_ref(t, ref['hip_x'], speed)
    mids, offs = [], []
    for p0, p1, _ in passes:
        m = (tau + off >= p0) & (tau + off <= p1)
        bb = (-2, off)
        for (c, j), q0 in raw.items():
            y = q0[m]
            ok = ~np.isnan(y) & (y < 75)
            if ok.sum() < 30:
                continue
            for o in np.arange(off - 1.0, off + 1.0, 0.02):
                r_ = np.corrcoef(np.interp(tau[m][ok] + o, t, F[:, j]), y[ok])[0, 1]
                if r_ > bb[0]:
                    bb = (r_, o)
        if bb[0] > 0.6:
            mids.append(np.mean(tau[m]))
            offs.append(bb[1])
    mids, offs = np.array(mids), np.array(offs)
    keep = np.ones(len(mids), bool)
    drift, b = 0.0, off
    for _ in range(3):
        if keep.sum() < 3:
            break
        drift, b = np.polyfit(mids[keep], offs[keep], 1)
        keep = np.abs(offs - (b + drift * mids)) <= 0.15
    print('  clock: sync r %.2f (needs > 0.9), t_ref = t_pi + %.2f + %.5f t_pi, %d passes; per-pass offsets %s%s' % (
        best[0], b, drift, len(passes), np.round(offs, 2).tolist(),
        '' if keep.all() else ' (dropped %d)' % (~keep).sum()))
    return t_pi_to_ref(tau, b, drift), F, t, passes


def t_pi_to_ref(tau, b, drift):
    return tau + b + drift * tau


def prepare(rec, refp, a):
    print(rec)
    tp, P = load_session(rec)
    tau = tp - tp[0]
    ref = np.genfromtxt(refp, delimiter=',', names=True, comments='#')
    tr, F, t, passes = align(tau, P, ref, tuple(map(float, a.sync.split(','))), a.off_min, a.off_max, a.speed)
    Y = np.column_stack([np.interp(tr, t, F[:, j]) for j in range(2)])
    grp = -np.ones(len(tau), int)
    for i, (p0, p1, _) in enumerate(passes):
        grp[(tr >= p0) & (tr <= p1)] = i
    raw = np.column_stack([flex(P[:, 0, :, :2], LEGS[0]), flex(P[:, 1, :, :2], LEGS[1])])
    return dict(P=P, X=features(P), V=valid(P), Y=Y, grp=grp, tr=tr, F=F, t=t, passes=passes, raw=raw)


def strides(pred, s, j):
    """Per reference swing peak: (pass index, reference peak, prediction max within +-0.2 s)."""
    out = []
    for i, (p0, p1, _) in enumerate(s['passes']):
        ii = np.where((s['t'] >= p0) & (s['t'] <= p1))[0]
        for p in find_peaks(s['F'][ii, j], prominence=12, distance=24)[0]:
            k = ii[p]
            w = (s['tr'] >= s['t'][k] - 0.2) & (s['tr'] <= s['t'][k] + 0.2)
            q = pred[w, j]
            q = q[~np.isnan(q)]
            if len(q):
                out.append((i, s['F'][k, j], q.max()))
    return np.array(out) if out else np.zeros((0, 3))


def agreement(name, pred, s, gates=None):
    inp = s['grp'] >= 0
    lines, ok_all = [], True
    for j, leg in ((0, 'L'), (1, 'R')):
        m = inp & ~np.isnan(pred[:, j])
        e = pred[m, j] - s['Y'][m, j]
        st = strides(pred, s, j)
        d = st[:, 2] - st[:, 1] if len(st) else np.array([np.nan])
        bias, sd = np.mean(d), np.std(d, ddof=1) if len(d) > 1 else np.nan
        med = np.median(np.abs(d))
        line = ('  %-8s %s  frames r %.2f RMSE %4.1f | swing peaks n %3d: ref %4.1f, model %4.1f, |err| median %4.1f, '
                'bias %+4.1f, 95%% LoA %+5.1f..%+5.1f' % (name, leg, np.corrcoef(pred[m, j], s['Y'][m, j])[0, 1],
                                                       np.sqrt(np.mean(e ** 2)), len(st), np.median(st[:, 1]),
                                                       np.median(st[:, 2]), med, bias, bias - 1.96 * sd,
                                                       bias + 1.96 * sd))
        if gates:
            g = med <= gates[0] and abs(bias) <= gates[1] and abs(bias) + 1.96 * sd <= gates[2]
            ok_all &= bool(g)
            line += '  -> %s' % ('PASS' if g else 'FAIL')
        lines.append(line)
    print('\n'.join(lines))
    return ok_all


def by_style(pred, s, styles):
    """Median swing peak per walking style: reference vs model vs the raw own-camera 2D angle."""
    print('  per walking style (median swing peak, deg):   ref   model   raw 2D')
    for st_name in dict.fromkeys(styles):
        for j, leg in ((0, 'L'), (1, 'R')):
            rows = [r for r in strides(pred, s, j) if r[0] < len(styles) and styles[int(r[0])] == st_name]
            raw = [r for r in strides(s['raw'], s, j) if r[0] < len(styles) and styles[int(r[0])] == st_name]
            if rows:
                rows = np.array(rows)
                print('    %-8s %s  n %2d                       %5.1f  %5.1f   %5.1f' % (
                    st_name, leg, len(rows), np.median(rows[:, 1]), np.median(rows[:, 2]),
                    np.median(np.array(raw)[:, 2]) if raw else np.nan))
    if len(styles) != len(s['passes']):
        print('  note: %d styles given for %d passes found' % (len(styles), len(s['passes'])))


def fit(Xa, Ya, kind):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    # 800 x 0.1: on the 13:01 walk slightly better swing peaks than 300 x 0.05 (R peak bias -5.1 vs -5.2, median
    # |peak err| 7.4 vs 8.0); weighting high angles did not help (v3.3.1 comparison)
    mk = {'gbm': lambda: HistGradientBoostingRegressor(max_iter=800, learning_rate=0.1), 'ridge': lambda: Ridge(1.0)}
    return [mk[kind]().fit(Xa, Ya[:, j]) for j in range(2)]


def predict(models, X, V):
    Y = np.column_stack([m.predict(X) for m in models])
    Y[~V] = np.nan
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', nargs=2, action='append', metavar=('REC_DIR', 'REF_CSV'), default=[])
    ap.add_argument('--test', nargs=2, metavar=('REC_DIR', 'REF_CSV'))
    ap.add_argument('--load', help='a saved model to test (no training)')
    ap.add_argument('--styles', default='', help='comma list, one per pass of the test walk, in walking order')
    ap.add_argument('--gates', default='5,3,10', help='max median |peak err|, max |bias|, max 95%% LoA (deg)')
    ap.add_argument('--sync', default='3,30')
    ap.add_argument('--off-min', type=float, default=0.0)
    ap.add_argument('--off-max', type=float, default=40.0)
    ap.add_argument('--speed', type=float, default=12.0)
    ap.add_argument('--out', default='config/angle_model.pkl')
    a = ap.parse_args()
    gates = tuple(map(float, a.gates.split(',')))

    S = [prepare(rec, refp, a) for rec, refp in a.session]
    if S and not a.load:
        for kind in ('gbm', 'ridge'):
            print('cross-validation, %s (%s):' % (kind, 'leave one session out' if len(S) > 1 else 'leave one pass out'))
            for si, s in enumerate(S):
                pred = np.full_like(s['Y'], np.nan)
                folds = [None] if len(S) > 1 else sorted(set(s['grp'][s['grp'] >= 0]))
                for f in folds:
                    if f is None:
                        Xtr = np.vstack([q['X'][q['grp'] >= 0] for k, q in enumerate(S) if k != si])
                        Ytr = np.vstack([q['Y'][q['grp'] >= 0] for k, q in enumerate(S) if k != si])
                        te = np.ones(len(s['X']), bool)
                    else:
                        tr_ = (s['grp'] >= 0) & (s['grp'] != f)
                        Xtr, Ytr, te = s['X'][tr_], s['Y'][tr_], s['grp'] == f
                    ms = fit(Xtr, Ytr, kind)
                    pred[te] = predict(ms, s['X'][te], s['V'][te])
                agreement(kind, pred, s)
        Xa = np.vstack([q['X'][q['grp'] >= 0] for q in S])
        Ya = np.vstack([q['Y'][q['grp'] >= 0] for q in S])
        gbm, ridge = fit(Xa, Ya, 'gbm'), fit(Xa, Ya, 'ridge')
        rj = {'coef': [m.coef_.tolist() for m in ridge], 'intercept': [float(m.intercept_) for m in ridge]}
        with open(a.out, 'wb') as fh:
            pickle.dump({'feature_version': FEATURE_VERSION, 'gbm': gbm, 'ridge': rj,
                         'sessions': [x[0] for x in a.session]}, fh)
        import json
        with open(a.out.replace('.pkl', '_ridge.json'), 'w') as fh:
            json.dump({'feature_version': FEATURE_VERSION, **rj}, fh)
        print('saved %s and %s (trained on %d frames from %d sessions)' % (a.out, a.out.replace('.pkl', '_ridge.json'),
                                                                         len(Xa), len(S)))
    if a.test:
        model = AngleModel(a.load or a.out)
        T = prepare(a.test[0], a.test[1], a)
        pred = model.predict_sequence(T['P'])
        print('HELD-OUT TEST (%s model, gates: median |peak err| <= %g, |bias| <= %g, 95%% LoA within +-%g deg):' % (
            model.kind, *gates))
        ok = agreement('model', pred, T, gates)
        agreement('raw 2D', T['raw'], T)
        if a.styles:
            by_style(pred, T, a.styles.split(','))
        np.savetxt(os.path.join(a.test[0], 'model_test_pred.csv'),
                   np.column_stack([T['tr'], T['Y'], pred]), delimiter=',', fmt='%.3f',
                   header='t_ref,ref_L,ref_R,model_L,model_R', comments='')
        print('RESULT:', 'PASS on both legs' if ok else 'FAIL (see above)')


if __name__ == '__main__':
    main()
