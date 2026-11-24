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

v3.7.0 (after the independent review, validation/review_20261124):
  --features own (default): each knee's model sees only its own leg (gait3d/angle_model.py, FEATURE_VERSION 3), so
      it cannot copy the other leg's movement; --features both = the version-2 features.
  Training uses only frames where both cameras see that leg (--all-frames = old behaviour).
  STRICT scoring, printed next to the old one and fixed before any new test walk: strides are cut on the reference
      alone (between knee-flexion minima, 0.7-3.5 s, near-leg passes), each signal's own peak and own minimum in the
      stride, a stride counts when the model has values for >= 80 % of it; reported: strides scored / found, median
      |peak error|, bias, 95 % limits, share of strides within +-5 deg, stance-minimum bias, with 95 % confidence
      intervals from resampling whole passes. Goal: 95 % of strides within +-5 deg.
  DESYNC test (--test): the other leg's points are replaced by points from other walking moments; a model that
      measures the leg must not change (own-leg models are unaffected by construction).
"""
import argparse
import os
import pickle
import sys

import numpy as np
from scipy.signal import find_peaks, medfilt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gait3d.angle_model import (FEATURE_VERSION, LEGS, OWN_FEATURE_VERSION, AngleModel, flex, frame_points,  # noqa: E402
                                leg_features, valid)
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
    def walking_match():
        # v3.3.7: match the walking itself: every Pi camera/leg angle against the same reference leg, all passes.
        w = (-2, off)
        for (c, j), q0 in raw.items():
            Fj = medfilt(ref['flex_L' if j == 0 else 'flex_R'], 5)
            for o in np.arange(off_min, off_max, 0.05):
                tt = tau + o
                m = ~np.isnan(q0) & (q0 < 75) & (tt > passes[0][0]) & (tt < passes[-1][1])
                if m.sum() < 100:
                    continue
                r_ = np.corrcoef(np.interp(tt[m], t, Fj), q0[m])[0, 1]
                if r_ > w[0]:
                    w = (r_, o)
        return w

    def per_pass(off):
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
        return mids, offs

    if best[0] < 0.9 and passes:
        # knee bends missed (23 Nov walks 2-3: bends at 27-36 s, start of the iPad video unusable)
        w = walking_match()
        print('  clock: knee-bend sync r %.2f too low; walking match r %.2f at offset %.2f s' % (best[0], *w))
        off = w[1]
    mids, offs = per_pass(off)
    if passes and len(offs) < len(passes) / 2:
        # v3.4.1: the knee-bend match passed (r > 0.9) but fewer than half the passes confirm it: a false match
        # (24 Nov test walk: bends at 48-57 s, outside the window; r 0.91 at -12.6 s, no pass confirmed; the
        # walking gives +47.8 s, all passes within 0.2 s).
        w = walking_match()
        print('  clock: only %d of %d passes confirm offset %.2f s -> walking match r %.2f at %.2f s' % (
            len(offs), len(passes), off, *w))
        off = w[1]
        mids, offs = per_pass(off)
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
    near = near_legs(t, F, passes)
    Yt = Y.copy()
    if getattr(a, 'labels', 'near') == 'near':
        for i in range(len(passes)):
            Yt[grp == i, 1 - near[i]] = np.nan
    version = OWN_FEATURE_VERSION if getattr(a, 'features', 'own') == 'own' else FEATURE_VERSION
    return dict(P=P, Xs=leg_features(P, version), V=valid(P), Y=Y, Yt=Yt, grp=grp, tr=tr, F=F, t=t, passes=passes,
                raw=raw, near=near, labels=getattr(a, 'labels', 'near'))


def near_legs(t, F, passes):
    """v3.3.6: which leg is nearer the reference camera in each pass. The far leg is partly hidden behind the near
    one, so its reference angle is unreliable (23 Nov walk: a stiff-right pass shows up only on the near-side R
    trace, 18.6 deg, and a far-side reading jumped to 71 deg). In a normal (not mirrored) video, a person walking
    towards image-right shows their right side, walking towards image-left their left side, whichever side of the
    walkway the camera stands on. The direction comes from the reference mid-hip motion."""
    out = [1 if d > 0 else 0 for _, _, d in passes]
    print('  near leg per pass (reference camera): %s' % ''.join('LR'[k] for k in out))
    return out

def strides(pred, s, j):
    """Per reference swing peak: (pass index, reference peak, prediction max within +-0.2 s)."""
    out = []
    for i, (p0, p1, _) in enumerate(s['passes']):
        if s.get('labels') == 'near' and s['near'][i] != j:
            continue
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
        m = inp & ~np.isnan(pred[:, j]) & ~np.isnan(s['Yt'][:, j])
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


def fit(Xs, Ya, kind, version=OWN_FEATURE_VERSION):
    """Xs = [X for L, X for R] (rows already selected); Ya = (N, 2) labels, NaN = not used for that leg."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    # both-leg features: 800 x 0.1 (v3.3.1). Own-leg features (v3.7.0): smaller, regularised trees, chosen on pooled
    # leave-one-walk-out strict scoring of the three 23 Nov walks (median |peak err| L 4.0 / R 4.4 vs 4.5 / 4.8 for
    # 800 x 0.1); the 800 x 0.1 model fitted its training data to 0.02 deg (review finding m5).
    if version == OWN_FEATURE_VERSION:
        g = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=60,
                                                  l2_regularization=1.0, max_leaf_nodes=15)
    else:
        g = lambda: HistGradientBoostingRegressor(max_iter=800, learning_rate=0.1)
    mk = {'gbm': g, 'ridge': lambda: Ridge(1.0)}
    out = []
    for j in range(2):
        ok = ~np.isnan(Ya[:, j])      # v3.3.6: far-leg labels are NaN with --labels near; v3.7.0: unseen frames too
        out.append(mk[kind]().fit(Xs[j][ok], Ya[ok, j]))
    return out


def predict(models, Xs, V):
    Y = np.column_stack([m.predict(X) for m, X in zip(models, Xs)])
    Y[~V] = np.nan
    return Y


def train_rows(q, a):
    """Rows of a session used for training and their labels (unseen frames' labels set to NaN unless --all-frames)."""
    m = q['grp'] >= 0
    Y = q['Yt'][m].copy()
    if not getattr(a, 'all_frames', False):
        Y[~q['V'][m]] = np.nan
    return [X[m] for X in q['Xs']], Y


def stack_rows(sessions, a):
    parts = [train_rows(q, a) for q in sessions]
    return [np.vstack([p[0][j] for p in parts]) for j in range(2)], np.vstack([p[1] for p in parts])


def strict_strides(pred, s, j, min_valid=0.8):
    """v3.7.0 strict scoring: strides cut on the reference alone. Rows (pass, ref_peak, model_peak, ref_min,
    model_min, scored)."""
    t, F, tr = s['t'], s['F'], s['tr']
    rows = []
    for i, (p0, p1, _) in enumerate(s['passes']):
        if s.get('labels') == 'near' and s['near'][i] != j:
            continue
        ii = np.where((t >= p0) & (t <= p1))[0]
        f = F[ii, j]
        mins = find_peaks(-f, distance=18, prominence=4)[0]
        for m0, m1 in zip(mins[:-1], mins[1:]):
            if not 0.7 <= t[ii[m1]] - t[ii[m0]] <= 3.5:
                continue
            seg = f[m0:m1 + 1]
            q = pred[(tr >= t[ii[m0]]) & (tr <= t[ii[m1]]), j]
            ok = len(q) >= 5 and np.mean(~np.isnan(q)) >= min_valid
            rows.append((i, seg.max(), np.nanmax(q) if ok else np.nan, seg.min(), np.nanmin(q) if ok else np.nan,
                         float(ok)))
    return np.array(rows) if rows else np.zeros((0, 6))


def strict_report(name, rows_by_leg, nboot=2000, seed=0):
    """rows_by_leg: {0: rows, 1: rows}, rows with a 7th column = pass key (unique across walks)."""
    rng = np.random.default_rng(seed)
    st = lambda d: (np.median(np.abs(d)), d.mean(), d.mean() - 1.96 * d.std(ddof=1), d.mean() + 1.96 * d.std(ddof=1),
                    100 * np.mean(np.abs(d) <= 5))
    res = {}
    for j, leg in ((0, 'L'), (1, 'R')):
        R = rows_by_leg[j]
        sc = R[R[:, 5] == 1]
        if len(sc) < 3:
            print('  %-8s %s  strict: too few strides (%d)' % (name, leg, len(sc)))
            continue
        d, ds, key = sc[:, 2] - sc[:, 1], sc[:, 4] - sc[:, 3], sc[:, 6]
        base, u = st(d), np.unique(key)
        B = np.array([st(np.concatenate([d[key == k] for k in rng.choice(u, len(u))])) for _ in range(nboot)])
        ci = np.percentile(B, [2.5, 97.5], axis=0)
        print('  %-8s %s  STRICT %d/%d strides, %d passes: |peak err| median %.1f [%.1f-%.1f], bias %+.1f [%+.1f, %+.1f], '
              '95%% limits %+.1f..%+.1f, within +-5 deg %.0f%% [%.0f-%.0f] (goal 95%%), stance-min bias %+.1f' % (
                  name, leg, len(sc), len(R), len(u), base[0], ci[0, 0], ci[1, 0], base[1], ci[0, 1], ci[1, 1],
                  base[2], base[3], base[4], ci[0, 4], ci[1, 4], ds.mean()))
        res[leg] = base
    return res


def strict_rows(pred, s, key0=0):
    out = {}
    for j in range(2):
        r = strict_strides(pred, s, j)
        out[j] = np.column_stack([r, key0 + r[:, 0]]) if len(r) else np.zeros((0, 7))
    return out


def desync_test(model, s, shifts=(37, 83, 151, 211, 307)):
    """Other leg's points (both cameras) taken from other walking frames; prints the change in each leg's score."""
    inpass = np.where(s['grp'] >= 0)[0]
    base = model.predict_sequence(s['P'])
    print('  DESYNC test (other leg from other walking moments, %d shifts): mean |change| of each knee\'s angle '
          'inside passes' % len(shifts))
    for j, leg in ((0, 'L'), (1, 'R')):
        other = list(LEGS[1 - j])
        ch = []
        for sh in shifts:
            P2 = s['P'].copy()
            donor = np.roll(inpass, sh)
            P2[np.ix_(inpass, [0, 1], other, [0, 1, 2])] = s['P'][np.ix_(donor, [0, 1], other, [0, 1, 2])]
            p = model.predict_sequence(P2)[inpass, j]
            ch.append(np.nanmean(np.abs(p - base[inpass, j])))
        print('    %s: %s deg -> %s' % (leg, ' '.join('%.2f' % c for c in ch),
                                       'unaffected' if max(ch) < 0.05 else 'DEPENDS ON THE OTHER LEG'))


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
    ap.add_argument('--labels', default='near', choices=['near', 'both'],
                    help='v3.3.6: near = per pass, learn and score only the leg nearer the reference camera')
    ap.add_argument('--features', default='own', choices=['own', 'both'],
                    help='v3.7.0: own = each knee from its own leg only (default); both = version-2 features')
    ap.add_argument('--all-frames', action='store_true', help='also train on frames where a camera misses the leg')
    a = ap.parse_args()
    gates = tuple(map(float, a.gates.split(',')))

    S = [prepare(rec, refp, a) for rec, refp in a.session]
    version = OWN_FEATURE_VERSION if a.features == 'own' else FEATURE_VERSION
    if S and not a.load:
        for kind in ('gbm', 'ridge'):
            print('cross-validation, %s, %s-leg features (%s):' % (
                kind, a.features, 'leave one session out' if len(S) > 1 else 'leave one pass out'))
            pooled = {0: [], 1: []}
            for si, s in enumerate(S):
                pred = np.full_like(s['Y'], np.nan)
                folds = [None] if len(S) > 1 else sorted(set(s['grp'][s['grp'] >= 0]))
                for f in folds:
                    if f is None:
                        Xtr, Ytr = stack_rows([q for k, q in enumerate(S) if k != si], a)
                        te = np.ones(len(s['P']), bool)
                    else:
                        keep = dict(s, grp=np.where(s['grp'] != f, s['grp'], -1))
                        Xtr, Ytr = train_rows(keep, a)
                        te = s['grp'] == f
                    ms = fit(Xtr, Ytr, kind, version)
                    pred[te] = predict(ms, [X[te] for X in s['Xs']], s['V'][te])
                agreement(kind, pred, s)
                r = strict_rows(pred, s, key0=1000 * si)
                for j in range(2):
                    pooled[j].append(r[j])
            strict_report('pooled', {j: np.vstack(pooled[j]) for j in range(2)})
        Xa, Ya = stack_rows(S, a)
        gbm, ridge = fit(Xa, Ya, 'gbm', version), fit(Xa, Ya, 'ridge', version)
        rj = {'coef': [m.coef_.tolist() for m in ridge], 'intercept': [float(m.intercept_) for m in ridge]}
        with open(a.out, 'wb') as fh:
            pickle.dump({'feature_version': version, 'gbm': gbm, 'ridge': rj,
                         'sessions': [x[0] for x in a.session]}, fh)
        import json
        with open(a.out.replace('.pkl', '_ridge.json'), 'w') as fh:
            json.dump({'feature_version': version, **rj}, fh)
        print('saved %s and %s (feature version %d, trained on %d / %d frames (L / R) from %d sessions)' % (
            a.out, a.out.replace('.pkl', '_ridge.json'), version, (~np.isnan(Ya[:, 0])).sum(),
            (~np.isnan(Ya[:, 1])).sum(), len(S)))
    if a.test:
        model = AngleModel(a.load or a.out)
        T = prepare(a.test[0], a.test[1], a)
        pred = model.predict_sequence(T['P'])
        print('HELD-OUT TEST (%s model, gates: median |peak err| <= %g, |bias| <= %g, 95%% LoA within +-%g deg):' % (
            model.kind, *gates))
        ok = agreement('model', pred, T, gates)
        agreement('raw 2D', T['raw'], T)
        strict_report('model', strict_rows(pred, T))
        desync_test(model, T)
        if a.styles:
            by_style(pred, T, a.styles.split(','))
        np.savetxt(os.path.join(a.test[0], 'model_test_pred.csv'),
                   np.column_stack([T['tr'], T['Y'], pred]), delimiter=',', fmt='%.3f',
                   header='t_ref,ref_L,ref_R,model_L,model_R', comments='')
        print('RESULT:', 'PASS on both legs' if ok else 'FAIL (see above)')


if __name__ == '__main__':
    main()
