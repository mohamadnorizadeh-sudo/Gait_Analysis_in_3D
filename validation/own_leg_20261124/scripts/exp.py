"""v3.7.0 prototype: own-leg features vs the deployed both-leg features; old and strict scoring; desync test.

Usage: python3 exp.py [gbm|ridge]
"""
import os
import pickle
import sys

import numpy as np
from scipy.signal import find_peaks

REPO = '/home/claude/gait_analysis_in_3d'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from gait3d.angle_model import LEGS, MIN_CONF, features, flex, valid  # noqa: E402
from train_angle_model import strides  # noqa: E402

KIND = sys.argv[1] if len(sys.argv) > 1 else 'gbm'
S = pickle.load(open('/home/claude/work/v37/prepared.pkl', 'rb'))
TRAIN = ['w1', 'w2', 'w3']


def own_frame(P, j):
    """(N, 2, 17, 3) -> (N, 22): leg j only, per camera: knee and ankle relative to that leg's hip / its own length,
    3 confidences, raw 2D knee angle, log leg length (px), thigh/shank ratio, seen flag."""
    h, k, a = LEGS[j]
    out = []
    for ci in range(2):
        xy, c = P[:, ci, :, :2], P[:, ci, :, 2]
        th = np.linalg.norm(xy[:, h] - xy[:, k], axis=1)
        sh = np.linalg.norm(xy[:, k] - xy[:, a], axis=1)
        L = th + sh + 1e-6
        seen = np.all(np.nan_to_num(c[:, [h, k, a]], nan=0.0) >= MIN_CONF, axis=1).astype(float)
        out += [(xy[:, k] - xy[:, h]) / L[:, None], (xy[:, a] - xy[:, h]) / L[:, None], c[:, [h, k, a]],
                flex(xy, LEGS[j])[:, None], np.log(L)[:, None], (th / (sh + 1e-6))[:, None], seen[:, None]]
    X = np.hstack(out)
    # unseen camera: zero its numbers (the seen flag says so) instead of leaving NaN or stale values
    for ci in range(2):
        cols = slice(ci * 11, ci * 11 + 10)
        X[X[:, ci * 11 + 10] == 0, cols] = 0.0
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def own_features(P, j, context=3):
    X = own_frame(P, j)
    lag = [X] + [np.vstack([np.repeat(X[:1], d, 0), X[:-d]]) for d in range(1, context)]
    return np.hstack(lag)


def feats(P, mode):
    if mode == 'both':
        X = features(P)
        return [X, X]
    return [own_features(P, 0), own_features(P, 1)]


def mk():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    return HistGradientBoostingRegressor(max_iter=800, learning_rate=0.1) if KIND == 'gbm' else Ridge(1.0)


def fit(sessions, mode, valid_only):
    models = []
    for j in range(2):
        Xs, ys = [], []
        for s in sessions:
            Xj = feats(s['P'], mode)[j]
            m = (s['grp'] >= 0) & ~np.isnan(s['Yt'][:, j])
            if valid_only:
                m &= s['V'][:, j]
            Xs.append(Xj[m]), ys.append(s['Yt'][m, j])
        models.append(mk().fit(np.vstack(Xs), np.concatenate(ys)))
    return models


def predict(models, P, mode):
    X = feats(P, mode)
    Y = np.column_stack([models[j].predict(X[j]) for j in range(2)])
    Y[~valid(P)] = np.nan
    return Y


# ---------- scoring ----------
def old_scores(pred, s):
    out = {}
    for j in range(2):
        st = strides(pred, s, j)
        d = st[:, 2] - st[:, 1]
        sd = np.std(d, ddof=1)
        out['LR'[j]] = dict(n=len(d), med=np.median(np.abs(d)), bias=d.mean(), lo=d.mean() - 1.96 * sd,
                            hi=d.mean() + 1.96 * sd)
    return out


def strict_strides(pred, s, j, min_valid=0.8):
    """Strides segmented on the reference alone (between knee-flexion minima, 0.7-3.5 s), near-leg passes only.
    Each signal's own peak and own minimum inside the stride. Returns rows
    (pass, ref_peak, model_peak, ref_min, model_min, scored)."""
    t, F, tr = s['t'], s['F'], s['tr']
    rows = []
    for i, (p0, p1, _) in enumerate(s['passes']):
        if s['near'][i] != j:
            continue
        ii = np.where((t >= p0) & (t <= p1))[0]
        f = F[ii, j]
        mins = find_peaks(-f, distance=18, prominence=4)[0]
        for m0, m1 in zip(mins[:-1], mins[1:]):
            dur = t[ii[m1]] - t[ii[m0]]
            if not 0.7 <= dur <= 3.5:
                continue
            seg = f[m0:m1 + 1]
            w = (tr >= t[ii[m0]]) & (tr <= t[ii[m1]])
            q = pred[w, j]
            ok = (len(q) >= 5) and np.mean(~np.isnan(q)) >= min_valid
            rows.append((i, seg.max(), np.nanmax(q) if ok else np.nan, seg.min(), np.nanmin(q) if ok else np.nan,
                         float(ok)))
    return np.array(rows) if rows else np.zeros((0, 6))


def strict_scores(pred, s, nboot=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for j in range(2):
        R = strict_strides(pred, s, j)
        sc = R[R[:, 5] == 1]
        d, ds = sc[:, 2] - sc[:, 1], sc[:, 4] - sc[:, 3]

        def stats(dd):
            sd = np.std(dd, ddof=1)
            return np.array([np.median(np.abs(dd)), dd.mean(), dd.mean() - 1.96 * sd, dd.mean() + 1.96 * sd,
                             100 * np.mean(np.abs(dd) <= 5)])
        base = stats(d)
        passes = np.unique(sc[:, 0])
        B = []
        for _ in range(nboot):   # resample whole passes: strides in one pass are not independent
            pick = rng.choice(passes, len(passes))
            dd = np.concatenate([d[sc[:, 0] == p] for p in pick])
            if len(dd) > 2:
                B.append(stats(dd))
        B = np.array(B)
        ci = np.percentile(B, [2.5, 97.5], axis=0)
        old_gate = np.mean((B[:, 0] <= 5) & (np.abs(B[:, 1]) <= 3) & (np.maximum(-B[:, 2], B[:, 3]) <= 10))
        out['LR'[j]] = dict(total=len(R), scored=len(sc), med=base[0], bias=base[1], lo=base[2], hi=base[3],
                            within5=base[4], ci=ci, p_old_gates=old_gate,
                            stance_bias=ds.mean(), stance_med=np.median(np.abs(ds)), passes=len(passes))
    return out


def show(tag, old, new):
    for leg in 'LR':
        o, n = old[leg], new[leg]
        print(f'  {tag:<22} {leg} old: n{o["n"]:3d} med {o["med"]:4.1f} bias {o["bias"]:+5.1f} LoA {o["lo"]:+6.1f}..{o["hi"]:+5.1f}'
              f' | strict: {n["scored"]:2d}/{n["total"]:2d} strides ({n["passes"]} passes) med {n["med"]:4.1f}'
              f' [{n["ci"][0,0]:.1f}-{n["ci"][1,0]:.1f}] bias {n["bias"]:+5.1f} [{n["ci"][0,1]:+.1f},{n["ci"][1,1]:+.1f}]'
              f' LoA {n["lo"]:+6.1f}..{n["hi"]:+5.1f} within5 {n["within5"]:3.0f}% stance bias {n["stance_bias"]:+5.1f}'
              f' P(old gates) {n["p_old_gates"]:.2f}')


def desync(models, s, mode, shifts=(37, 83, 151, 211, 307)):
    """Replace the OTHER leg's points with those from other walking frames; report how each leg's score moves."""
    inpass = np.where(s['grp'] >= 0)[0]
    res = []
    for sh in shifts:
        preds = []
        for j in range(2):
            P2 = s['P'].copy()
            other = list(LEGS[1 - j])
            donor = np.roll(inpass, sh)
            # frames inside passes get the other leg's points (both cameras) from another walking frame
            P2[np.ix_(inpass, [0, 1], other, [0, 1, 2])] = s['P'][np.ix_(donor, [0, 1], other, [0, 1, 2])]
            preds.append(predict(models, P2, mode)[:, j])
        res.append(strict_scores(np.column_stack(preds), s, nboot=200))
    return res


if __name__ == '__main__':
    for mode in ('both', 'own'):
        for vo in ((False, True) if mode == 'both' else (True,)):
            tag = f'{mode}{"+validonly" if vo else ""}'
            print(f'=== {KIND} features={tag}')
            # leave one walk out
            allp = {}
            for k in TRAIN:
                ms = fit([S[x] for x in TRAIN if x != k], mode, vo)
                p = predict(ms, S[k]['P'], mode)
                allp[k] = p
                show(f'LOWO {k}', old_scores(p, S[k]), strict_scores(p, S[k], nboot=300))
            # held-out test, model on all three walks
            ms = fit([S[x] for x in TRAIN], mode, vo)
            p = predict(ms, S['test']['P'], mode)
            show('TEST', old_scores(p, S['test']), strict_scores(p, S['test']))
            np.save(f'/home/claude/work/v37/pred_test_{KIND}_{tag}.npy', p)
            if mode == 'both' and not vo or mode == 'own':
                ds = desync(ms, S['test'], mode)
                for leg in 'LR':
                    print(f'  desync TEST {leg}: med ' + ' '.join(f'{d[leg]["med"]:.1f}' for d in ds) +
                          ' | bias ' + ' '.join(f'{d[leg]["bias"]:+.1f}' for d in ds) +
                          ' | within5 ' + ' '.join(f'{d[leg]["within5"]:.0f}' for d in ds))
