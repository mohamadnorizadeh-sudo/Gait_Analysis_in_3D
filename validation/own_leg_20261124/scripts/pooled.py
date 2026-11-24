"""Pooled leave-one-walk-out (strict scoring) to CHOOSE the model on training walks only; the test walk is scored
once afterwards for the chosen model (and, for the record, the others)."""
import sys, numpy as np
import exp
from exp import S, TRAIN, fit, predict, strict_strides
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

CONFIGS = {
    'both gbm (deployed)': ('both', lambda: HistGradientBoostingRegressor(max_iter=800, learning_rate=0.1)),
    'own gbm 800x0.1':     ('own',  lambda: HistGradientBoostingRegressor(max_iter=800, learning_rate=0.1)),
    'own gbm regularised': ('own',  lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=60, l2_regularization=1.0, max_leaf_nodes=15)),
    'own ridge':           ('own',  lambda: Ridge(1.0)),
    'both ridge':          ('both', lambda: Ridge(1.0)),
}

def summarise(rows_by_key, rng=np.random.default_rng(0), nboot=2000):
    out = {}
    for leg in 'LR':
        R = np.vstack(rows_by_key[leg]); sc = R[R[:, 5] == 1]
        d = sc[:, 2] - sc[:, 1]; ds = sc[:, 4] - sc[:, 3]
        keys = sc[:, 6]
        st = lambda dd: (np.median(np.abs(dd)), dd.mean(), dd.mean() - 1.96 * dd.std(ddof=1), dd.mean() + 1.96 * dd.std(ddof=1), 100 * np.mean(np.abs(dd) <= 5))
        base = st(d); u = np.unique(keys)
        B = np.array([st(np.concatenate([d[keys == k] for k in rng.choice(u, len(u))])) for _ in range(nboot)])
        ci = np.percentile(B, [2.5, 97.5], axis=0)
        out[leg] = (len(R), len(sc), base, ci, ds.mean())
    return out

def fmt(tag, o):
    for leg in 'LR':
        n, ns, b, ci, sb = o[leg]
        print(f'  {tag:<22}{leg}: {ns}/{n} strides  median |err| {b[0]:.1f} [{ci[0,0]:.1f}-{ci[1,0]:.1f}]  bias {b[1]:+.1f} [{ci[0,1]:+.1f},{ci[1,1]:+.1f}]'
              f'  LoA {b[2]:+.1f}..{b[3]:+.1f}  within +-5: {b[4]:.0f}% [{ci[0,4]:.0f}-{ci[1,4]:.0f}]  stance bias {sb:+.1f}')

which = sys.argv[1:] or list(CONFIGS)
for name in which:
    mode, maker = CONFIGS[name]
    exp.mk = maker
    rows = {'L': [], 'R': []}
    for wi, k in enumerate(TRAIN):
        ms = fit([S[x] for x in TRAIN if x != k], mode, True)
        p = predict(ms, S[k]['P'], mode)
        for j, leg in enumerate('LR'):
            r = strict_strides(p, S[k], j)
            rows[leg].append(np.column_stack([r, wi * 100 + r[:, 0]]))
    print(f'=== {name}')
    fmt('pooled LOWO (3 walks)', summarise(rows))
    ms = fit([S[x] for x in TRAIN], mode, True)
    p = predict(ms, S['test']['P'], mode)
    rt = {leg: [np.column_stack([r, r[:, 0]])] for j, leg in enumerate('LR') for r in [strict_strides(p, S['test'], j)]}
    fmt('held-out test', summarise(rt))
    np.save(f'pred_test_{name.split()[0]}_{name.split()[1]}_{len(name)}.npy', p)
