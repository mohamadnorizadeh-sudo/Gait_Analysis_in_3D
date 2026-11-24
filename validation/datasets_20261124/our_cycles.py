"""Our gait cycles (reference and Pi model) for comparison with published healthy-gait datasets.
Walks 1-3: leave-one-walk-out predictions; test walk: model trained on walks 1-3. Near-leg passes only."""
import sys, argparse, numpy as np, pickle
sys.path.insert(0, '/home/claude/gait_analysis_in_3d'); sys.path.insert(0, '/home/claude/gait_analysis_in_3d/scripts')
import train_angle_model as T
from scipy.signal import find_peaks
W = '/home/claude/work'
a = argparse.Namespace(sync='3,30', off_min=-30, off_max=60, speed=12, labels='near')
S = {}
for name, rec, ref in (('walk1', 'w1/rec', 'w1/ref_ipad_movenet_clean.csv'), ('walk2', 'w2/rec', 'w2/ref_ipad_movenet_clean.csv'),
                       ('walk3', 'w3/rec', 'w3/ref_ipad_movenet_clean.csv'), ('test', 'test1/rec', 'test1/ref_ipad_movenet_clean.csv')):
    S[name] = T.prepare(f'{W}/{rec}', f'{W}/{ref}', a)
train = ['walk1', 'walk2', 'walk3']
pred = {}
for name in S:
    tr_names = [n for n in train if n != name]
    Xa = np.vstack([S[n]['X'][S[n]['grp'] >= 0] for n in tr_names]); Ya = np.vstack([S[n]['Yt'][S[n]['grp'] >= 0] for n in tr_names])
    ms = T.fit(Xa, Ya, 'gbm')
    pred[name] = T.predict(ms, S[name]['X'], S[name]['V'])

def period(tt, y, p0, p1):
    m = (tt >= p0) & (tt <= p1) & np.isfinite(y)
    if m.sum() < 30: return np.nan
    t = tt[m]; dt = np.median(np.diff(t)); u = np.arange(t[0], t[-1], dt); v = np.interp(u, t, y[m]); v = v - v.mean()
    ac = np.correlate(v, v, 'full')[len(v) - 1:]; ac /= ac[0]
    lo, hi = int(0.7 / dt), min(len(ac) - 1, int(2.6 / dt))
    if hi <= lo: return np.nan
    k = lo + int(np.argmax(ac[lo:hi])); return k * dt if ac[k] > 0.3 else np.nan

def cycles(tt, y, p0, p1, fs_min=8, Tp=None):
    """cycles between consecutive swing peaks inside a pass, resampled to 101 points (0 = swing peak)."""
    m = (tt >= p0) & (tt <= p1) & np.isfinite(y)
    t, v = tt[m], y[m]
    if len(t) < 10:
        return []
    dt = np.median(np.diff(t))
    thr = max(15.0, 0.6 * np.percentile(v, 95))
    pk, _ = find_peaks(v, height=thr, distance=max(1, int(0.7 / dt)), prominence=10)
    out = []
    for a_, b_ in zip(pk[:-1], pk[1:]):
        T_ = t[b_] - t[a_]
        if 0.7 <= T_ <= 2.5 and (b_ - a_) >= fs_min * T_ * 0.5 and (Tp is None or not np.isfinite(Tp) or abs(T_ - Tp) <= 0.25 * Tp):
            u = np.linspace(t[a_], t[b_], 101)
            out.append((np.interp(u, t, v), T_))
    return out

rows = []
for name, s in S.items():
    for i, (p0, p1, d) in enumerate(s['passes']):
        j = s['near'][i]
        # reference at its own 30 fps clock, model at Pi frames mapped to reference time
        Tp = period(s['t'], s['F'][:, j], p0, p1)
        rc = cycles(s['t'], s['F'][:, j], p0, p1, Tp=Tp)
        mc = cycles(s['tr'], pred[name][:, j], p0, p1, Tp=Tp)
        refpk = np.median([c.max() for c, _ in rc]) if rc else np.nan
        rows.append(dict(walk=name, pass_=i, leg='LR'[j], ref_cycles=rc, model_cycles=mc, ref_peak=refpk, period=Tp))
pickle.dump(rows, open('/home/claude/work/norm/our_cycles.pkl', 'wb'))
for r in rows:
    print(r['walk'], r['pass_'], r['leg'], 'ref cycles', len(r['ref_cycles']), 'model cycles', len(r['model_cycles']), 'ref peak %.1f' % r['ref_peak'],
          'stride %.2f s' % (np.median([T_ for _, T_ in r['ref_cycles']]) if r['ref_cycles'] else np.nan), 'autocorr period %.2f s' % r['period'])
