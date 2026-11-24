"""Our gait cycles with the deployed v3.7.0 own-leg model (training walks: leave-one-walk-out; test walk: model trained
on the three training walks) and the iPad MoveNet reference. Near-leg passes only; cycles from swing peak to swing peak."""
import pickle, sys
import numpy as np
from scipy.signal import find_peaks
sys.path.insert(0, '/home/claude/gait_analysis_in_3d'); sys.path.insert(0, '/home/claude/gait_analysis_in_3d/scripts')
from gait3d.angle_model import OWN_FEATURE_VERSION, leg_features
import train_angle_model as T
from types import SimpleNamespace

S = pickle.load(open('/home/claude/work/v37/prepared.pkl', 'rb'))
for s in S.values():
    s['Xs'] = leg_features(s['P'], OWN_FEATURE_VERSION)
a = SimpleNamespace(all_frames=False)
train = ['w1', 'w2', 'w3']
pred = {}
for name in S:
    tr = [n for n in train if n != name]
    Xs, Y = T.stack_rows([S[n] for n in tr], a)
    ms = T.fit(Xs, Y, 'gbm', OWN_FEATURE_VERSION)
    pred[name] = T.predict(ms, S[name]['Xs'], S[name]['V'])

def period(tt, y, p0, p1):
    m = (tt >= p0) & (tt <= p1) & np.isfinite(y)
    if m.sum() < 30: return np.nan
    t = tt[m]; dt = np.median(np.diff(t)); u = np.arange(t[0], t[-1], dt); v = np.interp(u, t, y[m]); v = v - v.mean()
    ac = np.correlate(v, v, 'full')[len(v) - 1:]; ac /= ac[0]
    lo, hi = int(0.7 / dt), min(len(ac) - 1, int(2.6 / dt))
    if hi <= lo: return np.nan
    k = lo + int(np.argmax(ac[lo:hi])); return k * dt if ac[k] > 0.3 else np.nan

def cycles(tt, y, p0, p1, Tp=None, fs_min=8):
    m = (tt >= p0) & (tt <= p1) & np.isfinite(y)
    t, v = tt[m], y[m]
    if len(t) < 10: return []
    dt = np.median(np.diff(t))
    thr = max(15.0, 0.6 * np.percentile(v, 95))
    pk, _ = find_peaks(v, height=thr, distance=max(1, int(0.7 / dt)), prominence=10)
    out = []
    for a_, b_ in zip(pk[:-1], pk[1:]):
        T_ = t[b_] - t[a_]
        if 0.7 <= T_ <= 2.5 and (b_ - a_) >= fs_min * T_ * 0.5 and (Tp is None or not np.isfinite(Tp) or abs(T_ - Tp) <= 0.25 * Tp):
            out.append((np.interp(np.linspace(t[a_], t[b_], 101), t, v), T_))
    return out

rows = []
for name, s in S.items():
    for i, (p0, p1, d) in enumerate(s['passes']):
        j = s['near'][i]
        Tp = period(s['t'], s['F'][:, j], p0, p1)
        rc = cycles(s['t'], s['F'][:, j], p0, p1, Tp=Tp)
        mc = cycles(s['tr'], pred[name][:, j], p0, p1, Tp=Tp)
        refpk = np.median([c.max() for c, _ in rc]) if rc else np.nan
        rows.append(dict(walk=name, pass_=i, leg='LR'[j], ref_cycles=rc, model_cycles=mc, ref_peak=refpk, period=Tp))
pickle.dump(dict(rows=rows, pred=pred), open('/home/claude/work/deep/ours.pkl', 'wb'))
nl = lambda r: 40 <= r['ref_peak'] <= 75
st = lambda r: r['ref_peak'] < 35
for lab, sel in (('normal-like', nl), ('stiff/shuffle', st)):
    R = [r for r in rows if sel(r)]
    print(lab, 'passes', len(R), 'ref cycles', sum(len(r['ref_cycles']) for r in R),
          'model cycles L', sum(len(r['model_cycles']) for r in R if r['leg'] == 'L'), 'R', sum(len(r['model_cycles']) for r in R if r['leg'] == 'R'),
          'stride %.2f s' % np.median([T_ for r in R for _, T_ in r['ref_cycles']]))
