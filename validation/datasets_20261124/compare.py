import numpy as np, json, pickle
N = json.load(open('norm_curves.json')); rows = pickle.load(open('our_cycles.pkl', 'rb'))
def rot(m):                       # re-reference a heel-strike cycle so that index 0 = swing peak (cyclic)
    m = np.asarray(m)[:100]; k = int(np.argmax(m)); r = np.r_[m[k:], m[:k]]; return np.r_[r, r[0]]
def group(sel):
    R = [c for r in rows if sel(r) for c, _ in r['ref_cycles']]
    M = {L: [c for r in rows if sel(r) and r['leg'] == L for c, _ in r['model_cycles']] for L in 'LR'}
    Rl = {L: [c for r in rows if sel(r) and r['leg'] == L for c, _ in r['ref_cycles']] for L in 'LR'}
    T = [T_ for r in rows if sel(r) for _, T_ in r['ref_cycles']]
    return R, Rl, M, T
normal = lambda r: 40 <= r['ref_peak'] <= 75
reduced = lambda r: r['ref_peak'] < 35
excl = [r for r in rows if not (r['ref_peak'] <= 75) or np.isnan(r['ref_peak'])]
res = {}
for gname, sel in (('normal-like passes', normal), ('stiff/shuffle passes', reduced)):
    R, Rl, M, T = group(sel)
    res[gname] = dict(ref=np.array(R), refL=np.array(Rl['L']), refR=np.array(Rl['R']), modL=np.array(M['L']), modR=np.array(M['R']), stride=np.array(T))
    print(gname, 'passes', sum(sel(r) for r in rows), 'ref cycles', len(R), 'model cycles L/R', len(M['L']), len(M['R']), 'stride median %.2f s (IQR %.2f-%.2f)' % (np.median(T), *np.percentile(T, [25, 75])))
print('excluded passes (reference peak > 75 deg or missing):', [(r['walk'], r['pass_']) for r in excl])
def metrics(curves):
    c = np.array(curves); m = c.mean(0)
    return dict(n=len(c), peak=float(np.mean(c.max(1))), peak_sd=float(np.std(c.max(1), ddof=1)), minimum=float(np.mean(c.min(1))), rom=float(np.mean(c.max(1) - c.min(1))))
def vs(curve_mean, key):
    n = N[key]; m = rot(n['mean']); sd = np.asarray(n['sd'])[:100]; k = int(np.argmax(np.asarray(n['mean'])[:100])); sd = np.r_[sd[k:], sd[:k]]; sd = np.r_[sd, sd[0]]
    d = curve_mean - m
    return dict(rmse=float(np.sqrt(np.mean(d ** 2))), r=float(np.corrcoef(curve_mean, m)[0, 1]), within2sd=float(np.mean(np.abs(d) <= 2 * sd) * 100), peak_diff=float(curve_mean.max() - m.max()))
NORMS = ['Moreira2021 0.56 m/s', 'Moreira2021 0.42 m/s', 'Moreira2021 0.69 m/s', 'Fukuchi2018 treadmill speed 2 (All)', 'Fukuchi2018 treadmill speed 1 (All)', 'Fukuchi2018 overground slow (All)', 'UCI Helwig2016 unbraced']
table = []
for gname in res:
    for lab, arr in (('iPad reference (both legs, near side)', res[gname]['ref']), ('Pi model left', res[gname]['modL']), ('Pi model right', res[gname]['modR'])):
        if len(arr) < 3: continue
        mt = metrics(arr); row = dict(group=gname, series=lab, **mt)
        for key in (NORMS if gname.startswith('normal') else ['UCI Helwig2016 knee brace on right knee right leg', 'Moreira2021 0.56 m/s']):
            row[key] = vs(arr.mean(0), key)
        table.append(row)
for key in NORMS + ['UCI Helwig2016 knee brace on right knee right leg']:
    m = np.array(N[key]['mean'])
    print('NORM %-50s n=%2d peak %.1f min %.1f ROM %.1f stride %s' % (key, N[key]['n'], m.max(), m.min(), m.max() - m.min(), N[key].get('stride_s')))
for t in table:
    print('\n%s | %s: n=%d cycles, peak %.1f +- %.1f, min %.1f, ROM %.1f' % (t['group'], t['series'], t['n'], t['peak'], t['peak_sd'], t['minimum'], t['rom']))
    for k, v in t.items():
        if isinstance(v, dict): print('    vs %-48s RMSE %5.1f  r %.2f  within 2SD %3.0f%%  peak diff %+5.1f' % (k, v['rmse'], v['r'], v['within2sd'], v['peak_diff']))
pickle.dump(dict(res=res, table=table, N=N, NORMS=NORMS), open('compare.pkl', 'wb'))
