import numpy as np, pandas as pd, glob, os, json
out = {}
info = pd.read_csv('fukuchi/WBDSinfo.csv')
age = {int(r.Subject): r.AgeGroup for r in info.itertuples()}
def spd(tag):
    v = []
    for r in info.itertuples():
        f = r.FileName
        if f.endswith('mkr.txt') and ('walk' + tag) in f.replace('O0', 'O').replace('O1', 'O'):
            try: v.append(float(r._12))
            except: pass
    return v
for tag, label in (('T01', 'treadmill speed 1'), ('T02', 'treadmill speed 2'), ('OS', 'overground slow'), ('OC', 'overground comfortable')):
    for grp in ('Young', 'Older', 'All'):
        C = []
        for f in sorted(glob.glob(f'fukuchi/WBDS*walk{tag}ang.txt')):
            sid = int(os.path.basename(f)[4:6])
            if grp != 'All' and age[sid] != grp: continue
            a = pd.read_csv(f, sep='\t')
            c = (a['RKneeAngleZ'].values + a['LKneeAngleZ'].values) / 2
            if np.isfinite(c).all(): C.append(c)
        C = np.array(C)
        out[f'Fukuchi2018 {label} ({grp})'] = dict(mean=C.mean(0).tolist(), sd=C.std(0, ddof=1).tolist(), n=len(C))
g = pd.read_csv('/mnt/user-data/uploads/Downloads/gait_check/datasets/UCI_multivariate_gait/gait.csv')
k = g[g.joint == 2]
for cond, label in ((1, 'unbraced'), (2, 'knee brace on right knee')):
    for leg in ((1, 2) if cond == 1 else (2,)):
        d = k[(k.condition == cond) & (k.leg == leg)]
        S = d.groupby(['subject', 'time']).angle.mean().unstack().values
        key = f'UCI Helwig2016 {label}' + ('' if cond == 1 and leg == 1 else (' right leg' if cond == 2 else ' (right)'))
        out[key] = dict(mean=S.mean(0).tolist(), sd=S.std(0, ddof=1).tolist(), n=len(S))
json.dump(out, open('norm_curves.json', 'w'))
for k_, v in out.items():
    m = np.array(v['mean'])
    print('%-55s n=%2d  peak %.1f at %d%%  loading %.1f  min %.1f  ROM %.1f' % (k_, v['n'], m.max(), m.argmax(), m[5:30].max(), m.min(), m.max() - m.min()))

# Moreira et al. 2021: 16 young adults, 7 controlled overground speeds
base = 'moreira/ASCII files'
SP = {'V1': 0.28, 'V15': 0.42, 'V2': 0.56, 'V25': 0.69, 'V3': 0.83, 'V35': 0.97, 'V4': 1.11}
for v, ms in SP.items():
    C, ST = [], []
    for p in sorted(os.listdir(base)):
        cs = []
        for leg in 'LR':
            for f in glob.glob(f'{base}/{p}/Processed_Data/{v}/{leg}/Angles/T*.txt'):
                a = pd.read_csv(f, sep='\t')
                for col in [c for c in a.columns if c.endswith('_Knee_X')]:
                    y = a[col].values
                    if np.isfinite(y).all(): cs.append(np.interp(np.linspace(0, 1, 101), np.linspace(0, 1, len(y)), y))
            for f in glob.glob(f'{base}/{p}/Processed_Data/{v}/{leg}/StrideTime/*.txt'):
                st = pd.read_csv(f, sep='\t').drop(columns=['Trial']).values.ravel()
                ST += [x for x in st if np.isfinite(x) and x > 0]
        if cs: C.append(np.mean(cs, 0))
    C = np.array(C)
    out[f'Moreira2021 {ms:.2f} m/s'] = dict(mean=C.mean(0).tolist(), sd=C.std(0, ddof=1).tolist(), n=len(C), stride_s=float(np.median(ST)) if ST else None)
json.dump(out, open('norm_curves.json', 'w'))
for k_ in [k for k in out if k.startswith('Moreira')]:
    v = out[k_]; m = np.array(v['mean'])
    print('%-30s n=%2d stride %.2f s peak %.1f at %d%% loading %.1f min %.1f ROM %.1f' % (k_, v['n'], v['stride_s'], m.max(), m.argmax(), m[5:30].max(), m.min(), m.max() - m.min()))
