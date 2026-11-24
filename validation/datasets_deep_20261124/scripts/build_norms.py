"""Per-subject healthy knee-flexion curves from every dataset we hold, all re-referenced the same way.

Output norms.pkl: {key: dict(dataset, condition, speed (m/s or None), n, subjects (n x 101, None if only mean/sd),
mean (101), sd (101), population, note)}. Curves start at heel strike as published; analyse.py rotates them."""
import glob, os, pickle, re
import numpy as np, pandas as pd

N = '/home/claude/work/norm'
OUT = {}
r101 = lambda y: np.interp(np.linspace(0, 1, 101), np.linspace(0, 1, len(y)), y)

# 1) Moreira et al. 2021: 16 young adults, 7 controlled overground speeds; per participant = mean of trials, both legs
base = f'{N}/moreira/ASCII files'
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
                    if np.isfinite(y).all():
                        cs.append(r101(y))
            for f in glob.glob(f'{base}/{p}/Processed_Data/{v}/{leg}/StrideTime/*.txt'):
                st = pd.read_csv(f, sep='\t').drop(columns=['Trial']).values.ravel()
                ST += [x for x in st if np.isfinite(x) and x > 0]
        if cs:
            C.append(np.mean(cs, 0))
    C = np.array(C)
    OUT[f'Moreira {ms:.2f}'] = dict(dataset='Moreira 2021', condition=f'{ms:.2f} m/s overground', speed=ms, subjects=C,
                                    n=len(C), stride=float(np.median(ST)), population='16 young adults')

# 2) Fukuchi et al. 2018: 42 young and older adults; treadmill T01-T08 and overground slow/comfortable/fast
info = pd.read_csv(f'{N}/fukuchi/WBDSinfo.csv')
age = {int(r.Subject): r.AgeGroup for r in info.itertuples()}
def cond_speed(tag):
    sp = []
    for r in info.itertuples():
        f = r.FileName
        if tag.startswith('T'):
            ok = f'walk{tag}' in f
        else:
            ok = re.search(r'walkO\d\d' + tag[1] + r'\.c3d$', f) is not None
        if ok:
            try:
                sp.append(float(r._12))
            except Exception:
                pass
    return float(np.mean(sp)) if sp else None
for tag in ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'OS', 'OC', 'OF']:
    for grp in ('All', 'Young', 'Older'):
        C = []
        for f in sorted(glob.glob(f'{N}/fukuchi/WBDS*walk{tag}ang.txt')):
            sid = int(os.path.basename(f)[4:6])
            if grp != 'All' and age[sid] != grp:
                continue
            a = pd.read_csv(f, sep='\t')
            c = (a['RKneeAngleZ'].values + a['LKneeAngleZ'].values) / 2
            if np.isfinite(c).all():
                C.append(r101(c))
        if len(C) >= 3:
            lab = {'OS': 'overground slow', 'OC': 'overground comfortable', 'OF': 'overground fast'}.get(tag, 'treadmill ' + tag)
            OUT[f'Fukuchi {tag} {grp}'] = dict(dataset='Fukuchi 2018', condition=lab, speed=cond_speed(tag), subjects=np.array(C),
                                               n=len(C), population={'All': '42 adults', 'Young': 'young adults',
                                                                     'Older': 'older adults'}[grp], group=grp)

# 3) Helwig et al. 2016 (UCI): 10 adults on a treadmill, unbraced and with a right-knee brace
g = pd.read_csv('/mnt/user-data/uploads/Downloads/gait_check/datasets/UCI_multivariate_gait/gait.csv')
k = g[g.joint == 2]
for cond, lab in ((1, 'unbraced'), (2, 'right knee braced')):
    d = k[(k.condition == cond) & (k.leg == (2 if cond == 2 else 1))] if cond == 2 else k[(k.condition == 1)]
    S = d.groupby(['subject', 'time']).angle.mean().unstack().values
    OUT[f'Helwig {lab}'] = dict(dataset='Helwig 2016 (UCI)', condition=lab + ' treadmill', speed=None, subjects=S, n=len(S),
                                population='10 adults')

# 4) Schwartz et al. 2008 and 5) Pinzone et al. 2014: published mean +- SD only (pyCGM2 open-access normative files)
P = '/home/claude/work/ds_ext/pycgm2/Data/normativeData/openAccessData'
def msd(path, cond):
    s = pd.read_excel(path, 'Descriptive Kinematics')
    kk = s[(s.Label == 'KneeAngles') & (s.Axis == 'X') & (s.Condition == cond)]
    m = kk[kk.StatType == 'mean'].iloc[0, 5:].values.astype(float)
    sd = kk[kk.StatType == 'sd'].iloc[0, 5:].values.astype(float)
    return r101(m), r101(sd)
for cond in ('verySlow', 'slow', 'free', 'fast', 'veryFast'):
    m, sd = msd(f'{P}/schwartz 2008/pyCGM2-Schwartz2008.xlsx', cond)
    OUT[f'Schwartz {cond}'] = dict(dataset='Schwartz 2008', condition=cond, speed=None, subjects=None, mean=m, sd=sd, n=83,
                                   population='83 typically developing children/adolescents')
for cond in ('centreOne', 'centreTwo'):
    m, sd = msd(f'{P}/Pinzone 2014/pyCGM2-Pinzone2014.xlsx', cond)
    OUT[f'Pinzone {cond}'] = dict(dataset='Pinzone 2014', condition=cond.replace('centre', 'centre ').replace('One', '1').replace('Two', '2'),
                                  speed=None, subjects=None, mean=m, sd=sd, n=None, population='healthy children and adolescents, 2 labs')

for k_, v in OUT.items():
    if v.get('subjects') is not None:
        v['mean'], v['sd'] = v['subjects'].mean(0), v['subjects'].std(0, ddof=1)
pickle.dump(OUT, open('/home/claude/work/deep/norms.pkl', 'wb'))
for k_, v in OUT.items():
    print('%-22s n=%-4s speed %-5s peak %5.1f min %5.1f %s' % (k_, v['n'], ('%.2f' % v['speed']) if v['speed'] else '-', v['mean'].max(), v['mean'].min(), v['population']))
