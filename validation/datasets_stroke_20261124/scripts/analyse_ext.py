"""v3.7.2: add Van Criekinge 2023 (138 able-bodied adults 21-86 y + 50 stroke survivors) and Lencioni 2019 (50 people,
trial-level walking speed) to the deep validation of our knee curves (iPad reference, Pi v3.7.0 left/right).

Inputs: deep/norms.pkl and deep/ours.pkl (v3.7.1), newds/raw_curves.pkl (extract.py), Fukuchi raw per-leg files.
Writes newds/fig/*.png, newds/fig_deck/*.png, newds/results_ext.json."""
import glob, json, os, pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D, N = '/home/claude/work/deep', '/home/claude/work/newds'
for d in ('fig', 'fig_deck'):
    os.makedirs(f'{N}/{d}', exist_ok=True)
NORM = pickle.load(open(f'{D}/norms.pkl', 'rb'))
OURS = pickle.load(open(f'{D}/ours.pkl', 'rb'))['rows']
RAW = pickle.load(open(f'{N}/raw_curves.pkl', 'rb'))
rng = np.random.default_rng(7)

BLUE, ORANGE, AQUA, GREY, DARK, LIGHT, PLUM = '#2a78d6', '#eb6834', '#1baf7a', '#8a8f98', '#1f2328', '#d9dbde', '#9b3f8f'
SER = {'ipad': ('iPad video reference', BLUE), 'piL': ('Pi model, left knee', ORANGE), 'piR': ('Pi model, right knee', AQUA)}
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9aa0a6',
                     'axes.labelcolor': '#3c4043', 'xtick.color': '#5f6368', 'ytick.color': '#5f6368', 'axes.titleweight': 'bold',
                     'axes.titlesize': 13, 'figure.dpi': 100, 'savefig.dpi': 200, 'axes.grid': True, 'grid.color': '#eceef0',
                     'grid.linewidth': 0.8, 'legend.frameon': False})
x = np.arange(101)
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def rot(c):
    """Start the cycle at the swing peak (searched in 45-95 % of a heel-strike cycle, robust for stiff stroke knees)."""
    c = np.asarray(c, float)[:100]
    k = 45 + int(np.argmax(c[45:95]))
    r = np.r_[c[k:], c[:k]]
    return np.r_[r, r[0]]


def feat(C):
    """Swing peak = maximum around the cycle start/end (cycles begin at the swing peak); stance minimum over 20-80 %."""
    C = np.atleast_2d(C)
    pk = C[:, np.r_[0:15, 86:101]].max(1)
    st = C[:, 20:80].min(1)
    return dict(peak=pk, stance=st, rom=pk - st)


def auc(a, b):  # P(a > b)
    a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return float(np.mean(a > b) + 0.5 * np.mean(a == b))


# ---------- references: v3.7.1 healthy set (peak-aligned the same way) ----------
REF = {}
for key, v in NORM.items():
    if v.get('subjects') is not None:
        S = np.array([rot(c) for c in v['subjects']])
        REF[key] = dict(v, S=S, m=S.mean(0), sd=S.std(0, ddof=1))
    else:
        k = 45 + int(np.argmax(v['mean'][45:95]))
        sd = np.asarray(v['sd'])[:100]; sd = np.r_[sd[k:], sd[:k]]; sd = np.r_[sd, sd[0]]
        REF[key] = dict(v, S=None, m=rot(v['mean']), sd=sd)

# ---------- new: Van Criekinge 2023 able-bodied ----------
VA = np.array([rot(c) for c in RAW['vc_able'].values()])
REF['VC able'] = dict(dataset='Van Criekinge 2023', condition='preferred speed, barefoot', speed=None, S=VA, m=VA.mean(0),
                      sd=VA.std(0, ddof=1), n=len(VA), population='138 able-bodied adults, 21-86 y')

# ---------- new: Lencioni 2019, per-subject means in walking-speed bins (adults >= 18 y) ----------
L = pd.DataFrame([{k: v for k, v in r.items() if k != 'knee'} for r in RAW['len']])
LK = np.array([rot(r['knee']) for r in RAW['len']])
BINS = [(0.25, 0.8, 'Lencioni <0.8'), (0.8, 1.0, 'Lencioni 0.8-1.0'), (1.0, 1.2, 'Lencioni 1.0-1.2'),
        (1.2, 1.4, 'Lencioni 1.2-1.4'), (1.4, 2.4, 'Lencioni >1.4')]
LEN_SUBJ = []  # (subject, age, bin, speed, curve) for the speed trend
for lo, hi, key in BINS:
    S, sp = [], []
    for sid, g in L[(L.age >= 18) & (L.speed >= lo) & (L.speed < hi)].groupby('subject'):
        c = LK[g.index].mean(0); S.append(c); sp.append(g.speed.mean())
        LEN_SUBJ.append((sid, g.age.iloc[0], key, g.speed.mean(), c))
    S = np.array(S)
    REF[key] = dict(dataset='Lencioni 2019', condition='below 0.8 m/s' if lo < 0.5 else (f'above {lo:.1f} m/s' if hi > 2 else f'{lo:.1f}–{hi:.1f} m/s'), speed=float(np.mean(sp)), S=S, m=S.mean(0),
                    sd=S.std(0, ddof=1), n=len(S), population='adults 19-72 y')

# ---------- new: Van Criekinge 2023 stroke survivors ----------
P_lab = np.array([rot(RAW['vc_stroke_P'][k]) for k in sorted(RAW['vc_stroke_P'])])
N_lab = np.array([rot(RAW['vc_stroke_N'][k]) for k in sorted(RAW['vc_stroke_N'])])
# Label check: clinically the paretic knee bends LESS in swing. In this release the "Pside" leg bends more in 40/50.
lab_check = dict(pside_peak_higher=int(np.sum(P_lab.max(1) > N_lab.max(1))), n=len(P_lab),
                 pside_peak=float(P_lab.max(1).mean()), nside_peak=float(N_lab.max(1).mean()))
# Label-free definition used for all stroke analyses: per person, the leg with the lower swing peak = more-affected leg.
more = np.where((P_lab.max(1) < N_lab.max(1))[:, None], P_lab, N_lab)
less = np.where((P_lab.max(1) < N_lab.max(1))[:, None], N_lab, P_lab)
REF['VC stroke more'] = dict(dataset='Van Criekinge 2023', condition='stroke, more-affected leg', speed=None, S=more, m=more.mean(0),
                             sd=more.std(0, ddof=1), n=len(more), population='50 stroke survivors, 19-85 y, 14-139 days after stroke')
REF['VC stroke less'] = dict(dataset='Van Criekinge 2023', condition='stroke, less-affected leg', speed=None, S=less, m=less.mean(0),
                             sd=less.std(0, ddof=1), n=len(less), population='50 stroke survivors')

# ---------- our cycles ----------
normal = lambda r: 40 <= r['ref_peak'] <= 75
stiff = lambda r: r['ref_peak'] < 35


def ours(sel):
    out = {'ipad': [(c, r['walk'] + str(r['pass_'])) for r in OURS if sel(r) for c, _ in r['ref_cycles']],
           'piL': [(c, r['walk'] + str(r['pass_'])) for r in OURS if sel(r) and r['leg'] == 'L' for c, _ in r['model_cycles']],
           'piR': [(c, r['walk'] + str(r['pass_'])) for r in OURS if sel(r) and r['leg'] == 'R' for c, _ in r['model_cycles']]}
    return {k: (np.array([c for c, _ in v]), np.array([p for _, p in v])) for k, v in out.items()}


ON, OS = ours(normal), ours(stiff)
OM = {s: ON[s][0].mean(0) for s in SER}
stride = float(np.median([T_ for r in OURS if normal(r) for _, T_ in r['ref_cycles']]))
ms = sorted([(v['speed'], v['stride']) for k, v in NORM.items() if k.startswith('Moreira')])
our_speed = float(np.interp(-stride, [-s for _, s in ms], [sp for sp, _ in ms]))
R = {'our_speed_m_s': our_speed, 'stride_s': stride, 'label_check': lab_check,
     'refs': {k: dict(n=int(REF[k]['n']), speed=REF[k]['speed'], peak=float(feat(REF[k]['S'])['peak'].mean()),
                      stance=float(feat(REF[k]['S'])['stance'].mean()), rom=float(feat(REF[k]['S'])['rom'].mean()))
              for k in ['VC able', 'VC stroke more', 'VC stroke less'] + [b[2] for b in BINS]}}

# ---------- A: curve agreement (all healthy conditions, old + new) ----------
OLD = ['Moreira 0.28', 'Moreira 0.42', 'Moreira 0.56', 'Moreira 0.69', 'Moreira 0.83', 'Moreira 0.97', 'Moreira 1.11',
       'Fukuchi T01 All', 'Fukuchi T02 All', 'Fukuchi T03 All', 'Fukuchi T04 All', 'Fukuchi OS All', 'Fukuchi OC All',
       'Fukuchi T01 Older', 'Fukuchi OS Older', 'Helwig unbraced', 'Schwartz verySlow', 'Schwartz slow', 'Schwartz free',
       'Pinzone centreOne', 'Pinzone centreTwo']
NEW = ['VC able'] + [b[2] for b in BINS]
COND = OLD + NEW
LAB = {}
for k in COND:
    v = REF[k]
    LAB[k] = f"{v['dataset'].split(' (')[0]} · " + (f"{v['speed']:.2f} m/s" if v['speed'] else v['condition'])
LAB.update({'Fukuchi OS All': 'Fukuchi 2018 · overground slow', 'Fukuchi OC All': 'Fukuchi 2018 · overground comfortable',
            'Fukuchi T01 Older': 'Fukuchi 2018 · 0.49 m/s · older', 'Fukuchi OS Older': 'Fukuchi 2018 · overground slow · older',
            'Helwig unbraced': 'Helwig 2016 · treadmill comfortable', 'Schwartz verySlow': 'Schwartz 2008 · very slow',
            'Schwartz slow': 'Schwartz 2008 · slow', 'Schwartz free': 'Schwartz 2008 · self-selected',
            'Pinzone centreOne': 'Pinzone 2014 · lab 1', 'Pinzone centreTwo': 'Pinzone 2014 · lab 2',
            'VC able': 'Van Criekinge 2023 · preferred, 138 adults'})
for lo, hi, key in BINS:
    LAB[key] = f"Lencioni 2019 · {REF[key]['speed']:.2f} m/s (n={REF[key]['n']})"
A = {s: {k: dict(rmse=rmse(OM[s], REF[k]['m']), r=float(np.corrcoef(OM[s], REF[k]['m'])[0, 1]),
                 band=float(np.mean(np.abs(OM[s] - REF[k]['m']) <= 2 * REF[k]['sd']) * 100),
                 offset=float(np.mean(OM[s] - REF[k]['m']))) for k in COND + ['VC stroke more', 'VC stroke less']} for s in SER}
R['A'] = {s: {k: A[s][k] for k in NEW + ['VC stroke more', 'VC stroke less']} for s in SER}

# ---------- B1: person-to-group tolerance, B2: lab-to-lab ----------
B1 = {}
for k in ('Moreira 0.56', 'Fukuchi T01 All', 'VC able', 'Lencioni <0.8', 'Lencioni 0.8-1.0'):
    S = REF[k]['S']
    B1[k] = np.array([rmse(S[i], np.delete(S, i, 0).mean(0)) for i in range(len(S))])
R['B1'] = {k: dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)), n=len(v)) for k, v in B1.items()}
R['percentile'] = {s: {k: float(np.mean(B1[k] <= A[s][k]['rmse']) * 100) for k in B1} for s in SER}
R['rmse_vs_group'] = {s: {k: A[s][k]['rmse'] for k in B1} for s in SER}
SLOW = ['Moreira 0.42', 'Moreira 0.56', 'Moreira 0.69', 'Fukuchi T01 All', 'Fukuchi T02 All', 'Fukuchi OS All',
        'Schwartz verySlow', 'Schwartz slow', 'Pinzone centreOne', 'Pinzone centreTwo', 'Helwig unbraced',
        'Lencioni <0.8', 'Lencioni 0.8-1.0', 'VC able']
B2 = []
for i, a in enumerate(SLOW):
    for b in SLOW[i + 1:]:
        if REF[a]['dataset'] != REF[b]['dataset'] or a.startswith('Pinzone'):
            B2.append((a, b, rmse(REF[a]['m'], REF[b]['m'])))
B2v = np.array([v for *_, v in B2])
R['B2'] = dict(n_pairs=len(B2), median=float(np.median(B2v)), p90=float(np.percentile(B2v, 90)),
               with_new=[(a, b, round(v, 2)) for a, b, v in B2 if a in NEW or b in NEW])

# ---------- C: speed trend from Lencioni (person-level, adults) and expected healthy values at our pace ----------
LS = pd.DataFrame([dict(subject=s, age=a, bin=b, speed=v, **{f: float(feat(c[None])[f][0]) for f in ('peak', 'stance', 'rom')})
                   for s, a, b, v, c in LEN_SUBJ])
C = {}
for f in ('peak', 'stance', 'rom'):
    X = np.c_[np.ones(len(LS)), LS.speed]; y = LS[f].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta; s2 = res @ res / (len(y) - 2); XtXi = np.linalg.inv(X.T @ X)
    x0 = np.array([1, our_speed]); pred = float(x0 @ beta)
    pi = 1.96 * np.sqrt(s2 * (1 + x0 @ XtXi @ x0))
    C[f] = dict(slope=float(beta[1]), intercept=float(beta[0]), expected=pred, pi95=float(pi), n_points=len(y),
                n_subjects=int(LS.subject.nunique()), min_speed=float(LS.speed.min()),
                ours={s: float(feat(ON[s][0])[f].mean()) for s in SER},
                ours_z={s: float((feat(ON[s][0])[f].mean() - pred) / np.sqrt(s2)) for s in SER})
R['C_speed_trend'] = C

# ---------- EQ: our features vs Lencioni slow adults (matched-ish speed) and VC able ----------
def boot_ci(vals, keys, n=2000):
    u = np.unique(keys); b = []
    for _ in range(n):
        pick = rng.choice(u, len(u)); b.append(np.mean(np.concatenate([vals[keys == p] for p in pick])))
    return np.percentile(b, [5, 95])
EQ = {}
for k in ('Lencioni <0.8', 'VC able'):
    hf = feat(REF[k]['S'])
    for s in SER:
        for f in ('peak', 'stance', 'rom'):
            v = feat(ON[s][0])[f]; lo, hi = boot_ci(v, ON[s][1]); se = hf[f].std(ddof=1) / np.sqrt(len(hf[f]))
            EQ[f'{k}|{s}|{f}'] = dict(diff=float(v.mean() - hf[f].mean()), lo=float(lo - hf[f].mean() - 1.645 * se),
                                      hi=float(hi - hf[f].mean() + 1.645 * se), ours=float(v.mean()), healthy=float(hf[f].mean()),
                                      sd=float(hf[f].std(ddof=1)))
R['EQ'] = EQ

# ---------- S: stroke ----------
hp = feat(VA)['peak']; sp_more = feat(more)['peak']; sp_less = feat(less)['peak']
cut = float(np.percentile(hp, 5))
S_ = dict(healthy_peak=float(hp.mean()), healthy_sd=float(hp.std(ddof=1)), more_peak=float(sp_more.mean()), more_sd=float(sp_more.std(ddof=1)),
          less_peak=float(sp_less.mean()), cut_p5=cut, frac_more_below_cut=float(np.mean(sp_more < cut) * 100),
          frac_less_below_cut=float(np.mean(sp_less < cut) * 100), auc_more=auc(hp, sp_more), auc_less=auc(hp, sp_less),
          more_stance=float(feat(more)['stance'].mean()), healthy_stance=float(feat(VA)['stance'].mean()),
          more_rom=float(feat(more)['rom'].mean()), healthy_rom=float(feat(VA)['rom'].mean()))
# our normal / stiff walking against the healthy cut-off (pass level: what the walker would report per pass)
def pass_peaks(sel, series):
    out = []
    for r in OURS:
        if not sel(r):
            continue
        if series == 'ipad' and r['ref_cycles']:
            out.append(np.mean([c.max() for c, _ in r['ref_cycles']]))
        if series == 'pi' + r['leg'] and r['model_cycles']:
            out.append(np.mean([c.max() for c, _ in r['model_cycles']]))
    return np.array(out)
PP = {s: dict(normal=pass_peaks(normal, s), stiff=pass_peaks(stiff, s)) for s in SER}
# label-based and fairer variants of the detection AUC (the lower-peak choice flatters the stroke side a little)
S_['auc_pside_label'] = auc(hp, P_lab.max(1)); S_['auc_nside_label'] = auc(hp, N_lab.max(1))
S_['ours_vs_cut'] = {s: dict(normal_above=float(np.mean(PP[s]['normal'] >= cut) * 100), stiff_below=float(np.mean(PP[s]['stiff'] < cut) * 100),
                             n=(len(PP[s]['normal']), len(PP[s]['stiff'])), normal_mean=float(PP[s]['normal'].mean()),
                             stiff_mean=float(PP[s]['stiff'].mean())) for s in SER}
S_['rmse_stiff_vs_more'] = {s: rmse(OS[s][0].mean(0), more.mean(0)) for s in SER}
S_['rmse_normal_vs_less'] = {s: rmse(OM[s], less.mean(0)) for s in SER}
S_['rmse_normal_vs_vcable'] = {s: rmse(OM[s], VA.mean(0)) for s in SER}
# measurement error of the Pi swing peak per pass, against the iPad reference
def pass_err(leg):
    e = [np.mean([c.max() for c, _ in r['model_cycles']]) - np.mean([c.max() for c, _ in r['ref_cycles']])
         for r in OURS if (normal(r) or stiff(r)) and r['leg'] == leg and r['ref_cycles'] and r['model_cycles']]
    return np.array(e)
EL, ER = pass_err('L'), pass_err('R')
S_['pi_pass_error'] = dict(L=dict(bias=float(EL.mean()), sd=float(EL.std(ddof=1)), n=len(EL)), R=dict(bias=float(ER.mean()), sd=float(ER.std(ddof=1)), n=len(ER)))
# detection of the more-affected leg vs healthy when our measurement noise is added (Monte Carlo)
NOISE = np.arange(0, 15.5, 0.5)
def detect(noise, reps=200):
    a_, s_ = [], []
    for _ in range(reps):
        h = hp + rng.normal(0, noise, len(hp)); m = sp_more + rng.normal(0, noise, len(sp_more))
        a_.append(auc(h, m)); c5 = np.percentile(h, 5); s_.append(np.mean(m < c5) * 100)
    return float(np.mean(a_)), float(np.mean(s_))
DET = np.array([detect(n) for n in NOISE])
S_['detect_curve'] = dict(noise=NOISE.tolist(), auc=DET[:, 0].tolist(), sens_at_95spec=DET[:, 1].tolist())
# left-right asymmetry: healthy per-leg (Fukuchi raw files, overground slow + treadmill 0.49 m/s) vs stroke
asym_h = []
for tag in ('OS', 'T01'):
    for f in sorted(glob.glob(f'/home/claude/work/norm/fukuchi/WBDS*walk{tag}ang.txt')):
        a = pd.read_csv(f, sep='\t')
        if np.isfinite(a['RKneeAngleZ']).all() and np.isfinite(a['LKneeAngleZ']).all():
            asym_h.append((tag, abs(a['RKneeAngleZ'].max() - a['LKneeAngleZ'].max())))
AH = np.array([v for t, v in asym_h if t == 'OS']); AH1 = np.array([v for t, v in asym_h if t == 'T01'])
AS = np.abs(P_lab.max(1) - N_lab.max(1))
S_['asym'] = dict(healthy_OS_median=float(np.median(AH)), healthy_OS_p95=float(np.percentile(AH, 95)), n_OS=len(AH),
                  healthy_T01_median=float(np.median(AH1)), healthy_T01_p95=float(np.percentile(AH1, 95)),
                  stroke_median=float(np.median(AS)), stroke_frac_above_healthy_p95=float(np.mean(AS > np.percentile(AH, 95)) * 100),
                  auc=auc(AS, AH), our_LR_bias=float(EL.mean() - ER.mean()))
AH_draw = rng.choice(AH, (500, len(hp)))
S_['auc_more_fair'] = float(np.mean([auc(hp - d / 2, sp_more) for d in AH_draw[:200]]))
cut_speed = C['peak']['expected'] - C['peak']['pi95']
S_['cut_speed_matched'] = float(cut_speed)
S_['ours_vs_speed_cut'] = {s: dict(normal_above=float(np.mean(PP[s]['normal'] >= cut_speed) * 100),
                                   stiff_below=float(np.mean(PP[s]['stiff'] < cut_speed) * 100)) for s in SER}
S_['more_below_speed_cut'] = float(np.mean(sp_more < cut_speed) * 100)
S_['more_peak_range'] = [float(sp_more.min()), float(sp_more.max())]
S_['normal_passes_below_speed_cut'] = {s: sorted(round(float(v), 1) for v in PP[s]['normal'] if v < cut_speed) for s in SER}
R['S'] = S_
# RMSE ranges by walking-speed group of the healthy condition (old + new)
SPD = {k: (REF[k]['speed'] if REF[k]['speed'] else {'Schwartz verySlow': 0.6, 'Schwartz slow': 0.9, 'Helwig unbraced': 1.1,
       'Pinzone centreOne': 1.2, 'Pinzone centreTwo': 1.2, 'Schwartz free': 1.25, 'VC able': 1.2}[k]) for k in COND}
R['rmse_by_speed'] = {g: {s: [round(min(A[s][k]['rmse'] for k in COND if lo <= SPD[k] < hi), 1),
                              round(max(A[s][k]['rmse'] for k in COND if lo <= SPD[k] < hi), 1)] for s in SER}
                      for g, lo, hi in (('<=0.7', 0, 0.71), ('0.7-1.0', 0.71, 1.0), ('>=1.0', 1.0, 3))}
R['r_range'] = {s: [round(min(A[s][k]['r'] for k in COND), 3), round(max(A[s][k]['r'] for k in COND), 3),
                    round(min(A[s][k]['r'] for k in COND if SPD[k] < 1.4), 3)] for s in SER}
json.dump(R, open(f'{N}/results_ext.json', 'w'), indent=1, default=float)


# ====================== FIGURES ======================
def save(fig, name, keep_texts=False):
    fig.savefig(f'{N}/fig/{name}.png', bbox_inches='tight', facecolor='white')
    if fig._suptitle is not None:
        fig._suptitle.set_visible(False)
    axs_ = [a for a in fig.axes if a.get_label() != '<colorbar>']
    if len(axs_) == 1:
        axs_[0].set_title('')
    if not keep_texts:
        for t in fig.texts:
            if t.get_position()[1] > 0.85:
                t.set_visible(False)
    fig.savefig(f'{N}/fig_deck/{name}.png', bbox_inches='tight', facecolor='white', dpi=160); plt.close(fig)


# F9: our curves vs the new healthy references
SM = ['VC able', 'Lencioni <0.8', 'Lencioni 0.8-1.0']
fig, axs = plt.subplots(1, 3, figsize=(14, 4.9), sharey=True)
for ax, k in zip(axs, SM):
    m, sd = REF[k]['m'], REF[k]['sd']
    ax.fill_between(x, m - 2 * sd, m + 2 * sd, color=LIGHT, lw=0, label='healthy ±2 SD')
    ax.plot(x, m, color=GREY, lw=1.6, ls='--', label='healthy mean')
    for s, (lab, col) in SER.items():
        ax.plot(x, OM[s], color=col, lw=2, label=lab)
    ttl = (f"Van Criekinge 2023 · {REF[k]['n']} adults 21–86 y\npreferred speed" if k == 'VC able' else
           f"Lencioni 2019 · {REF[k]['n']} adults\n{REF[k]['condition']} (mean {REF[k]['speed']:.2f} m/s)")
    ax.set_title(ttl, fontsize=11.5, loc='left')
    ax.text(0.98, 0.97, 'iPad / Pi L / Pi R\nRMSE ' + ' / '.join(f"{A[s][k]['rmse']:.1f}" for s in SER) + '°\nin ±2 SD band ' +
            ' / '.join(f"{A[s][k]['band']:.0f}" for s in SER) + ' %', transform=ax.transAxes, ha='right', va='top', fontsize=9.5, color='#3c4043')
    ax.set_ylim(-12, 80); ax.set_xlabel('% gait cycle (0 = swing peak)')
axs[0].set_ylabel('knee flexion (°)')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.06))
fig.suptitle('Against the two new datasets: same shape, same gaps (stance more bent, swing peak lower)', x=0.01, y=1.02, ha='left', fontsize=15, fontweight='bold')
fig.tight_layout(rect=(0, 0.04, 1, 0.97)); save(fig, 'f9_new_references')

# F10: RMSE heatmap over all 27 healthy conditions (7 datasets)
order = sorted(COND, key=lambda k: (REF[k]['speed'] if REF[k]['speed'] else {'Schwartz verySlow': 0.6, 'Schwartz slow': 0.9,
              'Helwig unbraced': 1.1, 'Pinzone centreOne': 1.2, 'Pinzone centreTwo': 1.2, 'Schwartz free': 1.25, 'VC able': 1.2}[k]))
M = np.array([[A[s][k]['rmse'] for k in order] for s in SER])
fig, ax = plt.subplots(figsize=(14, 4.6))
im = ax.imshow(M, cmap='Blues', vmin=3, vmax=14, aspect='auto')
ax.set_yticks(range(3)); ax.set_yticklabels([SER[s][0] for s in SER])
ax.set_xticks(range(len(order)))
ax.set_xticklabels([LAB[k] for k in order], rotation=60, ha='right', fontsize=8.5)
for t, k in zip(ax.get_xticklabels(), order):
    if k in NEW:
        t.set_fontweight('bold'); t.set_color('#1d5fae')
for i in range(3):
    for j in range(len(order)):
        ax.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=8, color='white' if M[i, j] > 9 else DARK)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.01); cb.set_label('RMSE (°)')
ax.set_title('Sorted by walking speed: closest at slow speeds in every dataset, including the two new ones (bold)', loc='left')
save(fig, 'f10_rmse_heatmap_all')

# F11: tolerance with the larger groups
fig, ax = plt.subplots(figsize=(13.5, 5.6))
groups = [('Moreira 0.56', 'Moreira 2021\n0.56 m/s (16)'), ('Fukuchi T01 All', 'Fukuchi 2018\n0.49 m/s (42)'),
          ('Lencioni <0.8', f"Lencioni 2019\n<0.8 m/s ({REF['Lencioni <0.8']['n']})"),
          ('Lencioni 0.8-1.0', f"Lencioni 2019\n0.8–1.0 m/s ({REF['Lencioni 0.8-1.0']['n']})"),
          ('VC able', 'Van Criekinge 2023\npreferred (138)')]
for i, (k, lab) in enumerate(groups):
    v = B1[k]
    ax.boxplot(v, positions=[i], widths=0.5, showfliers=False, patch_artist=True, boxprops=dict(facecolor=LIGHT, color=GREY),
               medianprops=dict(color=DARK, lw=2), whiskerprops=dict(color=GREY), capprops=dict(color=GREY))
    ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=8, color=GREY, alpha=0.5, zorder=3)
    for j, s in enumerate(SER):
        ax.scatter(i + 0.42 + (j - 1) * 0.09, A[s][k]['rmse'], s=60, color=SER[s][1], zorder=4, edgecolor='white',
                   label=SER[s][0] if i == 0 else None)
    ax.text(i, -0.25, ' / '.join(f"{R['percentile'][s][k]:.0f}" for s in SER), ha='center', va='top', fontsize=9, color='#3c4043',
            transform=ax.get_xaxis_transform())
i = len(groups)
ax.boxplot(B2v, positions=[i], widths=0.5, showfliers=False, patch_artist=True, boxprops=dict(facecolor=LIGHT, color=GREY),
           medianprops=dict(color=DARK, lw=2), whiskerprops=dict(color=GREY), capprops=dict(color=GREY))
ax.scatter(i + rng.uniform(-0.17, 0.17, len(B2v)), B2v, s=8, color=GREY, alpha=0.5, zorder=3)
ax.set_xticks(range(len(groups) + 1)); ax.set_xticklabels([g[1] for g in groups] + [f'lab vs lab\n{len(B2v)} pairs of means'], fontsize=9.5)
ax.set_ylabel('curve difference, RMSE (°)'); ax.set_ylim(0, 16)
ax.legend(loc='upper left', fontsize=9.5, ncol=3)
ax.text(-0.5, -0.25, 'our percentile among\nhealthy people\n(iPad / Pi L / Pi R)', ha='right', va='top', fontsize=8.5, color='#5f6368', transform=ax.get_xaxis_transform())
ax.set_title('Grey: each healthy person vs their own group. Dots: our mean curve vs the same group mean', loc='left')
fig.suptitle('With 7 datasets the picture holds: iPad inside healthy spread, Pi at its upper edge', x=0.01, y=1.0, ha='left', fontsize=15, fontweight='bold')
fig.tight_layout(rect=(0, 0.02, 1, 0.96)); save(fig, 'f11_tolerance_7_datasets')

# F12: speed trend (Lencioni person-level) with expected values at our pace
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2))
for ax, f, ttl in ((axs[0], 'peak', 'Swing-phase peak flexion'), (axs[1], 'stance', 'Mid-stance minimum flexion')):
    ax.scatter(LS.speed, LS[f], s=16, color=GREY, alpha=0.55, label='Lencioni 2019, one adult at one speed')
    mo = [(REF[k]['speed'], feat(REF[k]['S'])[f]) for k in REF if k.startswith('Moreira')]
    for sp, v in mo:
        ax.errorbar(sp, v.mean(), yerr=v.std(ddof=1), fmt='o', color=DARK, mfc='white', capsize=2, ms=5,
                    label='Moreira 2021 group mean ± SD' if sp == mo[0][0] else None)
    xs = np.linspace(0.25, 2.3, 50); c = C[f]
    yl = c['intercept'] + c['slope'] * xs
    ax.plot(xs, yl, color=DARK, lw=1.4, label='Lencioni trend')
    ax.fill_between(xs, yl - c['pi95'], yl + c['pi95'], color=LIGHT, alpha=0.6, lw=0, label='95% range for one healthy person')
    for j, s in enumerate(SER):
        v = feat(ON[s][0])[f]
        ax.errorbar(our_speed + (j - 1) * 0.03, v.mean(), yerr=v.std(ddof=1), fmt='D', color=SER[s][1], ms=8, capsize=3, label=SER[s][0])
    ax.axvline(our_speed, color='#a04a1c', lw=1, ls=':')
    ax.set_xlabel('walking speed (m/s)'); ax.set_ylabel('knee flexion (°)'); ax.set_title(ttl, loc='left')
axs[0].text(our_speed + 0.04, 30, f'our pace ≈ {our_speed:.2f} m/s', fontsize=9, color='#a04a1c')
axs[0].set_ylim(25, 85); axs[1].set_ylim(-10, 30)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=9.5)
fig.suptitle(f"Expected for a healthy adult at our pace: swing peak {C['peak']['expected']:.0f}° ± {C['peak']['pi95']:.0f}°, "
             f"stance {C['stance']['expected']:.0f}° ± {C['stance']['pi95']:.0f}°", x=0.01, y=1.0, ha='left', fontsize=15, fontweight='bold')
fig.tight_layout(rect=(0, 0.04, 1, 0.95)); save(fig, 'f12_speed_trend_lencioni')

# F13: stroke curves vs healthy vs ours
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
for ax, title, src, sub in ((axs[0], 'Our normal walking', ON, 'normal'), (axs[1], 'Our stiff-knee walking', OS, 'stiff')):
    ax.fill_between(x, VA.mean(0) - 2 * VA.std(0, ddof=1), VA.mean(0) + 2 * VA.std(0, ddof=1), color=LIGHT, lw=0, label='healthy ±2 SD (138 adults)')
    ax.plot(x, VA.mean(0), color=GREY, ls='--', lw=1.5, label='healthy mean')
    ax.fill_between(x, more.mean(0) - more.std(0, ddof=1), more.mean(0) + more.std(0, ddof=1), color=PLUM, alpha=0.13, lw=0,
                    label='stroke, more-affected leg ±1 SD')
    ax.plot(x, more.mean(0), color=PLUM, lw=2.2, label='stroke, more-affected leg (50)')
    ax.plot(x, less.mean(0), color=PLUM, lw=1.6, ls=(0, (4, 2)), label='stroke, less-affected leg')
    for s in SER:
        ax.plot(x, src[s][0].mean(0), color=SER[s][1], lw=2, label=SER[s][0])
    ax.set_title(title, loc='left'); ax.set_xlabel('% gait cycle (0 = swing peak)'); ax.set_ylim(-8, 75)
axs[0].set_ylabel('knee flexion (°)')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.12), fontsize=9.5)
fig.suptitle('Group-average curves cannot separate them: our slow walker-assisted walking overlays the stroke average', x=0.01, y=1.0, ha='left', fontsize=15, fontweight='bold')
fig.tight_layout(rect=(0, 0.05, 1, 0.95)); save(fig, 'f13_stroke_curves')

# F14: per-person map: swing peak vs range of motion
fig, ax = plt.subplots(figsize=(11.5, 6))
fh, fm, fl = feat(VA), feat(more), feat(less)
ax.scatter(fh['peak'], fh['rom'], s=22, color=GREY, alpha=0.55, label='healthy adult (Van Criekinge, 138)')
ax.scatter(fm['peak'], fm['rom'], s=34, color=PLUM, alpha=0.85, label='stroke, more-affected leg (50)')
ax.scatter(fl['peak'], fl['rom'], s=34, facecolor='none', edgecolor=PLUM, alpha=0.8, label='stroke, less-affected leg')
for s in SER:
    for sub, src, mk in (('normal', ON, 'o'), ('stiff', OS, 'D')):
        F = feat(src[s][0]); keys = src[s][1]
        pk = [F['peak'][keys == p].mean() for p in np.unique(keys)]; rm = [F['rom'][keys == p].mean() for p in np.unique(keys)]
        ax.scatter(pk, rm, s=70, marker=mk, color=SER[s][1], edgecolor='white', lw=1, zorder=4,
                   label=f"{SER[s][0]} · {'normal' if sub == 'normal' else 'stiff-knee'} pass")
ax.axvline(cut, color=DARK, lw=1, ls=':')
ax.text(cut + 0.5, 4, f'healthy 5th percentile,\npreferred speed: {cut:.0f}°', ha='left', va='bottom', fontsize=9, color=DARK)
ax.axvline(cut_speed, color='#a04a1c', lw=1, ls='--')
ax.text(cut_speed - 0.5, 4, f'healthy lower limit\nat our 0.55 m/s: {cut_speed:.0f}°', ha='right', va='bottom', fontsize=9, color='#a04a1c')
ax.set_xlabel('swing peak flexion (°)'); ax.set_ylabel('knee range of motion (°)'); ax.set_xlim(0, 80); ax.set_ylim(0, 70)
ax.legend(loc='upper left', fontsize=8.5, ncol=1, bbox_to_anchor=(1.01, 1))
ax.set_title('Each dot is one person (datasets) or one walking pass (ours)', loc='left')
fig.suptitle(f"{S_['frac_more_below_cut']:.0f}% of stroke-affected knees sit below the healthy limit, as do all our stiff passes and some normal ones",
             x=0.01, y=1.0, ha='left', fontsize=14.5, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, 'f14_stroke_map')

# F15: detection vs measurement noise, and left-right asymmetry
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axs[0]
ax.plot(NOISE, DET[:, 0] * 100, color=DARK, lw=2, label='AUC, healthy vs stroke more-affected (%)')
ax.plot(NOISE, DET[:, 1], color=PLUM, lw=2, ls=(0, (5, 2)), label='stroke legs caught at 95% specificity (%)')
for leg, e, col in (('Pi left', EL, ORANGE), ('Pi right', ER, AQUA)):
    sd = e.std(ddof=1); ax.axvline(sd, color=col, lw=1.6)
    ax.text(sd + 0.15 if leg == 'Pi right' else sd - 0.15, 97, f'{leg}\nerror SD {sd:.1f}°', color=col, fontsize=9,
            ha='left' if leg == 'Pi right' else 'right', va='top')
ax.set_xlabel('measurement error added to every knee (SD, °)'); ax.set_ylabel('%'); ax.set_ylim(0, 100)
ax.set_title('Effect of our measurement error on detection', loc='left'); ax.legend(loc='lower left', fontsize=9)
ax = axs[1]
bins_ = np.arange(0, 42, 2)
ax.hist(AH, bins=bins_, color=GREY, alpha=0.6, label=f'healthy, |left − right| (Fukuchi, {len(AH)})', weights=np.full(len(AH), 100 / len(AH)))
ax.hist(AS, bins=bins_, color=PLUM, alpha=0.55, label='stroke, |left − right| (50)', weights=np.full(len(AS), 100 / len(AS)))
ax.axvline(S_['asym']['healthy_OS_p95'], color=DARK, lw=1, ls=':')
ax.text(S_['asym']['healthy_OS_p95'] + 0.4, ax.get_ylim()[1] * 0.6, f"healthy 95th\npercentile {S_['asym']['healthy_OS_p95']:.1f}°", fontsize=9, color=DARK)
b = abs(S_['asym']['our_LR_bias'])
ax.axvline(b, color=ORANGE, lw=2)
ax.annotate(f'our Pi left−right bias {b:.1f}°', xy=(b, 31), xytext=(9, 31), color='#a04a1c', fontsize=9, va='center',
            arrowprops=dict(arrowstyle='->', color='#a04a1c', lw=1))
ax.set_xlabel('difference in swing peak between legs (°)'); ax.set_ylabel('% of people')
ax.set_title('Leg-to-leg difference: a second stroke marker', loc='left'); ax.legend(fontsize=9, loc='center right')
lo_, hi_ = sorted([np.interp(e.std(ddof=1), NOISE, DET[:, 1]) for e in (EL, ER)])
fig.suptitle(f"Our error costs little in AUC ({S_['auc_more'] * 100:.0f}% → {np.interp(max(EL.std(ddof=1), ER.std(ddof=1)), NOISE, DET[:, 0]) * 100:.0f}%) "
             f"but more at a strict cut-off ({DET[0, 1]:.0f}% → {lo_:.0f}–{hi_:.0f}% of stroke knees caught)",
             x=0.01, y=1.0, ha='left', fontsize=15, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, 'f15_stroke_detection')

print(json.dumps({k: R[k] for k in ('our_speed_m_s', 'label_check', 'refs', 'B1', 'percentile', 'rmse_vs_group', 'C_speed_trend')}, indent=1, default=float))
print(json.dumps({k: v for k, v in R['S'].items() if k != 'detect_curve'}, indent=1, default=float))
print('B2', R['B2']['n_pairs'], R['B2']['median'], R['B2']['p90'])
for s in SER:
    print(s, {k: (round(A[s][k]['rmse'], 1), round(A[s][k]['r'], 3), round(A[s][k]['band'])) for k in NEW + ['VC stroke more', 'VC stroke less']})
for k, e in EQ.items():
    print(k, {a: round(b, 1) for a, b in e.items()})


# ====================== v3.7.2b: the remaining slides redone with all 7 datasets ======================
rng2 = np.random.default_rng(11)
DEEP = json.load(open(f'{D}/results.json'))
EQ4 = {}
for k in ('Fukuchi T01 All', 'Moreira 0.56'):
    for s in SER:
        for f in ('peak', 'stance', 'rom'):
            EQ4[(k, s, f)] = DEEP['EQ'][f'{k}|{s}|{f}']
for k in ('Lencioni <0.8', 'VC able'):
    for s in SER:
        for f in ('peak', 'stance', 'rom'):
            EQ4[(k, s, f)] = EQ[f'{k}|{s}|{f}']
SHORT = {'Fukuchi T01 All': 'Fukuchi 2018\n0.49 m/s (42)', 'Moreira 0.56': 'Moreira 2021\n0.56 m/s (16)',
         'Lencioni <0.8': f"Lencioni 2019\n<0.8 m/s ({REF['Lencioni <0.8']['n']})", 'VC able': 'Van Criekinge 2023\npreferred (138)'}

# F16: one panel per dataset (plus older adults)
SM8 = ['Moreira 0.56', 'Fukuchi T01 All', 'Fukuchi T01 Older', 'Helwig unbraced', 'Schwartz verySlow', 'Pinzone centreOne', 'VC able', 'Lencioni <0.8']
T8 = {'Moreira 0.56': 'Moreira 2021 · 0.56 m/s', 'Fukuchi T01 All': 'Fukuchi 2018 · 0.49 m/s', 'Fukuchi T01 Older': 'Fukuchi 2018 · 0.49 m/s, older',
      'Helwig unbraced': 'Helwig 2016 · treadmill', 'Schwartz verySlow': 'Schwartz 2008 · very slow', 'Pinzone centreOne': 'Pinzone 2014 · lab 1',
      'VC able': 'Van Criekinge 2023 · preferred', 'Lencioni <0.8': 'Lencioni 2019 · below 0.8 m/s'}
fig, axs = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
for ax, k in zip(axs.flat, SM8):
    m, sd = REF[k]['m'], REF[k]['sd']
    ax.fill_between(x, m - 2 * sd, m + 2 * sd, color=LIGHT, lw=0, label='healthy ±2 SD')
    ax.plot(x, m, color=GREY, lw=1.6, ls='--', label='healthy mean')
    for s, (lab, col) in SER.items():
        ax.plot(x, OM[s], color=col, lw=2, label=lab)
    ax.set_title(T8[k], fontsize=11.5, loc='left')
    ax.text(0.98, 0.96, ' / '.join(f"{A[s][k]['rmse']:.1f}" for s in SER) + '°', transform=ax.transAxes, ha='right', va='top', fontsize=10, color='#3c4043')
    ax.set_ylim(-12, 80)
for ax in axs[1]: ax.set_xlabel('% gait cycle (0 = swing peak)')
for ax in axs[:, 0]: ax.set_ylabel('knee flexion (°)')
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.04))
fig.suptitle('Same shape as healthy walking in all seven datasets; numbers are RMSE for iPad / Pi left / Pi right', x=0.01, y=1.0, ha='left', fontsize=15, fontweight='bold')
fig.tight_layout(rect=(0, 0.04, 1, 0.95)); save(fig, 'f16_curves_all_datasets')

# F17: swing peak and stance minimum vs speed, Moreira + Fukuchi + Lencioni speed groups
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2))
for ax, f, ttl in ((axs[0], 'peak', 'Swing-phase peak flexion'), (axs[1], 'stance', 'Mid-stance minimum flexion')):
    groups_ = [('Moreira', 'o', 'Moreira 2021 (overground, young)', [k for k in REF if k.startswith('Moreira')]),
               ('Fukuchi', 's', 'Fukuchi 2018 (treadmill, all ages)', [k for k in REF if k.startswith('Fukuchi T0') and k.endswith(' All')]),
               ('Lencioni', '^', 'Lencioni 2019 (overground, adults)', [b[2] for b in BINS])]
    for name, mk, lab, ks in groups_:
        ks = sorted(ks, key=lambda k: REF[k]['speed'])
        sp = [REF[k]['speed'] for k in ks]; hf = [feat(REF[k]['S'])[f] for k in ks]
        ax.errorbar(sp, [h_.mean() for h_ in hf], yerr=[h_.std(ddof=1) for h_ in hf], fmt=mk + '-', color=GREY if name != 'Lencioni' else DARK,
                    mfc='white', capsize=3, lw=1.2, label=lab)
    for j, s in enumerate(SER):
        v = feat(ON[s][0])[f]
        ax.errorbar(our_speed + (j - 1) * 0.025, v.mean(), yerr=v.std(ddof=1), fmt='D', color=SER[s][1], ms=8, capsize=3, label=SER[s][0])
    ax.axvspan(our_speed - 0.08, our_speed + 0.08, color='#fdf0e8', zorder=0)
    ax.set_xlabel('walking speed (m/s)'); ax.set_ylabel('knee flexion (°)'); ax.set_title(ttl, loc='left')
axs[0].legend(fontsize=9, loc='lower right')
axs[1].text(our_speed + 0.1, 15, f'shaded: our pace ≈ {our_speed:.2f} m/s', ha='left', va='center', fontsize=9, color='#a04a1c')
fig.suptitle('Three datasets with measured speeds: our swing peak sits at the low end of healthy, our stance 5–8° more bent',
             x=0.01, y=1.0, ha='left', fontsize=15, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, 'f17_features_vs_speed_all')

# F18: equivalence against four references, one panel per feature
fig, axs = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
refs4 = ['Fukuchi T01 All', 'Moreira 0.56', 'Lencioni <0.8', 'VC able']
for ax, (f, fl) in zip(axs, (('peak', 'Swing peak'), ('stance', 'Stance minimum'), ('rom', 'Range of motion'))):
    ax.axvspan(-5, 5, color='#e8f3ec', zorder=0)
    for i, k in enumerate(refs4):
        for j, s in enumerate(SER):
            e = EQ4[(k, s, f)]; yy = len(refs4) - 1 - i + (1 - j) * 0.22
            ax.plot([e['lo'], e['hi']], [yy, yy], color=SER[s][1], lw=2.2)
            ax.plot(e['diff'], yy, 'o', color=SER[s][1], ms=7, mec='white', label=SER[s][0] if (i == 0 and f == 'peak') else None)
    ax.axvline(0, color=GREY, lw=1); ax.set_title(fl, loc='left'); ax.set_xlabel('ours − healthy (°), 90% CI')
    ax.set_xlim(-27, 16)
axs[0].set_yticks(range(len(refs4))[::-1]); axs[0].set_yticklabels([SHORT[k] for k in refs4], fontsize=9.5)
axs[0].plot([], [], 's', color='#e8f3ec', ms=12, label='±5° (accuracy goal)')
fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.07))
fig.suptitle('Swing peak within ±5° for iPad and Pi right against most references; stance and range are not',
             x=0.01, y=1.0, ha='left', fontsize=15, fontweight='bold')
fig.tight_layout(rect=(0, 0.04, 1, 0.95)); save(fig, 'f18_equivalence_4refs')

# F19: where the gap is, against three references
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
GAPR = {}
for ax, k in zip(axs, ('Moreira 0.56', 'Lencioni <0.8', 'VC able')):
    sd = REF[k]['sd']
    ax.fill_between(x, -2 * sd, 2 * sd, color=LIGHT, lw=0, label='healthy ±2 SD')
    ax.fill_between(x, -sd, sd, color='#c8cbcf', lw=0, label='±1 SD')
    ax.axhspan(-5, 5, color='#e8f3ec', alpha=0.6, zorder=0, label='±5°')
    for s in SER:
        dcur = OM[s] - REF[k]['m']; ax.plot(x, dcur, color=SER[s][1], lw=2, label=SER[s][0])
        GAPR[f'{k}|{s}'] = dict(early_stance_max=float(dcur[15:40].max()), around_peak_min=float(np.r_[dcur[:8], dcur[93:]].min()),
                                midstance_inside_2sd=float(np.mean(np.abs(dcur[40:70]) <= 2 * sd[40:70]) * 100))
    ax.axhline(0, color=GREY, lw=1)
    for xa, txt in ((12, 'swing → heel strike'), (50, 'stance'), (88, 'push-off → swing')):
        ax.text(xa, -24, txt, ha='center', fontsize=8.5, color='#5f6368')
    ax.set_ylim(-26, 26); ax.set_xlabel('% gait cycle (0 = swing peak)'); ax.set_title(SHORT[k].replace('\n', ' · '), loc='left', fontsize=11.5)
axs[0].set_ylabel('ours − healthy (°)')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.07))
fig.suptitle('The gap sits in early stance and around the swing peak, against every reference', x=0.01, y=1.0, ha='left', fontsize=15, fontweight='bold')
fig.tight_layout(rect=(0, 0.04, 1, 0.95)); save(fig, 'f19_difference_curve_3refs')
R2 = json.load(open(f'{N}/results_ext.json'))
R2['gap_by_ref'] = GAPR
R2['EQ4'] = {'|'.join(k): v for k, v in EQ4.items()}
json.dump(R2, open(f'{N}/results_ext.json', 'w'), indent=1, default=float)
print(json.dumps(GAPR, indent=0))
