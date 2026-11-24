"""Deep validation analysis: our knee curves (iPad reference, Pi v3.7.0 left/right) against 5 published healthy datasets,
with tolerance benchmarks. Writes fig/*.png and results.json."""
import json, os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = '/home/claude/work/deep'; os.makedirs(f'{D}/fig', exist_ok=True)
NORM = pickle.load(open(f'{D}/norms.pkl', 'rb'))
OURS = pickle.load(open(f'{D}/ours.pkl', 'rb'))['rows']
rng = np.random.default_rng(1)

BLUE, ORANGE, AQUA, GREY, DARK, LIGHT = '#2a78d6', '#eb6834', '#1baf7a', '#8a8f98', '#1f2328', '#d9dbde'
SER = {'ipad': ('iPad video reference', BLUE), 'piL': ('Pi model, left knee', ORANGE), 'piR': ('Pi model, right knee', AQUA)}
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9aa0a6',
                     'axes.labelcolor': '#3c4043', 'xtick.color': '#5f6368', 'ytick.color': '#5f6368', 'axes.titleweight': 'bold',
                     'axes.titlesize': 13, 'figure.dpi': 100, 'savefig.dpi': 200, 'axes.grid': True, 'grid.color': '#eceef0',
                     'grid.linewidth': 0.8, 'legend.frameon': False})

def rot(c):
    c = np.asarray(c, float)[:100]; k = int(np.argmax(c)); r = np.r_[c[k:], c[:k]]; return np.r_[r, r[0]]

# ---------- healthy references, peak-aligned ----------
REF = {}
for key, v in NORM.items():
    if v.get('subjects') is not None:
        S = np.array([rot(c) for c in v['subjects']])
        REF[key] = dict(v, S=S, m=S.mean(0), sd=S.std(0, ddof=1))
    else:
        k = int(np.argmax(v['mean'][:100]))
        sd = np.asarray(v['sd'])[:100]; sd = np.r_[sd[k:], sd[:k]]; sd = np.r_[sd, sd[0]]
        REF[key] = dict(v, S=None, m=rot(v['mean']), sd=sd)

# ---------- our cycles ----------
normal = lambda r: 40 <= r['ref_peak'] <= 75
stiff = lambda r: r['ref_peak'] < 35
def ours(sel):
    out = {}
    out['ipad'] = [(c, r['walk'] + str(r['pass_'])) for r in OURS if sel(r) for c, _ in r['ref_cycles']]
    out['piL'] = [(c, r['walk'] + str(r['pass_'])) for r in OURS if sel(r) and r['leg'] == 'L' for c, _ in r['model_cycles']]
    out['piR'] = [(c, r['walk'] + str(r['pass_'])) for r in OURS if sel(r) and r['leg'] == 'R' for c, _ in r['model_cycles']]
    return {k: (np.array([c for c, _ in v]), np.array([p for _, p in v])) for k, v in out.items()}
ON, OS = ours(normal), ours(stiff)
stride = np.median([T_ for r in OURS if normal(r) for _, T_ in r['ref_cycles']])
# our walking speed from stride time, interpolated on Moreira's controlled speeds (stride time vs speed)
ms = sorted([(v['speed'], v['stride']) for k, v in NORM.items() if k.startswith('Moreira')])
our_speed = float(np.interp(-stride, [-s for _, s in ms], [sp for sp, _ in ms]))

rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
feat = lambda C: dict(peak=C.max(1), stance=C[:, 20:80].min(1), rom=C.max(1) - C[:, 20:80].min(1))

R = {'stride_s': float(stride), 'our_speed_m_s': our_speed}
# ---------- A: curve agreement with every healthy condition ----------
COND = ['Moreira 0.28', 'Moreira 0.42', 'Moreira 0.56', 'Moreira 0.69', 'Moreira 0.83', 'Moreira 0.97', 'Moreira 1.11',
        'Fukuchi T01 All', 'Fukuchi T02 All', 'Fukuchi T03 All', 'Fukuchi T04 All', 'Fukuchi OS All', 'Fukuchi OC All',
        'Fukuchi T01 Older', 'Fukuchi OS Older', 'Helwig unbraced', 'Schwartz verySlow', 'Schwartz slow', 'Schwartz free',
        'Pinzone centreOne', 'Pinzone centreTwo']
LAB = {k: (f"{REF[k]['dataset'].split(' (')[0]} · " + (f"{REF[k]['speed']:.2f} m/s" if REF[k]['speed'] else REF[k]['condition'])
           + (' · older' if 'Older' in k else '')) for k in COND}
LAB['Fukuchi OS All'] = 'Fukuchi 2018 · overground slow 0.86 m/s'; LAB['Fukuchi OC All'] = 'Fukuchi 2018 · overground comfortable 1.27 m/s'
LAB['Fukuchi OS Older'] = 'Fukuchi 2018 · overground slow · older'
LAB['Helwig unbraced'] = 'Helwig 2016 · treadmill, comfortable'
for k in ('Schwartz verySlow', 'Schwartz slow', 'Schwartz free'):
    LAB[k] = 'Schwartz 2008 · ' + {'verySlow': 'very slow', 'slow': 'slow', 'free': 'self-selected'}[k.split()[1]]
LAB['Pinzone centreOne'] = 'Pinzone 2014 · lab 1'; LAB['Pinzone centreTwo'] = 'Pinzone 2014 · lab 2'
A = {}
for s in SER:
    m = ON[s][0].mean(0)
    A[s] = {k: dict(rmse=rmse(m, REF[k]['m']), r=float(np.corrcoef(m, REF[k]['m'])[0, 1]),
                    band=float(np.mean(np.abs(m - REF[k]['m']) <= 2 * REF[k]['sd']) * 100),
                    offset=float(np.mean(m - REF[k]['m']))) for k in COND}
R['A'] = A

# ---------- B: tolerance benchmarks ----------
# B1 healthy person-to-group: each person's curve vs the mean of the others (same dataset and condition)
B1 = {}
for k in ('Moreira 0.42', 'Moreira 0.56', 'Fukuchi T01 All', 'Fukuchi T01 Older', 'Helwig unbraced'):
    S = REF[k]['S']; n = len(S)
    B1[k] = [rmse(S[i], np.delete(S, i, 0).mean(0)) for i in range(n)]
# B2 between labs: mean curve vs mean curve for slow/self-selected conditions of DIFFERENT datasets
SLOW = ['Moreira 0.42', 'Moreira 0.56', 'Moreira 0.69', 'Fukuchi T01 All', 'Fukuchi T02 All', 'Fukuchi OS All',
        'Schwartz verySlow', 'Schwartz slow', 'Pinzone centreOne', 'Pinzone centreTwo', 'Helwig unbraced']
B2 = []
for i, a in enumerate(SLOW):
    for b in SLOW[i + 1:]:
        if REF[a]['dataset'] != REF[b]['dataset'] or a.startswith('Pinzone'):
            B2.append((a, b, rmse(REF[a]['m'], REF[b]['m'])))
R['B1'] = {k: dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)), n=len(v)) for k, v in B1.items()}
R['B2'] = dict(pairs=[(a, b, round(x, 2)) for a, b, x in B2], median=float(np.median([x for *_, x in B2])),
               p90=float(np.percentile([x for *_, x in B2], 90)), pinzone_two_labs=rmse(REF['Pinzone centreOne']['m'], REF['Pinzone centreTwo']['m']))
# percentile of our RMSE among healthy individuals (vs the same group mean)
R['percentile'] = {s: {k: float(np.mean(np.array(B1[k]) <= A[s][k]['rmse']) * 100) for k in B1} for s in SER}

# ---------- C: features vs speed; equivalence at matched speed ----------
FE = {s: {f: v for f, v in feat(ON[s][0]).items()} for s in SER}
def boot_ci(vals, keys, stat=np.mean, n=2000):
    u = np.unique(keys); b = []
    for _ in range(n):
        pick = rng.choice(u, len(u)); b.append(stat(np.concatenate([vals[keys == p] for p in pick])))
    return np.percentile(b, [5, 95])
EQ = {}
for k in ('Moreira 0.56', 'Moreira 0.42', 'Fukuchi T01 All', 'Fukuchi T01 Older'):
    hf = feat(REF[k]['S'])
    for s in SER:
        for f in ('peak', 'stance', 'rom'):
            d = FE[s][f].mean() - hf[f].mean()
            lo, hi = boot_ci(FE[s][f], ON[s][1])
            se_h = hf[f].std(ddof=1) / np.sqrt(len(hf[f]))
            EQ[(k, s, f)] = dict(diff=float(d), lo=float(lo - hf[f].mean() - 1.645 * se_h), hi=float(hi - hf[f].mean() + 1.645 * se_h),
                                 sd=float(hf[f].std(ddof=1)), ours=float(FE[s][f].mean()), healthy=float(hf[f].mean()))
R['EQ'] = {'|'.join(k): v for k, v in EQ.items()}

# ---------- D: stiff-knee detection vs knee brace ----------
def auc(a, b):  # P(normal peak > stiff peak)
    a, b = np.asarray(a), np.asarray(b); return float(np.mean(a[:, None] > b[None, :]) + 0.5 * np.mean(a[:, None] == b[None, :]))
Dd = {}
for s in SER:
    a, b = ON[s][0].max(1), OS[s][0].max(1)
    Dd[s] = dict(normal=float(a.mean()), stiff=float(b.mean()), drop=float(a.mean() - b.mean()), auc=auc(a, b), n=(len(a), len(b)))
hu, hb = REF['Helwig unbraced']['S'].max(1), REF['Helwig right knee braced']['S'].max(1)
Dd['helwig'] = dict(normal=float(hu.mean()), stiff=float(hb.mean()), drop=float(hu.mean() - hb.mean()), auc=auc(hu, hb), n=(len(hu), len(hb)))
R['D'] = Dd
R['brace_vs_ours'] = {s: rmse(OS[s][0].mean(0), REF['Helwig right knee braced']['m']) for s in SER}
json.dump(R, open(f'{D}/results.json', 'w'), indent=1, default=float)

# ====================== FIGURES ======================
def save(fig, name):
    fig.savefig(f'{D}/fig/{name}.png', bbox_inches='tight', facecolor='white')
    # deck version: the slide heading carries the finding, so drop the figure's main title
    if fig._suptitle is not None:
        fig._suptitle.set_visible(False)
    axs_ = [a for a in fig.axes if a.get_label() != '<colorbar>']
    if len(axs_) == 1:
        axs_[0].set_title('')
    if name.startswith('f1'):
        for t in fig.texts:
            t.set_visible(False)
    os.makedirs(f'{D}/fig_deck', exist_ok=True)
    fig.savefig(f'{D}/fig_deck/{name}.png', bbox_inches='tight', facecolor='white', dpi=160); plt.close(fig)
x = np.arange(101)

# F1: small multiples, our three curves vs six references
SM = ['Moreira 0.42', 'Moreira 0.56', 'Fukuchi T01 All', 'Fukuchi T01 Older', 'Schwartz verySlow', 'Pinzone centreOne']
fig, axs = plt.subplots(2, 3, figsize=(13, 6.6), sharex=True, sharey=True)
for ax, k in zip(axs.flat, SM):
    m, sd = REF[k]['m'], REF[k]['sd']
    ax.fill_between(x, m - 2 * sd, m + 2 * sd, color=LIGHT, lw=0, label='healthy ±2 SD')
    ax.plot(x, m, color=GREY, lw=1.6, ls='--', label='healthy mean')
    for s, (lab, col) in SER.items():
        ax.plot(x, ON[s][0].mean(0), color=col, lw=2, label=lab)
    ax.set_title(LAB[k], fontsize=11.5)
    ax.text(0.98, 0.95, 'RMSE  ' + ' / '.join(f"{A[s][k]['rmse']:.1f}" for s in SER) + '°', transform=ax.transAxes, ha='right', va='top', fontsize=9.5, color='#3c4043')
    ax.set_ylim(-12, 80)
for ax in axs[1]: ax.set_xlabel('% gait cycle (0 = swing peak)')
for ax in axs[:, 0]: ax.set_ylabel('knee flexion (°)')
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.04))
fig.suptitle('Same shape as every healthy reference; the gap is largest in mid-stance', x=0.01, y=0.995, ha='left', fontsize=15, fontweight='bold')
fig.text(0.01, 0.925, 'RMSE in each panel: iPad / Pi left / Pi right', fontsize=10, color='#5f6368')
fig.tight_layout(rect=(0, 0.04, 1, 0.91)); save(fig, 'f1_curves_six_references')

# F2: heatmap of RMSE against all 21 healthy conditions
M = np.array([[A[s][k]['rmse'] for k in COND] for s in SER])
fig, ax = plt.subplots(figsize=(13, 4.2))
im = ax.imshow(M, cmap='Blues', vmin=3, vmax=14, aspect='auto')
ax.set_yticks(range(3)); ax.set_yticklabels([SER[s][0] for s in SER])
ax.set_xticks(range(len(COND))); ax.set_xticklabels([LAB[k] for k in COND], rotation=55, ha='right', fontsize=9)
for i in range(3):
    for j in range(len(COND)):
        ax.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=8.5, color='white' if M[i, j] > 9 else DARK)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label('RMSE (°)')
ax.set_title('Closest to healthy walking at slow speeds (0.3–0.7 m/s); furthest from fast and comfortable-speed norms', loc='left')
save(fig, 'f2_rmse_heatmap')

# F3: tolerance: our RMSE vs healthy person-to-group and lab-to-lab differences
fig, ax = plt.subplots(figsize=(13.5, 5.6))
groups = [('person vs group\nMoreira 0.56 m/s', B1['Moreira 0.56']), ('person vs group\nMoreira 0.42 m/s', B1['Moreira 0.42']),
          ('person vs group\nFukuchi 0.49 m/s', B1['Fukuchi T01 All']), ('older person vs group\nFukuchi 0.49 m/s', B1['Fukuchi T01 Older']),
          ('lab vs lab\nslow-speed means', [x_ for *_, x_ in B2])]
for i, (lab, v) in enumerate(groups):
    v = np.asarray(v)
    ax.boxplot(v, positions=[i], widths=0.5, showfliers=False, patch_artist=True, boxprops=dict(facecolor=LIGHT, color=GREY),
               medianprops=dict(color=DARK, lw=2), whiskerprops=dict(color=GREY), capprops=dict(color=GREY))
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=12, color=GREY, alpha=0.6, zorder=3)
xo = len(groups) + 0.4
for j, s in enumerate(SER):
    vals = [A[s]['Moreira 0.56']['rmse'], A[s]['Fukuchi T01 All']['rmse']]
    ax.scatter([xo + j * 0.35] * 2, vals, s=70, color=SER[s][1], zorder=4, label=SER[s][0], edgecolor='white')
    ax.text(xo + j * 0.35, max(vals) + 0.5, f'{min(vals):.1f}–{max(vals):.1f}', ha='center', fontsize=9, color=SER[s][1])
ax.set_xticks(list(range(len(groups))) + [xo + 0.35]); ax.set_xticklabels([g[0] for g in groups] + ['our system vs\nMoreira 0.56 / Fukuchi 0.49'], fontsize=9.5)
ax.set_ylabel('curve difference, RMSE (°)'); ax.set_ylim(0, max(16, ax.get_ylim()[1]))
ax.legend(loc='upper left', fontsize=9.5)
ax.set_title('The iPad reference sits mid-range of healthy variation; the Pi model sits at its upper edge', loc='left')
ax.text(0, -0.2, 'Grey: how far each healthy person is from their own group mean (Moreira, Fukuchi), and how far published labs are from each other.', transform=ax.transAxes, fontsize=9.5, color='#5f6368')
save(fig, 'f3_tolerance_benchmarks')

# F4: swing peak and stance minimum vs speed
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for ax, f, ttl in ((axs[0], 'peak', 'Swing-phase peak flexion'), (axs[1], 'stance', 'Mid-stance minimum flexion')):
    for prefix, mk, lab in (('Moreira', 'o', 'Moreira 2021 (overground, young)'), ('Fukuchi T0', 's', 'Fukuchi 2018 (treadmill, all ages)')):
        ks = [k for k in COND + ['Fukuchi T05 All', 'Fukuchi T06 All'] if k.startswith(prefix) and 'Older' not in k]
        ks = sorted(set(ks), key=lambda k: REF[k]['speed'])
        sp = [REF[k]['speed'] for k in ks]; hf = [feat(REF[k]['S'])[f] for k in ks]
        ax.errorbar(sp, [h.mean() for h in hf], yerr=[h.std(ddof=1) for h in hf], fmt=mk + '-', color=GREY, mfc='white', capsize=3, lw=1.2, label=lab)
    for j, s in enumerate(SER):
        v = FE[s][f]
        ax.errorbar(our_speed + (j - 1) * 0.025, v.mean(), yerr=v.std(ddof=1), fmt='D', color=SER[s][1], ms=8, capsize=3, label=SER[s][0])
    ax.axvspan(our_speed - 0.08, our_speed + 0.08, color='#fdf0e8', zorder=0)
    ax.set_xlabel('walking speed (m/s)'); ax.set_ylabel('knee flexion (°)'); ax.set_title(ttl, loc='left')
axs[0].legend(fontsize=9, loc='lower right')
axs[1].text(our_speed + 0.1, -3.5, f'shaded: our pace ≈ {our_speed:.2f} m/s (from stride {stride:.2f} s)', ha='left', va='bottom', fontsize=9, color='#a04a1c')
fig.suptitle('At our pace the swing peak is at the low end of healthy (Pi left lower); stance stays about 5–8° more bent', x=0.01, y=0.99, ha='left', fontsize=15, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, 'f4_features_vs_speed')

# F5: equivalence forest plot vs Moreira 0.56 and Fukuchi 0.49
fig, ax = plt.subplots(figsize=(12, 5.6))
rows_ = []
for k in ('Moreira 0.56', 'Fukuchi T01 All'):
    for f, fl in (('peak', 'swing peak'), ('stance', 'stance minimum'), ('rom', 'range of motion')):
        for s in SER:
            rows_.append((f"{fl} · {LAB[k].split(' · ')[0]} {LAB[k].split(' · ')[1]}", s, EQ[(k, s, f)]))
y = np.arange(len(rows_))[::-1]
sdmax = max(EQ[(k, 'ipad', 'peak')]['sd'] for k in ('Moreira 0.56', 'Fukuchi T01 All'))
ax.axvspan(-5, 5, color='#e8f3ec', zorder=0, label='±5° (accuracy goal)')
for yi, (lab, s, e) in zip(y, rows_):
    ax.plot([e['lo'], e['hi']], [yi, yi], color=SER[s][1], lw=2.2)
    ax.plot(e['diff'], yi, 'o', color=SER[s][1], ms=7, mec='white')
ax.axvline(0, color=GREY, lw=1)
ax.set_yticks(y[::3] - 1); ax.set_yticklabels([r[0] for r in rows_][::3], fontsize=9.5)
ax.set_xlabel('our mean minus healthy mean (°), with 90% CI')
for s in SER: ax.plot([], [], 'o-', color=SER[s][1], label=SER[s][0])
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9)
ax.set_title('Swing peak: within ±5° of slow healthy walking for iPad and Pi right; stance 4–8° more bent; range smaller', loc='left')
save(fig, 'f5_equivalence')

# F6: stiff-knee detection vs knee brace
fig, axs = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(width_ratios=[1.3, 1]))
ax = axs[0]
pos = 0
ticks, tl = [], []
for s in SER:
    for g, C, al in (('normal', ON[s][0], 0.35), ('stiff', OS[s][0], 0.9)):
        v = C.max(1)
        ax.boxplot(v, positions=[pos], widths=0.6, showfliers=False, patch_artist=True, boxprops=dict(facecolor=SER[s][1], alpha=al, color=SER[s][1]),
                   medianprops=dict(color=DARK), whiskerprops=dict(color=SER[s][1]), capprops=dict(color=SER[s][1]))
        ticks.append(pos); tl.append(g); pos += 1
    ax.text(pos - 1.5, 1.0, SER[s][0].replace('Pi model, ', 'Pi ').replace(' video reference', ''), transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=10, color=SER[s][1], fontweight='bold')
    pos += 0.6
for g, v, al in (('unbraced', hu, 0.35), ('braced', hb, 0.9)):
    ax.boxplot(v, positions=[pos], widths=0.6, showfliers=False, patch_artist=True, boxprops=dict(facecolor=GREY, alpha=al, color=GREY),
               medianprops=dict(color=DARK), whiskerprops=dict(color=GREY), capprops=dict(color=GREY))
    ticks.append(pos); tl.append(g); pos += 1
ax.text(pos - 1.5, 1.0, 'Helwig 2016', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=10, color=GREY, fontweight='bold')
ax.set_xticks(ticks); ax.set_xticklabels(tl, fontsize=8.5); ax.set_ylabel('swing peak flexion (°)')
ax.set_title('Stiff-knee walking drops the swing peak by 18–24°', loc='left', pad=22)
ax = axs[1]
ax.fill_between(x, REF['Helwig right knee braced']['m'] - 2 * REF['Helwig right knee braced']['sd'],
                REF['Helwig right knee braced']['m'] + 2 * REF['Helwig right knee braced']['sd'], color=LIGHT, lw=0, label='knee brace ±2 SD')
ax.plot(x, REF['Helwig right knee braced']['m'], ls='--', color=GREY, label='knee brace mean')
for s in SER: ax.plot(x, OS[s][0].mean(0), color=SER[s][1], lw=2, label=SER[s][0] + f" (RMSE {R['brace_vs_ours'][s]:.1f}°)")
ax.set_ylim(-5, 50); ax.set_xlabel('% gait cycle (0 = swing peak)'); ax.set_ylabel('knee flexion (°)')
ax.legend(fontsize=8.5, loc='upper center'); ax.set_title('Our stiff walking matches braced walking', loc='left', pad=22)
fig.tight_layout(); save(fig, 'f6_stiff_knee')

# F7: difference curve with healthy spread
fig, ax = plt.subplots(figsize=(12, 4.8))
k = 'Moreira 0.56'
ax.fill_between(x, -2 * REF[k]['sd'], 2 * REF[k]['sd'], color=LIGHT, lw=0, label='healthy spread ±2 SD (Moreira 0.56)')
ax.fill_between(x, -REF[k]['sd'], REF[k]['sd'], color='#c8cbcf', lw=0, label='±1 SD')
ax.axhspan(-5, 5, color='#e8f3ec', alpha=0.6, zorder=0, label='±5°')
for s in SER: ax.plot(x, ON[s][0].mean(0) - REF[k]['m'], color=SER[s][1], lw=2, label=SER[s][0])
ax.axhline(0, color=GREY, lw=1)
for xa, txt in ((10, 'swing → heel strike'), (50, 'stance'), (90, 'push-off → swing')):
    ax.text(xa, -24, txt, ha='center', fontsize=9, color='#5f6368')
ax.set_ylim(-26, 26); ax.set_xlabel('% gait cycle (0 = swing peak)'); ax.set_ylabel('ours − healthy (°)')
ax.legend(ncol=3, fontsize=9, loc='upper center')
ax.set_title('Where the gap is: early stance (+8 to +13°) and the swing peak (−9 to −18°); mid-stance stays inside healthy ±2 SD', loc='left')
save(fig, 'f7_difference_curve')
print(json.dumps({k: v for k, v in R.items() if k in ('stride_s', 'our_speed_m_s', 'B1', 'B2', 'percentile', 'D', 'brace_vs_ours')}, indent=1, default=float)[:4000])
for s in SER: print(s, {k: round(A[s][k]['rmse'], 1) for k in COND})
for k_, e in EQ.items(): print(k_, {a: round(b, 1) for a, b in e.items()})

# F8: accuracy in context (literature benchmarks; different metrics, labelled)
lit = [('Marker-based gait lab, repeat sessions\n(McGinley 2009 review: most errors < 5°)', None, 5.0, None, GREY),
       ('Markerless 3D (Theia3D), between days\nRMSD 1.0–3.7°', 1.0, 3.7, None, GREY),
       ('OpenPose 2D vs IMU, knee (Penev 2024)\nRMSE 4.0°, 95% limits ±10.3°', None, 3.98, 10.3, GREY),
       ('OpenPose 2D side video vs gait lab\n(Stenum 2021) knee MAE 5.6°', None, 5.6, None, GREY),
       ('OUR Pi vs iPad, held-out walk, left\nmedian 3.7°, limits −7.6..+14.6°', None, 3.7, 11.1, ORANGE),
       ('OUR Pi vs iPad, held-out walk, right\nmedian 4.1°, limits −11.8..+12.7°', None, 4.1, 12.25, AQUA),
       ('OUR Pi vs iPad, training walks left out, left\nmedian 4.1°, limits −18.6..+12.8°', None, 4.1, 15.7, ORANGE),
       ('OUR Pi vs iPad, training walks left out, right\nmedian 4.7°, limits −21.1..+18.4°', None, 4.7, 19.75, AQUA)]
fig, axs = plt.subplots(1, 2, figsize=(13.5, 6), sharey=True, gridspec_kw=dict(width_ratios=[1.1, 1]))
yy = np.arange(len(lit))[::-1]
for yi, (lab, lo, val, loa, col) in zip(yy, lit):
    if lo is not None:
        axs[0].barh(yi, val - lo, left=lo, color=col, alpha=0.5, height=0.55)
    else:
        axs[0].barh(yi, val, color=col, alpha=0.85 if col != GREY else 0.5, height=0.55)
    axs[0].text(val + 0.15, yi, f'{val:.1f}°' if lo is None else f'{lo:.1f}–{val:.1f}°', va='center', fontsize=9.5)
    if loa:
        axs[1].barh(yi, loa, color=col, alpha=0.85 if col != GREY else 0.5, height=0.55)
        axs[1].text(loa + 0.3, yi, f'±{loa:.1f}°' if col == GREY else f'≈±{loa:.1f}°', va='center', fontsize=9.5)
axs[0].axvline(5, color=DARK, ls=':', lw=1.2); axs[0].text(5.05, len(lit) - 0.45, '5° goal', fontsize=9)
axs[0].set_yticks(yy); axs[0].set_yticklabels([l[0] for l in lit], fontsize=9)
axs[0].set_xlabel('typical error (°): median, MAE or RMSE as published'); axs[0].set_xlim(0, 8)
axs[1].set_xlabel('half-width of 95% limits of agreement (°)'); axs[1].set_xlim(0, 24)
axs[0].set_title('Typical error', loc='left'); axs[1].set_title('Spread (95% limits)', loc='left')
fig.suptitle('Our typical error is in the range of published markerless systems; our spread is wider', x=0.01, y=0.99, ha='left', fontsize=15, fontweight='bold')
fig.text(0.01, 0.005, 'Ours: swing-peak error against the iPad MoveNet reference (strict scoring, v3.7.0 on the Pi), not against a gait lab. Published values use different metrics and setups.', fontsize=9, color='#5f6368')
fig.tight_layout(rect=(0, 0.03, 1, 0.95)); save(fig, 'f8_accuracy_in_context')
