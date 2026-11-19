#!/usr/bin/env python3
"""Compare gait_cycles.json (scripts/side_gait_cycles.py) with the WBDS normative knee curve (v3.1.0).

Usage: python3 scripts/compare_normative.py runs/<session>/gait_cycles.json [--cond T01] [--png out.png]
For each leg and each tracker (marker / markerless) prints: Pearson r of the mean curve against the
normative mean, RMSE, RMSE after removing the mean offset, the share of the cycle inside mean +- 2 SD,
and the key features (loading-response peak, stance minimum, swing peak and its timing).
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('cycles')
ap.add_argument('--norm', default=os.path.join(os.path.dirname(__file__), '..', 'validation', 'normative',
                                               'wbds_knee_flexion.csv'))
ap.add_argument('--cond', default='T01')
ap.add_argument('--png', default='')
a = ap.parse_args()

N = np.genfromtxt(a.norm, delimiter=',', names=True)
nm, ns = N[a.cond + '_mean'], N[a.cond + '_sd']
R = json.load(open(a.cycles))


def feats(m):
    return dict(loading=m[:20].max(), stance_min=m[20:55].min(), swing=m[55:90].max(),
                swing_at=int(m[55:90].argmax() + 55), rom=m.max() - m.min())


fn = feats(nm)
print('normative %s: loading %.1f  stance-min %.1f  swing %.1f @%d%%  ROM %.1f' % (
    a.cond, fn['loading'], fn['stance_min'], fn['swing'], fn['swing_at'], fn['rom']))
rows = []
for leg in 'LR':
    r = R.get(leg, {})
    for k, name in (('mk', 'marker'), ('ps', 'markerless')):
        if not r.get(k + '_n'):
            continue
        m = np.array(r[k + '_mean'])
        e = m - nm
        f = feats(m)
        inside = np.mean(np.abs(e) <= 2 * ns) * 100
        row = (leg, name, r[k + '_n'], np.corrcoef(m, nm)[0, 1], np.sqrt((e ** 2).mean()),
               np.sqrt(((e - e.mean()) ** 2).mean()), e.mean(), inside, f)
        rows.append(row)
        print('%s %-10s strides %2d  r=%.3f  RMSE %.1f  RMSE-after-offset %.1f  offset %+.1f  in 2SD %.0f%%  '
              '| loading %.1f stance-min %.1f swing %.1f @%d%% ROM %.1f' % (
                  leg, name, row[2], row[3], row[4], row[5], row[6], row[7],
                  f['loading'], f['stance_min'], f['swing'], f['swing_at'], f['rom']))

if a.png:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    x = np.arange(101)
    for ax, leg in zip(axs, 'LR'):
        ax.fill_between(x, nm - 2 * ns, nm + 2 * ns, color='#bbb', alpha=.5, lw=0, label='WBDS %s mean ± 2 SD' % a.cond)
        ax.plot(x, nm, color='#555', lw=2, label='WBDS %s mean' % a.cond)
        r = R.get(leg, {})
        for k, name, col in (('mk', 'markers', '#1f6fb5'), ('ps', 'markerless', '#d9822b')):
            if r.get(k + '_n'):
                m, s = np.array(r[k + '_mean']), np.array(r[k + '_sd'])
                ax.plot(x, m, color=col, lw=2, label='%s (n=%d strides)' % (name, r[k + '_n']))
                ax.fill_between(x, m - s, m + s, color=col, alpha=.15, lw=0)
        ax.set_title('%s knee' % ('Left' if leg == 'L' else 'Right'))
        ax.set_xlabel('gait cycle (%)')
        ax.grid(alpha=.3)
    axs[0].set_ylabel('knee flexion (deg)')
    axs[1].legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(a.png, dpi=130)
    print('saved', a.png)
