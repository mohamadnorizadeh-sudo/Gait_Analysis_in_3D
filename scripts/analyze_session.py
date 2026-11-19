#!/usr/bin/env python3
"""Summarise a tracker CSV: per-leg angles, depth sources, stereo shin length,
and frame-to-frame joint jumps (a jump of a whole shin length usually means the
L/R labels swapped between legs)."""
import csv, sys
import numpy as np

rows = list(csv.DictReader(open(sys.argv[1])))
print('rows', len(rows))
t0 = float(rows[0]['t'])
for s in 'LR':
    x = [y for y in rows if y['side'] == s]
    if not x:
        continue
    fl = np.array([float(y['flexion_deg']) for y in x])
    sh = [float(y['shin_stereo_mm']) for y in x if y['shin_stereo_mm']]
    src = {k: sum(1 for y in x for j in ('hip', 'knee', 'ankle') if y['src_' + j] == k) for k in ('measured', 'constrained', 'prior')}
    print(f'{s}: n {len(x)}  flexion median {np.median(fl):.1f}  range {fl.min():.1f}..{fl.max():.1f}  '
          f'stereo shin median {np.median(sh) if sh else 0:.0f} mm (n {len(sh)})  sources {src}')
prev, jumps, swaps = {}, [], 0
by_t = {}
for y in rows:
    by_t.setdefault(y['t'], {})[y['side']] = y
P = lambda y, j: np.array([float(y[f'{j}_{c}']) for c in 'xyz'])
for t in sorted(by_t, key=float):
    cur = by_t[t]
    for s, y in cur.items():
        if s in prev and float(t) - prev[s][0] < 0.5:
            jumps.append(np.linalg.norm(P(y, 'ankle') - prev[s][1]))
            o = 'R' if s == 'L' else 'L'
            if o in prev and np.linalg.norm(P(y, 'ankle') - prev[o][1]) < 0.5 * np.linalg.norm(P(y, 'ankle') - prev[s][1]):
                swaps += 1
    for s, y in cur.items():
        prev[s] = (float(t), P(y, 'ankle'))
j = np.array(jumps)
print(f'ankle movement per frame: median {np.median(j):.0f} mm, p95 {np.percentile(j, 95):.0f}, max {j.max():.0f}; '
      f'frames > 150 mm: {(j > 150).sum()} of {len(j)}; probable L/R swaps: {swaps}')
