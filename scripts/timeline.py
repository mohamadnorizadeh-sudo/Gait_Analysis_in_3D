#!/usr/bin/env python3
"""Text timeline of a tracker CSV in 1-second bins: per leg the median flexion,
its spread (max-min inside the bin), and frames whose ankle jumped > 150 mm."""
import csv, sys
import numpy as np
rows = list(csv.DictReader(open(sys.argv[1])))
t0 = float(rows[0]['t'])
binsz = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
P = lambda y: np.array([float(y[f'ankle_{c}']) for c in 'xyz'])
prev, bins = {}, {}
for y in sorted(rows, key=lambda r: float(r['t'])):
    t, s = float(y['t']), y['side']
    b = bins.setdefault(int((t - t0) // binsz), {'L': [], 'R': [], 'jL': 0, 'jR': 0})
    b[s].append(float(y['flexion_filtered_deg']))
    if s in prev and t - prev[s][0] < 0.5 and np.linalg.norm(P(y) - prev[s][1]) > 150:
        b['j' + s] += 1
    prev[s] = (t, P(y))
print(' sec |  L flex (spread) jumps |  R flex (spread) jumps')
for k in sorted(bins):
    b = bins[k]
    f = lambda a: f'{np.median(a):6.1f} ({max(a) - min(a):5.1f})' if a else '     -        '
    print(f'{k * binsz:4.0f} | {f(b["L"])} {b["jL"]:3d}   | {f(b["R"])} {b["jR"]:3d}')
