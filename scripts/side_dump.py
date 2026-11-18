#!/usr/bin/env python3
"""Compact dump of a side session's angles.csv: median raw flexion per 0.5 s bin as "L,R;L,R;..."
(v3.1.1 - used to copy a recording's trace out of the Pi remote shell for validation)."""
import csv, sys
import numpy as np
r = list(csv.DictReader(open(sys.argv[1] + '/angles.csv')))
col = sys.argv[2] if len(sys.argv) > 2 else 'raw'
f = lambda v: float(v) if v not in ('', 'nan', 'None') else np.nan
t0 = float(r[0]['t'])
B = {}
for x in r:
    B.setdefault(int((float(x['t']) - t0) / 0.5), []).append((f(x['L_' + col]), f(x['R_' + col])))
m = lambda v: '' if np.all(np.isnan(v)) else '%d' % round(np.nanmedian(v))
print('BINS ' + ';'.join(m([p[0] for p in B.get(b, [])] or [np.nan]) + ',' + m([p[1] for p in B.get(b, [])] or [np.nan])
                         for b in range(max(B) + 1)))
