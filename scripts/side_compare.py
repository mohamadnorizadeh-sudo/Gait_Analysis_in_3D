#!/usr/bin/env python3
"""v3.0.5 - compare a side-camera session with a phone reference trace.

  python3 scripts/side_compare.py runs/side_session_X validation/ref_20261116_IMG_4865.csv [--leg L] [--col raw]

The reference CSV has t_s, flexion_deg (same tapes tracked in the phone video).  The two clocks are aligned
by the time shift that maximises the correlation of the two angle traces; then every reference hold
(angle > 25 deg, stable within 6 deg for >= 1 s) is compared with the system median over the same time.
"""
import argparse, csv, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('session'); ap.add_argument('ref')
ap.add_argument('--leg', default='L'); ap.add_argument('--col', default='raw', choices=['raw', 'flexion'])
ap.add_argument('--ref-window', default='', help='lo-hi seconds of the reference to use (e.g. 0-75 for the left leg part)')
a = ap.parse_args()


def load(path, tcol, vcol):
    t, v = [], []
    for r in csv.DictReader(open(path)):
        if r[vcol] not in ('', None):
            t.append(float(r[tcol])); v.append(float(r[vcol]))
    return np.array(t), np.array(v)


def grid(t, v, dt=0.1, gap=0.6):
    g = np.arange(t[0], t[-1], dt)
    y = np.interp(g, t, v)
    near = np.array([np.min(np.abs(t - x)) for x in g])
    y[near > gap] = np.nan
    return g, y


def best_shift(gs, ys, gr, yr, lo=-120.0, hi=120.0, step=0.1):
    best = (-2, 0.0)
    for sh in np.arange(lo, hi, step):
        y = np.interp(gr + sh, gs, np.nan_to_num(ys, nan=-999), left=-999, right=-999)
        ok = (y > -900) & np.isfinite(yr)
        if ok.sum() < 100 or np.std(y[ok]) < 1 or np.std(yr[ok]) < 1:
            continue
        c = np.corrcoef(y[ok], yr[ok])[0, 1]
        if c > best[0]:
            best = (c, sh)
    return best


def holds(g, y, min_s=1.0, tol=6.0, floor=25.0):
    out, i, n = [], 0, len(g)
    while i < n:
        if not np.isfinite(y[i]) or y[i] < floor:
            i += 1; continue
        j = i
        while j + 1 < n and np.isfinite(y[j + 1]) and abs(y[j + 1] - np.median(y[i:j + 1])) <= tol:
            j += 1
        if g[j] - g[i] >= min_s:
            out.append((g[i], g[j], float(np.median(y[i:j + 1]))))
        i = j + 1
    return out


if __name__ == '__main__':
    ts, vs = load(a.session + '/angles.csv', 't', f'{a.leg}_{a.col}')
    tr, vr = load(a.ref, 't_s', 'flexion_deg')
    if a.ref_window:
        lo, hi = map(float, a.ref_window.split('-')); k = (tr >= lo) & (tr <= hi); tr, vr = tr[k], vr[k]
    ts = ts - ts[0]
    gs, ys = grid(ts, vs); gr, yr = grid(tr, vr)
    c, sh = best_shift(gs, ys, gr, yr)
    print(f'{a.leg} leg ({a.col}): system t = reference t {sh:+.1f} s, correlation {c:.2f}')
    errs = []
    print('ref t      ref   sys   diff  valid%')
    for t0, t1, m in holds(gr, yr):
        k = (gs >= t0 + sh) & (gs <= t1 + sh)
        v = ys[k][np.isfinite(ys[k])]
        if len(v) == 0:
            print(f'{t0:5.1f}-{t1:5.1f} {m:5.0f}     -     -      0'); continue
        d = float(np.median(v)) - m; errs.append(d)
        print(f'{t0:5.1f}-{t1:5.1f} {m:5.0f} {np.median(v):5.0f} {d:+6.1f} {100 * len(v) / max(1, k.sum()):6.0f}')
    st = np.isfinite(yr) & (yr < 12)
    y = np.interp(gr[st] + sh, gs, np.nan_to_num(ys, nan=np.nan))
    if np.isfinite(y).any():
        print(f'straight leg: ref {np.nanmedian(yr[st]):.0f}, sys {np.nanmedian(y):.0f}')
    if errs:
        e = np.array(errs)
        print(f'holds {len(e)}: bias {e.mean():+.1f}, mean abs {np.abs(e).mean():.1f}, max abs {np.abs(e).max():.1f} deg')
