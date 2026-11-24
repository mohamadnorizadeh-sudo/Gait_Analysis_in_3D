#!/usr/bin/env python3
"""Compare a Pi walking session with a far-view reference video, stride by stride (v3.1.9).

Inputs
  pi    : base64 line from scripts/side_export_series.py (per-frame ungated markerless + live output)
  ref   : CSV from scripts/ref_movenet_track.py (t, flex_L, flex_R, conf_L, conf_R, hip_x)
Steps
  1. clock offset: t_ref = t_pi + offset, searched in [--off-min, --off-max] by correlating max(L, R) flexion
     inside a sync window of Pi time (--sync, e.g. the two knee bends after standing still);
  2. walking passes from the reference mid-hip x speed (> --speed px/s for > 6 s); direction -1/+1;
  3. swing peaks: reference peaks (prominence 12 deg, >= 0.8 s apart) per leg and pass; the Pi value is the
     max of its ungated markerless angle (conf >= 0.5) within +-0.3 s; also the live tracker output.
Prints the offset, passes, per-pass correlation, and swing-peak medians (near / far side), and writes
<out>_peaks.csv.

  python3 scripts/walk_compare_ref.py series.b64 ref.csv --sync 10,20 --off-min 5 --off-max 30 --out walk
"""
import argparse
import os
import sys

import numpy as np
from scipy.signal import find_peaks, medfilt

sys.path.insert(0, os.path.dirname(__file__))
from side_export_series import decode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pi')
    ap.add_argument('ref')
    ap.add_argument('--sync', default='10,20', help='Pi-time window (s) holding the sync knee bends')
    ap.add_argument('--off-min', type=float, default=0.0)
    ap.add_argument('--off-max', type=float, default=40.0)
    ap.add_argument('--speed', type=float, default=12.0, help='walking threshold, reference px/s')
    ap.add_argument('--out', default='walk')
    a = ap.parse_args()
    d = decode(open(a.pi).read())
    tau = d['kp_t'] - d['kp_t'][0]
    pl = d['kp'].copy()
    pl[d['kp_c'] < 0.5] = np.nan
    live = d['an'][:, :2]
    r = np.genfromtxt(a.ref, delimiter=',', names=True)
    t = r['t']
    F = {'L': medfilt(r['flex_L'], 5), 'R': medfilt(r['flex_R'], 5)}

    s0, s1 = map(float, a.sync.split(','))
    sel = (tau > s0) & (tau < s1)
    best = (-2, 0.0, '')
    for col, s in ((0, 'L'), (1, 'R')):           # per leg: one camera may be unreliable while standing
        q = pl[sel, col]
        ok = ~np.isnan(q)
        if ok.sum() < 20:
            continue
        for o in np.arange(a.off_min, a.off_max, 0.02):
            c = np.corrcoef(np.interp(tau[sel][ok] + o, t, np.maximum(F['L'], F['R'])), q[ok])[0, 1]
            if c > best[0]:
                best = (c, o, s)
    off = best[1]
    print('offset t_ref = t_pi + %.2f s (r = %.3f, Pi %s leg, sync window %g-%g s)' % (off, best[0], best[2], s0, s1))

    hx = np.convolve(r['hip_x'], np.ones(15) / 15, 'same')
    v = np.convolve(np.gradient(hx, t), np.ones(61) / 61, 'same')
    walk = np.abs(v) > a.speed
    passes, i = [], 0
    while i < len(t):
        if walk[i]:
            j = i
            while j + 1 < len(t) and walk[j + 1] and np.sign(v[j + 1]) == np.sign(v[i]):
                j += 1
            if t[j] - t[i] > 6:
                passes.append((t[i], t[j], int(np.sign(v[i]))))
            i = j + 1
        else:
            i += 1
    print('passes (ref time):', [(round(p0, 1), round(p1, 1), dd) for p0, p1, dd in passes])

    rows = []
    for p0, p1, dd in passes:
        near = 'R' if dd > 0 else 'L'   # v3.6.1: same rule as train_angle_model.near_legs (was inverted)
        for s, col in (('L', 0), ('R', 1)):
            idx = np.where((t >= p0) & (t <= p1))[0]
            pk, _ = find_peaks(F[s][idx], prominence=12, distance=24)
            for p in pk:
                k = idx[p]
                w = (tau >= t[k] - off - 0.3) & (tau <= t[k] - off + 0.3)
                pv = np.nanmax(pl[w, col]) if np.any(~np.isnan(pl[w, col])) else np.nan
                lv = np.nanmax(live[w, col]) if np.any(~np.isnan(live[w, col])) else np.nan
                rows.append((p0, s, s == near, F[s][k], pv, lv))
    with open(a.out + '_peaks.csv', 'w') as f:
        f.write('pass_start,leg,near,ref_peak,pi_peak,pi_live_peak\n')
        for rr in rows:
            f.write('%.1f,%s,%d,%.1f,%s,%s\n' % (rr[0], rr[1], rr[2], rr[3], '' if np.isnan(rr[4]) else '%.1f' % rr[4],
                                                '' if np.isnan(rr[5]) else '%.1f' % rr[5]))
    for s in 'LR':
        for nr in (True, False):
            z = np.array([(rr[3], rr[4], rr[5]) for rr in rows if rr[1] == s and rr[2] == nr], float)
            if len(z):
                print('%s %s n=%2d  ref %.1f  Pi markerless %.1f  ratio %.2f  live %.1f' % (
                    s, 'near' if nr else 'far ', len(z), np.median(z[:, 0]), np.nanmedian(z[:, 1]),
                    np.nanmedian(z[:, 1] / z[:, 0]), np.nanmedian(z[:, 2]) if np.any(~np.isnan(z[:, 2])) else np.nan))


if __name__ == '__main__':
    main()
