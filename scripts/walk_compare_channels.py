#!/usr/bin/env python3
"""Compare several Pi knee-angle channels with a far-view reference video, stride by stride (v3.2.3).

Like walk_compare_ref.py, but for any set of per-frame Pi channels in one npz:
  t (Pi seconds) and (N, 2) arrays [left, right] such as
  kp  - plane flexion, the leg's own camera (what the live system uses)
  k3  - 3D flexion triangulated from both cameras (v3.2.2)
  k2x - plane flexion from the other camera
  e3  - triangulation reprojection rms (px), used with --max-e3 to drop bad 3D frames
The npz is made from side_export_series.py output (see validation/walk_20261122b/README.md).

Steps: clock offset from the sync knee bends (the --sync window, Pi time from the first frame); walking passes
from the reference mid-hip speed; per pass and leg, frame-by-frame agreement (reference interpolated to the Pi
frames, both legs' angles median-filtered over 5 frames) and swing peaks (reference peaks with prominence 12 deg,
the Pi maximum within +-0.3 s).

  python3 scripts/walk_compare_channels.py pi.npz ref.csv --sync 3,25 --off-min 5 --off-max 25 --out walk
"""
import argparse

import numpy as np
from scipy.signal import find_peaks, medfilt


def passes_from_ref(t, hip_x, speed):
    hx = np.convolve(hip_x, np.ones(15) / 15, 'same')
    v = np.convolve(np.gradient(hx, t), np.ones(61) / 61, 'same')
    walk = np.abs(v) > speed
    out, i = [], 0
    while i < len(t):
        if walk[i]:
            j = i
            while j + 1 < len(t) and walk[j + 1] and np.sign(v[j + 1]) == np.sign(v[i]):
                j += 1
            if t[j] - t[i] > 6:
                out.append((t[i], t[j], int(np.sign(v[i]))))
            i = j + 1
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pi')
    ap.add_argument('ref')
    ap.add_argument('--channels', default='kp,k3,k2x')
    ap.add_argument('--sync', default='3,25')
    ap.add_argument('--sync-channel', default='kp')
    ap.add_argument('--off-min', type=float, default=0.0)
    ap.add_argument('--off-max', type=float, default=40.0)
    ap.add_argument('--offset', type=float, default=None, help='skip the search')
    ap.add_argument('--speed', type=float, default=12.0)
    ap.add_argument('--max-e3', type=float, default=1e9)
    ap.add_argument('--out', default='walk')
    a = ap.parse_args()
    d = np.load(a.pi)
    tau = d['t'] - d['t'][0]
    chans = a.channels.split(',')
    P = {}
    for c in chans:
        x = d[c].astype(float).copy()
        if c == 'k3' and 'e3' in d:
            x[~(d['e3'] <= a.max_e3)] = np.nan
        for j in range(2):                              # 5-frame median, NaN-aware
            y = x[:, j].copy()
            ok = ~np.isnan(y)
            if ok.sum() > 5:
                y[ok] = medfilt(y[ok], 5)
            x[:, j] = y
        P[c] = x
    r = np.genfromtxt(a.ref, delimiter=',', names=True)
    t = r['t']
    F = np.column_stack([medfilt(r['flex_L'], 5), medfilt(r['flex_R'], 5)])

    if a.offset is None:
        s0, s1 = map(float, a.sync.split(','))
        sel = (tau > s0) & (tau < s1)
        best = (-2, 0.0, '')
        for j, s in ((0, 'L'), (1, 'R')):
            q = P[a.sync_channel][sel, j]
            ok = ~np.isnan(q)
            if ok.sum() < 20:
                continue
            for o in np.arange(a.off_min, a.off_max, 0.02):
                c = np.corrcoef(np.interp(tau[sel][ok] + o, t, np.maximum(F[:, 0], F[:, 1])), q[ok])[0, 1]
                if c > best[0]:
                    best = (c, o, s)
        off = best[1]
        print('offset t_ref = t_pi + %.2f s (r = %.3f, %s leg, %s, sync %g-%g s)' % (off, best[0], best[2],
                                                                                    a.sync_channel, s0, s1))
    else:
        off = a.offset
    passes = passes_from_ref(t, r['hip_x'], a.speed)
    print('passes (ref time):', [(round(p0, 1), round(p1, 1), dd) for p0, p1, dd in passes])

    # frame-by-frame inside passes
    Fi = np.column_stack([np.interp(tau + off, t, F[:, j]) for j in range(2)])
    inp = np.zeros(len(tau), bool)
    for p0, p1, _ in passes:
        inp |= (tau + off >= p0) & (tau + off <= p1)
    lines = ['channel leg  n     r     slope  bias   rmse  (frames inside walking passes)']
    for c in chans:
        for j, s in ((0, 'L'), (1, 'R')):
            ok = inp & ~np.isnan(P[c][:, j])
            if ok.sum() < 30:
                continue
            x, y = Fi[ok, j], P[c][ok, j]
            sl, ic = np.polyfit(x, y, 1)
            lines.append('%-7s %s  %5d  %.2f  %.2f  %+5.1f  %5.1f' % (c, s, ok.sum(), np.corrcoef(x, y)[0, 1], sl,
                                                                   np.mean(y - x), np.sqrt(np.mean((y - x) ** 2))))
    print('\n'.join(lines))

    rows = []
    for p0, p1, dd in passes:
        near = 'R' if dd < 0 else 'L'
        idx = np.where((t >= p0) & (t <= p1))[0]
        for j, s in ((0, 'L'), (1, 'R')):
            pk, _ = find_peaks(F[idx, j], prominence=12, distance=24)
            for p in pk:
                k = idx[p]
                w = (tau >= t[k] - off - 0.3) & (tau <= t[k] - off + 0.3)
                vals = []
                for c in chans:
                    q = P[c][w, j]
                    vals.append(np.nanmax(q) if np.any(~np.isnan(q)) else np.nan)
                rows.append((p0, s, s == near, F[k, j], *vals))
    with open(a.out + '_peaks.csv', 'w') as f:
        f.write('pass_start,leg,near,ref_peak,' + ','.join(c + '_peak' for c in chans) + '\n')
        for rr in rows:
            f.write('%.1f,%s,%d,%.1f,' % rr[:4] + ','.join('' if np.isnan(v) else '%.1f' % v for v in rr[4:]) + '\n')
    print('swing peaks (median over strides):')
    print('leg side  n   ref   ' + '  '.join('%-14s' % c for c in chans))
    for s in 'LR':
        for nr in (True, False):
            z = np.array([rr[3:] for rr in rows if rr[1] == s and rr[2] == nr], float)
            if len(z):
                print('%s   %s %3d  %4.1f  ' % (s, 'near' if nr else 'far ', len(z), np.median(z[:, 0])) + '  '.join(
                    '%5.1f (x%.2f)  ' % (np.nanmedian(z[:, 1 + i]), np.nanmedian(z[:, 1 + i] / z[:, 0]))
                    for i in range(len(chans))))
    np.save(a.out + '_offset.npy', np.array([off]))


if __name__ == '__main__':
    main()
