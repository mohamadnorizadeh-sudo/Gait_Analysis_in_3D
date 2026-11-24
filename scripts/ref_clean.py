#!/usr/bin/env python3
"""Clean a MoveNet reference CSV from ref_movenet_track.py (v3.7.0; the step used since v3.3.7, now committed).

A frame is dropped when either leg's confidence is below 0.3 or either knee angle is above 110 deg (MoveNet loses
the legs at the start and end of the iPad video and when the person is far away). Dropped frames get their flexion
and hip_x linearly interpolated (by frame number) from the neighbouring good frames (edges hold the nearest good value) and their
confidences set to 0, so later steps can see which frames were filled in.

    python3 scripts/ref_clean.py ref_ipad_movenet.csv ref_ipad_movenet_clean.csv

Note (independent review, v3.6.1): the 110 deg cap also removes the top of deep knee bends in the reference.
"""
import sys

import numpy as np


def clean(r, min_conf=0.3, max_flex=110.0):
    i = np.arange(len(r))
    bad = (r['conf_L'] < min_conf) | (r['conf_R'] < min_conf) | (r['flex_L'] > max_flex) | (r['flex_R'] > max_flex)
    bad |= np.isnan(r['flex_L']) | np.isnan(r['flex_R'])
    out = {k: r[k].astype(float).copy() for k in r.dtype.names}
    for k in ('flex_L', 'flex_R', 'hip_x'):
        out[k][bad] = np.interp(i[bad], i[~bad], r[k][~bad])
    out['conf_L'][bad] = 0.0
    out['conf_R'][bad] = 0.0
    return out, int(bad.sum())


def main():
    src, dst = sys.argv[1], sys.argv[2]
    r = np.genfromtxt(src, delimiter=',', names=True, comments='#')
    out, n = clean(r)
    with open(dst, 'w') as f:
        f.write('t,flex_L,flex_R,conf_L,conf_R,hip_x\n')
        for i in range(len(r)):
            f.write('%.3f,%.1f,%.1f,%.2f,%.2f,%.1f\n' % (out['t'][i], out['flex_L'][i], out['flex_R'][i],
                                                       out['conf_L'][i], out['conf_R'][i], out['hip_x'][i]))
    print('%s: %d of %d frames filled in -> %s' % (src, n, len(r), dst))


if __name__ == '__main__':
    main()
