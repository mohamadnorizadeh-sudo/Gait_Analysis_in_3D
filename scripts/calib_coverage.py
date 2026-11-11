#!/usr/bin/env python3
"""Where in each camera picture did the calibration see the board?  Prints a 3x3 grid
(left/middle/right x top/middle/bottom) with the share of corner detections per cell,
for the single-camera views and the stereo pairs.
  python3 scripts/calib_coverage.py runs/calib_v5"""
import glob, os, sys
import cv2, numpy as np
D = sys.argv[1]
CB = (9, 6)
F = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
def corners(paths):
    pts = []
    for p in paths:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        ok, c = cv2.findChessboardCorners(g, CB, F)
        if ok:
            pts.append(c.reshape(-1, 2) / np.array(g.shape[::-1], float))
    return np.vstack(pts) if pts else np.zeros((0, 2))
for name, pats in (('cam0 (left)', [f'{D}/mono/cam0/*.png', f'{D}/left/*.png']),
                   ('cam1 (right)', [f'{D}/mono/cam1/*.png', f'{D}/right/*.png'])):
    P = corners(sorted(sum((glob.glob(p) for p in pats), [])))
    H = np.histogram2d(P[:, 1], P[:, 0], bins=3, range=[[0, 1], [0, 1]])[0] / max(1, len(P)) * 100
    print(f'{name}: {len(P)} corners   % per cell (rows top->bottom, cols left->right)')
    for r, lab in zip(H, ('top', 'mid', 'bot')):
        print(f'   {lab}  ' + '  '.join(f'{v:5.1f}' for v in r))
