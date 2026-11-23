#!/usr/bin/env python3
"""Has a side camera moved on the walker since a reference recording? (v3.1.7)

The plane calibration (config/side_planes.json) is only valid while each camera keeps its pose relative to the
walker. This compares the current view (runs/side_view.jpg, written by scripts/side_view.py) with a frame of a
reference recording (raw.mp4 = cam0|cam1) using ORB features. Two motions are told apart:

  walker features (lower part of the image: tubes, wheel, handle) move with the camera -> if THEY shift, the
  camera moved on the walker and the board calibration must be redone;
  room features shift when the walker was simply pushed somewhere else -> harmless.

For each camera it prints the dominant similarity transform of all matches and the median shift of the
matches in the lower --walker-frac of the image (where the walker frame is), plus how many matches there stayed
within 3 px. Rule of thumb from 18 Nov: <= 3 px / 0.5 deg = unchanged; the screw incident was 17-27 px.

  python3 scripts/side_view.py --no-pose --seconds 8      # refreshes runs/side_view.jpg (and shows the view)
  python3 scripts/cam_check.py runs/side_session_20261119_102313 [--frame 0] [--cur runs/side_view.jpg]
Writes runs/cam_check.jpg (reference over current, both cameras) for a visual check.
"""
import argparse
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('ref', help='reference recording folder (raw.mp4 = cam0|cam1)')
ap.add_argument('--frame', type=int, default=0)
ap.add_argument('--cur', default='runs/side_view.jpg')
ap.add_argument('--walker-frac', type=float, default=0.4, help='lower fraction of the image treated as walker')
a = ap.parse_args()

cap = cv2.VideoCapture(os.path.join(a.ref, 'raw.mp4'))
cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
ok, ref = cap.read()
if not ok:
    raise SystemExit('cannot read frame %d of %s/raw.mp4' % (a.frame, a.ref))
cur = cv2.imread(a.cur)
if cur is None:
    raise SystemExit('cannot read ' + a.cur)
W = ref.shape[1] // 2
refs = {'cam0': ref[:, :W], 'cam1': ref[:, W:]}
sep = cur.shape[1] - 2 * W                       # side_view.jpg has a 6 px white separator
curs = {'cam0': cur[:, :W], 'cam1': cur[:, W + sep:]}
orb = cv2.ORB_create(4000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
sheet = []
for cam in ('cam0', 'cam1'):
    g0 = cv2.cvtColor(refs[cam], cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(curs[cam], cv2.COLOR_BGR2GRAY)
    m1 = np.full(g1.shape, 255, np.uint8)
    m1[:40, :130] = 0                            # 'cam0'/'cam1' label drawn by side_view.py
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, m1)
    if d0 is None or d1 is None:
        print(cam, 'no features')
        continue
    ms = sorted(bf.match(d0, d1), key=lambda m: m.distance)[:800]
    p0 = np.float32([k0[m.queryIdx].pt for m in ms])
    p1 = np.float32([k1[m.trainIdx].pt for m in ms])
    d = p1 - p0
    M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    msg = '%s matches %d' % (cam, len(ms))
    if M is not None:
        rot = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
        sc = np.hypot(M[0, 0], M[1, 0])
        msg += ' | dominant: inliers %d shift %+.1f,%+.1f px rot %+.2f deg scale %.3f' % (
            int(inl.sum()), M[0, 2], M[1, 2], rot, sc)
    low = p0[:, 1] > (1 - a.walker_frac) * g0.shape[0]
    if low.sum() >= 5:
        med = np.median(d[low], axis=0)
        still = int((np.linalg.norm(d[low], axis=1) < 3).sum())
        msg += ' | walker zone: %d matches, median shift %+.1f,%+.1f px, %d within 3 px' % (
            int(low.sum()), med[0], med[1], still)
    print(msg)
    sheet.append(np.vstack([g0, np.full((6, g0.shape[1]), 255, np.uint8), g1]))
if sheet:
    out = os.path.join(os.path.dirname(a.cur) or '.', 'cam_check.jpg')
    cv2.imwrite(out, np.hstack([sheet[0], np.full((sheet[0].shape[0], 6), 255, np.uint8)] + sheet[1:]))
    print('wrote', out)
