#!/usr/bin/env python3
"""Render chosen frames of a side session with what the tracker sees (v3.1.4).

For each requested time: the camera image of one leg with
  yellow crosses = pose keypoints (hip, knee, ankle) -> markerless angle,
  red circles     = marker blobs found,
  green line      = marker chain (snapped hip, knee, shin) -> marker angle.
Tiles are put side by side, saved as <rec>/frames_<leg>.jpg and (with --show) shown on the Pi screen.
No tracker state is used (no gates), so the picture shows exactly what one frame allows.

Usage: python3 scripts/side_frames.py runs/side_session_... --leg L --times 54.2,54.4,103.7 [--show]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.side2d import (PlaneMap, SideConfig, flexion_2d, leg_keypoints, marker_blobs,  # noqa: E402
                           snap_to_markers)

ap = argparse.ArgumentParser()
ap.add_argument('rec')
ap.add_argument('--leg', default='L')
ap.add_argument('--times', required=True)
ap.add_argument('--planes', default='config/side_planes.json')
ap.add_argument('--source', default='L=cam0,R=cam1')
ap.add_argument('--show', action='store_true')
ap.add_argument('--scale', type=float, default=0.75)
ap.add_argument('--crop', type=int, default=0, help='crop each tile to the leg and resize to this height')
a = ap.parse_args()

src = dict(kv.split('=') for kv in a.source.split(','))
cam = src[a.leg]
pm = PlaneMap.load(a.planes)
cfg = SideConfig(source=src, markers=True, marker_color='tan', marker_radius_px=140, shank_seed=0.4)
rows = [json.loads(l) for l in open(os.path.join(a.rec, 'keypoints.jsonl'))]
T = np.array([r['t'] for r in rows]) - rows[0]['t']
want = sorted(float(x) for x in a.times.split(','))
idx = [int(np.argmin(np.abs(T - w))) for w in want]
cap = cv2.VideoCapture(os.path.join(a.rec, 'raw.mp4'))
tiles = []
for i in idx:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, fr = cap.read()
    if not ok:
        continue
    W = fr.shape[1] // 2
    img = (fr[:, :W] if cam == 'cam0' else fr[:, W:]).copy()
    kp = leg_keypoints(rows[i].get(cam, []), a.leg, 170)
    txt = 't=%.1f' % T[i]
    if kp is not None:
        pts = np.asarray(kp[0], float)
        for q in pts:
            cv2.drawMarker(img, (int(q[0]), int(q[1])), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
        ml = flexion_2d(*pm.to_plane(cam, a.leg, pts))
        blobs = marker_blobs(img, cfg)
        for x, y, _ in blobs:
            cv2.circle(img, (int(x), int(y)), 9, (0, 0, 255), 2)
        seed = pts.copy()
        seed[2] = pts[1] + cfg.shank_seed * (pts[2] - pts[1])
        fl = []
        sp, _ = snap_to_markers(img, seed, cfg.marker_radius_px, blobs=blobs, flags=fl, shank=(pts[1], pts[2]))
        mk = flexion_2d(*pm.to_plane(cam, a.leg, np.asarray(sp, float))) if all(fl) else None
        if all(fl):
            cv2.polylines(img, [np.asarray(sp, np.int32)], False, (0, 255, 0), 2)
        txt += ' pose %s mk %s' % ('-' if ml is None else '%.0f' % ml, '-' if mk is None else '%.0f' % mk)
        print('t=%.2f  markerless %s  marker %s  tapes %s' % (T[i], ml, mk, ''.join('HKS'[j] if f else '-' for j, f in enumerate(fl))))
    if a.crop and kp is not None:                   # v3.1.4: tight crop around the leg (small file)
        x0 = int(max(0, pts[:, 0].min() - 70)); x1 = int(min(img.shape[1], pts[:, 0].max() + 70))
        y0 = int(max(0, pts[:, 1].min() - 40)); y1 = int(min(img.shape[0], pts[:, 1].max() + 40))
        img = img[y0:y1, x0:x1]
        img = cv2.resize(img, (int(img.shape[1] * a.crop / img.shape[0]), a.crop))
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(img, txt, (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    tiles.append(img if a.crop else cv2.resize(img, None, fx=a.scale, fy=a.scale))
if tiles:
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 4, cv2.BORDER_CONSTANT) for t in tiles]
    m = cv2.hconcat(tiles)
    out = os.path.join(a.rec, 'frames_%s.jpg' % a.leg)
    cv2.imwrite(out, m)
    print('saved', out, m.shape)
    if a.show:
        cv2.namedWindow('frames', cv2.WINDOW_NORMAL)
        cv2.imshow('frames', m)
        cv2.waitKey(120000)
