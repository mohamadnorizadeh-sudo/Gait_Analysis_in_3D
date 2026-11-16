#!/usr/bin/env python3
"""v3.0 - offline knee angles from a side-camera recording (side_view.py --record or 07_side_session.py).

  python3 scripts/side_replay.py runs/side_20261115_122209                 # keypoints only
  python3 scripts/side_replay.py runs/side_... --markers                   # snap to tape markers (reads raw.mp4)
  python3 scripts/side_replay.py runs/side_... --phases stand:3-10,L1:14.5-19.5 --source L=cam1,R=cam0

Prints per phase the median flexion of each leg from BOTH cameras (near and far) so the
cameras can be compared, plus the tracker output (filtered, standing-offset) per leg.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.side2d import JOINTS, PlaneMap, SideConfig, SideTracker, flexion_2d, leg_keypoints, snap_to_markers

ap = argparse.ArgumentParser()
ap.add_argument('rec')
ap.add_argument('--planes', default='config/side_planes.json')
ap.add_argument('--markers', action='store_true')
ap.add_argument('--phases', default='')
ap.add_argument('--source', default='L=cam0,R=cam1')
ap.add_argument('--timeline', type=float, default=0.5, help='bin width (s) for the timeline, 0 = off')
a = ap.parse_args()

pm = PlaneMap.load(a.planes)
src = dict(kv.split('=') for kv in a.source.split(','))
cfg = SideConfig(source=src, markers=a.markers)
tr = SideTracker(pm, cfg)
rows = [json.loads(l) for l in open(os.path.join(a.rec, 'keypoints.jsonl'))]
cap = None
if a.markers:
    import cv2
    cap = cv2.VideoCapture(os.path.join(a.rec, 'raw.mp4'))
t0 = rows[0]['t']
per = []          # (t, {(cam,leg): flex}, {leg: tracker flex})
for r in rows:
    imgs = None
    if cap is not None:
        ok, fr = cap.read()
        if ok:
            imgs = {'cam0': fr[:, :fr.shape[1] // 2], 'cam1': fr[:, fr.shape[1] // 2:]}
    t = r['t'] - t0
    both = {}
    for cam in ('cam0', 'cam1'):
        for leg in 'LR':
            kp = leg_keypoints(r.get(cam, []), leg)
            if kp is None or kp[1].min() < cfg.kp_conf:
                continue
            pts = kp[0]
            if imgs is not None:
                fl = [False] * 3
                pts, n = snap_to_markers(imgs[cam], pts, cfg.marker_radius_px, flags=fl)
                if not all(fl[j] for j, ch in enumerate('HKA') if ch in cfg.marker_joints):
                    continue
            f = flexion_2d(*pm.to_plane(cam, leg, pts))
            if f is not None:
                both[(cam, leg)] = f
    out = tr.process(t, {'cam0': r.get('cam0', []), 'cam1': r.get('cam1', [])}, imgs)
    per.append((t, both, {s: o.flexion for s, o in out.items()}))

print(a.rec, len(rows), 'frames', round(per[-1][0], 1), 's | planes:', sorted(pm.planes) or 'none (raw pixels)',
      '| markers' if a.markers else '', '| offset', tr.offset)
med = lambda v: '%5.0f' % np.median(v) if v else '    -'
if a.phases:
    print('phase      cam0L cam0R cam1L cam1R | out L out R | valid% L R | raw peak L R')
    for ph in a.phases.split(','):
        name, rng = ph.split(':')
        lo, hi = map(float, rng.split('-'))
        sel = [p for p in per if lo <= p[0] < hi]
        cols = [med([p[1][(c, l)] for p in sel if (c, l) in p[1]]) for c in ('cam0', 'cam1') for l in 'LR']
        outs = [med([p[2][l] for p in sel if p[2][l] is not None]) for l in 'LR']
        src_c = {l: src[l] for l in 'LR'}
        valid = [100.0 * sum((src_c[l], l) in p[1] for p in sel) / max(1, len(sel)) for l in 'LR']
        peak = [max([p[1][(src_c[l], l)] for p in sel if (src_c[l], l) in p[1]] or [float('nan')]) for l in 'LR']
        print(f'{name:9s} ' + ' '.join(cols) + ' | ' + ' '.join(outs) + ' | %5.0f %3.0f | %5.0f %3.0f' % (*valid, *peak))
if a.timeline:
    line = []
    for b in np.arange(0, per[-1][0], a.timeline):
        sel = [p for p in per if b <= p[0] < b + a.timeline]
        o = [med([p[2][l] for p in sel if p[2][l] is not None]).strip() for l in 'LR']
        line.append(f'{b:.1f}:{o[0]}/{o[1]}')
    print('tracker L/R:', ' '.join(line))
