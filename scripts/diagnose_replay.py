#!/usr/bin/env python3
"""Replay a recording and explain, per protocol phase, WHY legs are lost or jump.

  python3 scripts/diagnose_replay.py runs/<session>/keypoints.jsonl --phases "stand:136-145,Lhold:148-160,..."
Per phase and side:
  det   fraction of frames the leg passed the 2D gate (after identity)
  solv  fraction with a 3D solution
  meas  fraction of joints triangulated (rest = bone-length constrained)
  cK/cA median knee / ankle confidence cam0|cam1 for that side (best person)
  jump  frames where the flexion changes > 15 deg from the previous frame
  sd    spread (std) of flexion
  idsw  identity swaps performed
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.pose import best_person, people_from_result
from gait3d.profile import SubjectProfile
from gait3d.recorder import read_recording
from gait3d.stereo_geometry import load_geometry
from gait3d.tracker import GaitTracker, TrackerConfig

ap = argparse.ArgumentParser()
ap.add_argument('recording')
ap.add_argument('--calib', default='stereo_calibration.npz')
ap.add_argument('--thigh-mm', type=float, default=420)
ap.add_argument('--shin-mm', type=float, default=410)
ap.add_argument('--lengths', default='', help='Lthigh,Lshin,Rthigh,Rshin (mm)')
ap.add_argument('--phases', default='stand:136-145,Lhold:148-160,Ltwist:161-169,Rhold:170-184,Rtwist:185-198,walk:199-226')
a = ap.parse_args()
phases = [(p.split(':')[0], *map(float, p.split(':')[1].split('-'))) for p in a.phases.split(',')]
prof = SubjectProfile.manual(a.thigh_mm, a.shin_mm)
if a.lengths:
    lt, ls, rt, rs = map(float, a.lengths.split(','))
    prof = SubjectProfile(lt, ls, rt, rs, method='manual')
tr = GaitTracker(load_geometry(a.calib), prof, TrackerConfig())
KI = {'L': (13, 15), 'R': (14, 16)}
rows, t0, sw0 = [], None, 0
for t, pl, pr, _ in read_recording(a.recording):
    t0 = t if t0 is None else t0
    out = tr.process(t, pl, pr)
    bl, br = best_person(people_from_result(pl)), best_person(people_from_result(pr))
    for s in 'LR':
        r = out.legs.get(s)
        cl = bl.conf[list(KI[s])] if bl is not None else (0, 0)
        cr = br.conf[list(KI[s])] if br is not None else (0, 0)
        rows.append(dict(t=t - t0, s=s, det=s in out.obs, solv=r is not None,
                         meas=(sum(v == 'measured' for v in r.source.values()) / 3) if r else np.nan,
                         f=out.flexion_corrected.get(s, np.nan), cK0=cl[0], cA0=cl[1], cK1=cr[0], cA1=cr[1]))
    rows[-1]['sw'] = tr.identity.swaps - sw0; sw0 = tr.identity.swaps
print(f'{"phase":7} side  det  solv meas  cK0  cK1  cA0  cA1 jump    sd   med idsw')
for name, lo, hi in phases:
    for s in 'LR':
        R = [r for r in rows if r['s'] == s and lo <= r['t'] < hi]
        if not R:
            continue
        f = np.array([r['f'] for r in R])
        ff = f[np.isfinite(f)]
        jumps = int(np.sum(np.abs(np.diff(ff)) > 15)) if len(ff) > 1 else 0
        sw = sum(r.get('sw', 0) for r in rows if r['s'] == 'R' and lo <= r['t'] < hi) if s == 'L' else 0
        m = lambda k: np.nanmedian([r[k] for r in R])
        print(f'{name:7} {s}    {np.mean([r["det"] for r in R]):.2f} {np.mean([r["solv"] for r in R]):.2f} '
              f'{np.nanmean([r["meas"] for r in R]) if np.any(np.isfinite([r["meas"] for r in R])) else 0:.2f} '
              f'{m("cK0"):.2f} {m("cK1"):.2f} {m("cA0"):.2f} {m("cA1"):.2f} {jumps:4d} '
              f'{(np.std(ff) if len(ff) else np.nan):5.1f} {(np.median(ff) if len(ff) else np.nan):5.1f} {sw:4d}')
