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
ap.add_argument('--swap-ratio', type=float, default=None)
ap.add_argument('--knee-gate', type=float, default=None)
ap.add_argument('--dump', default='', help='lo-hi seconds: print L/R flexion every 0.25 s')
ap.add_argument('--max-rate', type=float, default=None)
ap.add_argument('--anatomy', type=int, default=0)
ap.add_argument('--stereo-first', type=int, default=1)
ap.add_argument('--adapt', type=float, default=0.1)
ap.add_argument('--median', type=int, default=0)
ap.add_argument('--probe', default='', help='lo-hi:side  print keypoints & 3D per frame')
ap.add_argument('--phases', default='stand:136-145,Lhold:148-160,Ltwist:161-169,Rhold:170-184,Rtwist:185-198,walk:199-226')
a = ap.parse_args()
phases = [(p.split(':')[0], *map(float, p.split(':')[1].split('-'))) for p in a.phases.split(',')]
prof = SubjectProfile.manual(a.thigh_mm, a.shin_mm)
if a.lengths:
    lt, ls, rt, rs = map(float, a.lengths.split(','))
    prof = SubjectProfile(lt, ls, rt, rs, method='manual')
tr = GaitTracker(load_geometry(a.calib), prof, TrackerConfig(anatomy=a.anatomy, stereo_first=a.stereo_first, adapt_lengths=a.adapt, **({'max_rate_dps': a.max_rate} if a.max_rate else {})))
if a.swap_ratio is not None:
    tr.identity.swap_ratio = a.swap_ratio
KI = {'L': (13, 15), 'R': (14, 16)}
rows, t0, sw0 = [], None, 0
for t, pl, pr, _ in read_recording(a.recording):
    t0 = t if t0 is None else t0
    out = tr.process(t, pl, pr)
    if a.probe:
        _r, _s = a.probe.split(':'); _lo, _hi = map(float, _r.split('-'))
        if _lo <= t - t0 < _hi:
            o = out.obs.get(_s); r = out.legs.get(_s)
            txt = f'{t - t0:7.2f} '
            if o is not None:
                txt += 'c0 ' + ' '.join(f'{x:4.0f},{y:4.0f}/{c:.2f}' for (x, y), c in zip(o.xy, o.conf))
                if o.xy_r is not None:
                    txt += ' | c1 ' + ' '.join(f'{x:4.0f},{y:4.0f}/{c:.2f}' for (x, y), c in zip(o.xy_r, o.conf_r))
            if r is not None:
                txt += f' | z {r.hip[2]:4.0f} {r.knee[2]:4.0f} {r.ankle[2]:4.0f} f{out.flexion_corrected.get(_s, float("nan")):4.0f} ' + ''.join(v[0] for v in r.source.values())
            print(txt)
    bl, br = best_person(people_from_result(pl)), best_person(people_from_result(pr))
    for s in 'LR':
        r = out.legs.get(s)
        cl = bl.conf[list(KI[s])] if bl is not None else (0, 0)
        cr = br.conf[list(KI[s])] if br is not None else (0, 0)
        raw = np.nan                       # angle straight from triangulated joints (no bone lengths)
        if s in out.obs:
            _z, _p = tr._measure(out.obs[s])
            if len(_p) == 3:
                u, v = _p['hip'] - _p['knee'], _p['ankle'] - _p['knee']
                raw = 180 - float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
        rows.append(dict(t=t - t0, s=s, raw=raw, sol=(180 - r.angle_deg) if r is not None else np.nan, det=s in out.obs, solv=r is not None,
                         meas=(sum(v == 'measured' for v in r.source.values()) / 3) if r else np.nan,
                         f=out.flexion_corrected.get(s, np.nan), cK0=cl[0], cA0=cl[1], cK1=cr[0], cA1=cr[1]))
    rows[-1]['sw'] = tr.identity.swaps - sw0; sw0 = tr.identity.swaps
print(f'{"phase":7} side  det  solv meas  cK0  cK1  cA0  cA1 jump    sd   med idsw | solver  stereo(n)')
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
              f'{(np.std(ff) if len(ff) else np.nan):5.1f} {(np.median(ff) if len(ff) else np.nan):5.1f} {sw:4d} | '
              f'{np.nanmedian([r["sol"] for r in R]) if np.any(np.isfinite([r["sol"] for r in R])) else np.nan:6.1f} '
              f'{np.nanmedian([r["raw"] for r in R]) if np.any(np.isfinite([r["raw"] for r in R])) else np.nan:6.1f}({int(np.sum(np.isfinite([r["raw"] for r in R])))})')

if a.dump:
    lo, hi = map(float, a.dump.split('-'))
    out = []
    for b in np.arange(lo, hi, 0.25):
        v = {s: [r['f'] for r in rows if r['s'] == s and b <= r['t'] < b + 0.25 and np.isfinite(r['f'])] for s in 'LR'}
        out.append(f"{b:6.2f} " + " ".join(f"{np.median(v[s]):4.0f}" if v[s] else "   -" for s in 'LR'))
    for i in range(0, len(out), 4):
        print(' | '.join(out[i:i + 4]))
