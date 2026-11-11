#!/usr/bin/env python3
"""A/B test: tape-measured bone lengths vs lengths measured by the stereo rig itself.

Pass 1 measures thigh/shin per side from fully triangulated legs in the standing
window; pass 2 replays the whole recording with those lengths and prints the
diagnose table for both variants."""
import os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.profile import SubjectProfile
from gait3d.recorder import read_recording
from gait3d.stereo_geometry import load_geometry
from gait3d.tracker import GaitTracker, TrackerConfig

rec, lo, hi = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
geo = load_geometry('stereo_calibration.npz')
tr = GaitTracker(geo, SubjectProfile.manual(420, 410), TrackerConfig())
L = {s: {'thigh': [], 'shin': []} for s in 'LR'}
t0 = None
for t, pl, pr, _ in read_recording(rec):
    t0 = t if t0 is None else t0
    out = tr.process(t, pl, pr)
    if lo <= t - t0 < hi:
        for s, obs in out.obs.items():
            zs, p = tr._measure(obs)
            if len(p) == 3:
                L[s]['thigh'].append(np.linalg.norm(p['hip'] - p['knee']))
                L[s]['shin'].append(np.linalg.norm(p['ankle'] - p['knee']))
m = {s: {k: float(np.median(v)) if v else np.nan for k, v in d.items()} for s, d in L.items()}
for s in 'LR':
    print(f"stereo {s}: thigh {m[s]['thigh']:.0f} mm  shin {m[s]['shin']:.0f} mm  (n {len(L[s]['thigh'])})")
print(' '.join(f"{s}{k}={m[s][k]:.0f}" for s in 'LR' for k in ('thigh', 'shin')))
