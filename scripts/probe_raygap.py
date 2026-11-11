#!/usr/bin/env python3
"""Why are confident joints not triangulated?  Ray gap (mm) between cam0 and cam1
for the same joint, and for the mirrored label (L<->R), per protocol phase."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.pose import best_person, people_from_result
from gait3d.recorder import read_recording
from gait3d.stereo_geometry import load_geometry
geo = load_geometry(sys.argv[2] if len(sys.argv) > 2 else 'stereo_calibration.npz')
names = {11: 'Lhip', 12: 'Rhip', 13: 'Lkn', 14: 'Rkn', 15: 'Lank', 16: 'Rank'}
mirror = {11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15}
ph = [('stand', 136, 145), ('Lhold', 148, 160), ('Rhold', 170, 184), ('walk', 199, 226)]
if len(sys.argv) > 3:   # 'name:lo-hi,name:lo-hi'
    ph = [(q.split(':')[0], *map(float, q.split(':')[1].split('-'))) for q in sys.argv[3].split(',')]
same = {p[0]: {k: [] for k in names} for p in ph}; mir = {p[0]: {k: [] for k in names} for p in ph}
zs = {p[0]: {k: [] for k in names} for p in ph}
t0 = None
for t, pl, pr, _ in read_recording(sys.argv[1]):
    t0 = t if t0 is None else t0
    a, b = best_person(people_from_result(pl)), best_person(people_from_result(pr))
    if a is None or b is None:
        continue
    for name, lo, hi in ph:
        if lo <= t - t0 < hi:
            for k in names:
                if a.conf[k] > .5 and b.conf[k] > .5:
                    same[name][k].append(geo.ray_gap_mm(*a.xy[k], *b.xy[k]))
                    zs[name][k].append(geo.triangulate(*a.xy[k], *b.xy[k])[2])
                if a.conf[k] > .5 and b.conf[mirror[k]] > .5:
                    mir[name][k].append(geo.ray_gap_mm(*a.xy[k], *b.xy[mirror[k]]))
print('median ray gap mm  same-label / mirrored-label   [z mm]   (tracker limit 80 mm)')
for name, _, _ in ph:
    print(name.ljust(6), '  '.join(f'{names[k]} {np.median(same[name][k]) if same[name][k] else -1:4.0f}/'
                                     f'{np.median(mir[name][k]) if mir[name][k] else -1:4.0f} [{np.nanmedian(zs[name][k]) if zs[name][k] else -1:4.0f}]'
                                     for k in names))
