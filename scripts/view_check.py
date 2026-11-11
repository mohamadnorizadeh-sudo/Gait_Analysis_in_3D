#!/usr/bin/env python3
"""Is the right camera the weak view?  Per camera and leg joint:
  seen   = conf >= 0.5
  edge   = keypoint within 12 px of the image border (leg leaving the frame)
  overlap= the L and R joint are < 40 px apart (one leg hidden behind the other)
Then writes a contact sheet of video frames where a leg's flexion jumps > 15 deg.

  python3 scripts/view_check.py runs/<session>        -> prints table, writes <session>/jumps.png
"""
import csv, json, os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.pose import best_person, people_from_result
from gait3d.recorder import read_recording

S = sys.argv[1]
T_MAX = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9   # only the first T_MAX seconds
t_first = None
J = {'hip': (11, 12), 'knee': (13, 14), 'ankle': (15, 16)}
stats = {c: {k: {'seen': [], 'edge': [], 'overlap': []} for k in J} for c in ('cam0', 'cam1')}
ts = []
W = H = None
for t, pl, pr, _ in read_recording(os.path.join(S, 'keypoints.jsonl')):
    ts.append(t)
    t_first = t if t_first is None else t_first
    if t - t_first > T_MAX:
        continue
    p0 = best_person(people_from_result(pl))
    if p0 is None or np.mean(p0.conf[11:17]) < 0.5:       # nobody clearly in front of the walker
        continue
    for cam, res in (('cam0', pl), ('cam1', pr)):
        p = best_person(people_from_result(res))
        for k, (a, b) in J.items():
            if p is None:
                stats[cam][k]['seen'].append(0.0); continue
            stats[cam][k]['seen'].append(float((p.conf[a] >= .5) + (p.conf[b] >= .5)) / 2)
            xy = p.xy[[a, b]]
            if min(p.conf[a], p.conf[b]) < .5:
                continue
            stats[cam][k]['edge'].append(float(np.any((xy[:, 0] < 12) | (xy[:, 0] > 348) | (xy[:, 1] > 628) | (xy[:, 1] < 12))))
            stats[cam][k]['overlap'].append(float(np.linalg.norm(xy[0] - xy[1]) < 40))
print(f'frames {len(stats["cam0"]["knee"]["seen"])} with a person in the first {T_MAX:.0f} s   (fraction of frames)')
print('camera joint   seen  edge  overlap')
for cam in stats:
    for k in J:
        d = stats[cam][k]
        print(f'{cam:6} {k:6} {np.mean(d["seen"]):5.2f} {np.mean(d["edge"]) if d["edge"] else 0:5.2f} {np.mean(d["overlap"]) if d["overlap"] else 0:7.2f}')

# frames with flexion jumps
rows = list(csv.DictReader(open(os.path.join(S, 'angles.csv'))))
last, jumps = {}, []
for r in rows:
    t, s = float(r['t']), r['side']
    if t - float(rows[0]['t']) > T_MAX:
        break
    f = float(r['flexion_corrected_deg'] or r['flexion_filtered_deg'])
    if s in last and t - last[s][0] < 0.3 and abs(f - last[s][1]) > 15:
        jumps.append((t, s, last[s][1], f))
    last[s] = (t, f)
print(f'jumps > 15 deg: {len(jumps)}  (L {sum(j[1]=="L" for j in jumps)}, R {sum(j[1]=="R" for j in jumps)})')
vid = os.path.join(S, 'video.mp4')
if not os.path.exists(vid) or not jumps:
    sys.exit()
ts = np.array(ts)
pick = [jumps[i] for i in np.linspace(0, len(jumps) - 1, min(8, len(jumps))).astype(int)]
want = {int(np.argmin(np.abs(ts - j[0]))): j for j in pick}
cap, i, tiles = cv2.VideoCapture(vid), 0, []
while len(tiles) < len(want):
    ok, fr = cap.read()
    if not ok:
        break
    if i in want:
        t, s, a, b = want[i]
        fr = cv2.resize(fr, (fr.shape[1] // 2, fr.shape[0] // 2))
        cv2.putText(fr, f't{t - ts[0]:.1f}s {s} {a:.0f}->{b:.0f}', (4, fr.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(fr)
    i += 1
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
sheet = np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])
cv2.imwrite(os.path.join(S, 'jumps.png'), sheet)
print('contact sheet:', os.path.join(S, 'jumps.png'), sheet.shape)
