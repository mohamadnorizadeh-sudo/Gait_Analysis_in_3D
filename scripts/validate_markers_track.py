import cv2, numpy as np, json
from validate_markers_detect import detect
c = cv2.VideoCapture('v.mov'); fps = c.get(5)
prev, rows, i = None, [], -1
while True:
    ok, f = c.read(); i += 1
    if not ok: break
    if i % 2: continue
    cand = [(x, y) for x, y, a, r in detect(f, 1250) if 150 <= a <= 650 and r < 65]
    t = i / fps
    if prev is None:
        if len(cand) == 3:
            prev = sorted(cand, key=lambda p: p[1])
        else:
            rows.append((t, None)); continue
    cur, okall = [], True
    for p in prev:
        if not cand: okall = False; cur.append(p); continue
        d = [np.hypot(x - p[0], y - p[1]) for x, y in cand]
        j = int(np.argmin(d))
        if d[j] > 120: okall = False; cur.append(p)
        else: cur.append(cand.pop(j))
    if okall:
        prev = cur
        H, K, A = map(np.array, cur)
        u, v = H - K, A - K
        ang = np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
        rows.append((t, 180 - ang, [list(map(float, p)) for p in cur]))
    else:
        rows.append((t, None))
json.dump(rows, open('marker_angles.json', 'w'))
v = [r for r in rows if r[1] is not None]
print('frames', len(rows), 'tracked', len(v), 'fps', fps)
for s in range(0, int(rows[-1][0]) + 1, 1):
    b = [r[1] for r in v if s <= r[0] < s + 1]
    print(f'{s:3d}s', f'{np.median(b):6.1f}' if b else '   -', end=' |' if s % 6 != 5 else '\n')
