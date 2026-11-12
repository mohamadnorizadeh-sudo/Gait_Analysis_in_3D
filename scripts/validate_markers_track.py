import cv2, numpy as np, json, itertools
from validate_markers_detect import detect
c = cv2.VideoCapture('v.mov'); fps = c.get(5)
def pick(cand):
    best = None
    for tri in itertools.combinations(cand, 3):
        H = min(tri, key=lambda p: p[1])                     # hip = highest marker
        rest = [p for p in tri if p is not H]
        K, A = sorted(rest, key=lambda p: np.hypot(p[0] - H[0], p[1] - H[1]))   # knee = nearer to hip
        d1, d2 = np.hypot(K[0] - H[0], K[1] - H[1]), np.hypot(A[0] - K[0], A[1] - K[1])
        if not (200 < d1 < 420 and 220 < d2 < 450) or K[1] - H[1] < 150:
            continue
        s = abs(d1 - 290) + abs(d2 - 340)
        if best is None or s < best[0]:
            best = (s, (H, K, A))
    return None if best is None else best[1]
rows, i = [], -1
while True:
    ok, f = c.read(); i += 1
    if not ok: break
    if i % 2: continue
    t = i / fps
    cand = [(x, y) for x, y, a in detect(f) if a <= 650]
    tri = pick(cand)
    if tri is None:
        rows.append((t, None)); continue
    H, K, A = map(np.array, tri)
    u, v = H - K, A - K
    rows.append((t, 180 - float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))),
                 [list(map(float, p)) for p in tri]))
json.dump(rows, open('marker_angles3.json', 'w'))
v = [r for r in rows if r[1] is not None]
print('frames', len(rows), 'tracked', len(v))
for s in range(0, int(rows[-1][0]) + 1):
    b = [r[1] for r in v if s <= r[0] < s + 1]
    print(f'{s:3d}s', f'{np.median(b):6.1f}({len(b):2d})' if b else '     -    ', end=' |' if s % 6 != 5 else '\n')
