import cv2, numpy as np, json, itertools
from validate_markers_detect import detect
c = cv2.VideoCapture('v.mov'); fps = c.get(5)
rows, i, prevA, prevAt = [], -1, None, -9
while True:
    ok, f = c.read(); i += 1
    if not ok: break
    if i % 2: continue
    t = i / fps
    cand = detect(f)
    strong = [(x, y) for x, y, a, s in cand if s]
    allc = [(x, y) for x, y, a, s in cand]
    best = None
    for H, K in itertools.permutations(strong, 2):
        d1 = np.hypot(K[0] - H[0], K[1] - H[1])
        if not (200 < d1 < 420) or K[1] - H[1] < 150:
            continue
        for A in allc:
            if A in (H, K): continue
            d2 = np.hypot(A[0] - K[0], A[1] - K[1])
            if not (220 < d2 < 450): continue
            s = abs(d1 - 290) + abs(d2 - 340)
            if prevA is not None and t - prevAt < 0.5:
                s += 2.0 * np.hypot(A[0] - prevA[0], A[1] - prevA[1])
            if best is None or s < best[0]:
                best = (s, H, K, A)
    if best is None:
        rows.append((t, None)); continue
    _, H, K, A = best; prevA, prevAt = A, t
    H, K, A = map(np.array, (H, K, A)); u, v = H - K, A - K
    rows.append((t, 180 - float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))),
                 [H.tolist(), K.tolist(), A.tolist()]))
json.dump(rows, open('marker_angles.json', 'w'))
v = [r for r in rows if r[1] is not None]
print('frames', len(rows), 'tracked', len(v))
for s in range(0, int(rows[-1][0]) + 1):
    b = [r[1] for r in v if s <= r[0] < s + 1]
    print(f'{s:3d}s', f'{np.median(b):6.1f}({len(b):2d})' if b else '     -    ', end=' |' if s % 6 != 5 else '\n')
