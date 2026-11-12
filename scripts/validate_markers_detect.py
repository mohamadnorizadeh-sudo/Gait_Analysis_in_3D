import cv2, numpy as np, sys
def detect(f, x0=1000):
    roi = f[:, x0:]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    m = ((g > 170) & (hsv[..., 1] < 70)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (12 <= a <= 900) or w > 45 or h > 45 or max(w, h) > 3 * min(w, h):
            continue
        y0, y1, xa, xb = max(0, y - 8), min(g.shape[0], y + h + 8), max(0, x - 8), min(g.shape[1], x + w + 8)
        patch = g[y0:y1, xa:xb].astype(float)
        mask = m[y0:y1, xa:xb] == 0
        ring = patch[mask]
        if ring.size and np.median(ring) < 90:
            out.append((cen[i][0] + x0, cen[i][1], a, float(np.median(ring))))
    return out
if __name__ == '__main__':
    c = cv2.VideoCapture('v.mov')
    for t in map(float, sys.argv[1:]):
        c.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = c.read()
        print(t, [(round(x), round(y), a, round(r)) for x, y, a, r in detect(f)])
