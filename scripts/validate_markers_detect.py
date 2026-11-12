import cv2, numpy as np
def detect(f):
    """white tape squares on dark trousers: bright, unsaturated, compact blobs whose
    whole neighbourhood (+-22 px) is dark fabric"""
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    m = ((g > 160) & (hsv[..., 1] < 80)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (100 <= a <= 800) or max(w, h) > 2.2 * min(w, h) or a < 0.5 * w * h:
            continue
        P = 22
        y0, y1, xa, xb = max(0, y - P), min(g.shape[0], y + h + P), max(0, x - P), min(g.shape[1], x + w + P)
        ring = g[y0:y1, xa:xb][lab[y0:y1, xa:xb] != i]
        if ring.size and np.percentile(ring, 85) < 75:
            out.append((float(cen[i][0]), float(cen[i][1]), int(a)))
    return out
if __name__ == '__main__':
    import sys
    c = cv2.VideoCapture('v.mov')
    for t in map(float, sys.argv[1:]):
        c.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = c.read()
        print(t, [(round(x), round(y), a) for x, y, a in detect(f)])
