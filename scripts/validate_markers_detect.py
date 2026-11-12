import cv2, numpy as np
def detect(f):
    """returns (x, y, area, strong): strong = fully surrounded by dark fabric (hip/knee);
    weak = at least ~35 % dark surround (ankle marker sits at the trouser hem)"""
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    m = ((g > 160) & (hsv[..., 1] < 80)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (70 <= a <= 700) or max(w, h) > 2.2 * min(w, h) or a < 0.5 * w * h:
            continue
        cnt, _ = cv2.findContours((lab[y:y + h, x:x + w] == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnt:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(max(cnt, key=cv2.contourArea))
        fill, asp = a / max(1.0, rw * rh), max(rw, rh) / max(1.0, min(rw, rh))
        if fill < 0.82 or asp > (2.4 if fill >= 0.9 else 1.7):
            continue                      # tape is a solid square; wheel hubs / labels are round or long
        P = 22
        y0, y1, xa, xb = max(0, y - P), min(g.shape[0], y + h + P), max(0, x - P), min(g.shape[1], x + w + P)
        ring = g[y0:y1, xa:xb][lab[y0:y1, xa:xb] != i]
        if not ring.size:
            continue
        strong = np.percentile(ring, 85) < 75
        dark = np.mean(ring < 70)
        # tight ring (6 px) must be dark on most sides: tape on fabric, not a bright patch on a bright floor
        P2 = 6
        y0, y1, xa, xb = max(0, y - P2), min(g.shape[0], y + h + P2), max(0, x - P2), min(g.shape[1], x + w + P2)
        tight = g[y0:y1, xa:xb][lab[y0:y1, xa:xb] != i]
        if strong or (dark > 0.35 and np.mean(tight < 90) > 0.45):
            out.append((float(cen[i][0]), float(cen[i][1]), int(a), bool(strong)))
    return out
