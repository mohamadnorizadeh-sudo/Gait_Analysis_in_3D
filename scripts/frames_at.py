#!/usr/bin/env python3
"""Contact sheet of session video frames at given times (s from start), optionally shown on the Pi screen.
  python3 scripts/frames_at.py runs/<session> 10 40 57 63 69 72 75 81 [--show]"""
import json, os, sys
import cv2, numpy as np
S = sys.argv[1]
show = '--show' in sys.argv
want = [float(x) for x in sys.argv[2:] if x != '--show']
ts = np.array([json.loads(l)['t'] for l in open(os.path.join(S, 'keypoints.jsonl'))])
ts -= ts[0]
idx = {int(np.argmin(np.abs(ts - w))): w for w in want}
cap, i, tiles = cv2.VideoCapture(os.path.join(S, 'video.mp4')), 0, {}
while len(tiles) < len(idx):
    ok, f = cap.read()
    if not ok:
        break
    if i in idx:
        f = cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2))
        cv2.putText(f, f't {idx[i]:.0f}s', (5, f.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles[idx[i]] = f
    i += 1
T = [tiles[w] for w in want if w in tiles]
while len(T) % 4:
    T.append(np.zeros_like(T[0]))
sheet = np.vstack([np.hstack(T[k:k + 4]) for k in range(0, len(T), 4)])
out = os.path.join(S, 'frames.png')
cv2.imwrite(out, sheet)
print('saved', out, sheet.shape)
if show:
    cv2.imshow('session frames', cv2.resize(sheet, None, fx=1.4, fy=1.4))
    cv2.waitKey(120000)
