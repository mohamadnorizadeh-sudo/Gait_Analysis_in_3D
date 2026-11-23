#!/usr/bin/env python3
"""Far-view walking reference: MoveNet Thunder with a person-following crop (v3.1.9).

scripts/ref_movenet.py uses one fixed crop, fine for a sit-down in one place. When walking past the camera,
the person is small in a wide frame, so each frame is cropped around the previous frame's keypoints (square,
1.5 x the keypoint box, min 260 px) before the 256 x 256 model input. If the mean keypoint score drops below
0.25, the frame is also run on the full frame and the better result is kept.

Input: a video already reduced to e.g. 960x540 at 30 fps, for example:
  ffmpeg -i IMG_0017.MOV -vf fps=30,scale=960:540 -c:v libx264 -preset veryfast -crf 18 -an walk_540p30.mp4
Output CSV: t (s from the video start), flex_L, flex_R (image-plane knee flexion), conf_L, conf_R, hip_x (mid-hip
x in pixels, for pass detection); --kp-json keeps all 17 keypoints.

  python3 scripts/ref_movenet_track.py walk_540p30.mp4 ref.csv --model movenet_singlepose_thunder_4.onnx
Model source: https://github.com/Kazuhito00/MoveNet-Python-Example (onnx/), Google MoveNet, Apache-2.0.
"""
import argparse
import json

import cv2
import numpy as np


def flex(h, k, a):
    u, v = h - k, a - k
    c = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    return 180.0 - np.degrees(np.arccos(np.clip(c, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('out')
    ap.add_argument('--model', required=True)
    ap.add_argument('--kp-json', default='')
    a = ap.parse_args()
    import onnxruntime as ort
    sess = ort.InferenceSession(a.model, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    cap = cv2.VideoCapture(a.video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    def run(img, cx, cy, side):
        x0, y0, S = int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))
        crop = np.zeros((S, S, 3), np.uint8)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + S), min(H, y0 + S)
        if sx1 > sx0 and sy1 > sy0:
            crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
        inp = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (256, 256)).reshape(1, 256, 256, 3).astype('int32')
        k = sess.run(None, {name: inp})[0][0, 0]
        return np.c_[x0 + k[:, 1] * S, y0 + k[:, 0] * S, k[:, 2]]

    rows, kps, prev = [], [], None
    full = (W / 2, H / 2, float(max(W, H)))
    while True:
        ok, img = cap.read()
        if not ok:
            break
        box = full
        if prev is not None:
            m = prev[:, 2] > 0.3
            if m.sum() >= 6:
                xs, ys = prev[m, 0], prev[m, 1]
                box = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2,
                       float(np.clip(max(np.ptp(xs), np.ptp(ys)) * 1.5, 260, max(W, H))))
        kp = run(img, *box)
        if prev is None or kp[:, 2].mean() < 0.25:
            kp2 = run(img, *full)
            if kp2[:, 2].mean() > kp[:, 2].mean():
                kp = kp2
        prev = kp
        i = len(rows)
        rows.append((i / fps, flex(kp[11, :2], kp[13, :2], kp[15, :2]), flex(kp[12, :2], kp[14, :2], kp[16, :2]),
                     kp[[11, 13, 15], 2].min(), kp[[12, 14, 16], 2].min(), (kp[11, 0] + kp[12, 0]) / 2))
        if a.kp_json:
            kps.append(kp.round(2).tolist())
    with open(a.out, 'w') as f:
        f.write('t,flex_L,flex_R,conf_L,conf_R,hip_x\n')
        for r in rows:
            f.write('%.3f,%.1f,%.1f,%.2f,%.2f,%.1f\n' % r)
    if a.kp_json:
        json.dump({'fps': fps, 'W': W, 'H': H, 'kp': kps}, open(a.kp_json, 'w'))
    print('frames', len(rows), '->', a.out)


if __name__ == '__main__':
    main()
