#!/usr/bin/env python3
"""Marker-free reference knee angle from a FAR side-view phone/tablet video (v3.1.6).

Why: cardboard markers on clothing do not follow the bones (19 Nov: the phone videos' own markers read ~55 deg
when the user sat at ~95 deg), so they cannot validate the Pi.  A far camera (2-4 m, side-on, whole body in
view) has almost no perspective error, and a standard single-person pose model on it gives the sagittal knee
angle to a few degrees.  This script runs MoveNet SinglePose Thunder v4 (ONNX, onnxruntime, CPU) on frames and
writes the image-plane knee flexion of both legs.

Model: movenet_singlepose_thunder_4.onnx from https://github.com/Kazuhito00/MoveNet-Python-Example (onnx/),
a conversion of Google's MoveNet (Apache-2.0).  It is not stored in this repo.

Input: a video (read with ffmpeg at --fps) or a folder of frames (sorted).  --crop x,y,w,h (pixels of the
source) selects a square-ish region around the walking/sitting area; frames are resized to 256x256.
Output CSV: t (s from --t0 of the video), flex_L, flex_R, conf_L, conf_R (conf = min of hip/knee/ankle score).
The 17 keypoints are also saved (--kp-json) so frames can be re-checked.

Usage:
  python3 scripts/ref_movenet.py IMG_0012.MOV out.csv --model movenet_singlepose_thunder_4.onnx \
      --t0 18 --dur 254 --fps 4 --crop 360,0,1080,1080
  python3 scripts/ref_movenet.py frames_dir/ out.csv --model ... --fps 4
"""
import argparse
import glob
import json
import os
import subprocess

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('src')
ap.add_argument('out')
ap.add_argument('--model', required=True)
ap.add_argument('--fps', type=float, default=4.0)
ap.add_argument('--t0', type=float, default=0.0)
ap.add_argument('--dur', type=float, default=0.0)
ap.add_argument('--crop', default='', help='x,y,w,h in source pixels')
ap.add_argument('--kp-json', default='')
a = ap.parse_args()

import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(a.model, providers=['CPUExecutionProvider'])
IN = sess.get_inputs()[0].name


def frames():
    if os.path.isdir(a.src):
        for f in sorted(glob.glob(os.path.join(a.src, '*.jpg')) + glob.glob(os.path.join(a.src, '*.png'))):
            yield cv2.imread(f)
        return
    vf = ['fps=%g' % a.fps]
    if a.crop:
        x, y, w, h = a.crop.split(',')
        vf.append('crop=%s:%s:%s:%s' % (w, h, x, y))
    vf.append('scale=256:256')
    cmd = ['ffmpeg', '-v', 'error', '-ss', str(a.t0)] + (['-t', str(a.dur)] if a.dur else []) + \
          ['-i', a.src, '-vf', ','.join(vf), '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(256 * 256 * 3)
        if len(b) < 256 * 256 * 3:
            break
        yield np.frombuffer(b, np.uint8).reshape(256, 256, 3)


def flex(h, k, an):
    u, v = h - k, an - k
    c = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    return 180.0 - np.degrees(np.arccos(np.clip(c, -1, 1)))


rows, kps = [], []
for i, im in enumerate(frames()):
    h, w = im.shape[:2]
    x = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (256, 256)).reshape(1, 256, 256, 3).astype('int32')
    k = sess.run(None, {IN: x})[0][0, 0]                       # 17 x (y, x, score), normalised
    xy = np.c_[k[:, 1] * w, k[:, 0] * h]
    r = [i / a.fps]
    for hip, knee, ank in ((11, 13, 15), (12, 14, 16)):
        r.append(flex(xy[hip], xy[knee], xy[ank]))
    r += [float(k[[11, 13, 15], 2].min()), float(k[[12, 14, 16], 2].min())]
    rows.append(r)
    kps.append(np.c_[xy, k[:, 2]].round(2).tolist())
with open(a.out, 'w') as f:
    f.write('t,flex_L,flex_R,conf_L,conf_R\n')
    for r in rows:
        f.write('%.2f,%.1f,%.1f,%.2f,%.2f\n' % tuple(r))
if a.kp_json:
    json.dump(kps, open(a.kp_json, 'w'))
print('frames', len(rows), '->', a.out)
