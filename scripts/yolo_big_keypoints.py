#!/usr/bin/env python3
"""Run a large Ultralytics YOLO pose model (default YOLOv8x-pose) on a recorded video and write keypoints in the
same JSON-lines shape as the Pi's keypoints.jsonl (v3.3.3).

Why: the Pi runs YOLOv8m-pose on the AI HAT. The largest YOLOv8 pose model (x: 69.2 vs 65.0 COCO pose mAP, ~3x the
computation) is not compiled for the HAT. This runs it offline on a laptop to see whether a bigger pose model sees
more of the knee bend, (a) on the Pi's own recorded video (raw.mp4 = cam0 | cam1 side by side), and (b) on the
iPad reference video, where it may give cleaner reference angles than MoveNet.

Run on the Mac (needs internet once for the package and the model weights):
  pip install ultralytics
  python3 yolo_big_keypoints.py pi_walk_1301.mp4 kp_pi_yolov8x.jsonl              # Pi video: two cameras
  python3 yolo_big_keypoints.py walk_0018_540p30.mp4 kp_ipad_yolov8x.jsonl --split 1   # iPad video: one view

Each output line: {"i": frame index, "cam0": [people], "cam1": [people]} (one view -> only "cam0"), each person
{"xy": 17 x [x, y] pixels, "c": 17 confidences, "s": box score}. Frame i of the Pi video is line i of the Pi's
keypoints.jsonl, so the Pi time stamps can be attached later.

Apple GPU: Ultralytics has warned of an MPS bug for pose models, so the first 5 frames are run on both 'mps' and
'cpu' and compared; 'mps' is used only if they agree within 2 px.
"""
import argparse
import json
import time

import cv2
import numpy as np


def people(r):
    out = []
    if r.keypoints is None or len(r.keypoints) == 0:
        return out
    xy = r.keypoints.xy.cpu().numpy()
    c = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(xy.shape[:2])
    s = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.ones(len(xy))
    for k in range(len(xy)):
        out.append({'xy': [[round(float(x), 1), round(float(y), 1)] for x, y in xy[k]],
                    'c': [round(float(v), 3) for v in c[k]], 's': round(float(s[k]), 3)})
    return out


def views(im, split):
    w = im.shape[1] // split
    return [im[:, k * w:(k + 1) * w] for k in range(split)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('out')
    ap.add_argument('--model', default='yolov8x-pose.pt')
    ap.add_argument('--split', type=int, default=2, help='2 = Pi video (cam0 | cam1), 1 = one view (iPad)')
    ap.add_argument('--device', default='auto', help='auto | mps | cpu | cuda')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--fps-out', type=float, default=0.0,
                    help='v3.3.5: keep about this many frames per second (e.g. 30 for a 60 fps iPad video); 0 = all')
    a = ap.parse_args()
    import torch
    from ultralytics import YOLO
    model = YOLO(a.model)
    cap = cv2.VideoCapture(a.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = fps_in / a.fps_out if a.fps_out and a.fps_out < fps_in else 1.0
    cams = ['cam0', 'cam1'][:a.split]

    dev = a.device
    if dev == 'auto':
        dev = 'cpu'
        if torch.backends.mps.is_available():
            test = []
            for _ in range(5):
                ok, im = cap.read()
                if ok:
                    test.append(im)
            worst = 0.0
            for im in test:
                for v in views(im, a.split):
                    pm = people(model.predict(v, imgsz=a.imgsz, conf=a.conf, device='mps', verbose=False)[0])
                    pc = people(model.predict(v, imgsz=a.imgsz, conf=a.conf, device='cpu', verbose=False)[0])
                    if pm and pc:
                        worst = max(worst, float(np.max(np.abs(np.array(pm[0]['xy']) - np.array(pc[0]['xy'])))))
                    elif bool(pm) != bool(pc):
                        worst = 1e9
            dev = 'mps' if worst <= 2.0 else 'cpu'
            print(f'Apple GPU vs CPU on 5 frames: max difference {worst:.2f} px -> using {dev}', flush=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f'{a.model} on {a.video}: {n} frames x {a.split} view(s), device {dev}', flush=True)

    t0, i, k, nxt = time.time(), 0, -1, 0.0
    n_out = int(n / step) if n > 0 else 0
    with open(a.out, 'w') as f:
        while True:
            ok, im = cap.read()
            if not ok:
                break
            k += 1
            if k + 1e-6 < nxt:
                continue
            nxt += step
            res = model.predict(views(im, a.split), imgsz=a.imgsz, conf=a.conf, device=dev, verbose=False)
            rec = {'i': i, 't': round(k / fps_in, 4)}
            if i == 0:
                rec.update({'W': W, 'H': H, 'fps_in': fps_in})
            for cam, r in zip(cams, res):
                rec[cam] = people(r)
            f.write(json.dumps(rec) + '\n')
            i += 1
            if i % 100 == 0:
                dt = (time.time() - t0) / i
                print(f'{i}/{n_out} frames, {dt:.2f} s per frame, about {dt * (n_out - i) / 60:.0f} min left', flush=True)
    print(f'done: {i} frames -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
