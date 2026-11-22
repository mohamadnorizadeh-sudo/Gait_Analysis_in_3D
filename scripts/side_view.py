#!/usr/bin/env python3
"""Live side-by-side view of both cameras (v2.20, side-camera setup; aiming aid v3.2.4).

Shows cam0 | cam1 next to each other on the Pi screen with the leg keypoints drawn,
prints once a second whether hip, knee and ankle of each leg are inside each image,
and saves the latest view to runs/side_view.jpg.

  DISPLAY=:0 python3 scripts/side_view.py [--rot0 0] [--rot1 0] [--no-pose]

--rot0/--rot1: extra rotation (0/90/180/270) applied after camera_shm_writer's own rotation.
Stop the tracker first (the AI HAT is used here).
"""
import argparse, os, sys, time
import cv2
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.camera import ShmStereoSource
from gait3d.pose import best_person

ap = argparse.ArgumentParser()
ap.add_argument('--rot0', type=int, default=0)
ap.add_argument('--rot1', type=int, default=0)
ap.add_argument('--no-pose', action='store_true')
ap.add_argument('--hef', default='/usr/share/hailo-models/yolov8m_pose_h10.hef')
ap.add_argument('--seconds', type=float, default=1800)
ap.add_argument('--record', default='', help='folder: save raw cam0|cam1 video + keypoints.jsonl')
a = ap.parse_args()

ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
rot = lambda im, r: im if ROT[r % 360] is None else cv2.rotate(im, ROT[r % 360])
hp = None
if not a.no_pose:
    from gait3d.hailo_pose import HailoPoseEstimator
    hp = HailoPoseEstimator(a.hef)
JOINTS = {'L': (11, 13, 15), 'R': (12, 14, 16)}
COL = {'L': (255, 160, 0), 'R': (0, 120, 255)}


def annotate(img, name, people_out=None):
    status = name
    if hp is not None:
        people = hp([img])[0]
        if people_out is not None:
            people_out.extend(people)
        p = best_person(people)
        if p is None:
            status += ' no person'
        else:
            h, w = img.shape[:2]
            for side, ix in JOINTS.items():
                pts = [(int(p.xy[j][0]), int(p.xy[j][1])) for j in ix]
                ok = [p.conf[j] >= 0.5 and 8 < pts[k][0] < w - 8 and 8 < pts[k][1] < h - 8 for k, j in enumerate(ix)]
                for k in range(2):
                    if ok[k] and ok[k + 1]:
                        cv2.line(img, pts[k], pts[k + 1], COL[side], 3)
                for k, pt in enumerate(pts):
                    if ok[k]:
                        cv2.circle(img, pt, 6, COL[side], -1)
                status += f" {side}:" + ''.join('HKA'[k] if ok[k] else '-' for k in range(3))
                status += '(%.2f y%s)' % (min(p.conf[j] for j in ix), '/'.join('%.0f' % (100 * p.xy[j][1] / h) for j in ix))
        if p is not None:
            status += aim(img, name, p)
    cv2.putText(img, name, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return status


def aim(img, name, p):
    """v3.2.4 aiming aid: how side-on this camera sees its leg (cam0 -> left, cam1 -> right).

    In a true side view the two legs overlap in the image; seen from the front they are a hip-width apart. The
    mean horizontal gap between the left and right hip, knee and ankle, relative to the near leg's length (~85 cm),
    gives roughly the angle off a side view: asin(gap * 85 / 25 cm). Drawn large on the image and printed."""
    h, w = img.shape[:2]
    near = 'L' if name == 'cam0' else 'R'
    xy, c = np.asarray(p.xy, float), np.asarray(p.conf, float)
    ix = JOINTS[near]
    leg = np.linalg.norm(xy[ix[0]] - xy[ix[1]]) + np.linalg.norm(xy[ix[2]] - xy[ix[1]])
    inview = all(c[j] >= 0.5 and 4 < xy[j][0] < w - 4 and 4 < xy[j][1] < h - 4 for j in ix)
    pairs = [(a_, b_) for a_, b_ in zip(JOINTS['L'], JOINTS['R']) if c[a_] >= 0.3 and c[b_] >= 0.3]
    if leg < 40 or not pairs:
        msg, ang = 'aim: leg not found', None
    else:
        gap = float(np.mean([abs(xy[a_][0] - xy[b_][0]) for a_, b_ in pairs])) / leg
        ang = float(np.degrees(np.arcsin(min(1.0, gap * 85.0 / 25.0))))
        msg = 'off side ~%2.0f deg  leg %2.0f%% of height%s' % (ang, 100 * leg / h, '' if inview else '  HIP/KNEE/ANKLE NOT ALL IN VIEW')
    col = (0, 200, 0) if ang is not None and ang < 20 and inview else (0, 200, 255) if ang is not None and ang < 35 else (0, 0, 255)
    cv2.rectangle(img, (0, h - 34), (w, h), (0, 0, 0), -1)
    cv2.putText(img, msg[:44], (6, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    return ' AIM[%s]' % msg


last_print, t_end = 0.0, time.time() + a.seconds
out = os.path.join(os.path.dirname(__file__), '..', 'runs', 'side_view.jpg')
import json, signal
STOP = []
signal.signal(signal.SIGTERM, lambda *_: STOP.append(1))
signal.signal(signal.SIGINT, lambda *_: STOP.append(1))
rec_v = rec_k = None
if a.record:
    os.makedirs(a.record, exist_ok=True)
    rec_k = open(os.path.join(a.record, 'keypoints.jsonl'), 'w')
pj = lambda ps: [{'xy': [[round(float(x), 1), round(float(y), 1)] for x, y in p.xy], 'c': [round(float(c), 3) for c in p.conf], 's': round(float(p.score), 3)} for p in ps]
with ShmStereoSource() as src:
    while time.time() < t_end and not STOP:
        t, i0, i1, _ = src.read()
        i0, i1 = rot(i0, a.rot0), rot(i1, a.rot1)
        if a.record:
            raw = np.hstack([i0, i1])
            if rec_v is None:
                rec_v = cv2.VideoWriter(os.path.join(a.record, 'raw.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (raw.shape[1], raw.shape[0]))
            rec_v.write(raw)
        p0, p1 = [], []
        s0, s1 = annotate(i0, 'cam0', p0), annotate(i1, 'cam1', p1)
        if rec_k is not None:
            rec_k.write(json.dumps({'t': round(t, 4), 'cam0': pj(p0), 'cam1': pj(p1)}) + '\n')
        h = max(i0.shape[0], i1.shape[0])
        pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        view = np.hstack([pad(i0), np.full((h, 6, 3), 255, np.uint8), pad(i1)])
        if os.environ.get('DISPLAY'):
            cv2.imshow('side view: cam0 | cam1', cv2.resize(view, None, fx=1.4, fy=1.4))
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
        if time.time() - last_print > 1.0:
            last_print = time.time()
            print(f'{s0} | {s1}', flush=True)
            cv2.imwrite(out, view)
if rec_v is not None:
    rec_v.release()
if rec_k is not None:
    rec_k.close()
if hp is not None:
    hp.close()
