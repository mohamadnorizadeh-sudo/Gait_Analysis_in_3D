#!/usr/bin/env python3
"""v3.0 live session with the two SIDE cameras (cam0 = left leg, cam1 = right leg); learned angle v3.3.1.

Pi screen: cam0 | cam1 with hip-knee-ankle drawn + live knee-angle graphs.
Web: same dashboard as before (http://<pi>:8080).  Records raw video, all keypoints and angles.

  python3 scripts/07_side_session.py --out runs/side_session_X [--markers] [--seconds 600]

Stand still and straight for ~3 s in the first 10 s (sets 0 deg), then work as usual.
Stop with Ctrl+C / SIGTERM / q in the window (files are closed cleanly).
"""
import argparse, csv, json, os, signal, sys, time
import cv2
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.camera import ShmStereoSource
from gait3d.dashboard import LiveData, WebDashboard, render_screen
from gait3d.side2d import JOINTS, PlaneMap, SideConfig, SideTracker

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=time.strftime('runs/side_session_%Y%m%d_%H%M%S'))
ap.add_argument('--hef', default='/usr/share/hailo-models/yolov8m_pose_h10.hef')
ap.add_argument('--planes', default='config/side_planes.json')
ap.add_argument('--markers', action='store_true', help='snap hip/knee/ankle to white tape markers')
ap.add_argument('--marker-joints', default='HKA')
ap.add_argument('--marker-radius', type=float, default=50.0)
ap.add_argument('--marker-vmin', type=int, default=200)
ap.add_argument('--marker-color', default='white', choices=['white', 'tan'])
ap.add_argument('--shank-seed', type=float, default=0.7, help='shin tape search start: fraction knee->ankle')
ap.add_argument('--source', default='L=cam0,R=cam1', help='which camera measures which leg')
ap.add_argument('--rot0', type=int, default=0)
ap.add_argument('--rot1', type=int, default=0)
ap.add_argument('--seconds', type=float, default=3600)
ap.add_argument('--web-port', type=int, default=8080)
ap.add_argument('--headless', action='store_true')
ap.add_argument('--no-video', action='store_true')
ap.add_argument('--model', default='config/angle_model.pkl',
                help='v3.3.1 learned angle model from train_angle_model.py; shown when present, "" = off')
a = ap.parse_args()

from gait3d.hailo_pose import HailoPoseEstimator
hp = HailoPoseEstimator(a.hef)
ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
rot = lambda im, r: im if ROT[r % 360] is None else cv2.rotate(im, ROT[r % 360])
src_map = dict(kv.split('=') for kv in a.source.split(','))
planes = PlaneMap.load(a.planes)
tracker = SideTracker(planes, SideConfig(source=src_map, markers=a.markers, marker_joints=a.marker_joints,
                                             shank_seed=a.shank_seed,
                                             marker_radius_px=a.marker_radius, marker_v_min=a.marker_vmin,
                                             marker_color=a.marker_color))
model = None
if a.model and os.path.exists(a.model):
    from gait3d.angle_model import AngleModel
    try:
        model = AngleModel(a.model)
    except Exception as e:                                  # wrong feature version, missing file parts, ...
        print('[side] angle model NOT used:', e, flush=True)
os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, 'config.json'), 'w') as f:
    json.dump({'source': src_map, 'markers': a.markers, 'planes': sorted(planes.planes), 'started': time.ctime(),
               'model': (a.model + ' (' + model.kind + ')') if model else None}, f)
live = LiveData()
live.csv_path = os.path.join(a.out, 'angles.csv')
web = WebDashboard(live, a.web_port).start() if a.web_port else None
fcsv = open(live.csv_path, 'w', newline='')
w = csv.writer(fcsv)
w.writerow(['t', 'L_flexion', 'R_flexion', 'L_raw', 'R_raw', 'L_status', 'R_status', 'L_model', 'R_model'])
fkp = open(os.path.join(a.out, 'keypoints.jsonl'), 'w')
vid = None
STOP = []
signal.signal(signal.SIGTERM, lambda *_: STOP.append(1))
signal.signal(signal.SIGINT, lambda *_: STOP.append(1))
pj = lambda ps: [{'xy': [[round(float(x), 1), round(float(y), 1)] for x, y in p.xy],
                  'c': [round(float(c), 3) for c in p.conf], 's': round(float(p.score), 3)} for p in ps]
COL = {'L': (255, 170, 0), 'R': (0, 140, 255)}
t_end, n, t_fps, fps = time.time() + a.seconds, 0, time.time(), 0.0
print('[side] planes:', sorted(planes.planes) or 'NONE (uncalibrated pixels)', '| markers' if a.markers else '',
      '| model: %s (%s)' % (a.model, model.kind) if model else '| model: none (plane method)', flush=True)
try:
    with ShmStereoSource() as src:
        while time.time() < t_end and not STOP:
            t, i0, i1, _ = src.read()
            imgs = {'cam0': rot(i0, a.rot0), 'cam1': rot(i1, a.rot1)}
            raw = np.hstack([imgs['cam0'], imgs['cam1']])
            if not a.no_video:
                if vid is None:
                    vid = cv2.VideoWriter(os.path.join(a.out, 'raw.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10,
                                          (raw.shape[1], raw.shape[0]))
                vid.write(raw)
            people = {c: hp([im])[0] for c, im in imgs.items()}
            fkp.write(json.dumps({'t': round(t, 4), 'cam0': pj(people['cam0']), 'cam1': pj(people['cam1'])}) + '\n')
            out = tracker.process(t, people, imgs)
            ml = dict(zip('LR', model.push(people))) if model else {'L': None, 'R': None}
            w.writerow([f'{t:.4f}'] + [('' if out[s].flexion is None else f'{out[s].flexion:.1f}') for s in 'LR']
                       + [('' if out[s].raw is None else f'{out[s].raw:.1f}') for s in 'LR'] + [out[s].status for s in 'LR']
                       + [('' if ml[s] is None else f'{ml[s]:.1f}') for s in 'LR'])
            shown = ml if model else {s: out[s].flexion for s in 'LR'}
            vis = raw.copy()
            for s in 'LR':
                o, cam = out[s], src_map[s]
                if o.pts_px is None:
                    continue
                dx = 0 if cam == 'cam0' else imgs['cam0'].shape[1]
                p = [(int(x + dx), int(y)) for x, y in o.pts_px]
                cv2.line(vis, p[0], p[1], COL[s], 3)
                cv2.line(vis, p[1], p[2], COL[s], 3)
                for q in p:
                    cv2.circle(vis, q, 6, COL[s], -1)
                txt = '--' if shown[s] is None else f'{shown[s]:.0f}'
                cv2.putText(vis, f'{s} {txt}' + (' ML' if model else ''), (dx + 8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            COL[s], 2)
            n += 1
            if time.time() - t_fps > 2:
                fps, n, t_fps = n / (time.time() - t_fps), 0, time.time()
            status = ('stand still to set 0' if tracker.offset is None else 'tracking') + \
                     f" | L {out['L'].status} | R {out['R'].status}"
            live.push(t, {s: shown[s] for s in 'LR' if shown[s] is not None}, status, fps, vis)
            if not a.headless and os.environ.get('DISPLAY'):
                cv2.imshow('Gait - side cameras', render_screen(vis, live))
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
finally:
    fcsv.close()
    fkp.close()
    if vid is not None:
        vid.release()
    hp.close()
    if web:
        web.stop()
    print('[side] saved', a.out, '| offset', tracker.offset, flush=True)
