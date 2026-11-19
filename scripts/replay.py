#!/usr/bin/env python3
"""Replay a keypoint recording through the tracker with chosen settings.

  python3 scripts/replay.py runs/live10/keypoints.jsonl --calib stereo_calibration.npz \
      --thigh-mm 420 --shin-mm 410 --dir-min-cutoff 1.0 --dir-beta 2.0 --csv /tmp/replay.csv
Then: python3 scripts/analyze_session.py /tmp/replay.csv ; python3 scripts/timeline.py /tmp/replay.csv 2
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gait3d.profile import SubjectProfile
from gait3d.recorder import read_recording
from gait3d.stereo_geometry import load_geometry
from gait3d.tracker import CsvLogger, GaitTracker, TrackerConfig

ap = argparse.ArgumentParser()
ap.add_argument('recording')
ap.add_argument('--calib', default='stereo_calibration.npz')
ap.add_argument('--thigh-mm', type=float, required=True)
ap.add_argument('--shin-mm', type=float, required=True)
ap.add_argument('--csv', default='/tmp/replay.csv')
for k, v in vars(TrackerConfig()).items():
    if isinstance(v, (int, float)):
        ap.add_argument('--' + k.replace('_', '-'), type=float, default=v)
a = ap.parse_args()
cfg = TrackerConfig(**{k: getattr(a, k) for k, v in vars(TrackerConfig()).items() if isinstance(v, (int, float))})
tr = GaitTracker(load_geometry(a.calib), SubjectProfile.manual(a.thigh_mm, a.shin_mm), cfg)
log = CsvLogger(a.csv)
n = 0
for t, pl, pr, _ in read_recording(a.recording):
    log.log(t, tr.process(t, pl, pr)); n += 1
log.close()
print(f'replayed {n} frames -> {a.csv}')
