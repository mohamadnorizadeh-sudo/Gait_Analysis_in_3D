"""Per-session knee-angle CSV in Qatar time (v3.5.0).

One file per run in ~/gait_data/knee_angles/, named by the session's start in Qatar time:

    knee_angles_2026-11-24_12-30-05.csv

    actual_time_qatar,session_time_s,left_knee_deg,right_knee_deg
    2026-11-24 12:30:05.112,0.000,8.4,10.1
    2026-11-24 12:30:05.213,0.101,8.6,10.3

actual_time_qatar is the wall-clock time the camera frame was taken (Qatar is UTC+3 all year, no daylight saving),
session_time_s counts from the first frame of the run. The knee columns hold the angle shown on screen (learned
model when loaded, plane method otherwise); an empty cell means the leg was not measured in that frame.
"""
import csv
import os
import time
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
    QATAR = ZoneInfo('Asia/Qatar')
except Exception:                       # no tz database: Qatar has had a fixed UTC+3 offset since 1972
    QATAR = timezone(timedelta(hours=3), 'AST')

DEFAULT_DIR = os.path.expanduser('~/gait_data/knee_angles')
HEADER = ['actual_time_qatar', 'session_time_s', 'left_knee_deg', 'right_knee_deg']


def frame_wall_time(t_frame):
    """Wall-clock seconds for a frame stamped with CLOCK_MONOTONIC (the Pi camera source); falls back to now."""
    mono = time.clock_gettime(time.CLOCK_MONOTONIC)
    if abs(mono - t_frame) < 5.0:
        return time.time() - (mono - t_frame)
    return time.time()


class KneeLog:
    def __init__(self, folder=DEFAULT_DIR, start_wall=None):
        os.makedirs(folder, exist_ok=True)
        start = datetime.fromtimestamp(time.time() if start_wall is None else start_wall, QATAR)
        self.path = os.path.join(folder, 'knee_angles_%s.csv' % start.strftime('%Y-%m-%d_%H-%M-%S'))
        self.f = open(self.path, 'w', newline='', buffering=1)          # line-buffered: rows survive a crash
        self.w = csv.writer(self.f)
        self.w.writerow(HEADER)
        self.t0 = None

    def write(self, t_frame, left, right, wall=None):
        wall = frame_wall_time(t_frame) if wall is None else wall
        if self.t0 is None:
            self.t0 = t_frame
        stamp = datetime.fromtimestamp(wall, QATAR).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        fmt = lambda v: '' if v is None else '%.1f' % v
        self.w.writerow([stamp, '%.3f' % (t_frame - self.t0), fmt(left), fmt(right)])

    def close(self):
        self.f.close()
