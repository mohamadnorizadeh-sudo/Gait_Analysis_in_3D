import json, urllib.request
import numpy as np
from gait3d.dashboard import LiveData, WebDashboard, render_screen, report_html


def fill(d, secs=20.0, fps=12.0):
    t = 0.0
    while t < secs:
        d.push(t, {"L": 30 + 30 * np.sin(2 * np.pi * 0.9 * t), "R": 25 + 25 * np.sin(2 * np.pi * 0.9 * t + 3)}, "3D rigid lock", 12.0,
               np.zeros((640, 360, 3), np.uint8))
        t += 1 / fps


def test_summary_and_steps():
    d = LiveData(); fill(d)
    s = d.summary()
    assert 15 <= s["legs"]["L"]["steps"] <= 19          # 0.9 Hz for 20 s -> 18 swing peaks
    assert abs(s["legs"]["L"]["flexion_max_p95"] - 59) < 3
    assert "symmetry" in s and "Left knee" in report_html(s)


def test_screen_and_web():
    d = LiveData(); fill(d, 5)
    img = render_screen(np.zeros((640, 360, 3), np.uint8), d)
    assert img.shape[0] >= 480 and img.shape[1] == 360 + 520
    w = WebDashboard(d, 0).start()
    port = w.server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    assert b"live knee flexion" in urllib.request.urlopen(base + "/").read()
    live = json.loads(urllib.request.urlopen(base + "/api/live?since=-1").read())
    assert len(live["t"]) > 10 and live["L"][0] is not None
    assert urllib.request.urlopen(base + "/snapshot.jpg").read()[:2] == b"\xff\xd8"
    w.stop()
