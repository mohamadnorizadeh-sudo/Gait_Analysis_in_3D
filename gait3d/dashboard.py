"""Doctor-facing live display (v2.7): Pi screen panel + web page.

* LiveData        thread-safe store of the corrected knee flexion per leg,
                  session statistics, step peaks and the latest overlay frame.
* render_screen   OpenCV canvas for the Pi screen: camera overlay on the left,
                  live LEFT / RIGHT knee-flexion graphs and big numbers on the right.
* WebDashboard    stdlib HTTP server (no extra packages): live graphs in any
                  browser on the network, camera snapshot, CSV download and a
                  printable session report (browser "Print -> Save as PDF").
"""

from __future__ import annotations

import io
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

SIDES = ("L", "R")
COLORS_BGR = {"L": (255, 170, 0), "R": (0, 140, 255)}      # blue-ish / orange
COLORS_CSS = {"L": "#00aaff", "R": "#ff8c00"}


class LiveData:
    def __init__(self, window_s=30.0, step_min_peak=25.0, step_min_gap_s=0.6):
        self.lock = threading.Lock()
        self.window_s = window_s
        self.t0 = None
        self.hist = deque()                      # (t_rel, {side: flexion or None})
        self.all = {s: [] for s in SIDES}        # full session (t_rel, flexion)
        self.status = ""
        self.fps = 0.0
        self.jpeg = None
        self.csv_path = None
        self.subject = {}
        self.step_min_peak, self.step_min_gap_s = step_min_peak, step_min_gap_s

    def reset(self):
        with self.lock:
            self.t0, self.hist = None, deque()
            self.all = {s: [] for s in SIDES}

    def push(self, t, flexion: dict, status="", fps=0.0, frame=None):
        with self.lock:
            if self.t0 is None:
                self.t0 = t
            tr = t - self.t0
            vals = {s: (float(flexion[s]) if s in flexion else None) for s in SIDES}
            self.hist.append((tr, vals))
            while self.hist and tr - self.hist[0][0] > self.window_s:
                self.hist.popleft()
            for s, v in vals.items():
                if v is not None:
                    self.all[s].append((tr, v))
            self.status, self.fps = status, fps
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self.jpeg = buf.tobytes()

    def live(self, since=-1.0):
        with self.lock:
            pts = [(t, v) for t, v in self.hist if t > since]
            return {"t": [round(t, 3) for t, _ in pts],
                    "L": [None if v["L"] is None else round(v["L"], 1) for _, v in pts],
                    "R": [None if v["R"] is None else round(v["R"], 1) for _, v in pts],
                    "status": self.status, "fps": round(self.fps, 1),
                    "now": round(self.hist[-1][0], 3) if self.hist else 0.0}

    def _peaks(self, series):
        """Swing-phase peaks: local maxima above step_min_peak, >= step_min_gap_s apart."""
        if len(series) < 5:
            return []
        t = np.array([p[0] for p in series]); v = np.array([p[1] for p in series])
        peaks, last = [], -1e9
        for i in range(2, len(v) - 2):
            if v[i] >= self.step_min_peak and v[i] == v[max(0, i - 4):i + 5].max() and t[i] - last >= self.step_min_gap_s:
                peaks.append((float(t[i]), float(v[i])))
                last = t[i]
        return peaks

    def summary(self):
        with self.lock:
            out = {"duration_s": round(self.hist[-1][0], 1) if self.hist else 0.0, "subject": self.subject, "legs": {}}
            for s in SIDES:
                v = np.array([p[1] for p in self.all[s]])
                if v.size < 5:
                    out["legs"][s] = None
                    continue
                pk = self._peaks(self.all[s])
                out["legs"][s] = {
                    "samples": int(v.size),
                    "flexion_mean": round(float(v.mean()), 1),
                    "flexion_min_p5": round(float(np.percentile(v, 5)), 1),
                    "flexion_max_p95": round(float(np.percentile(v, 95)), 1),
                    "rom_p5_p95": round(float(np.percentile(v, 95) - np.percentile(v, 5)), 1),
                    "peak_flexion_max": round(float(v.max()), 1),
                    "steps": len(pk),
                    "mean_step_peak": round(float(np.mean([p[1] for p in pk])), 1) if pk else None,
                }
            L, R = out["legs"].get("L"), out["legs"].get("R")
            if L and R:
                out["symmetry"] = {"rom_diff_L_minus_R": round(L["rom_p5_p95"] - R["rom_p5_p95"], 1),
                                   "peak_diff_L_minus_R": round(L["flexion_max_p95"] - R["flexion_max_p95"], 1)}
            return out


# ---------------------------------------------------------------------- #
# Pi screen
# ---------------------------------------------------------------------- #
def _graph(img, x0, y0, w, h, data: LiveData, side, ymax=120.0):
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (40, 40, 40), -1)
    for a in range(0, int(ymax) + 1, 30):
        y = int(y0 + h - a / ymax * h)
        cv2.line(img, (x0, y), (x0 + w, y), (70, 70, 70), 1)
        cv2.putText(img, f"{a}", (x0 + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    with data.lock:
        pts = [(t, v[side]) for t, v in data.hist]
    if not pts:
        return None
    tnow = pts[-1][0]
    prev, last = None, None
    for t, v in pts:
        if v is None:
            prev = None
            continue
        p = (int(x0 + w - (tnow - t) / data.window_s * w), int(y0 + h - min(max(v, 0), ymax) / ymax * h))
        if prev is not None:
            cv2.line(img, prev, p, COLORS_BGR[side], 2)
        prev, last = p, v
    return last


def render_screen(overlay, data: LiveData, panel_w=520):
    h = max(overlay.shape[0], 480)
    canvas = np.zeros((h, overlay.shape[1] + panel_w, 3), np.uint8)
    canvas[:overlay.shape[0], :overlay.shape[1]] = overlay
    x0 = overlay.shape[1] + 10
    gh = (h - 90) // 2
    for i, side in enumerate(SIDES):
        y0 = 40 + i * (gh + 25)
        name = "LEFT knee" if side == "L" else "RIGHT knee"
        last = _graph(canvas, x0 + 120, y0, panel_w - 140, gh, data, side)
        cv2.putText(canvas, name, (x0, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS_BGR[side], 2)
        cv2.putText(canvas, "--" if last is None else f"{last:4.0f}", (x0, y0 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, COLORS_BGR[side], 3)
        cv2.putText(canvas, "deg flexion", (x0, y0 + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    cv2.putText(canvas, f"last {int(data.window_s)} s   |   {data.status}   |   {data.fps:.1f} fps",
                (x0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    return canvas


# ---------------------------------------------------------------------- #
# Web page
# ---------------------------------------------------------------------- #
PAGE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gait 3D - live knee angles</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
header{display:flex;gap:16px;align-items:center;padding:10px 16px;background:#1b1b1b;flex-wrap:wrap}
h1{font-size:18px;margin:0;font-weight:500}.st{color:#aaa;font-size:13px}
main{display:grid;grid-template-columns:minmax(260px,360px) 1fr;gap:12px;padding:12px}
@media(max-width:800px){main{grid-template-columns:1fr}}
.card{background:#1b1b1b;border-radius:10px;padding:10px}
.big{font-size:44px;font-weight:600;line-height:1}.lab{font-size:13px;color:#aaa}
canvas{width:100%;height:220px;display:block}img{width:100%;border-radius:6px}
button,a.b{background:#2c2c2c;color:#eee;border:1px solid #444;border-radius:6px;padding:6px 12px;font-size:13px;cursor:pointer;text-decoration:none}
.row{display:flex;gap:16px;align-items:baseline}
</style></head><body>
<header><h1>Gait 3D &mdash; live knee flexion</h1><span class="st" id="st"></span>
<span style="flex:1"></span><button onclick="fetch('/api/reset',{method:'POST'})">Reset session</button>
<a class="b" href="/session.csv">Download CSV</a><a class="b" href="/report" target="_blank">Session report</a></header>
<main><div class="card"><img id="cam" src="/snapshot.jpg" alt="camera"></div>
<div><div class="card"><div class="row"><span class="lab" style="color:#00aaff">LEFT knee</span><span class="big" id="vL" style="color:#00aaff">--</span><span class="lab">deg</span></div><canvas id="cL"></canvas></div>
<div class="card" style="margin-top:12px"><div class="row"><span class="lab" style="color:#ff8c00">RIGHT knee</span><span class="big" id="vR" style="color:#ff8c00">--</span><span class="lab">deg</span></div><canvas id="cR"></canvas></div></div></main>
<script>
const W=30,YMAX=120;let buf={t:[],L:[],R:[]},last=-1;
function draw(id,side,col){const c=document.getElementById(id),d=window.devicePixelRatio||1;c.width=c.clientWidth*d;c.height=c.clientHeight*d;
const g=c.getContext('2d');g.scale(d,d);const w=c.clientWidth,h=c.clientHeight;g.clearRect(0,0,w,h);g.font='11px system-ui';
for(let a=0;a<=YMAX;a+=30){const y=h-a/YMAX*h;g.strokeStyle='#333';g.beginPath();g.moveTo(28,y);g.lineTo(w,y);g.stroke();g.fillStyle='#888';g.fillText(a,2,y<14?y+12:y-2)}
const now=buf.t.length?buf.t[buf.t.length-1]:0;g.strokeStyle=col;g.lineWidth=2;g.beginPath();let pen=false;
for(let i=0;i<buf.t.length;i++){const v=buf[side][i];if(v===null){pen=false;continue}
const x=w-(now-buf.t[i])/W*(w-28),y=h-Math.min(Math.max(v,0),YMAX)/YMAX*h;pen?g.lineTo(x,y):g.moveTo(x,y);pen=true}g.stroke()}
async function tick(){try{const r=await fetch('/api/live?since='+last);const d=await r.json();
if(d.t.length){if(last>=0&&d.t[0]<last-1){buf={t:[],L:[],R:[]}}buf.t.push(...d.t);buf.L.push(...d.L);buf.R.push(...d.R);last=d.t[d.t.length-1]}
while(buf.t.length&&buf.t[buf.t.length-1]-buf.t[0]>W){buf.t.shift();buf.L.shift();buf.R.shift()}
for(const s of['L','R']){let v=null;for(let i=buf[s].length-1;i>=0;i--){if(buf[s][i]!==null){v=buf[s][i];break}}document.getElementById('v'+s).textContent=v===null?'--':Math.round(v)}
document.getElementById('st').textContent=d.status+'  |  '+d.fps+' fps';draw('cL','L','#00aaff');draw('cR','R','#ff8c00')}catch(e){document.getElementById('st').textContent='connection lost - retrying'}}
setInterval(tick,250);setInterval(()=>{document.getElementById('cam').src='/snapshot.jpg?'+Date.now()},400);
</script></body></html>"""


def report_html(s):
    def leg(k, name):
        v = s["legs"].get(k)
        if not v:
            return f"<h3>{name}</h3><p>No data.</p>"
        rows = [("Samples", v["samples"]), ("Mean flexion (deg)", v["flexion_mean"]),
                ("Range of motion p5-p95 (deg)", v["rom_p5_p95"]), ("Flexion p5 / p95 (deg)", f'{v["flexion_min_p5"]} / {v["flexion_max_p95"]}'),
                ("Peak flexion (deg)", v["peak_flexion_max"]), ("Steps detected (swing peaks)", v["steps"]),
                ("Mean swing peak (deg)", v["mean_step_peak"])]
        return f"<h3>{name}</h3><table>" + "".join(f"<tr><td>{a}</td><td><b>{b}</b></td></tr>" for a, b in rows) + "</table>"
    sym = s.get("symmetry")
    symh = (f"<h3>Symmetry</h3><table><tr><td>ROM difference L-R (deg)</td><td><b>{sym['rom_diff_L_minus_R']}</b></td></tr>"
            f"<tr><td>p95 flexion difference L-R (deg)</td><td><b>{sym['peak_diff_L_minus_R']}</b></td></tr></table>") if sym else ""
    subj = s.get("subject") or {}
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Gait session report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:760px;margin:24px auto;color:#222}}table{{border-collapse:collapse;width:100%}}
td{{border-bottom:1px solid #ddd;padding:6px}}h1{{font-size:22px}}.n{{color:#666;font-size:12px}}</style></head><body>
<h1>Gait session report &mdash; knee flexion</h1>
<p>Date: {time.strftime('%Y-%m-%d %H:%M')} &nbsp; Duration: {s['duration_s']} s &nbsp;
Thigh/shin (mm): L {subj.get('L', '?')} &nbsp; R {subj.get('R', '?')}</p>
{leg('L', 'Left knee')}{leg('R', 'Right knee')}{symh}
<p class="n">Flexion: 0 deg = straight knee, corrected by the standing (neutral) trial. Measured with a calibrated stereo camera pair and a
bone-length-constrained 3D model. Research prototype, not a certified medical device: validate against a goniometer before clinical use.
Use the browser's Print &rarr; Save as PDF to keep this report.</p></body></html>"""


class WebDashboard:
    def __init__(self, data: LiveData, port=8080):
        self.data, self.port = data, port
        d = data

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path, _, q = self.path.partition("?")
                if path == "/":
                    return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                if path == "/api/live":
                    since = float(q.split("=")[1]) if q.startswith("since=") else -1.0
                    return self._send(200, json.dumps(d.live(since)).encode(), "application/json")
                if path == "/api/summary":
                    return self._send(200, json.dumps(d.summary()).encode(), "application/json")
                if path == "/report":
                    return self._send(200, report_html(d.summary()).encode(), "text/html; charset=utf-8")
                if path.startswith("/snapshot.jpg"):
                    return self._send(200, d.jpeg or b"", "image/jpeg")
                if path == "/session.csv" and d.csv_path:
                    try:
                        with open(d.csv_path, "rb") as f:
                            return self._send(200, f.read(), "text/csv")
                    except OSError:
                        pass
                return self._send(404, b"not found", "text/plain")

            def do_POST(self):
                if self.path == "/api/reset":
                    d.reset()
                    return self._send(200, b"ok", "text/plain")
                return self._send(404, b"not found", "text/plain")

        self.server = ThreadingHTTPServer(("0.0.0.0", port), H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()
