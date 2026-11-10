"""Raw keypoint recorder / replayer (v2.8).

Every frame's YOLO people (17 keypoints + confidence, both cameras) are written
as one JSON line.  scripts/replay.py feeds a recording back through the full
tracker with any settings, so filters and gates can be tuned offline without
another walker session.
"""

from __future__ import annotations

import json

import numpy as np

from .pose import Person, people_from_result


class KeypointRecorder:
    def __init__(self, path):
        self.f = open(path, "w")

    def write(self, t, res_l, res_r, meta=None):
        rec = {"t": round(float(t), 4)}
        for k, r in (("L", res_l), ("R", res_r)):
            rec[k] = [{"xy": np.round(p.xy, 1).tolist(), "c": np.round(p.conf, 3).tolist(), "s": round(p.score, 3)}
                      for p in people_from_result(r)]
        if meta:
            rec["meta"] = meta
        self.f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def read_recording(path):
    """Yields (t, people_left, people_right, meta)."""
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            mk = lambda lst: [Person(np.array(p["xy"], float), np.array(p["c"], float), float(p["s"])) for p in lst]
            yield r["t"], mk(r["L"]), mk(r["R"]), r.get("meta", {})
