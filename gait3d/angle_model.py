"""Learned knee angle from both cameras' leg keypoints (v3.3.1).

One feature definition is shared by training (scripts/train_angle_model.py) and the live session
(scripts/07_side_session.py), so what the model sees live is exactly what it was trained on.

Features of frame k (FEATURE_VERSION 2, causal: no future frames, so no live delay):
  for each camera: hip, knee and ankle of both legs relative to the hip midpoint / leg length (12 numbers),
  their 6 confidences and both raw 2D knee angles -> 20 per camera, 40 per frame;
  then the same 40 for frames k-1 and k-2 -> 120.
"""
from collections import deque

import json
import pickle

import numpy as np

LEGS = ((11, 13, 15), (12, 14, 16))
FEATURE_VERSION = 2
CONTEXT = 3                      # frames k, k-1, k-2
MIN_CONF = 0.3                   # a leg point below this in either camera -> no model value for that frame


def best_person(people):
    """(xy 17x2, conf 17) of the person whose two legs are longest x most confident, or None."""
    best, bs = None, -1.0
    for p in people or []:
        xy = np.asarray(p['xy'] if isinstance(p, dict) else p.xy, float)
        c = np.asarray(p['c'] if isinstance(p, dict) else p.conf, float)
        L = sum(np.linalg.norm(xy[i] - xy[k]) + np.linalg.norm(xy[a] - xy[k]) for i, k, a in LEGS)
        q = L * float(np.mean(c[11:17]))
        if q > bs:
            best, bs = (xy, c), q
    return best


def frame_points(people_by_cam):
    """(2 cams, 17, 3) array [x, y, conf] for one frame; NaN where a camera has no person."""
    fr = np.full((2, 17, 3), np.nan)
    for ci, cam in enumerate(('cam0', 'cam1')):
        b = best_person(people_by_cam.get(cam))
        if b is not None:
            fr[ci, :, :2], fr[ci, :, 2] = b[0], b[1]
    return fr


def flex(xy, idx):
    u, v = xy[:, idx[0]] - xy[:, idx[1]], xy[:, idx[2]] - xy[:, idx[1]]
    c = np.sum(u * v, 1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return 180 - np.degrees(np.arccos(np.clip(c, -1, 1)))


def frame_features(P):
    """(N, 2, 17, 3) -> (N, 40) per-frame features (no context)."""
    out = []
    for ci in range(2):
        xy, c = P[:, ci, :, :2], P[:, ci, :, 2]
        mid = (xy[:, 11] + xy[:, 12]) / 2
        leg = np.linalg.norm(xy[:, 11] - xy[:, 13], axis=1) + np.linalg.norm(xy[:, 13] - xy[:, 15], axis=1) + 1e-6
        out += [((xy[:, 11:17] - mid[:, None]) / leg[:, None, None]).reshape(len(P), -1), c[:, 11:17]]
        out += [flex(xy, idx)[:, None] for idx in LEGS]
    return np.hstack(out)


def features(P):
    """(N, 2, 17, 3) sequence -> (N, 120): frame k with k-1 and k-2 (the first frames repeat frame 0)."""
    X = frame_features(P)
    lag = [X] + [np.vstack([np.repeat(X[:1], d, 0), X[:-d]]) for d in range(1, CONTEXT)]
    return np.nan_to_num(np.hstack(lag), nan=0.0)


def valid(P):
    """(N, 2) bool: both cameras see that leg's hip, knee and ankle with conf >= MIN_CONF."""
    ok = []
    for idx in LEGS:
        c = P[:, :, list(idx), 2]
        ok.append(np.all(np.nan_to_num(c, nan=0.0) >= MIN_CONF, axis=(1, 2)))
    return np.column_stack(ok)


class AngleModel:
    """Loads a model saved by train_angle_model.py; predicts (L, R) frame by frame or for a sequence."""

    def __init__(self, path):
        if path.endswith('.json'):
            d = json.load(open(path))
            self.kind, self.ridge = 'ridge', d
        else:
            d = pickle.load(open(path, 'rb'))
            self.ridge = d['ridge']
            try:
                import sklearn  # noqa: F401
                self.kind, self.gbm = 'gbm', d['gbm']
            except ImportError:
                self.kind = 'ridge'
        if d.get('feature_version') != FEATURE_VERSION:
            raise ValueError(f"model feature_version {d.get('feature_version')} != {FEATURE_VERSION}: retrain")
        self.buf = deque(maxlen=CONTEXT)

    def predict_features(self, X):
        if self.kind == 'gbm':
            return np.column_stack([m.predict(X) for m in self.gbm])
        W, b = np.array(self.ridge['coef'], float), np.array(self.ridge['intercept'], float)
        return X @ W.T + b

    def predict_sequence(self, P):
        Y = self.predict_features(features(P))
        Y[~valid(P)] = np.nan
        return Y

    def push(self, people_by_cam):
        """Live: one frame in, (L, R) out for that frame (None where a camera misses the leg)."""
        fr = frame_points(people_by_cam)
        self.buf.append(fr)
        seq = list(self.buf)
        seq = [seq[0]] * (CONTEXT - len(seq)) + seq
        P = np.array(seq)
        y = self.predict_features(features(P)[-1:])[0]
        ok = valid(P[-1:])[0]
        return tuple(float(np.clip(y[j], -10, 150)) if ok[j] else None for j in range(2))
