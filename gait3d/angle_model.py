"""Learned knee angle from both cameras' leg keypoints (v3.3.1; own-leg features v3.7.0).

One feature definition is shared by training (scripts/train_angle_model.py) and the live session
(scripts/07_side_session.py), so what the model sees live is exactly what it was trained on.

Features of frame k (FEATURE_VERSION 2, causal: no future frames, so no live delay):
  for each camera: hip, knee and ankle of both legs relative to the hip midpoint / leg length (12 numbers),
  their 6 confidences and both raw 2D knee angles -> 20 per camera, 40 per frame;
  then the same 40 for frames k-1 and k-2 -> 120.

Own-leg features (FEATURE_VERSION 3, v3.7.0, default for new models): each knee's model sees ONLY its own leg.
The independent review (validation/review_20261124) showed that with the version-2 features one leg's angle can be
predicted from the other leg's movement in symmetric walking, which would fail for a stroke patient. Per leg and
camera: knee and ankle relative to that leg's hip / its own length (4), the leg's 3 confidences, its raw 2D knee
angle, log of its length in pixels, thigh/shank length ratio and a seen flag -> 11 per camera, 22 per frame, 66 with
frames k-1 and k-2. A camera that does not see the leg gives zeros with seen = 0 (not stale values).
"""
from collections import deque

import json
import pickle

import numpy as np

LEGS = ((11, 13, 15), (12, 14, 16))
FEATURE_VERSION = 2              # both-leg features (models trained before v3.7.0)
OWN_FEATURE_VERSION = 3          # own-leg features (v3.7.0)
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


def own_frame(P, j):
    """(N, 2, 17, 3) -> (N, 22) features of leg j only (v3.7.0)."""
    h, k, a = LEGS[j]
    out = []
    for ci in range(2):
        xy, c = P[:, ci, :, :2], P[:, ci, :, 2]
        th = np.linalg.norm(xy[:, h] - xy[:, k], axis=1)
        sh = np.linalg.norm(xy[:, k] - xy[:, a], axis=1)
        L = th + sh + 1e-6
        seen = np.all(np.nan_to_num(c[:, [h, k, a]], nan=0.0) >= MIN_CONF, axis=1).astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            out += [(xy[:, k] - xy[:, h]) / L[:, None], (xy[:, a] - xy[:, h]) / L[:, None], c[:, [h, k, a]],
                    flex(xy, LEGS[j])[:, None], np.log(L)[:, None], (th / (sh + 1e-6))[:, None], seen[:, None]]
    X = np.hstack(out)
    for ci in range(2):
        X[X[:, ci * 11 + 10] == 0, ci * 11:ci * 11 + 10] = 0.0
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def own_features(P, j):
    """(N, 2, 17, 3) sequence -> (N, 66): leg j's frame k with k-1 and k-2 (v3.7.0)."""
    X = own_frame(P, j)
    lag = [X] + [np.vstack([np.repeat(X[:1], d, 0), X[:-d]]) for d in range(1, CONTEXT)]
    return np.hstack(lag)


def leg_features(P, version):
    """[X for the left knee, X for the right knee] for a model of this feature version."""
    if version == OWN_FEATURE_VERSION:
        return [own_features(P, 0), own_features(P, 1)]
    X = features(P)
    return [X, X]


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
        d = None
        if path.endswith('.json'):
            d = json.load(open(path))
            self.kind, self.ridge = 'ridge', d
        else:
            try:
                d = pickle.load(open(path, 'rb'))
            except (ImportError, ModuleNotFoundError, AttributeError) as e:
                # v3.7.0: scikit-learn missing or a different version -> use the ridge copy saved beside the model
                # (before, the pickle failed here and the ridge fallback inside it was unreachable)
                alt = path.replace('.pkl', '_ridge.json')
                print('angle model: %s (%s); using %s' % (type(e).__name__, e, alt))
                d = json.load(open(alt))
                self.kind, self.ridge = 'ridge', d
            if self.__dict__.get('kind') is None:
                self.ridge = d['ridge']
                try:
                    import sklearn  # noqa: F401
                    self.kind, self.gbm = 'gbm', d['gbm']
                    try:   # v3.3.9: one frame at a time is faster on one thread (Pi 5: 23.6 vs 36.9 ms per frame)
                        from threadpoolctl import threadpool_limits
                        self._threads = threadpool_limits(limits=1, user_api='openmp')
                    except Exception:
                        pass
                except ImportError:
                    self.kind = 'ridge'
        self.version = d.get('feature_version')
        if self.version not in (FEATURE_VERSION, OWN_FEATURE_VERSION):
            raise ValueError(f"model feature_version {self.version} unknown: retrain")
        self.buf = deque(maxlen=CONTEXT)

    def predict_features(self, Xs):
        """Xs = [X_left, X_right] (leg_features); returns (N, 2)."""
        if self.kind == 'gbm':
            return np.column_stack([m.predict(X) for m, X in zip(self.gbm, Xs)])
        W, b = self.ridge['coef'], self.ridge['intercept']
        return np.column_stack([X @ np.array(w, float) + float(c) for X, w, c in zip(Xs, W, b)])

    def predict_sequence(self, P):
        Y = self.predict_features(leg_features(P, self.version))
        Y[~valid(P)] = np.nan
        return Y

    def push(self, people_by_cam):
        """Live: one frame in, (L, R) out for that frame (None where a camera misses the leg)."""
        fr = frame_points(people_by_cam)
        self.buf.append(fr)
        seq = list(self.buf)
        seq = [seq[0]] * (CONTEXT - len(seq)) + seq
        P = np.array(seq)
        y = self.predict_features([X[-1:] for X in leg_features(P, self.version)])[0]
        ok = valid(P[-1:])[0]
        return tuple(float(np.clip(y[j], -10, 150)) if ok[j] else None for j in range(2))
