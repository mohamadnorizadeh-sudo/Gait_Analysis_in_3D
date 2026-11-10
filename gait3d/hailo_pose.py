"""YOLOv8-pose on the Raspberry Pi AI HAT+ 2 (Hailo-10H)  -- v2.4.

The Pi already ships compiled pose models for the Hailo-10H:
    /usr/share/hailo-models/yolov8s_pose_h10.hef
    /usr/share/hailo-models/yolov8m_pose_h10.hef   (more accurate)
They take a 640x640 UINT8 RGB image and return, for each stride (8, 16, 32):
    box  (H, W, 64)  DFL distances (4 sides x 16 bins)
    cls  (H, W, 1)   person score
    kpt  (H, W, 51)  17 keypoints x (dx, dy, visibility)
This module runs the HEF with HailoRT and decodes those tensors into the same
Person objects the CPU path produces, so the rest of the tracker is unchanged.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .pose import Person

DEFAULT_HEF = "/usr/share/hailo-models/yolov8m_pose_h10.hef"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def letterbox(img, size=640):
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    out = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, s, left, top


def decode(outputs: dict, conf_thr=0.25, iou_thr=0.6, size=640, sigmoid_cls=True, sigmoid_kpt=True):
    """outputs: name -> float array (H, W, C). Returns (boxes xyxy, scores, kpts (N,17,3)) in letterbox pixels."""
    by_hw = {}
    for name, a in outputs.items():
        a = np.asarray(a, np.float32)
        if a.ndim == 4:
            a = a[0]
        by_hw.setdefault(a.shape[0], {})[a.shape[2]] = a
    boxes, scores, kpts = [], [], []
    for hgt, d in sorted(by_hw.items(), reverse=True):
        stride = size // hgt
        box, cls, kp = d[64], d[1], d[51]
        sc = _sigmoid(cls[..., 0]) if sigmoid_cls else cls[..., 0]
        ys, xs = np.where(sc > conf_thr)
        if len(ys) == 0:
            continue
        b = box[ys, xs].reshape(-1, 4, 16)
        e = np.exp(b - b.max(axis=2, keepdims=True))
        dist = (e / e.sum(axis=2, keepdims=True)) @ np.arange(16, dtype=np.float32)   # (n, 4) l,t,r,b
        cx, cy = xs + 0.5, ys + 0.5
        x1, y1 = (cx - dist[:, 0]) * stride, (cy - dist[:, 1]) * stride
        x2, y2 = (cx + dist[:, 2]) * stride, (cy + dist[:, 3]) * stride
        k = kp[ys, xs].reshape(-1, 17, 3)
        kx = (k[..., 0] * 2.0 + xs[:, None]) * stride
        ky = (k[..., 1] * 2.0 + ys[:, None]) * stride
        kv = _sigmoid(k[..., 2]) if sigmoid_kpt else k[..., 2]
        boxes.append(np.stack([x1, y1, x2, y2], 1))
        scores.append(sc[ys, xs])
        kpts.append(np.stack([kx, ky, kv], 2))
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 17, 3))
    boxes, scores, kpts = np.concatenate(boxes), np.concatenate(scores), np.concatenate(kpts)
    keep = cv2.dnn.NMSBoxes([[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in boxes],
                            scores.tolist(), conf_thr, iou_thr)
    keep = np.array(keep).reshape(-1)
    return boxes[keep], scores[keep], kpts[keep]


class HailoPoseEstimator:
    """Drop-in for PoseEstimator: __call__([imgL, imgR]) -> [list[Person], list[Person]]."""

    def __init__(self, hef=DEFAULT_HEF, det_conf=0.25):
        from hailo_platform import FormatType, VDevice
        if not os.path.exists(hef):
            raise FileNotFoundError(hef)
        self.det_conf = det_conf
        params = VDevice.create_params()
        self.vdevice = VDevice(params)
        self.model = self.vdevice.create_infer_model(hef)
        self.model.input().set_format_type(FormatType.UINT8)
        for o in self.model.outputs:
            o.set_format_type(FormatType.FLOAT32)       # HailoRT de-quantises for us
        self.names = [o.name for o in self.model.outputs]
        self.shapes = {o.name: tuple(o.shape) for o in self.model.outputs}
        self.cfg = self.model.configure()
        self.configured = self.cfg.__enter__()
        h, w = self.model.input().shape[:2]
        self.size = int(h)

    def _infer(self, img_rgb):
        bind = self.configured.create_bindings(
            output_buffers={n: np.empty(self.shapes[n], np.float32) for n in self.names})
        bind.input().set_buffer(np.ascontiguousarray(img_rgb))
        self.configured.run([bind], 2000)
        return {n: bind.output(n).get_buffer() for n in self.names}

    def detect(self, img_bgr):
        lb, s, left, top = letterbox(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), self.size)
        boxes, scores, kpts = decode(self._infer(lb), self.det_conf, size=self.size)
        people = []
        for sc, k in zip(scores, kpts):
            xy = np.stack([(k[:, 0] - left) / s, (k[:, 1] - top) / s], 1)
            people.append(Person(xy, k[:, 2].copy(), float(sc)))
        return people

    def __call__(self, images):
        return [self.detect(im) for im in images]

    def close(self):
        try:
            self.cfg.__exit__(None, None, None)
        finally:
            self.vdevice.release()
