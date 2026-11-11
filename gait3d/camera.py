"""Synchronised dual IMX708 capture on the Raspberry Pi 5.

Issues fixed versus v1
----------------------
1. **No synchronisation.**  v1 called ``cam_l.capture_array()`` then
   ``cam_r.capture_array()``; the two frames could be up to one frame period
   (33 ms) apart.  A swinging shank moves ~3-5 cm in 33 ms, which is a huge
   disparity error.  v2 runs one capture thread per camera, tags each frame
   with its ``SensorTimestamp`` and only returns pairs whose timestamps are
   within ``max_skew_ms``.  Both cameras also get identical, fixed
   ``FrameDurationLimits`` so they run at exactly the same rate.
2. **Focus at infinity.**  ``LensPosition = 0.0`` focuses at infinity, but the
   legs are ~0.3-0.6 m away -> blurry, low-texture images -> SGBM/keypoint
   noise.  LensPosition is in dioptres (1/metres): 2.5 = 40 cm, 3.3 = 30 cm.
   NOTE: changing focus slightly changes the focal length, so recalibrate
   at the same LensPosition you track with.
3. **Independent auto-exposure / white balance.**  Different brightness in L
   and R hurts stereo matching.  v2 meters once on the left camera, caps the
   exposure time (motion blur during gait) and copies the exact same
   exposure / gain / colour gains to both cameras.
4. **Colour order.**  Picamera2 "RGB888" arrays are already in OpenCV BGR
   order; v1 converted them with RGB2BGR/RGB2GRAY (swapping colours).
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np


class _CamThread(threading.Thread):
    def __init__(self, cam, name):
        super().__init__(daemon=True, name=name)
        self.cam = cam
        self.buf = deque(maxlen=4)
        self.cond = threading.Condition()
        self.running = True
        self.error = None

    def run(self):
        try:
            while self.running:
                req = self.cam.capture_request()
                try:
                    arr = req.make_array("main")
                    ts = req.get_metadata().get("SensorTimestamp", time.monotonic_ns())
                finally:
                    req.release()
                with self.cond:
                    self.buf.append((int(ts), arr))
                    self.cond.notify_all()
        except Exception as e:      # surface errors to the main thread
            self.error = e


class StereoCamera:
    def __init__(self, size=(640, 480), fps=30.0, lens_position=3.0,
                 max_exposure_us=8000, max_skew_ms=6.0, left_index=0, right_index=1,
                 software_sync=True):
        self.software_sync = bool(software_sync)
        self.size, self.fps = tuple(size), float(fps)
        self.lens_position = float(lens_position)
        self.max_exposure_us = int(max_exposure_us)
        self.max_skew_ns = int(max_skew_ms * 1e6)
        self.indices = (left_index, right_index)
        self.cams, self.threads = [], []
        self._last_pair_ts = None

    # ------------------------------------------------------------------ #
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        from picamera2 import Picamera2
        from libcamera import controls

        frame_us = int(1e6 / self.fps)
        for n, idx in enumerate(self.indices):
            cam = Picamera2(idx)
            ctrl = {"FrameDurationLimits": (frame_us, frame_us)}
            if self.software_sync:
                # Raspberry Pi libcamera software camera sync (left = server, right = client).
                # Silently skipped on libcamera builds without the control.
                try:
                    mode = controls.rpi.SyncModeEnum.Server if n == 0 else controls.rpi.SyncModeEnum.Client
                    if "SyncMode" in cam.camera_controls:
                        ctrl["SyncMode"] = mode
                except AttributeError:
                    pass
            # NOTE: use the SAME configuration type (video, 640x480) for calibration and tracking,
            # otherwise the sensor crop differs and the calibration no longer applies.
            cfg = cam.create_video_configuration(
                main={"size": self.size, "format": "RGB888"},
                controls=ctrl,
                buffer_count=4,
            )
            cam.configure(cfg)
            cam.set_controls({"AfMode": controls.AfModeEnum.Manual,
                              "LensPosition": self.lens_position})
            self.cams.append(cam)

        # staggered start (avoids RP1 bandwidth spikes, as in v1)
        for cam in self.cams:
            cam.start()
            time.sleep(0.3)

        self._lock_exposure()

        for i, cam in enumerate(self.cams):
            t = _CamThread(cam, f"cam{i}")
            t.start()
            self.threads.append(t)

    def _lock_exposure(self):
        """Meter on the left camera, then apply identical fixed settings to both."""
        time.sleep(1.5)                       # let AE/AWB converge
        md = self.cams[0].capture_metadata()
        exp = int(md.get("ExposureTime", 8000))
        gain = float(md.get("AnalogueGain", 2.0))
        cg = md.get("ColourGains", (1.8, 1.6))
        if exp > self.max_exposure_us:        # trade exposure for gain -> less motion blur
            gain = min(16.0, gain * exp / self.max_exposure_us)
            exp = self.max_exposure_us
        ctrl = {"AeEnable": False, "ExposureTime": exp, "AnalogueGain": gain,
                "AwbEnable": False, "ColourGains": tuple(cg)}
        for cam in self.cams:
            cam.set_controls(ctrl)
        time.sleep(0.3)
        self.exposure = ctrl
        print(f"[camera] locked exposure={exp}us gain={gain:.2f} colour_gains={tuple(round(c, 2) for c in cg)} "
              f"lens={self.lens_position} dioptres")

    # ------------------------------------------------------------------ #
    def read(self, timeout=1.0, strict_wait=0.08):
        """Return (t_seconds, left_bgr, right_bgr, skew_ms) for the newest best-synced pair.

        Two free-running sensors at the same frame rate keep a constant phase
        offset.  If that offset is larger than max_skew_ms we still return the
        closest pair after `strict_wait` seconds (the HUD shows the skew), and
        recommend enabling hardware-free software sync (SyncMode, see README).
        """
        t0 = time.monotonic()
        tl, tr = self.threads
        while time.monotonic() - t0 < timeout:
            for t in self.threads:
                if t.error:
                    raise RuntimeError(f"camera thread failed: {t.error}")
            with tl.cond:
                L = list(tl.buf)
            with tr.cond:
                R = list(tr.buf)
            best = None
            for ts_l, a_l in L:
                if self._last_pair_ts is not None and ts_l <= self._last_pair_ts:
                    continue
                for ts_r, a_r in R:
                    skew = abs(ts_l - ts_r)
                    if best is None or skew < best[0] or (skew == best[0] and ts_l > best[1]):
                        best = (skew, ts_l, a_l, a_r)
            if best and (best[0] <= self.max_skew_ns or time.monotonic() - t0 > strict_wait):
                skew, ts, a_l, a_r = best
                self._last_pair_ts = ts
                return ts * 1e-9, a_l, a_r, skew * 1e-6
            with tl.cond:
                tl.cond.wait(0.005)
        raise TimeoutError("no stereo pair received (check both cameras are connected)")

    def stop(self):
        self._stop_threads()

    def _stop_threads(self):
        for t in self.threads:
            t.running = False
        for t in self.threads:
            t.join(timeout=1.0)
        for cam in self.cams:
            try:
                cam.stop()
                cam.close()
            except Exception:
                pass
        self.cams, self.threads = [], []


class ShmStereoSource:
    """Read synchronised stereo frames from the Gait-Pose-Estimation shared memory
    (rpi_src/camera_shm_writer.py) instead of opening the cameras.

    Why (v2.1): on the walker Pi that writer runs at boot and owns both cameras,
    so Picamera2 cannot open them a second time.  This source is strictly
    read-only (SeqLock read, ``track=False`` so Python never unlinks the
    segment) and lets v2 run side-by-side with the existing pipeline.

    Frame format of that writer: msgpack {"cam0","cam1": JPEG, "ts0","ts1": ns,
    "dt_ms"}, 640x360 from the 2304x1296 sensor mode, cam0 rotated right and
    cam1 rotated left (portrait 360x640).  A calibration must be captured from
    THIS source to be valid for it (scripts/01_capture_calibration.py --shm).
    """

    def __init__(self, name="cam_frame_shm", left="cam0", swap=False, poll_s=0.002, rot=(0, 0)):
        self.name, self.poll_s = name, poll_s
        # extra rotation (deg, multiple of 90) applied to (left, right) AFTER the writer's own
        # rotation -- e.g. a camera re-mounted upside down (v2.5.1). Calibrate with the same value.
        self.rot = tuple(int(r) % 360 for r in rot)
        self.keys = ("cam1", "cam0") if swap else (left, "cam1" if left == "cam0" else "cam0")
        self.shm = None
        self._last_seq = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        from multiprocessing import shared_memory
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False, track=False)
        except TypeError:                       # Python < 3.13 has no track=
            from multiprocessing import resource_tracker
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)
            resource_tracker.unregister(self.shm._name, "shared_memory")

    def _snapshot(self):
        import struct
        buf = self.shm.buf
        for _ in range(1000):
            s1 = struct.unpack("<I", bytes(buf[0:4]))[0]
            if s1 % 2 == 0 and s1 != self._last_seq:
                n = struct.unpack("<I", bytes(buf[4:8]))[0]
                data = bytes(buf[8:8 + n])
                if struct.unpack("<I", bytes(buf[0:4]))[0] == s1:
                    self._last_seq = s1
                    return data
            time.sleep(self.poll_s)
        raise TimeoutError("no new frame in shared memory (is camera_shm_writer.py running?)")

    def read(self, timeout=1.0):
        import cv2
        import msgpack
        p = msgpack.unpackb(self._snapshot())
        imgs = [cv2.imdecode(np.frombuffer(p[k], np.uint8), cv2.IMREAD_COLOR) for k in self.keys]
        codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        imgs = [cv2.rotate(im, codes[r]) if r in codes else im for im, r in zip(imgs, self.rot)]
        ts = int(p.get("ts0", time.monotonic_ns()))
        return ts * 1e-9, imgs[0], imgs[1], float(p.get("dt_ms", 0.0))

    def stop(self):
        if self.shm is not None:
            self.shm.close()            # never unlink: the writer owns the segment
            self.shm = None


def open_source(kind="picamera", **kw):
    """'picamera' -> StereoCamera (needs free cameras), 'shm' -> ShmStereoSource."""
    if kind == "shm":
        return ShmStereoSource(**{k: v for k, v in kw.items() if k in ("name", "left", "swap", "rot")})
    return StereoCamera(**{k: v for k, v in kw.items() if k not in ("name", "left", "swap", "rot")})
