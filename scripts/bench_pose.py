# Benchmark YOLOv8n-pose on the Pi 5 CPU with a real frame pair (read-only, no camera access)
import time, sys, cv2, numpy as np, importlib.util
from ultralytics import YOLO
L, R = cv2.imread('/tmp/shm_cam0.png'), cv2.imread('/tmp/shm_cam1.png')
print('ncnn installed:', importlib.util.find_spec('ncnn') is not None)
for w in sys.argv[1:] or ['/home/fruit/yolov8n-pose.pt']:
    m = YOLO(w, task='pose')
    for sz in (640, 480, 416, 320):
        for _ in range(2):
            m([L, R], imgsz=sz, verbose=False) if w.endswith('.pt') else [m(x, imgsz=sz, verbose=False) for x in (L, R)]
        t = time.perf_counter(); n = 8
        for _ in range(n):
            m([L, R], imgsz=sz, verbose=False) if w.endswith('.pt') else [m(x, imgsz=sz, verbose=False) for x in (L, R)]
        dt = (time.perf_counter() - t) / n
        print(f'{w.split("/")[-1]:28s} imgsz {sz}: {dt*1000:6.0f} ms per L+R pair -> {1/dt:5.1f} stereo fps')
