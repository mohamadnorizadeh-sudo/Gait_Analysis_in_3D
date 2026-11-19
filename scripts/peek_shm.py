# Read-only peek at frames from the running camera_shm_writer.py (does NOT stop or modify it).
import struct, sys, time, cv2, msgpack, numpy as np
from multiprocessing import shared_memory
out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/shm'
shm = shared_memory.SharedMemory(name='cam_frame_shm', create=False, track=False)
for _ in range(500):
    s1 = struct.unpack('<I', bytes(shm.buf[0:4]))[0]
    if s1 % 2:
        time.sleep(0.002); continue
    n = struct.unpack('<I', bytes(shm.buf[4:8]))[0]
    data = bytes(shm.buf[8:8 + n])
    if struct.unpack('<I', bytes(shm.buf[0:4]))[0] == s1:
        break
p = msgpack.unpackb(data)
shm.close()
print('dt_ms %.2f' % p['dt_ms'])
for k in ('cam0', 'cam1'):
    im = cv2.imdecode(np.frombuffer(p[k], np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(f'{out}_{k}.png', im)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    res = []
    for cb in ((9, 6), (6, 9)):
        ok, c = cv2.findChessboardCorners(g, cb, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        res.append(ok)
    print(k, im.shape, 'mean', round(float(g.mean()), 1), 'sharp', round(cv2.Laplacian(g, cv2.CV_64F).var(), 1), 'board', res)
