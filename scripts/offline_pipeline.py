#!/usr/bin/env python3
"""One-command offline analysis on the laptop: iPad videos -> YOLOv8x reference -> train -> held-out test (v3.3.5).

Folder layout (one folder per walk, inside ROOT, default ~/Desktop/Gait_Train):

  ROOT/walk1/IMG_xxxx.MOV                 the iPad side video of that walk (any .MOV/.mov/.mp4 at the top level)
  ROOT/walk1/side_session_2026.../        the Pi recording folder copied from ~/gait_v2/runs (has keypoints.jsonl)
  ROOT/walk2/...  ROOT/walk3/...  ROOT/walk4/...
  ROOT/test/...                           a folder whose name starts with "test" is the held-out test walk

  python3 scripts/offline_pipeline.py [ROOT] [--styles normal,normal,shuffle,shuffle,stiffL,stiffL,stiffR,stiffR]

For each walk it runs YOLOv8x-pose on the iPad video at 30 frames per second (skipped when the reference CSV already
exists), writes ref_ipad_yolov8x.csv next to the video, then trains the learned angle model on all walk folders
(leave-one-walk-out cross-validation) and tests it once on the test folder. Everything printed is also written to
ROOT/report.txt. The model goes to ROOT/model/angle_model.pkl (+ _ridge.json).

Note for the Pi: the .pkl contains a scikit-learn model and loads only with the same scikit-learn version (Pi has
1.4.2). The ridge .json works everywhere. For the gbm model on the Pi, copy the reference CSVs to the Pi and run the
training command printed at the end of the report there.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_EXT = ('.mov', '.mp4', '.m4v')
DEFAULT_STYLES = 'normal,normal,shuffle,shuffle,stiffL,stiffL,stiffR,stiffR'


class Tee:
    def __init__(self, path):
        self.f = open(path, 'a')

    def __call__(self, *a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + '\n')
        self.f.flush()


def run(cmd, log):
    log('$ ' + ' '.join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        log(line.rstrip())
    return p.wait()


def find_walk(d):
    kp = sorted(glob.glob(os.path.join(d, '**', 'keypoints.jsonl'), recursive=True))
    vids = [f for f in sorted(os.listdir(d)) if f.lower().endswith(VIDEO_EXT) and not f.startswith('.')]
    vids = sorted(vids, key=lambda f: -os.path.getsize(os.path.join(d, f)))
    return (os.path.dirname(kp[0]) if kp else None), (os.path.join(d, vids[0]) if vids else None), len(kp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?', default=os.path.expanduser('~/Desktop/Gait_Train'))
    ap.add_argument('--styles', default=DEFAULT_STYLES, help='walking style per pass of the test walk, in order')
    ap.add_argument('--model', default='yolov8x-pose.pt')
    ap.add_argument('--sync', default='3,30', help='Pi seconds window that contains the knee bends')
    ap.add_argument('--off-min', default='-30', help='negative: the iPad started after the Pi (v3.3.7)')
    ap.add_argument('--off-max', default='60', help='the iPad may start up to this many seconds before the Pi')
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    os.makedirs(os.path.join(root, 'model'), exist_ok=True)
    log = Tee(os.path.join(root, 'report.txt'))
    log('=' * 100)
    log('offline pipeline v3.3.5  %s  root %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), root))
    py = sys.executable

    train, test = [], None
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        if not os.path.isdir(d) or os.path.basename(d) in ('model',):
            continue
        name = os.path.basename(d)
        rec, vid, nkp = find_walk(d)
        if rec is None or vid is None:
            log('%-8s SKIPPED: needs one iPad video at the top of the folder and the Pi recording folder (found '
                'video: %s, keypoints.jsonl: %d)' % (name, 'yes' if vid else 'no', nkp))
            continue
        if nkp > 1:
            log('%-8s WARNING: %d keypoints.jsonl files inside; using %s' % (name, nkp, rec))
        ref = os.path.join(d, 'ref_ipad_yolov8x.csv')
        if not os.path.exists(ref):
            kp = os.path.join(d, 'kp_ipad_yolov8x.jsonl')
            if not os.path.exists(kp) or os.path.getsize(kp) == 0:
                log('%-8s YOLOv8x on %s ...' % (name, os.path.basename(vid)))
                if run([py, '-u', os.path.join(HERE, 'yolo_big_keypoints.py'), vid, kp + '.part', '--split', '1',
                        '--fps-out', '30', '--model', a.model], log) != 0:
                    log('%-8s YOLO FAILED, skipping this walk' % name)
                    continue
                os.replace(kp + '.part', kp)
            if run([py, os.path.join(HERE, 'ref_from_yolo_jsonl.py'), kp, ref], log) != 0:
                continue
        log('%-8s Pi: %s   iPad: %s   reference: %s' % (name, os.path.relpath(rec, root), os.path.basename(vid),
                                                        os.path.basename(ref)))
        if name.lower().startswith('test'):
            test = (rec, ref)
        else:
            train.append((rec, ref))

    if not train:
        log('nothing to train on: put each walk in its own folder (see the top of this script)')
        return 1
    out = os.path.join(root, 'model', 'angle_model.pkl')
    cmd = [py, os.path.join(HERE, 'train_angle_model.py')]
    for rec, ref in train:
        cmd += ['--session', rec, ref]
    if test:
        cmd += ['--test', test[0], test[1], '--styles', a.styles]
    cmd += ['--sync', a.sync, '--off-min', a.off_min, '--off-max', a.off_max, '--out', out]
    log('\ntraining on %d walk(s)%s' % (len(train), ' + held-out test' if test else ' (no test folder yet)'))
    rc = run(cmd, log)

    log('\nTo train the same model on the Pi (its scikit-learn 1.4.2 cannot load a .pkl made here):')
    log('  1) copy each walk\'s ref_ipad_yolov8x.csv to the Pi next to its recording folder, e.g.')
    log('     scp ROOT/walk1/ref_ipad_yolov8x.csv fruit@10.75.55.70:gait_v2/runs/<that side_session folder>/')
    pi_cmd = 'cd ~/gait_v2 && OMP_NUM_THREADS=1 nice python3 scripts/train_angle_model.py'
    for rec, ref in train:
        r = 'runs/' + os.path.basename(rec)
        pi_cmd += ' --session %s %s/ref_ipad_yolov8x.csv' % (r, r)
    pi_cmd += ' --sync %s --off-min %s --off-max %s --out config/angle_model.pkl' % (a.sync, a.off_min, a.off_max)
    log('  2) on the Pi:  ' + pi_cmd)
    log('  or, quicker: copy ROOT/model/angle_model_ridge.json to the Pi as config/angle_model_ridge.json and start'
        ' the session with --model config/angle_model_ridge.json')
    log('finished %s (exit %d)' % (time.strftime('%H:%M:%S'), rc))
    return rc


if __name__ == '__main__':
    sys.exit(main())
