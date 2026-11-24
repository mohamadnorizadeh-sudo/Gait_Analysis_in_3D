"""Prepare (align + label) the three training walks and the held-out walk once; cache to prepared.pkl."""
import os
import pickle
import sys
from types import SimpleNamespace

REPO = '/home/claude/gait_analysis_in_3d'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from train_angle_model import prepare  # noqa: E402

a = SimpleNamespace(sync='3,30', off_min=-30.0, off_max=60.0, speed=12.0, labels='near')
V = os.path.join(REPO, 'validation')
W = '/home/claude/work/v37'
S = {f'w{n}': prepare(f'{W}/w{n}', f'{V}/walk_20261123/walk{n}_ref_ipad_movenet_clean.csv', a) for n in (1, 2, 3)}
S['test'] = prepare(f'{W}/test', f'{V}/test_20261124/test_ref_ipad_movenet_clean.csv', a)
pickle.dump(S, open(f'{W}/prepared.pkl', 'wb'))
print('saved', {k: (len(v['P']), len(v['passes'])) for k, v in S.items()})
