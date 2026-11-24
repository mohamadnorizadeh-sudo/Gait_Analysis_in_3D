from common import *
import json
S,Tt,_=load_all()
ref=np.genfromtxt(D+'/test_ref_ipad_movenet_clean.csv',delimiter=',',names=True)
from scipy.signal import medfilt
F=np.maximum(medfilt(ref['flex_L'],5),medfilt(ref['flex_R'],5))
tp=np.array([json.loads(l)['t'] for l in open(D+'/test/keypoints.jsonl')]); tau=tp-tp[0]
raw=np.nanmax(np.column_stack([flex(Tt['P'][:,0,:,:2],LEGS[0]),flex(Tt['P'][:,1,:,:2],LEGS[0])]),1)
m=(tau>46)&(tau<60)&~np.isnan(raw)
best=max(((np.corrcoef(np.interp(tau[m]+o,ref['t'],F),raw[m])[0,1],o) for o in np.arange(45,51,0.01)))
print('independent knee-bend offset (Pi 46-60 s, max-L raw vs ref max): r %.3f offset %.2f s; drift-fit offset at tau=53 s: %.3f'%(best[0],best[1],47.88-0.00105*53))
# offsets of a curve
for o in np.arange(47.5,48.2,0.1):
    print('  o %.2f r %.3f'%(o,np.corrcoef(np.interp(tau[m]+o,ref['t'],F),raw[m])[0,1]))
# per pass offsets: residuals and which passes dropped
