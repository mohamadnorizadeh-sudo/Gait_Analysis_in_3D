from common import *
import json, time
from gait3d.angle_model import AngleModel
M=AngleModel(D+'/review_model.pkl'); off=M.predict_sequence(Tt['P']) if False else None
S,Tt,_=load_all(); off=np.load(D+'/test_pred.npy')
M=AngleModel(D+'/review_model.pkl')
live=[];t0=time.time()
for line in open(D+'/test/keypoints.jsonl'):
    r=json.loads(line); live.append([np.nan if v is None else v for v in M.push({'cam0':r['cam0'],'cam1':r['cam1']})])
live=np.array(live); dt=(time.time()-t0)/len(live)
d=np.abs(np.clip(off,-10,150)-live); both=~np.isnan(off)&~np.isnan(live)
print('frames %d, live push %.1f ms/frame here; NaN pattern identical: %s; max |live-offline| %.2e; frames differing >0.01: %d'%(len(live),dt*1e3,(np.isnan(off)==np.isnan(live)).all(),np.nanmax(d[both]),(d[both]>0.01).sum()))
print('offline prediction range L %.1f..%.1f R %.1f..%.1f (live clips to -10..150)'%(np.nanmin(off[:,0]),np.nanmax(off[:,0]),np.nanmin(off[:,1]),np.nanmax(off[:,1])))
