from common import *
import json
from scipy.signal import medfilt
S,Tt,_=load_all()
ref=np.genfromtxt(D+'/test_ref_ipad_movenet_clean.csv',delimiter=',',names=True); t=ref['t']
F=np.column_stack([medfilt(ref['flex_L'],5),medfilt(ref['flex_R'],5)])
tp=np.array([json.loads(l)['t'] for l in open(D+'/test/keypoints.jsonl')]); tau=tp-tp[0]
P=Tt['P']; raw={(c,j):flex(P[:,c,:,:2],LEGS[j]) for c in range(2) for j in range(2)}
passes=Tt['passes']
def per_pass(off):
    offs=[]
    for p0,p1,_ in passes:
        m=(tau+off>=p0)&(tau+off<=p1); bb=(-2,off)
        for (c,j),q0 in raw.items():
            y=q0[m]; ok=~np.isnan(y)&(y<75)
            if ok.sum()<30: continue
            for o in np.arange(off-1.0,off+1.0,0.02):
                r_=np.corrcoef(np.interp(tau[m][ok]+o,t,F[:,j]),y[ok])[0,1]
                if r_>bb[0]: bb=(r_,o)
        if bb[0]>0.6: offs.append(bb[1])
    return offs
true=47.80
for dl in (-6,-3,-2,-1.5,-1,-0.6,0,0.6,1,1.5,2,3,6):
    o=per_pass(true+dl)
    print('global offset true%+.1f s: %d/%d passes "confirm" (r>0.6); their offsets rel. to true: %s'%(dl,len(o),len(passes),np.round(np.array(o)-true,2).tolist()))
