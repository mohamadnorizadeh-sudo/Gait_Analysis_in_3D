from common import *
import json, pickle
S,Tt,_=load_all()
src=open('a8_ablation.py').read(); exec(src[src.index('def cols'):src.index('assert')])
d=pickle.load(open(D+'/review_model.pkl','rb'))
tp=np.array([json.loads(l)['t'] for l in open(D+'/test/keypoints.jsonl')]); tau=tp-tp[0]
kb=np.where((tau>49)&(tau<58))[0]; stand=np.where((tau>44)&(tau<48.5))[0]
X=Tt['X']; base=np.column_stack([m.predict(X) for m in d['gbm']])
pkR=kb[np.argsort(-Tt['Y'][kb,1])[:8]]   # 8 deepest-bend frames by reference R
pkL=kb[np.argsort(-Tt['Y'][kb,0])[:8]]
ref_stand=X[stand].mean(0)
def sub(rows,colset):
    Xs=X[rows].copy(); Xs[:,colset]=ref_stand[colset]; return Xs
print('deepest knee-bend frames (8): ref R %.0f, ref L %.0f'%(Tt['Y'][pkR,1].mean(),Tt['Y'][pkL,0].mean()))
for name,cs in [('none',np.array([],int)),('L-leg features -> standing',cols(leg=0)),('R-leg cam1 -> standing',np.intersect1d(cols(leg=1),cols(cam=1))),('R-leg cam0 -> standing',np.intersect1d(cols(leg=1),cols(cam=0)))]:
    pr=d['gbm'][1].predict(sub(pkR,cs)) if len(cs) else base[pkR,1]
    print('  R prediction with %-28s: %.1f'%(name,pr.mean()))
for name,cs in [('none',np.array([],int)),('R-leg features -> standing',cols(leg=1)),('L-leg cam0 -> standing',np.intersect1d(cols(leg=0),cols(cam=0))),('L-leg cam1 -> standing',np.intersect1d(cols(leg=0),cols(cam=1)))]:
    pr=d['gbm'][0].predict(sub(pkL,cs)) if len(cs) else base[pkL,0]
    print('  L prediction with %-28s: %.1f'%(name,pr.mean()))
