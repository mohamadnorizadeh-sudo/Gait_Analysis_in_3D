import sys, os, pickle, types, io, contextlib
import numpy as np
REPO='/home/claude/gait_analysis_in_3d'
sys.path.insert(0, REPO); sys.path.insert(0, REPO+'/scripts')
import train_angle_model as T
from gait3d.angle_model import features, valid, flex, LEGS, frame_features, CONTEXT
D='/home/claude/work/review/data'
SESS=[(D+'/w1',D+'/walk1_ref_ipad_movenet_clean.csv'),(D+'/w2',D+'/walk2_ref_ipad_movenet_clean.csv'),(D+'/w3',D+'/walk3_ref_ipad_movenet_clean.csv')]
TEST=(D+'/test',D+'/test_ref_ipad_movenet_clean.csv')
A=types.SimpleNamespace(sync='3,30',off_min=-30.0,off_max=60.0,speed=12.0,labels='near')
def load_all(cache=D+'/prepared.pkl', quiet=True):
    if os.path.exists(cache):
        return pickle.load(open(cache,'rb'))
    f=io.StringIO()
    with contextlib.redirect_stdout(f):
        S=[T.prepare(r,c,A) for r,c in SESS]; Tt=T.prepare(*TEST,A)
    pickle.dump((S,Tt,f.getvalue()),open(cache,'wb'))
    return S,Tt,f.getvalue()
def peak_stats(st):
    if len(st)==0: return dict(n=0)
    d=st[:,2]-st[:,1]; b=d.mean(); sd=d.std(ddof=1)
    return dict(n=len(d),med=np.median(np.abs(d)),bias=b,lo=b-1.96*sd,hi=b+1.96*sd,sd=sd)
def fmt(s):
    return 'n %2d med|e| %4.1f bias %+5.1f LoA %+6.1f..%+5.1f'%(s['n'],s['med'],s['bias'],s['lo'],s['hi']) if s['n']>1 else 'n %d'%s['n']
def fit_predict(Xtr,Ytr,Xte,Vte,kind='gbm',cols=None):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    out=np.full((len(Xte),2),np.nan)
    for j in range(2):
        c=slice(None) if cols is None else cols[j]
        ok=~np.isnan(Ytr[:,j])
        m=(HistGradientBoostingRegressor(max_iter=800,learning_rate=0.1) if kind=='gbm' else Ridge(1.0)).fit(Xtr[ok][:,c],Ytr[ok,j])
        out[:,j]=m.predict(Xte[:,c])
    out[~Vte]=np.nan
    return out
def train_stack(S,key='X'):
    X=np.vstack([q[key][q['grp']>=0] for q in S]); Y=np.vstack([q['Yt'][q['grp']>=0] for q in S]); return X,Y
