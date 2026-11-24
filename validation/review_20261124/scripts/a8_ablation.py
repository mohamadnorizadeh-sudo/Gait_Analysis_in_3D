from common import *
import sys
S,Tt,_=load_all()
kind=sys.argv[1] if len(sys.argv)>1 else 'gbm'
# --- column groups in the original 120-feature layout
def cols(leg=None,cam=None):
    out=[]
    for lag in range(3):
        for c in range(2):
            if cam is not None and c!=cam: continue
            base=lag*40+c*20
            if leg in (None,0): out+= [base+i for i in (0,1,4,5,8,9,12,14,16,18)]
            if leg in (None,1): out+= [base+i for i in (2,3,6,7,10,11,13,15,17,19)]
    return np.array(sorted(out))
assert len(cols())==120
# --- pure single-leg features: knee & ankle relative to that leg's hip / that leg's length, confs, flex
def pure(P,leg):
    h,k,a=LEGS[leg]; out=[]
    for c in range(2):
        xy,cf=P[:,c,:,:2],P[:,c,:,2]
        L=np.linalg.norm(xy[:,h]-xy[:,k],axis=1)+np.linalg.norm(xy[:,k]-xy[:,a],axis=1)+1e-6
        out+=[(xy[:,k]-xy[:,h])/L[:,None],(xy[:,a]-xy[:,h])/L[:,None],cf[:,[h,k,a]],flex(xy,LEGS[leg])[:,None]]
    X=np.hstack(out); lag=[X]+[np.vstack([np.repeat(X[:1],d,0),X[:-d]]) for d in range(1,3)]
    return np.nan_to_num(np.hstack(lag),nan=0.0)
for s in S+[Tt]:
    s['XL']=pure(s['P'],0); s['XR']=pure(s['P'],1)
def run(label, getX):
    # getX(s, j) -> feature matrix for predicting leg j
    res={}
    # LOSO CV pooled over the three folds (strides concatenated)
    cv=[[],[]]
    for si,s in enumerate(S):
        pred=np.full_like(s['Y'],np.nan)
        for j in range(2):
            Xtr=np.vstack([getX(q,j)[q['grp']>=0] for k,q in enumerate(S) if k!=si]); Ytr=np.concatenate([q['Yt'][q['grp']>=0,j] for k,q in enumerate(S) if k!=si])
            ok=~np.isnan(Ytr)
            from sklearn.ensemble import HistGradientBoostingRegressor
            from sklearn.linear_model import Ridge
            m=(HistGradientBoostingRegressor(max_iter=800,learning_rate=0.1) if kind=='gbm' else Ridge(1.0)).fit(Xtr[ok],Ytr[ok])
            pred[:,j]=m.predict(getX(s,j)); pred[~s['V'][:,j],j]=np.nan
        for j in range(2): cv[j].append(T.strides(pred,s,j))
    # test
    pred=np.full_like(Tt['Y'],np.nan)
    for j in range(2):
        Xtr=np.vstack([getX(q,j)[q['grp']>=0] for q in S]); Ytr=np.concatenate([q['Yt'][q['grp']>=0,j] for q in S]); ok=~np.isnan(Ytr)
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
        m=(HistGradientBoostingRegressor(max_iter=800,learning_rate=0.1) if kind=='gbm' else Ridge(1.0)).fit(Xtr[ok],Ytr[ok])
        pred[:,j]=m.predict(getX(Tt,j)); pred[~Tt['V'][:,j],j]=np.nan
    line='%-34s'%label
    for j in range(2):
        c=peak_stats(np.vstack(cv[j])); t=peak_stats(T.strides(pred,Tt,j))
        mm=~np.isnan(Tt['Yt'][:,j])&(Tt['grp']>=0)&~np.isnan(pred[:,j])
        r=np.corrcoef(pred[mm,j],Tt['Y'][mm,j])[0,1]
        line+=' | %s CV med %4.1f bias %+5.1f sd %4.1f ; TEST med %4.1f bias %+5.1f LoA %+5.1f..%+5.1f r %.2f'%('LR'[j],c['med'],c['bias'],c['sd'],t['med'],t['bias'],t['lo'],t['hi'],r)
    print(line,flush=True)
    return pred
run('full (as deployed)', lambda s,j: s['X'])
run('own-leg columns (orig. normalisation)', lambda s,j: s['X'][:,cols(leg=j)])
run('other-leg columns only', lambda s,j: s['X'][:,cols(leg=1-j)])
run('pure own leg (own hip/own length)', lambda s,j: s['XL'] if j==0 else s['XR'])
run('pure OTHER leg only', lambda s,j: s['XR'] if j==0 else s['XL'])
run('cam0 only (both legs)', lambda s,j: s['X'][:,cols(cam=0)])
run('cam1 only (both legs)', lambda s,j: s['X'][:,cols(cam=1)])
run('own leg, cam0 only', lambda s,j: s['X'][:,np.intersect1d(cols(leg=j),cols(cam=0))])
run('own leg, cam1 only', lambda s,j: s['X'][:,np.intersect1d(cols(leg=j),cols(cam=1))])
