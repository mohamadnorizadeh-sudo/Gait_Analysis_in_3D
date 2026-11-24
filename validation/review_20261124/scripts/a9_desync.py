from common import *
import pickle
from scipy.signal import find_peaks
S,Tt,_=load_all()
exec(open('a8_ablation.py').read().split('# --- pure')[0].split('kind=')[0])  # nothing
def cols(leg=None,cam=None):
    out=[]
    for lag in range(3):
        for c in range(2):
            if cam is not None and c!=cam: continue
            base=lag*40+c*20
            if leg in (None,0): out+= [base+i for i in (0,1,4,5,8,9,12,14,16,18)]
            if leg in (None,1): out+= [base+i for i in (2,3,6,7,10,11,13,15,17,19)]
    return np.array(sorted(out))
d=pickle.load(open(D+'/review_model.pkl','rb'))
for kind in ('gbm','ridge'):
    def predict(X):
        if kind=='gbm': Y=np.column_stack([m.predict(X) for m in d['gbm']])
        else: Y=X@np.array(d['ridge']['coef']).T+np.array(d['ridge']['intercept'])
        Y[~Tt['V']]=np.nan; return Y
    X=Tt['X']; base=predict(X)
    print(kind,'baseline test:', ' | '.join('%s %s'%('LR'[j],fmt(peak_stats(T.strides(base,Tt,j)))) for j in range(2)))
    near=[~np.isnan(Tt['Yt'][:,j])&(Tt['grp']>=0) for j in range(2)]
    for what,leg in (('desync OTHER leg',None),('desync OWN leg',None)):
        for j in range(2):
            g=cols(leg=1-j) if 'OTHER' in what else cols(leg=j)
            res=[];chg=[];sdp=[]
            for N in (17,37,53,71):
                Xs=X.copy(); Xs[:,g]=np.roll(X[:,g],N,axis=0)
                p=predict(Xs)
                m=near[j]&~np.isnan(p[:,j])&~np.isnan(base[:,j])
                chg.append(np.mean(np.abs(p[m,j]-base[m,j])))
                res.append(np.corrcoef(p[m,j],Tt['Y'][m,j])[0,1])
                st=peak_stats(T.strides(p,Tt,j)); sdp.append((st['med'],st['bias'],st['sd']))
            sdp=np.array(sdp)
            print('  %-17s -> predicting %s: mean |change| %.1f deg, frame r with ref %.2f (base %.2f), swing peaks med %.1f bias %+.1f sd %.1f'%(
                what,'LR'[j],np.mean(chg),np.mean(res),np.corrcoef(base[near[j]&~np.isnan(base[:,j]),j],Tt['Y'][near[j]&~np.isnan(base[:,j]),j])[0,1],*sdp.mean(0)))
    # per camera desync of own leg (which camera does the own-leg prediction use?)
    for j in range(2):
        for c in range(2):
            g=np.intersect1d(cols(leg=j),cols(cam=c)); chg=[]
            for N in (17,37,53,71):
                Xs=X.copy(); Xs[:,g]=np.roll(X[:,g],N,axis=0); p=predict(Xs)
                m=near[j]&~np.isnan(p[:,j]); chg.append(np.mean(np.abs(p[m,j]-base[m,j])))
            print('  desync own leg %s in cam%d only: mean |change| %.1f'%('LR'[j],c,np.mean(chg)))
