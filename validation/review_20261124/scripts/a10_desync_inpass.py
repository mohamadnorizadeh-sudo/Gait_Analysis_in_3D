from common import *
import pickle
S,Tt,_=load_all()
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
X=Tt['X']
for kind in ('gbm','ridge'):
    def predict(X):
        Y=np.column_stack([m.predict(X) for m in d['gbm']]) if kind=='gbm' else X@np.array(d['ridge']['coef']).T+np.array(d['ridge']['intercept'])
        Y[~Tt['V']]=np.nan; return Y
    base=predict(X)
    for j in range(2):
        g=cols(leg=1-j); out=[]
        # only frames inside passes where leg j is near; other-leg features taken from another in-pass frame N steps later (walking-only donors)
        idx=np.where(Tt['grp']>=0)[0]
        for N in (7,13,23,41,59):
            Xs=X.copy(); Xs[idx[:,None],g[None,:]]=X[np.roll(idx,N)[:,None],g[None,:]]
            p=predict(Xs); st=peak_stats(T.strides(p,Tt,j))
            m=~np.isnan(Tt['Yt'][:,j])&(Tt['grp']>=0)&~np.isnan(p[:,j])
            out.append((st['med'],st['bias'],st['lo'],st['hi'],np.corrcoef(p[m,j],Tt['Y'][m,j])[0,1],np.mean(np.abs(p[m,j]-base[m,j]))))
        o=np.array(out)
        print('%s predicting %s with OTHER-leg features from other walking frames: med %.1f (range %.1f-%.1f) bias %+.1f LoA %+.1f..%+.1f r %.2f mean|change| %.1f; gates pass in %d/5 shifts'%(
            kind,'LR'[j],o[:,0].mean(),o[:,0].min(),o[:,0].max(),o[:,1].mean(),o[:,2].mean(),o[:,3].mean(),o[:,4].mean(),o[:,5].mean(),
            sum((r[0]<=5)&(abs(r[1])<=3)&(max(abs(r[2]),abs(r[3]))<=10) for r in out)))
