from common import *
S,Tt,_=load_all(); pred=np.load(D+'/test_pred.npy')
from sklearn.ensemble import HistGradientBoostingRegressor
cv=[[],[]]
for si,s in enumerate(S):
    Xtr,Ytr=train_stack([q for k,q in enumerate(S) if k!=si]); p=np.full_like(s['Y'],np.nan)
    for j in range(2):
        ok=~np.isnan(Ytr[:,j]); p[:,j]=HistGradientBoostingRegressor(max_iter=800,learning_rate=0.1).fit(Xtr[ok],Ytr[ok,j]).predict(s['X'])
    p[~s['V']]=np.nan
    for j in range(2): cv[j].append(T.strides(p,s,j))
for lab,sets in (('LOSO-CV pooled',[np.vstack(cv[j]) for j in range(2)]),('TEST',[T.strides(pred,Tt,j) for j in range(2)])):
    for j in range(2):
        st=sets[j]; r,m=st[:,1],st[:,2]; d=m-r
        sl,ic=np.polyfit(r,m,1); sd_,id_=np.polyfit((r+m)/2,d,1)
        lo=r<30; hi=r>55
        print('%-15s %s n %3d: model = %.2f*ref + %.1f ; BA slope of diff vs mean %.2f ; mean err for ref<30: %+.1f (n %d), ref>55: %+.1f (n %d)'%(lab,'LR'[j],len(r),sl,ic,sd_,d[lo].mean() if lo.any() else np.nan,lo.sum(),d[hi].mean() if hi.any() else np.nan,hi.sum()))
