from common import *
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
S,Tt,_=load_all()
X,Y=train_stack(S)
for j in range(2):
    ok=~np.isnan(Y[:,j]); m=HistGradientBoostingRegressor(max_iter=800,learning_rate=0.1).fit(X[ok],Y[ok,j])
    print('in-sample gbm %s RMSE %.2f (n %d frames, %d features); ridge in-sample RMSE %.2f'%('LR'[j],np.sqrt(np.mean((m.predict(X[ok])-Y[ok,j])**2)),ok.sum(),X.shape[1],
          np.sqrt(np.mean((Ridge(1.0).fit(X[ok],Y[ok,j]).predict(X[ok])-Y[ok,j])**2))))
print('LOSO CV pooled frame RMSE / swing-peak stats for other gbm settings, and TEST:')
for it,lr,leaf in ((800,0.1,20),(200,0.05,20),(100,0.05,50),(300,0.05,100)):
    cvst=[[],[]]; rm=[[],[]]
    for si,s in enumerate(S):
        Xtr,Ytr=train_stack([q for k,q in enumerate(S) if k!=si]); pred=np.full_like(s['Y'],np.nan)
        for j in range(2):
            ok=~np.isnan(Ytr[:,j]); pred[:,j]=HistGradientBoostingRegressor(max_iter=it,learning_rate=lr,min_samples_leaf=leaf).fit(Xtr[ok],Ytr[ok,j]).predict(s['X'])
        pred[~s['V']]=np.nan
        for j in range(2):
            cvst[j].append(T.strides(pred,s,j)); m=~np.isnan(s['Yt'][:,j])&(s['grp']>=0)&~np.isnan(pred[:,j]); rm[j].append((pred[m,j]-s['Y'][m,j])**2)
    pred=np.full_like(Tt['Y'],np.nan)
    for j in range(2):
        ok=~np.isnan(Y[:,j]); pred[:,j]=HistGradientBoostingRegressor(max_iter=it,learning_rate=lr,min_samples_leaf=leaf).fit(X[ok],Y[ok,j]).predict(Tt['X'])
    pred[~Tt['V']]=np.nan
    print('  iter %d lr %.2f leaf %d: '%(it,lr,leaf)+' | '.join('%s CV rmse %.1f %s ; TEST %s'%('LR'[j],np.sqrt(np.mean(np.concatenate(rm[j]))),fmt(peak_stats(np.vstack(cvst[j]))),fmt(peak_stats(T.strides(pred,Tt,j)))) for j in range(2)))
