from common import *
from scipy.signal import find_peaks
S,Tt,_=load_all(); pred=np.load(D+'/test_pred.npy')
print('peaks per near-leg pass: ref prom12 (used) / ref prom6 / model own peaks prom6 ; pass duration; implied stride time')
for name,s,pr in [('test',Tt,pred)]:
    for i,(p0,p1,d) in enumerate(s['passes']):
        j=s['near'][i]; ii=np.where((s['t']>=p0)&(s['t']<=p1))[0]
        a=len(find_peaks(s['F'][ii,j],prominence=12,distance=24)[0]); b=len(find_peaks(s['F'][ii,j],prominence=6,distance=15)[0])
        m=(s['tr']>=p0)&(s['tr']<=p1); q=pr[m,j]; q=np.where(np.isnan(q),np.nanmin(q),q)
        c=len(find_peaks(q,prominence=6,distance=5)[0])
        print('  pass %d %s: %2d / %2d / %2d  dur %.1f s  stride %.2f s'%(i,'LR'[j],a,b,c,p1-p0,(p1-p0)/max(c,1)))
# fraction of strides within +-5 deg
for j in range(2):
    st=T.strides(pred,Tt,j); d=st[:,2]-st[:,1]
    print('%s: strides within +-5 deg of reference: %d/%d (%.0f%%); within +-10: %.0f%%'%('LR'[j],(np.abs(d)<=5).sum(),len(d),100*(np.abs(d)<=5).mean(),100*(np.abs(d)<=10).mean()))
# pooled CV, gbm, same stat
