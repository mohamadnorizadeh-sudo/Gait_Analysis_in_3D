from common import *
from scipy.signal import find_peaks
S,Tt,_=load_all(); pred=np.load(D+'/test_pred.npy')
def own_peaks(x,m):
    q=x[m]; q=np.where(np.isnan(q),np.nanmin(q),q); return q[find_peaks(q,prominence=6,distance=5)[0]]
print('Each leg is only ever trained/scored in ONE walking direction. Model vs best-camera raw peaks by direction (test):')
for j,c in ((0,1),(1,0)):
    raw=flex(Tt['P'][:,c,:,:2],LEGS[j])
    for near in (True,False):
        mp=[];rp=[];conf=[]
        for i,(p0,p1,d) in enumerate(Tt['passes']):
            if (Tt['near'][i]==j)!=near: continue
            m=(Tt['tr']>=p0)&(Tt['tr']<=p1)
            mp+=list(own_peaks(pred[:,j],m)); rp+=list(own_peaks(raw,m)); conf.append(np.nanmean(Tt['P'][m][:,:,list(LEGS[j]),2]))
        print('  %s %s passes: model peak median %.1f, raw cam%d peak median %.1f, ratio %.2f, Pi leg conf %.2f'%('LR'[j],'scored (near)' if near else 'unscored (far)',np.median(mp),c,np.median(rp),np.median(mp)/np.median(rp),np.mean(conf)))
