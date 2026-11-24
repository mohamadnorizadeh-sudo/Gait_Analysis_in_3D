from common import *
from scipy.signal import find_peaks
S,Tt,_=load_all(); pred=np.load(D+'/test_pred.npy')
tau=Tt['tr']; tp=np.arange(len(tau))
# Pi time
import json
t0=json.loads(open(D+'/test/keypoints.jsonl').readline())['t']
tpi=np.array([json.loads(l)['t'] for l in open(D+'/test/keypoints.jsonl')])-t0
m=(tpi>44)&(tpi<62)
raw4={(c,j):flex(Tt['P'][:,c,:,:2],LEGS[j]) for c in range(2) for j in range(2)}
Y=Tt['Y']
print('Pi t | ref L R | model L R | raw c0L c0R c1L c1R | valid L R')
for k in np.where(m)[0][::4]:
    print('%5.1f | %5.1f %5.1f | %5.1f %5.1f | %5.1f %5.1f %5.1f %5.1f | %d %d'%(tpi[k],Y[k,0],Y[k,1],pred[k,0],pred[k,1],*[raw4[q][k] for q in [(0,0),(0,1),(1,0),(1,1)]],*Tt['V'][k]))
print('max in 44-62 s: ref L %.1f R %.1f; model L %.1f R %.1f; raw c0L %.1f c0R %.1f c1L %.1f c1R %.1f'%(np.nanmax(Y[m,0]),np.nanmax(Y[m,1]),np.nanmax(pred[m,0]),np.nanmax(pred[m,1]),*[np.nanmax(raw4[q][m]) for q in [(0,0),(0,1),(1,0),(1,1)]]))
# training label range
Xtr,Ytr=train_stack(S)
print('training labels: L max %.1f p99 %.1f, R max %.1f p99 %.1f; n L %d R %d'%(np.nanmax(Ytr[:,0]),np.nanpercentile(Ytr[:,0],99),np.nanmax(Ytr[:,1]),np.nanpercentile(Ytr[:,1],99),(~np.isnan(Ytr[:,0])).sum(),(~np.isnan(Ytr[:,1])).sum()))
for name,s in zip(['w1','w2','w3'],S):
    for j in range(2):
        st=T.strides(np.zeros_like(s['Y']),s,j); 
    print(name,'near-leg ref swing peak range L %.0f-%.0f R %.0f-%.0f'%(*[(lambda st:(st[:,1].min(),st[:,1].max()))(T.strides(s['Y'],s,j)) for j in range(2)][0],*[(lambda st:(st[:,1].min(),st[:,1].max()))(T.strides(s['Y'],s,j)) for j in range(2)][1]))
