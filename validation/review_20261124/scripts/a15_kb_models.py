from common import *
import json
S,Tt,_=load_all()
exec(open('a8_ablation.py').read().split('for s in S+[Tt]:')[0].split('kind=sys.argv')[0])
src=open('a8_ablation.py').read()
exec(src[src.index('def cols'):src.index('for s in S+[Tt]:')])
for s in S+[Tt]:
    s['XL']=pure(s['P'],0); s['XR']=pure(s['P'],1)
tp=np.array([json.loads(l)['t'] for l in open(D+'/test/keypoints.jsonl')]); tau=tp-tp[0]
kb=(tau>49)&(tau<58)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
variants={'full':lambda s,j:s['X'],'pure own leg':lambda s,j:(s['XL'] if j==0 else s['XR']),
          'own leg cam0':lambda s,j:s['X'][:,np.intersect1d(cols(leg=j),cols(cam=0))],'own leg cam1':lambda s,j:s['X'][:,np.intersect1d(cols(leg=j),cols(cam=1))]}
print('max output during test knee bends (Pi 49-58 s); reference max L 108.5 R 92.8')
for kind in ('gbm','ridge'):
    for name,g in variants.items():
        o=[]
        for j in range(2):
            Xtr=np.vstack([g(q,j)[q['grp']>=0] for q in S]); Ytr=np.concatenate([q['Yt'][q['grp']>=0,j] for q in S]); ok=~np.isnan(Ytr)
            m=(HistGradientBoostingRegressor(max_iter=800,learning_rate=0.1) if kind=='gbm' else Ridge(1.0)).fit(Xtr[ok],Ytr[ok])
            p=m.predict(g(Tt,j)); p[~Tt['V'][:,j]]=np.nan; o.append(np.nanmax(p[kb]))
        print('  %-6s %-13s L %.1f R %.1f'%(kind,name,*o))
