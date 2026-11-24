from common import *
S,Tt,_=load_all()
print('Median pixel positions in passes (x,y) per camera: Lhip Lknee Lank | Rhip Rknee Rank; leg length px L/R; conf L/R')
for name,s in zip(['w1','w2','w3','test'],S+[Tt]):
    P=s['P'][s['grp']>=0]
    for c in range(2):
        xy=np.nanmedian(P[:,c,11:17,:2],0); cf=np.nanmedian(P[:,c,11:17,2],0)
        lenL=np.nanmedian(np.linalg.norm(P[:,c,11,:2]-P[:,c,13,:2],axis=1)+np.linalg.norm(P[:,c,13,:2]-P[:,c,15,:2],axis=1))
        lenR=np.nanmedian(np.linalg.norm(P[:,c,12,:2]-P[:,c,14,:2],axis=1)+np.linalg.norm(P[:,c,14,:2]-P[:,c,16,:2],axis=1))
        # how often is R knee left of L knee in x (side swap indicator)
        sw=np.nanmean(P[:,c,14,0]<P[:,c,13,0])
        print('%-5s cam%d L %s | R %s | len %.0f/%.0f | conf L %.2f R %.2f | frac Rknee_x<Lknee_x %.2f'%(name,c,' '.join('(%.0f,%.0f)'%tuple(p) for p in xy[[0,2,4]]),' '.join('(%.0f,%.0f)'%tuple(p) for p in xy[[1,3,5]]),lenL,lenR,cf[[0,2,4]].min(),cf[[1,3,5]].min(),sw))
