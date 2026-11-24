from common import *
S,Tt,_=load_all()
print('Correlation of raw 2D angle (camera c, Pi leg j) with the NEAR-leg reference, frames in passes where that leg is near, both cams valid for leg')
print('session | ref L vs c0L c1L | ref R vs c0R c1R | crosstalk: ref R vs c0L c1L ; ref L vs c0R c1R')
for name,s in zip(['w1','w2','w3','test'],S+[Tt]):
    raw4={(c,j):flex(s['P'][:,c,:,:2],LEGS[j]) for c in range(2) for j in range(2)}
    out=[]
    for j in range(2):
        for (c,jj) in [(0,j),(1,j)]:
            m=~np.isnan(s['Yt'][:,j])&(s['grp']>=0)&s['V'][:,jj]&(raw4[(c,jj)]<150)
            out.append(np.corrcoef(raw4[(c,jj)][m],s['Y'][m,j])[0,1])
    ct=[]
    for j,oj in ((1,0),(0,1)):
        for c in range(2):
            m=~np.isnan(s['Yt'][:,j])&(s['grp']>=0)&s['V'][:,oj]
            ct.append(np.corrcoef(raw4[(c,oj)][m],s['Y'][m,j])[0,1])
    print('%-5s | %5.2f %5.2f | %5.2f %5.2f | %5.2f %5.2f ; %5.2f %5.2f'%(name,*out,*ct))
# check walk1 high R labels
s=S[0]; st=T.strides(s['Y'],s,1); print('w1 R near swing peaks by pass:',[(int(i),round(r,0)) for i,r,_ in st])
Xtr,Ytr=train_stack(S)
for j in range(2): print('labels >75 deg leg',j,(Ytr[:,j]>75).sum())
