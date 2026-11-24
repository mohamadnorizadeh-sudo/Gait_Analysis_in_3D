from common import *
S,Tt,_=load_all()
print('corr(ref leg j, Pi raw same leg j from cam0 / cam1) split by whether j is the near (geometric rule) or far leg in that pass')
for name,s in zip(['w1','w2','w3','test'],S+[Tt]):
    raw4={(c,j):flex(s['P'][:,c,:,:2],LEGS[j]) for c in range(2) for j in range(2)}
    row=[]
    for j in range(2):
        for nf in (True,False):
            for c in range(2):
                m=np.zeros(len(s['tr']),bool)
                for i,(p0,p1,d) in enumerate(s['passes']):
                    if (s['near'][i]==j)==nf: m|=(s['grp']==i)
                m&=s['V'][:,j]
                row.append(np.corrcoef(raw4[(c,j)][m],s['Y'][m,j])[0,1])
    print('%-5s L near c0 %.2f c1 %.2f | L far c0 %.2f c1 %.2f || R near c0 %.2f c1 %.2f | R far c0 %.2f c1 %.2f'%(name,*row))
# swing peak amplitude of ref near vs far
print('median reference swing peak (prominence 12) near vs far per leg')
from scipy.signal import find_peaks
for name,s in zip(['w1','w2','w3','test'],S+[Tt]):
    o=[]
    for j in range(2):
        for nf in (True,False):
            pk=[]
            for i,(p0,p1,d) in enumerate(s['passes']):
                if (s['near'][i]==j)!=nf: continue
                ii=np.where((s['t']>=p0)&(s['t']<=p1))[0]
                pk+=list(s['F'][ii[find_peaks(s['F'][ii,j],prominence=12,distance=24)[0]],j])
            o.append('%s %s n%2d %.0f'%('LR'[j],'near' if nf else 'far ',len(pk),np.median(pk)))
    print(name,' | '.join(o))
