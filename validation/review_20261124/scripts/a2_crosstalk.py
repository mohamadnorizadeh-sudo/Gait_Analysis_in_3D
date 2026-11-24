from common import *
S,Tt,_=load_all()
pred=np.load(D+'/test_pred.npy')
def p90(x): x=x[~np.isnan(x)]; return np.percentile(x,90) if len(x)>10 else np.nan
print('Per pass 90th percentile of each signal (deg). near = leg facing iPad; refFar is unreliable.')
for name,s,pr in [('test',Tt,pred)]:
    raw4={(c,j):flex(s['P'][:,c,:,:2],LEGS[j]) for c in range(2) for j in range(2)}
    print(name,' pass near | refL refR | modelL modelR | c0L c0R c1L c1R')
    for i,(p0,p1,dd) in enumerate(s['passes']):
        m=(s['tr']>=p0)&(s['tr']<=p1); mr=(s['t']>=p0)&(s['t']<=p1)
        print('   %d %s | %5.1f %5.1f | %5.1f %5.1f | %5.1f %5.1f %5.1f %5.1f'%(i,'LR'[s['near'][i]],p90(s['F'][mr,0]),p90(s['F'][mr,1]),p90(pr[m,0]),p90(pr[m,1]),*[p90(raw4[k][m]) for k in [(0,0),(0,1),(1,0),(1,1)]]))
