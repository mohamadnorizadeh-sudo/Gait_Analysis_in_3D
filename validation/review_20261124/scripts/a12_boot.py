from common import *
S,Tt,_=load_all(); pred=np.load(D+'/test_pred.npy')
rng=np.random.default_rng(1)
print('Test-walk swing-peak stats: analytic 95% CI of LoA (Bland-Altman, SE=sqrt(3 s^2/n)) and pass-cluster bootstrap')
for j in range(2):
    st=T.strides(pred,Tt,j); d=st[:,2]-st[:,1]; n=len(d); b=d.mean(); s=d.std(ddof=1)
    se_l=np.sqrt(3*s*s/n); se_b=s/np.sqrt(n)
    print(' %s n %d bias %+.1f (95%% CI %+.1f..%+.1f)  upper LoA %+.1f (CI %+.1f..%+.1f) lower LoA %+.1f (CI %+.1f..%+.1f)'%('LR'[j],n,b,b-2.07*se_b,b+2.07*se_b,b+1.96*s,b+1.96*s-2.07*se_l,b+1.96*s+2.07*se_l,b-1.96*s,b-1.96*s-2.07*se_l,b-1.96*s+2.07*se_l))
    ps=sorted(set(st[:,0].astype(int))); B=[]
    for _ in range(5000):
        pick=rng.choice(ps,len(ps)); dd=np.concatenate([d[st[:,0]==p] for p in pick])
        if len(dd)<3: continue
        bb=dd.mean(); ss=dd.std(ddof=1)
        B.append((np.median(np.abs(dd)),bb,max(abs(bb-1.96*ss),abs(bb+1.96*ss)),(np.median(np.abs(dd))<=5)&(abs(bb)<=3)&(abs(bb)+1.96*ss<=10)))
    B=np.array(B)
    print('    cluster bootstrap over %d passes: median|e| 95%% %.1f..%.1f, bias %.1f..%.1f, max|LoA| %.1f..%.1f, P(pass all gates) = %.2f'%(len(ps),*np.percentile(B[:,0],[2.5,97.5]),*np.percentile(B[:,1],[2.5,97.5]),*np.percentile(B[:,2],[2.5,97.5]),B[:,3].mean()))
    # within-pass ICC-like: between-pass variance of errors
    mp=[d[st[:,0]==p].mean() for p in ps]; print('    per-pass mean error',np.round(mp,1))
# constant-offset alignment instead of drift fit
b,dr=47.88,-0.00105
tau=(Tt['tr']-b)/(1+dr)
for off in (47.75,47.81,47.57):
    s2=dict(Tt); s2['tr']=tau+off
    print(' constant offset %.2f (no drift): '%off+' | '.join('%s %s'%('LR'[j],fmt(peak_stats(T.strides(pred,s2,j)))) for j in range(2)))
