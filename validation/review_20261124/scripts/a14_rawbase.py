from common import *
S,Tt,_=load_all(); pred=np.load(D+'/test_pred.npy')
def raw(s,c,j): x=flex(s['P'][:,c,:,:2],LEGS[j]); x=x.copy(); x[~s['V'][:,j]]=np.nan; return x
for name,s in [('test',Tt)]+list(zip(['w1','w2','w3'],S)):
    print(name)
    for lab,f in [('raw cam0 (as reported L)/cam1 (as reported R)',lambda c_j: c_j),]:
        pass
    R={}
    R['reported raw (cam0-L, cam1-R)']=np.column_stack([raw(s,0,0),raw(s,1,1)])
    R['raw cam0 both legs']=np.column_stack([raw(s,0,0),raw(s,0,1)])
    R['raw cam1 both legs']=np.column_stack([raw(s,1,0),raw(s,1,1)])
    R['raw mean of two cams']=np.column_stack([(raw(s,0,0)+raw(s,1,0))/2,(raw(s,0,1)+raw(s,1,1))/2])
    for k,v in R.items():
        print('   %-32s '%k+' | '.join('%s %s'%('LR'[j],fmt(peak_stats(T.strides(v,s,j)))) for j in range(2)))
