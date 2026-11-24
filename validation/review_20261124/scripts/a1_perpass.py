from common import *
from gait3d.angle_model import AngleModel
S,Tt,_=load_all()
for name,s in zip(['w1','w2','w3','test'],S+[Tt]):
    dt=np.diff(s['tr']); inp=(s['grp']>=0)[1:]
    print(name,'dt in passes mean %.3f p95 %.3f max %.2f'%(dt[inp].mean(),np.percentile(dt[inp],95),dt[inp].max()))
M=AngleModel(D+'/review_model.pkl'); pred=M.predict_sequence(Tt['P'])
np.save(D+'/test_pred.npy',pred)
raw4={(c,j):flex(Tt['P'][:,c,:,:2],LEGS[j]) for c in range(2) for j in range(2)}
print('\nTEST per pass (near leg only, swing peaks): ref median, model median, raw cam0/cam1 median; model other-leg median peak')
for j,leg in ((0,'L'),(1,'R')):
    st=T.strides(pred,Tt,j)
    for i in sorted(set(st[:,0].astype(int))):
        r=st[st[:,0]==i]; d=r[:,2]-r[:,1]
        rw0=T.strides(np.column_stack([raw4[(0,0)],raw4[(0,1)]]),Tt,j); rw0=rw0[rw0[:,0]==i]
        rw1=T.strides(np.column_stack([raw4[(1,0)],raw4[(1,1)]]),Tt,j); rw1=rw1[rw1[:,0]==i]
        p0,p1,dd=Tt['passes'][i]
        # model's own peaks of other leg within pass
        m=(Tt['tr']>=p0)&(Tt['tr']<=p1)
        oth=pred[m,1-j]; 
        from scipy.signal import find_peaks
        q=np.nan_to_num(oth,nan=0); pk=find_peaks(q,prominence=10,distance=6)[0]
        print('  %s pass %d (t %.0f-%.0f) n %2d ref %5.1f model %5.1f err mean %+5.1f sd %4.1f | raw cam0 %5.1f cam1 %5.1f | model other leg peaks med %5.1f'%(
            leg,i,p0,p1,len(r),np.median(r[:,1]),np.median(r[:,2]),d.mean(),d.std(ddof=1) if len(d)>1 else np.nan,
            np.median(rw0[:,2]) if len(rw0) else np.nan,np.median(rw1[:,2]) if len(rw1) else np.nan, np.median(q[pk]) if len(pk) else np.nan))
    print('  ',leg,'all:',fmt(peak_stats(st)))
