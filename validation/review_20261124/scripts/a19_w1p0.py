from common import *
S,Tt,_=load_all(); s=S[0]
ref=np.genfromtxt(D+'/walk1_ref_ipad_movenet_clean.csv',delimiter=',',names=True)
for t0 in np.arange(8,48,2):
    m=(ref['t']>=t0)&(ref['t']<t0+2)
    print('ref t %4.0f-%4.0f: hip_x %6.1f..%6.1f  flexL max %5.1f flexR max %5.1f confL %.2f confR %.2f'%(t0,t0+2,ref['hip_x'][m].min(),ref['hip_x'][m].max(),ref['flex_L'][m].max(),ref['flex_R'][m].max(),ref['conf_L'][m].mean(),ref['conf_R'][m].mean()))
m=s['grp']==0; print('Pi frames in pass 0:',m.sum(),'valid L/R',s['V'][m].mean(0),'labels R non-NaN',(~np.isnan(s['Yt'][m,1])).sum(), 'Pi tau range', (s['tr'][m]-14.22).min(),(s['tr'][m]-14.22).max())
# model vs ref lag on test
pred=np.load(D+'/test_pred.npy')
for j in range(2):
    m=~np.isnan(Tt['Yt'][:,j])&(Tt['grp']>=0)&~np.isnan(pred[:,j])
    best=max((np.corrcoef(pred[m,j],np.interp(Tt['tr'][m]+dl,Tt['t'],Tt['F'][:,j]))[0,1],dl) for dl in np.arange(-0.4,0.41,0.02))
    print('test %s: model best matches reference shifted by %+.2f s (r %.3f; at 0: %.3f)'%('LR'[j],best[1],best[0],np.corrcoef(pred[m,j],Tt['Y'][m,j])[0,1]))
