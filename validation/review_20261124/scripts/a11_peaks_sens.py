from common import *
from scipy.signal import find_peaks
S,Tt,_=load_all(); pred=np.load(D+'/test_pred.npy')
def strides_w(pred,s,j,w0=0.2,mode='max'):
    out=[]
    for i,(p0,p1,_) in enumerate(s['passes']):
        if s['near'][i]!=j: continue
        ii=np.where((s['t']>=p0)&(s['t']<=p1))[0]
        for p in find_peaks(s['F'][ii,j],prominence=12,distance=24)[0]:
            k=ii[p]; w=(s['tr']>=s['t'][k]-w0)&(s['tr']<=s['t'][k]+w0); q=pred[w,j]; q=q[~np.isnan(q)]
            if mode=='max' and len(q): out.append((i,s['F'][k,j],q.max()))
            if mode=='interp':
                ok=~np.isnan(pred[:,j]); 
                # nearest valid model frames bracket
                v=np.interp(s['t'][k],s['tr'][ok],pred[ok,j]); out.append((i,s['F'][k,j],v))
    return np.array(out)
print('1) window / pairing sensitivity (test, gbm):')
for w0 in (0.1,0.2,0.3,0.5):
    print('   max within +-%.1f s: '%w0+' | '.join('%s %s'%('LR'[j],fmt(peak_stats(strides_w(pred,Tt,j,w0)))) for j in range(2)))
print('   model interpolated at ref peak time: '+' | '.join('%s %s'%('LR'[j],fmt(peak_stats(strides_w(pred,Tt,j,0,'interp')))) for j in range(2)))
# model's own peak per stride: pair each ref peak with the model's local max between neighbouring ref troughs (+-0.5 s)
print('   max within +-0.5 s (whole swing):  '+' | '.join('%s %s'%('LR'[j],fmt(peak_stats(strides_w(pred,Tt,j,0.5)))) for j in range(2)))
# same window effect on ref-vs-ref: how much does max-of-window inflate a noise-free copy? use ref itself sampled at Pi frames
Yref=Tt['Y'].copy()
print('   control: reference sampled at Pi frames, max within +-0.2 s vs ref peak: '+' | '.join('%s bias %+.1f'%('LR'[j],peak_stats(strides_w(Yref,Tt,j,0.2))['bias']) for j in range(2)))
print('2) clock shift sensitivity (test, gbm, +-0.2 s window):')
for dl in (-0.3,-0.2,-0.1,0,0.1,0.2,0.3):
    s2=dict(Tt); s2['tr']=Tt['tr']+dl
    print('   shift %+.1f s: '%dl+' | '.join('%s %s'%('LR'[j],fmt(peak_stats(T.strides(pred,s2,j)))) for j in range(2)))
# constant offset (no drift)
tau=Tt['tr']-47.88+0.00105*np.nan  # placeholder
