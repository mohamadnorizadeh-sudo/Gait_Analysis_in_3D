from common import *
import io,contextlib
from scipy.signal import find_peaks, medfilt
S,Tt,_=load_all()
for n,s in zip((1,2,3),S):
    y=np.genfromtxt(D+'/walk%d_ref_ipad_yolov8x.csv'%n,delimiter=',',names=True)
    mref=np.genfromtxt(D+'/walk%d_ref_ipad_movenet_clean.csv'%n,delimiter=',',names=True)
    # both refs from the same iPad video; compare on MoveNet time base (assume same t origin; check lag by xcorr)
    Fy=np.column_stack([medfilt(y['flex_L'],5),medfilt(y['flex_R'],5)])
    best=max((np.corrcoef(np.interp(mref['t'][100:-100]+dl,y['t'],Fy[:,0]),s['F'][100:-100,0])[0,1],dl) for dl in np.arange(-1,1.01,0.033))
    out=[]
    for j in range(2):
        a=[];b=[]
        for i,(p0,p1,d) in enumerate(s['passes']):
            if s['near'][i]!=j: continue
            ii=np.where((s['t']>=p0)&(s['t']<=p1))[0]
            for p in find_peaks(s['F'][ii,j],prominence=12,distance=24)[0]:
                k=ii[p]; w=(y['t']>=s['t'][k]+best[1]-0.2)&(y['t']<=s['t'][k]+best[1]+0.2)
                a.append(s['F'][k,j]); b.append(Fy[w,j].max())
        a,b=np.array(a),np.array(b); d=b-a
        out.append('%s n %d MoveNet %.1f YOLOv8x %.1f diff median %+.1f (IQR %+.1f..%+.1f)'%('LR'[j],len(a),np.median(a),np.median(b),np.median(d),*np.percentile(d,[25,75])))
    print('walk%d (time lag YOLO vs MoveNet %+.2f s, r %.2f): '%(n,best[1],best[0])+' | '.join(out))
