from common import *
import io,contextlib
S,Tt,_=load_all()
s=S[0]
f=io.StringIO()
with contextlib.redirect_stdout(f):
    sy=T.prepare(D+'/w1',D+'/walk1_ref_ipad_yolov8x.csv',A)
print(f.getvalue()[-400:])
ref=np.genfromtxt(D+'/walk1_ref_ipad_movenet_clean.csv',delimiter=',',names=True)
print('MoveNet passes',[(round(a,1),round(b,1),d) for a,b,d in s['passes']])
print('YOLO passes   ',[(round(a,1),round(b,1),d) for a,b,d in sy['passes']])
raw4={(c,j):flex(s['P'][:,c,:,:2],LEGS[j]) for c in range(2) for j in range(2)}
for i in (4,6):
    p0,p1,_=s['passes'][i]; m=(s['tr']>=p0)&(s['tr']<=p1)
    mr=(ref['t']>=p0)&(ref['t']<=p1)
    print('pass',i,'MoveNet ref R p95 %.1f conf_R median %.2f frac conf<0.3 %.2f | YOLO-ref (on Pi frames) R p95 %.1f | Pi c0R p95 %.1f c1R p95 %.1f | Pi c0L %.1f'%(
        np.percentile(s['Y'][m,1],95),np.median(ref['conf_R'][mr]),(ref['conf_R'][mr]<0.3).mean(),np.percentile(sy['Y'][m,1],95),np.nanpercentile(raw4[(0,1)][m],95),np.nanpercentile(raw4[(1,1)][m],95),np.nanpercentile(raw4[(0,0)][m],95)))
# overall: interpolated fraction in near-leg frames
for name,rp,s_ in [('w1',D+'/walk1_ref_ipad_movenet_clean.csv',S[0]),('w2',D+'/walk2_ref_ipad_movenet_clean.csv',S[1]),('w3',D+'/walk3_ref_ipad_movenet_clean.csv',S[2]),('test',D+'/test_ref_ipad_movenet_clean.csv',Tt)]:
    r=np.genfromtxt(rp,delimiter=',',names=True)
    fr=[];cn=[];cf=[]
    for i,(p0,p1,d) in enumerate(s_['passes']):
        mr=(r['t']>=p0)&(r['t']<=p1); j=s_['near'][i]
        c=r['conf_L' if j==0 else 'conf_R'][mr]; co=r['conf_R' if j==0 else 'conf_L'][mr]
        fr.append((c<0.3).mean()); cn.append(np.median(c)); cf.append(np.median(co))
    print(name,'near-leg ref frames with conf<0.3 (interpolated) per pass',np.round(fr,2),' median near conf %.2f far conf %.2f'%(np.mean(cn),np.mean(cf)))
