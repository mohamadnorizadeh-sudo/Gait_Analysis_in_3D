import cv2,numpy as np,subprocess,json,sys
from blobs import blobs
src,out=sys.argv[1],sys.argv[2];t0,t1=float(sys.argv[3]),float(sys.argv[4])
seed=[tuple(map(float,s.split(':'))) for s in sys.argv[5].split(',')]
W,H=960,540;FPS=10;R=40
p=subprocess.Popen(['ffmpeg','-v','error','-ss',str(t0),'-t',str(t1-t0),'-i',src,'-vf','fps=%d,scale=%d:%d'%(FPS,W,H),'-f','rawvideo','-pix_fmt','bgr24','-'],stdout=subprocess.PIPE)
prev=[np.array(s) for s in seed];L0=(np.linalg.norm(prev[0]-prev[1]),np.linalg.norm(prev[2]-prev[1]))
age=[0,0,0];res=[];i=0
while True:
    b=p.stdout.read(W*H*3)
    if len(b)<W*H*3: break
    img=np.frombuffer(b,np.uint8).reshape(H,W,3)
    B=[np.array(x[:2]) for x in blobs(img)]
    cur=[];ok=True;used=set()
    for j in range(3):
        d=sorted((np.linalg.norm(q-prev[j]),k) for k,q in enumerate(B) if k not in used)
        d=[x for x in d if x[0]<R*(1+min(age[j],4))]
        if d: k=d[0][1];used.add(k);cur.append(B[k]);age[j]=0
        else: cur.append(prev[j]);age[j]+=1;ok=False
    h,k,s=cur
    l1,l2=np.linalg.norm(h-k),np.linalg.norm(s-k)
    if not(0.6*L0[0]<l1<1.5*L0[0] and 0.6*L0[1]<l2<1.5*L0[1]): ok=False
    else: prev=cur
    v1=h-k;v2=s-k;a=180-np.degrees(np.arccos(np.clip(v1@v2/np.linalg.norm(v1)/np.linalg.norm(v2),-1,1)))
    res.append((round(t0+i/FPS,1),round(float(a),1) if ok else None,[list(map(int,q)) for q in cur]))
    i+=1
json.dump(res,open(out,'w'))
t=np.array([r[0] for r in res]);a=np.array([np.nan if r[1] is None else r[1] for r in res])
B=[];tt=t[0]
while tt<t[-1]:
    m=(t>=tt)&(t<tt+0.5);v=a[m];B.append('' if np.all(np.isnan(v)) else '%d'%round(np.nanmedian(v)));tt+=0.5
print('valid %.0f%%'%(100*np.mean(~np.isnan(a))),'last',','.join('%d:%d'%tuple(q) for q in prev))
print('BINS0 %.1f '%t[0]+';'.join(B))
