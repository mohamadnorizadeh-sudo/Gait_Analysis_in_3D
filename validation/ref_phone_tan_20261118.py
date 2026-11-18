import cv2,numpy as np,subprocess,json,sys
from blobs import blobs
W,H=960,540;FPS=10
def run(t0,t1,seed,R=45):
    p=subprocess.Popen(['ffmpeg','-v','error','-ss',str(t0),'-t',str(t1-t0),'-i','/mnt/user-data/uploads/IMG_4933.MOV','-vf','fps=%d,scale=%d:%d'%(FPS,W,H),'-f','rawvideo','-pix_fmt','bgr24','-'],stdout=subprocess.PIPE)
    prev=[np.array(s,float) for s in seed];age=[0,0,0];out=[];i=0
    while True:
        b=p.stdout.read(W*H*3)
        if len(b)<W*H*3: break
        img=np.frombuffer(b,np.uint8).reshape(H,W,3)
        B=[np.array(x[:2]) for x in blobs(img,hue=(12,30),s_min=20,v_min=90,frac=0.3)]
        cur=[];ok=True;used=set()
        for j in range(3):
            # predict: use previous position; search radius grows when lost
            d=[(np.linalg.norm(q-prev[j]),k) for k,q in enumerate(B) if k not in used]
            d=[x for x in d if x[0]<R*(1+age[j])]
            if d:
                k=min(d)[1];used.add(k);cur.append(B[k]);age[j]=0
            else:
                cur.append(prev[j]);age[j]+=1;ok=False
        prev=cur
        h,k,s=cur
        v1=h-k;v2=s-k;a=180-np.degrees(np.arccos(np.clip(v1@v2/np.linalg.norm(v1)/np.linalg.norm(v2),-1,1)))
        out.append((round(t0+i/FPS,1),round(float(a),1) if ok else None,[list(map(int,q)) for q in cur]))
        i+=1
    return out
L=run(15,70,[(681,116),(697,204),(714,286)])
Rr=run(86,126,[(512,131),(501,206),(463,256)])
json.dump(L+Rr,open('ref5.json','w'))
for name,o in (('L',L),('R',Rr)):
    t=np.array([x[0] for x in o]);a=np.array([np.nan if x[1] is None else x[1] for x in o])
    print(name,' '.join('%d:%s'%(tt,'-' if np.all(np.isnan(a[(t>=tt)&(t<tt+1)])) else int(np.nanmedian(a[(t>=tt)&(t<tt+1)]))) for tt in range(int(t[0]),int(t[-1])+1)))
