import cv2,json,sys,os,math,numpy as np
sys.path.insert(0,'.')
from gait3d.side2d import PlaneMap,flexion_2d,leg_keypoints
D=sys.argv[1]
pm=PlaneMap.load('config/side_planes.json')
rows=[json.loads(l) for l in open(D+'/keypoints.jsonl')]
cap=cv2.VideoCapture(D+'/raw.mp4')
def blobs(im,fill=0.3):
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV); m=((hsv[...,2]>=165)&(hsv[...,1]<=70)).astype(np.uint8)
    n,lab,st,cen=cv2.connectedComponentsWithStats(m,8); o=[]
    for j in range(1,n):
        x,y,w,h,a=st[j]
        if a<40 or a>3000 or max(w,h)>3*min(w,h) or a<fill*w*h: continue
        y0,y1,x0,x1=max(0,y-6),min(640,y+h+6),max(0,x-6),min(360,x+w+6)
        p=hsv[y0:y1,x0:x1,2].astype(float); mk=np.ones(p.shape,bool); mk[y-y0:y-y0+h,x-x0:x-x0+w]=False
        if np.percentile(p[mk],30)>90: continue
        o.append((cen[j][0],cen[j][1]))
    return o
V={k:open('/tmp/v_%s/angles.csv'%k,'w') for k in ('kp','mk','hyb') if not os.makedirs('/tmp/v_%s'%k,exist_ok=True)}
for f in V.values(): f.write('t,L_raw,R_raw\n')
cnt={'mk':[0,0],'hyb':[0,0],'kp':[0,0]}
for r in rows:
    ok,fr=cap.read()
    if not ok: break
    out={k:['',''] for k in V}
    for li,(leg,cam) in enumerate((('L','cam0'),('R','cam1'))):
        kp=leg_keypoints(r[cam],leg,170.0)
        if kp is None or kp[1].min()<0.5: continue
        P=kp[0]; out['kp'][li]='%.1f'%flexion_2d(*pm.to_plane(cam,leg,P)); cnt['kp'][li]+=1
        im=fr[:,:360] if cam=='cam0' else fr[:,360:]
        bl=blobs(im); seed=P.copy(); seed[2]=P[1]+0.7*(P[2]-P[1]); Q=[]; used=set()
        for p in seed:
            c=[(math.hypot(b[0]-p[0],b[1]-p[1]),i) for i,b in enumerate(bl) if i not in used]
            c=[x for x in c if x[0]<80]
            if c: d,i=min(c); used.add(i); Q.append(np.array(bl[i]))
            else: Q.append(None)
        if all(q is not None for q in Q):
            out['mk'][li]='%.1f'%flexion_2d(*pm.to_plane(cam,leg,np.array(Q))); cnt['mk'][li]+=1
        if Q[2] is not None:   # hybrid: shin tape + keypoint hip/knee shifted by tape offset when seen
            H=Q[0] if Q[0] is not None else P[0]; K=Q[1] if Q[1] is not None else P[1]
            out['hyb'][li]='%.1f'%flexion_2d(*pm.to_plane(cam,leg,np.array([H,K,Q[2]]))); cnt['hyb'][li]+=1
    for k,f in V.items(): f.write('%.4f,%s,%s\n'%(r['t'],*out[k]))
for f in V.values(): f.close()
print('valid L/R of',len(rows),cnt)
