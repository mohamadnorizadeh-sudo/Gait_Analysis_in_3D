import cv2,numpy as np,sys
def blobs(img,hue=(12,26),s_min=60,v_min=80,ring_dark=110,frac=0.45):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    m=((hsv[...,0]>=hue[0])&(hsv[...,0]<=hue[1])&(hsv[...,1]>=s_min)&(hsv[...,2]>=v_min)).astype(np.uint8)
    n,lab,st,cen=cv2.connectedComponentsWithStats(m,8);out=[]
    H,W=m.shape
    for i in range(1,n):
        x,y,w,h,a=st[i]
        if a<15 or a>2500 or max(w,h)>3*min(w,h) or a<0.4*w*h: continue
        r=5;y0,y1,x0,x1=max(0,y-r),min(H,y+h+r),max(0,x-r),min(W,x+w+r)
        P=hsv[y0:y1,x0:x1,2];ring=np.ones(P.shape,bool);ring[y-y0:y-y0+h,x-x0:x-x0+w]=False
        if np.mean(P[ring]<=ring_dark)<frac: continue
        out.append((float(cen[i][0]),float(cen[i][1]),int(a)))
    return out
if __name__=='__main__':
    for f in sys.argv[1:]:
        print(f,[(int(x),int(y),a) for x,y,a in blobs(cv2.imread(f))])
