import cv2,numpy as np,subprocess,sys
from blobs import blobs
src,out,t=sys.argv[1],sys.argv[2],float(sys.argv[3])
x0,y0,x1,y1=[int(v) for v in sys.argv[4].split(',')] if len(sys.argv)>4 else (0,0,960,540)
subprocess.run(['ffmpeg','-v','error','-y','-ss',str(t),'-i',src,'-frames:v','1','-vf','scale=960:540','/tmp/p.png'])
im=cv2.imread('/tmp/p.png');B=blobs(im)
for i,(x,y,a) in enumerate(B):
    if x0<=x<x1 and y0<=y<y1:
        cv2.circle(im,(int(x),int(y)),8,(0,0,255),1);cv2.putText(im,str(i),(int(x)+6,int(y)-6),0,0.45,(0,0,255),1)
        print(i,int(x),int(y),a)
cv2.imwrite(out,im[y0:y1,x0:x1])
