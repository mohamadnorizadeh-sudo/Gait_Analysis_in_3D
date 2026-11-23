import cv2,numpy as np,subprocess,sys
# usage: grid.py SRC OUT t x0,y0,x1,y1 (coords in 960x540 space) ; renders at 2x (1920 space) crop with grid every 25 (960-space) px
src,out,t=sys.argv[1],sys.argv[2],float(sys.argv[3]); x0,y0,x1,y1=[int(v) for v in sys.argv[4].split(',')]
subprocess.run(['ffmpeg','-v','error','-y','-ss',str(t),'-i',src,'-frames:v','1','/tmp/g.png'])
im=cv2.imread('/tmp/g.png'); H,W=im.shape[:2]; s=W/960.0
c=im[int(y0*s):int(y1*s),int(x0*s):int(x1*s)].copy()
k=1.0  # display scale relative to native crop
for gx in range((x0//25+1)*25,x1,25):
    X=int((gx-x0)*s); cv2.line(c,(X,0),(X,c.shape[0]),(0,255,255) if gx%100==0 else (0,180,0),1)
    if gx%50==0: cv2.putText(c,str(gx),(X+2,14),0,0.45,(0,255,255),1)
for gy in range((y0//25+1)*25,y1,25):
    Y=int((gy-y0)*s); cv2.line(c,(0,Y),(c.shape[1],Y),(0,255,255) if gy%100==0 else (0,180,0),1)
    if gy%50==0: cv2.putText(c,str(gy),(2,Y-3),0,0.45,(0,255,255),1)
cv2.putText(c,'t=%.1f'%t,(c.shape[1]-90,c.shape[0]-8),0,0.6,(0,0,255),2)
cv2.imwrite(out,c,[cv2.IMWRITE_JPEG_QUALITY,88]); print(out,c.shape,W,H)
