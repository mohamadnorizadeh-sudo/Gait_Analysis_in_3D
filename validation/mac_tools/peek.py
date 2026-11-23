import cv2,numpy as np,subprocess,sys
from blobs import blobs
src,out=sys.argv[1],sys.argv[2];ts=[float(x) for x in sys.argv[3].split(',')]
tiles=[]
for t in ts:
    subprocess.run(['ffmpeg','-v','error','-y','-ss',str(t),'-i',src,'-frames:v','1','-vf','scale=960:540','/tmp/p.png'])
    im=cv2.imread('/tmp/p.png')
    B=blobs(im)
    for i,(x,y,a) in enumerate(B):
        cv2.circle(im,(int(x),int(y)),10,(0,0,255),2);cv2.putText(im,str(i),(int(x)+8,int(y)-8),0,0.6,(0,0,255),2)
    cv2.putText(im,'t=%.0f'%t,(10,30),0,1,(0,0,255),2)
    print(t,[(i,int(x),int(y),a) for i,(x,y,a) in enumerate(B)])
    tiles.append(cv2.resize(im,(480,270)))
rows=[cv2.hconcat(tiles[i:i+2]) for i in range(0,len(tiles),2)]
if len(rows)>1 and rows[-1].shape[1]!=rows[0].shape[1]: rows[-1]=cv2.copyMakeBorder(rows[-1],0,0,0,rows[0].shape[1]-rows[-1].shape[1],0)
cv2.imwrite(out,cv2.vconcat(rows))
