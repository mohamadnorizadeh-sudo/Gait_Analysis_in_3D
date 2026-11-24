import numpy as np, csv
rows=list(csv.DictReader(open('/home/claude/gait_analysis_in_3d/validation/static_20261124/phone_vs_models.csv')))
sel=[r for r in rows if float(r['still_spread_deg'])<=4 and 'ambig' not in r['phone_orientation'] and 'standing' not in r['phone_orientation']]
print('n selected',len(sel),[r['video_s'] for r in sel])
T=np.array([float(r['shin_tilt_true_deg']) for r in sel]);M=np.array([float(r['movenet_shin_deg']) for r in sel]);Y=np.array([float(r['yolov8x_shin_deg']) for r in sel])
eM,eY=M-T,Y-T
for n,e in (('MoveNet',eM),('YOLOv8x',eY)):
    print(n,'mean %.2f sd %.2f mean|e| %.2f max|e| %.1f'%(e.mean(),e.std(ddof=1),np.abs(e).mean(),np.abs(e).max()), 'per pose',np.round(e,1))
    b=np.polyfit(T,M if n=='MoveNet' else Y,1); print('   regression model = %.3f*true + %.2f'%tuple(b))
    for lo,hi in ((0,30),(30,60),(60,95)):
        m=(T>=lo)&(T<hi); print('   tilt %d-%d n %d mean err %.1f'%(lo,hi,m.sum(),e[m].mean() if m.any() else np.nan))
d=np.abs(eM)-np.abs(eY); print('paired |eM|-|eY|',np.round(d,1),'MoveNet better in',(d<0).sum(),'of',len(d))
from scipy.stats import wilcoxon, ttest_rel
print('wilcoxon |e| paired p=%.3f'%wilcoxon(np.abs(eM),np.abs(eY)).pvalue,'ttest p=%.3f'%ttest_rel(np.abs(eM),np.abs(eY)).pvalue)
# bootstrap CI of mean error
rng=np.random.default_rng(0)
for n,e in (('MoveNet',eM),('YOLOv8x',eY)):
    bs=[rng.choice(e,len(e)).mean() for _ in range(10000)]; print(n,'mean err 95%% bootstrap CI %.1f..%.1f'%tuple(np.percentile(bs,[2.5,97.5])))
# all rows incl moving/ambiguous
T2=np.array([float(r['shin_tilt_true_deg']) for r in rows]);M2=np.array([float(r['movenet_shin_deg']) for r in rows]);Y2=np.array([float(r['yolov8x_shin_deg']) for r in rows])
print('all 17 rows: MoveNet mean %.1f mean|e| %.1f ; YOLO mean %.1f mean|e| %.1f'%((M2-T2).mean(),np.abs(M2-T2).mean(),(Y2-T2).mean(),np.abs(Y2-T2).mean()))
