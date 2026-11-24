"""Per-subject knee flexion curves from Van Criekinge 2023 (able-bodied + stroke, Excel) and Lencioni 2019 (MAT).
Usage: python3 -I extract.py <vancriekinge_dir> <lencioni_mat_dir> <lencioni_demo_xlsx> <out.pkl>"""
import sys, glob, os, pickle
import numpy as np, openpyxl, pandas as pd, scipy.io as sio

vc, ld, demo, out = sys.argv[1:5]
r101 = lambda y: np.interp(np.linspace(0, 1, 101), np.linspace(0, 1, len(y)), np.asarray(y, float))

def sheets(path, cols):
    wb = openpyxl.load_workbook(path, read_only=True)
    res = {}
    for name in wb.sheetnames:
        if not name.startswith('Sub'):
            continue
        rows = list(wb[name].iter_rows(values_only=True))
        head = list(rows[0]); data = np.array([[np.nan if v is None else v for v in r] for r in rows[1:]], float)
        res[name] = {c: data[:, head.index(c)] for c in cols}
    return res

AB = sheets(f'{vc}/MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx', ['KneeAngles'])
ST = sheets(f'{vc}/MAT_normalizedData_PostStrokeAdults_v27-02-23.xlsx', ['Pside_KneeAngles', 'Nside_KneeAngles'])
O = {'vc_able': {}, 'vc_stroke_P': {}, 'vc_stroke_N': {}, 'len': []}
for k, d in AB.items():
    y = d['KneeAngles']
    if np.isfinite(y).sum() > 900:
        O['vc_able'][k] = r101(y[np.isfinite(y)])
for k, d in ST.items():
    for side in 'PN':
        y = d[f'{side}side_KneeAngles']
        if np.isfinite(y).sum() > 900:
            O[f'vc_stroke_{side}'][k] = r101(y[np.isfinite(y)])

dm = pd.read_excel(demo)
O['len_demo'] = dm
for f in sorted(glob.glob(f'{ld}/Subject*.mat'), key=lambda p: int(os.path.basename(p)[7:-4])):
    s = sio.loadmat(f, squeeze_me=True, struct_as_record=False)['s']
    names = [str(n).strip() for n in s.AngVarName]
    ki = names.index('KneeFlx')
    sid = int(os.path.basename(f)[7:-4])
    for t in np.atleast_1d(s.Data):
        if str(t.Task).strip() != 'Walking':
            continue
        y = np.asarray(t.Ang)[ki]
        if np.isfinite(y).all():
            O['len'].append(dict(subject=sid, age=float(s.Age), gender=str(s.Gender), height=float(s.BH), foot=str(t.Foot).strip(),
                                 speed=float(t.speed), stride=float(t.strideLength), cadence=float(t.cadence), knee=r101(y)))
pickle.dump(O, open(out, 'wb'))
print('VC able', len(O['vc_able']), 'stroke P', len(O['vc_stroke_P']), 'N', len(O['vc_stroke_N']))
L = pd.DataFrame([{k: v for k, v in r.items() if k != 'knee'} for r in O['len']])
print('Lencioni trials', len(L), 'subjects', L.subject.nunique())
print(L.groupby('subject').agg(age=('age', 'first'), n=('speed', 'size'), vmin=('speed', 'min'), vmax=('speed', 'max')).describe().round(2))
print(dm.head(8).to_string())
