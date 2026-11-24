#!/usr/bin/env python3
"""Gait summary, deviation flags and draft recommendations for the doctor, from one session's knee CSV (v3.6.0).

    python3 scripts/gait_report.py ~/gait_data/knee_angles/knee_angles_2026-11-24_12-30-05.csv
    python3 scripts/gait_report.py --latest            # newest file in ~/gait_data/knee_angles

Writes <name>_report.json and <name>_report.txt to ~/gait_data/reports/ (or --out-dir).

What it does:
  1. Splits each knee's angle signal into strides (swing peak to swing peak), keeping only strides whose length
     agrees with that walking bout's repeat period (autocorrelation), as in the 24 Nov dataset validation.
  2. Per leg: swing-peak flexion, stance minimum, range of motion, stride time and its variability.
  3. Matches the walking speed by stride time to healthy adults walking at controlled speeds
     (config/normative_knee.json, from Moreira et al. 2021) and flags deviations.
  4. Drafts recommendations for the doctor: walker speed and projected footprint (step-target) settings.

This is decision support, not a diagnosis: every flag and recommendation is for a clinician to confirm. The default
thresholds below are placeholders until the clinical team sets them. Right-knee values are not trustworthy with the
camera position of 23-24 Nov (CHANGELOG v3.5.0); the report says so while that warning is set in the config.
"""
import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
from scipy.signal import find_peaks, medfilt

HERE = os.path.dirname(os.path.abspath(__file__))
NORM = os.path.join(HERE, '..', 'config', 'normative_knee.json')
DEFAULTS = dict(
    swing_peak_low_sd=2.0,        # flag reduced swing flexion (stiff-knee pattern) below healthy mean - 2 SD
    asymmetry_pct=15.0,           # flag left/right swing-peak asymmetry above this symmetry index (%)
    stride_cv_pct=10.0,           # flag irregular stepping above this coefficient of variation of stride time (%)
    hyperextension_deg=-5.0,      # flag knee hyperextension in stance below this angle
    stance_flexion_sd=2.0,        # flag excessive stance flexion above healthy mean + 2 SD
    min_strides=4,                # fewer strides than this per leg -> not enough data for that leg
    right_leg_warning=True,       # CHANGELOG v3.5.0: right knee poorly seen by cam1
)


def load(path):
    t, L, R, wall = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            t.append(float(row['session_time_s']))
            L.append(float(row['left_knee_deg']) if row['left_knee_deg'] else np.nan)
            R.append(float(row['right_knee_deg']) if row['right_knee_deg'] else np.nan)
            wall.append(row['actual_time_qatar'])
    return np.array(t), np.array(L), np.array(R), wall


def resample(t, y, fs=30.0):
    ok = np.isfinite(y)
    if ok.sum() < 20:
        return None, None
    u = np.arange(t[ok][0], t[ok][-1], 1 / fs)
    v = np.interp(u, t[ok], y[ok])
    gap = np.interp(u, t, np.isfinite(y).astype(float)) < 0.5        # long dropouts stay marked as missing
    v = medfilt(v, 5)
    v[gap] = np.nan
    return u, v


def period(u, v):
    w = np.isfinite(v)
    if w.sum() < 60:
        return np.nan
    x = v[w] - np.nanmean(v[w])
    ac = np.correlate(x, x, 'full')[len(x) - 1:]
    ac /= ac[0]
    dt = u[1] - u[0]
    lo, hi = int(0.7 / dt), min(len(ac) - 1, int(3.0 / dt))
    if hi <= lo:
        return np.nan
    k = lo + int(np.argmax(ac[lo:hi]))
    return k * dt if ac[k] > 0.3 else np.nan


def strides(u, v):
    if u is None:
        return []
    out = []
    # walking bouts = runs of valid samples; strides are found inside each bout
    valid = np.isfinite(v)
    edges = np.flatnonzero(np.diff(np.r_[0, valid.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < 90:
            continue
        uu, vv = u[a:b], v[a:b]
        Tp = period(uu, vv)
        thr = max(15.0, 0.6 * np.percentile(vv, 95))
        pk, _ = find_peaks(vv, height=thr, distance=int(0.7 * 30), prominence=10)
        for i, j in zip(pk[:-1], pk[1:]):
            T = uu[j] - uu[i]
            if 0.7 <= T <= 3.0 and (not np.isfinite(Tp) or abs(T - Tp) <= 0.25 * Tp):
                seg = vv[i:j + 1]
                out.append(dict(t=float(uu[i]), T=float(T), peak=float(vv[i]), stance_min=float(np.min(seg)), rom=float(vv[i] - np.min(seg))))
    return out


def summary(st):
    if not st:
        return dict(n=0)
    a = lambda k: np.array([s[k] for s in st])
    T = a('T')
    return dict(n=len(st), swing_peak=round(float(np.median(a('peak'))), 1), swing_peak_sd=round(float(np.std(a('peak'))), 1),
                stance_min=round(float(np.median(a('stance_min'))), 1), rom=round(float(np.median(a('rom'))), 1),
                stride_time=round(float(np.median(T)), 2), stride_cv=round(float(np.std(T) / np.mean(T) * 100), 1),
                cadence_steps_min=round(float(120 / np.median(T)), 1))


def report(path, cfg, norm):
    t, L, R, wall = load(path)
    S = {}
    for leg, y in (('left', L), ('right', R)):
        u, v = resample(t, y)
        S[leg] = summary(strides(u, v))
    st_all = [S[k]['stride_time'] for k in S if S[k]['n'] >= cfg['min_strides']]      # speed match on legs with enough strides
    ref = None
    if st_all:
        T = float(np.median(st_all))
        ref = min(norm['speeds'], key=lambda s: abs(s['stride_time_s'] - T))
    flags, recs = [], []
    for leg in ('left', 'right'):
        s = S[leg]
        if s['n'] < cfg['min_strides']:
            flags.append(dict(leg=leg, code='insufficient_data', text=f'{leg} knee: only {s["n"]} clean stride(s); not enough to judge'))
            continue
        if ref:
            lo = ref['swing_peak_deg'] - cfg['swing_peak_low_sd'] * ref['swing_peak_sd']
            if s['swing_peak'] < lo:
                flags.append(dict(leg=leg, code='reduced_swing_flexion', value=s['swing_peak'], limit=round(lo, 1),
                                  text=f'{leg} knee: swing-phase flexion {s["swing_peak"]} deg, below the healthy range at this speed (>= {lo:.0f} deg): stiff-knee pattern'))
            hi = ref['stance_min_deg'] + cfg['stance_flexion_sd'] * ref['stance_min_sd']
            if s['stance_min'] > hi:
                flags.append(dict(leg=leg, code='excess_stance_flexion', value=s['stance_min'], limit=round(hi, 1),
                                  text=f'{leg} knee: stays flexed {s["stance_min"]} deg in stance (healthy <= {hi:.0f} deg): crouched stance or forward lean on the walker'))
        if s['stance_min'] < cfg['hyperextension_deg']:
            flags.append(dict(leg=leg, code='hyperextension', value=s['stance_min'],
                              text=f'{leg} knee: hyperextends to {s["stance_min"]} deg in stance'))
        if s['stride_cv'] > cfg['stride_cv_pct']:
            flags.append(dict(leg=leg, code='irregular_stepping', value=s['stride_cv'],
                              text=f'{leg} leg: stride time varies {s["stride_cv"]} % (limit {cfg["stride_cv_pct"]:.0f} %): irregular stepping'))
    if S['left']['n'] >= cfg['min_strides'] and S['right']['n'] >= cfg['min_strides']:
        a, b = S['left']['swing_peak'], S['right']['swing_peak']
        si = abs(a - b) / ((a + b) / 2) * 100
        if si > cfg['asymmetry_pct']:
            weaker = 'left' if a < b else 'right'
            flags.append(dict(leg='both', code='asymmetry', value=round(si, 1), weaker=weaker,
                              text=f'asymmetric knee flexion: symmetry index {si:.0f} % (limit {cfg["asymmetry_pct"]:.0f} %), less flexion on the {weaker}'))
    codes = {f['code'] for f in flags}
    cad = np.nanmedian([S[k]['cadence_steps_min'] for k in S if S[k]['n'] >= cfg['min_strides']]) if st_all else np.nan
    if codes & {'reduced_swing_flexion', 'asymmetry', 'irregular_stepping', 'hyperextension'}:
        recs.append(dict(target='walker_speed', action='reduce', amount_pct=15,
                         text='Walker speed: reduce by about 15 % so the patient can complete knee flexion in swing; reassess next session.'))
    elif flags == [] and st_all:
        recs.append(dict(target='walker_speed', action='keep_or_increase', amount_pct=10,
                         text='Walker speed: knee pattern within the healthy range at this speed; keep, or increase by up to 10 % if the clinician agrees.'))
    if 'reduced_swing_flexion' in codes or 'asymmetry' in codes:
        legs = sorted({f['leg'] for f in flags if f['code'] in ('reduced_swing_flexion',)} | {f.get('weaker') for f in flags if f['code'] == 'asymmetry'} - {None})
        recs.append(dict(target='footprint_projection', action='step_targets', legs=legs, cadence_steps_min=round(float(cad * 0.9), 1) if np.isfinite(cad) else None,
                         text=f'Projected footprints: show step targets for the {" and ".join(legs)} leg slightly ahead of the current step (encourage a full swing), '
                              f'at about {cad * 0.9:.0f} steps/min (10 % below the measured cadence) so there is time to bend the knee.'))
    if 'irregular_stepping' in codes:
        recs.append(dict(target='footprint_projection', action='rhythm', cadence_steps_min=round(float(cad), 1) if np.isfinite(cad) else None,
                         text=f'Projected footprints: evenly spaced targets with a steady rhythm at about {cad:.0f} steps/min to regularise stepping.'))
    warnings = ['Decision support only: thresholds are defaults until set by the clinical team.']
    if cfg['right_leg_warning']:
        warnings.append('Right-knee values are not reliable with the current camera position (CHANGELOG v3.5.0).')
    return dict(file=os.path.basename(path), session_start_qatar=wall[0] if wall else None, duration_s=round(float(t[-1] - t[0]), 1) if len(t) else 0,
                legs=S, matched_healthy_speed=ref, flags=flags, recommendations=recs, warnings=warnings,
                generated=datetime.now().isoformat(timespec='seconds'), version='3.6.0')


def text(r):
    L = [f'GAIT SUMMARY FOR THE DOCTOR  ({r["file"]}, start {r["session_start_qatar"]} Qatar time, {r["duration_s"]} s)', '']
    m = r['matched_healthy_speed']
    if m:
        L.append(f'Compared with healthy adults at {m["speed_m_s"]} m/s (stride {m["stride_time_s"]} s; swing flexion {m["swing_peak_deg"]} +- {m["swing_peak_sd"]} deg).')
    for leg in ('left', 'right'):
        s = r['legs'][leg]
        if s['n']:
            L.append(f'{leg:>5} knee: {s["n"]} strides, swing flexion {s["swing_peak"]} deg, stance minimum {s["stance_min"]} deg, '
                     f'range {s["rom"]} deg, stride {s["stride_time"]} s (CV {s["stride_cv"]} %), {s["cadence_steps_min"]} steps/min')
        else:
            L.append(f'{leg:>5} knee: no clean strides')
    L += ['', 'FINDINGS:'] + ([' - ' + f['text'] for f in r['flags']] or [' - none: within the healthy range at this speed'])
    L += ['', 'SUGGESTIONS (for the clinician to confirm):'] + ([' - ' + x['text'] for x in r['recommendations']] or [' - none'])
    L += ['', 'NOTES:'] + [' - ' + w for w in r['warnings']]
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='?')
    ap.add_argument('--latest', action='store_true')
    ap.add_argument('--knee-dir', default='~/gait_data/knee_angles')
    ap.add_argument('--out-dir', default='~/gait_data/reports')
    ap.add_argument('--norm', default=NORM)
    a = ap.parse_args()
    path = a.csv
    if a.latest or not path:
        files = sorted(glob.glob(os.path.join(os.path.expanduser(a.knee_dir), 'knee_angles_*.csv')))
        if not files:
            sys.exit('no knee CSV found')
        path = files[-1]
    norm = json.load(open(a.norm))
    cfg = dict(DEFAULTS, **norm.get('thresholds', {}))
    r = report(path, cfg, norm)
    od = os.path.expanduser(a.out_dir)
    os.makedirs(od, exist_ok=True)
    base = os.path.join(od, os.path.basename(path).replace('.csv', '_report'))
    json.dump(r, open(base + '.json', 'w'), indent=1)
    open(base + '.txt', 'w').write(text(r))
    print(text(r))
    print('saved', base + '.json', base + '.txt')


if __name__ == '__main__':
    main()
