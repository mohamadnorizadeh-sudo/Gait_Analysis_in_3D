VALIDATION AGAINST PUBLISHED HEALTHY-GAIT DATASETS (24 Nov 2026)

Datasets (copies in Downloads/gait_check/datasets/):
 1. Moreira et al. 2021, Scientific Data, doi 10.1038/s41597-021-00881-3; data doi 10.6084/m9.figshare.c.4923162.
    16 healthy young adults, overground, 7 controlled speeds 0.28-1.11 m/s; knee flexion (Knee_X) per stride.
    WHY: only open dataset with controlled VERY SLOW speeds like walker gait; our stride time (~2.2 s) matches its 0.56 m/s condition (2.10 s).
 2. Fukuchi, Fukuchi & Duarte 2018, PeerJ 6:e4640, doi 10.7717/peerj.4640; data doi 10.6084/m9.figshare.5722711.
    42 healthy young and older adults, overground (slow/comfortable/fast) and treadmill at 8 speeds from ~0.50 m/s.
    WHY: widely used normative set, includes older adults (closer to stroke patients' age) and a 0.50 m/s condition.
 3. Helwig et al. 2016 (UCI Machine Learning Repository "Multivariate Gait Data", CC BY 4.0).
    10 healthy adults on a treadmill, unbraced and with a KNEE BRACE on the right knee.
    WHY: the knee-brace condition is a published model of stiff-knee gait to compare with our stiff-knee passes.

Method: our knee angle signals from the three training walks (leave-one-walk-out model predictions) and the held-out test walk,
near-leg passes only, split into gait cycles between consecutive swing peaks (cycles kept only when their length matches the
pass's autocorrelation stride period within 25 %), resampled to 101 points. Published heel-strike-based curves were re-referenced
to start at their swing peak so both are aligned the same way. 'Normal-like' passes: iPad reference swing peak 40-75 deg;
'stiff/shuffle' passes: below 35 deg. RMSE, correlation r and the % of the cycle inside the healthy mean +-2 SD band are computed
on the mean curves.

IMPORTANT: this compares our measurements with OTHER healthy people. It checks plausibility (shape, ranges, speed effects),
it is NOT an accuracy validation of our sensor; that needs simultaneous gait-lab measurement of the same person.

Files: curves_normal_walking.csv, curves_stiff_passes.csv (mean curves), comparison_table.csv (all metrics),
datasets_summary.csv, fig1-3 PNG, and the scripts used (norm_curves.py, our_cycles.py, compare.py).
