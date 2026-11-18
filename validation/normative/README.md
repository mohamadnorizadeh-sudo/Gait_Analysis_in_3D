# Normative knee flexion: WBDS (Fukuchi, Fukuchi & Duarte 2018)

Source: Fukuchi CA, Fukuchi RK, Duarte M (2018). A public dataset of overground and treadmill walking
kinematics and kinetics in healthy individuals. PeerJ 6:e4640, doi:10.7717/peerj.4640.
Data: figshare doi:10.6084/m9.figshare.5722711 (CC BY 4.0). The processed `*ang.txt` files (Visual3D,
101 points per gait cycle, heel strike = 0 %) were taken from a public mirror
(github.com/ROAM-Lab-ND/Template-Model-Optimization, HumanData/WBDSascii). Speeds come from WBDSinfo.xlsx
(github.com/demotu/gait_score, data/Claudiane).

`wbds_knee_flexion.csv`: knee flexion (KneeAngleZ, deg) per % gait cycle, mean and SD over 42 healthy
adults (24 young, 18 older), each subject's left and right averaged.

| condition | speed (m/s) | loading-response peak | stance minimum | swing peak | swing peak at |
|---|---|---|---|---|---|
| T01 treadmill | 0.50 | 9.6 ± 7.3 | 2.4 ± 5.5 | 52.7 ± 4.9 | 76 % |
| T02 treadmill | 0.68 | 10.1 ± 7.7 | 2.5 ± 5.4 | 55.7 ± 4.3 | 75 % |
| T03 treadmill | 0.87 | 11.4 ± 7.7 | 2.8 ± 5.3 | 59.0 ± 4.2 | 75 % |
| T05 treadmill | 1.24 | 17.8 ± 5.3 | 3.1 ± 4.6 | 62.7 ± 3.8 | 74 % |
| OC overground comfortable | - | 15.8 ± 6.6 | 3.0 ± 4.7 | 62.6 ± 3.5 | 74 % |

Walker-assisted walking is slow (typically under 0.6 m/s), so T01/T02 are the closest references. They are
healthy adults without a walker, so they check the **pattern and range** of our curve, not the accuracy
of one person's numbers.
