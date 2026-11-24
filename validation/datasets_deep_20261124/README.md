# 24 Nov 2026: deep comparison with published healthy-gait datasets (v3.7.1), numbers and charts only

Question: is the knee curve measured on the walker (iPad MoveNet reference, Pi model v3.7.0 left and right) within the
normal tolerance of healthy walking? A plausibility check against other people, not an accuracy test.

## Datasets (263 healthy people)

| Dataset | People | Conditions | Source |
|---|---|---|---|
| Moreira et al. 2021, Scientific Data | 16 young adults | 7 overground speeds 0.28-1.11 m/s | figshare (copy on the Mac) |
| Fukuchi et al. 2018, PeerJ | 42 adults (18 older) | treadmill 0.49-1.79 m/s, overground slow/comfortable/fast | figshare (copy on the Mac) |
| Helwig et al. 2016, UCI Multivariate Gait Data | 10 adults | treadmill, unbraced and right-knee brace | UCI (copy on the Mac) |
| Schwartz et al. 2008, J Biomech | 83 children/teenagers | 5 speeds, very slow to very fast (mean +- SD only) | pyCGM2 open-access normative files (GitHub) |
| Pinzone et al. 2014, Gait Posture | 112 children/teenagers, 2 labs | self-selected speed per lab (mean +- SD only) | pyCGM2 open-access normative files (GitHub) |

Not yet added (browser download, needs the user's approval): Van Criekinge et al. 2023 (138 adults 21-86 y and 50
stroke survivors, figshare c.6503791), Lencioni et al. 2019 (50 people, walking at different speeds).

## Method

Our cycles: near-leg passes, swing peak to swing peak, 101 points; training walks predicted by the v3.7.0 own-leg
model trained on the other two walks, the held-out walk by the model trained on all three. Normal-like passes
(reference swing peak 40-75 deg): 53 iPad, 27 Pi left, 29 Pi right cycles; stiff passes (< 35 deg): 34 / 15 / 12.
Healthy curves: each person's curve rotated to start at its own swing peak. Our pace from the stride time (2.17 s)
on Moreira's stride-speed relation: about 0.55 m/s.

## Results (results.json)

- Shape: r 0.97-0.99 against all 21 healthy conditions, all three signals.
- Curve RMSE to slow healthy walking (Moreira 0.56 / Fukuchi 0.49 m/s): iPad 6.0 / 6.4, Pi left 9.2 / 9.0, Pi right
  8.3 / 8.7 deg; closest: Schwartz very slow 4.3 / 6.2 / 6.4.
- Tolerance: a healthy person's RMSE to their own group mean (leave-one-out) has median 4.9-6.3 and 90th percentile
  8.0-9.5 deg; lab-to-lab RMSE between slow-speed means has median 5.9, 90th percentile 9.1 (Pinzone's two labs: 1.3).
  Our gap ranks at the 62nd-69th percentile of healthy people (iPad), 81st-94th (Pi right), 86th-100th (Pi left).
- Features at our pace: swing peak iPad 53.4, Pi right 53.0, Pi left 47.4 deg vs healthy 52.8 (Fukuchi 0.49) and 58.6
  (Moreira 0.56); stance minimum ours 6.2-9.2 vs healthy 1.5; differences with 90 % CI in results.json (EQ).
- Stiff knee: swing peak drops 18-24 deg; normal vs stiff strides AUC 0.99 / 0.97 / 0.97; stiff curves vs Helwig
  knee-brace mean RMSE 4.0 / 3.4 / 3.9 deg.

Figures: `figures/f1`-`f8`. Scripts (`scripts/`): `build_norms.py` (healthy curves), `ours.py` (our cycles),
`analyse.py` (metrics and figures); paths at the top point to the working folders used on 24 Nov.
