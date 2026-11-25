# 24 Nov 2026: Van Criekinge 2023 (healthy + stroke) and Lencioni 2019 added (v3.7.2), numbers and charts only

Extends `validation/datasets_deep_20261124/` (v3.7.1) from 5 to 7 open datasets: 451 healthy people and, for the first
time, 50 stroke survivors. Our signals are unchanged: iPad MoveNet reference and the Pi v3.7.0 own-leg model (left and
right knee), near-leg passes, cycles from swing peak to swing peak, 101 points. Pace about 0.55 m/s (stride 2.17 s).

## New data

| Dataset | Used here | Note |
|---|---|---|
| Van Criekinge et al. 2023, Sci Data 10:852 | 138 able-bodied adults 21-86 y (one curve per person, L/R averaged, preferred speed); 50 stroke survivors 19-85 y, 14-139 days after stroke (one curve per leg) | Excel release; speeds are only in the paper's supplement |
| Lencioni et al. 2019, Sci Data 6:309 | 29 adults 19-72 y, 568 walks, each with its measured speed (0.29-2.29 m/s for all 50 people) | per-person mean curves in speed bins |

**Label check (Van Criekinge stroke Excel):** the leg labelled `Pside` (paretic) has the higher swing peak in 40 of 50
people (56.2 vs 44.9 deg on average) and the larger knee and hip range in 41-42 of 50. A paretic knee normally bends
less in swing, so the labels may be swapped in this release. All stroke results here use a label-free definition: for
each person, the leg with the lower swing peak is the **more-affected leg**. Label-based AUCs are reported too.

## Results

**Curve gap (RMSE, iPad / Pi left / Pi right):** Van Criekinge 138 adults 7.1 / 9.7 / 8.9 deg; Lencioni adults below
0.8 m/s 6.3 / 9.4 / 8.7; Lencioni 0.8-1.0 m/s 6.5 / 9.5 / 8.6. Shape r 0.95-0.99 for every healthy condition slower
than 1.4 m/s (27 conditions in 7 datasets; 0.93 for faster walking).

**Tolerance:** a healthy person's gap to their own group: median 3.7-5.0 deg, 90th percentile 8.0-10.1 (Moreira 0.56,
Fukuchi 0.49, Lencioni two speed bins, Van Criekinge). Our percentile among healthy people: iPad 62-87, Pi left 80-100,
Pi right 80-94. Lab vs lab (83 pairs of slow/preferred means): median 5.1, 90th percentile 8.4 deg.

**Speed-matched expectation (Lencioni adults, person-level trend):** at 0.55 m/s a healthy adult's swing peak is
56.5 +- 9.9 deg (95% range 46.6-66.4), stance minimum 1.0 +- 10.5 deg. Ours: swing peak iPad 53.4, Pi left 47.4,
Pi right 53.0 (z -0.6 / -1.8 / -0.7); stance 6.2 / 9.1 / 9.2. 0.55 m/s is at the slow end of Lencioni's data.

**Features vs group means (ours minus healthy, 90% CI):** swing peak vs Lencioni below 0.8 m/s -3.6 / -9.6 / -4.0,
vs Van Criekinge -4.3 / -10.3 / -4.7; stance +5.4 / +8.3 / +8.4 and +4.6 / +7.5 / +7.6.

**Stroke:**
- Swing peak: healthy 57.7 +- 5.4 deg (Van Criekinge), stroke more-affected leg 42.6 +- 14.3 (range 5.8-72.3),
  less-affected 58.5. Stance minimum healthy 1.6, more-affected 8.8, less-affected 13.1 deg.
- Group-average curves: our *normal* mean curves are 2.3-3.6 deg from the stroke more-affected average curve but
  7.1-9.7 from healthy adults. Both groups walk slowly and the stroke average blends very different patients, so
  curve similarity to a group average cannot be used to judge a person.
- Per person: 70% of more-affected knees are below the healthy 5th percentile (48.9 deg), 60% below the speed-matched
  limit (46.6 deg). Our stiff-knee passes: 100% below on all three signals (8 iPad, 3 + 3 Pi). Our normal passes above
  the speed-matched limit: iPad 81%, Pi left 70%, Pi right 67% (those below are mostly within 5 deg of it).
- Detection by swing peak, healthy vs more-affected leg: AUC 0.84 (0.81 when the healthy side is also taken as the
  lower of two legs; 0.76 with the `Nside` label as paretic, 0.51 with `Pside`). Adding our Pi pass-level error
  (SD 5.3 left, 6.3 right; bias vs iPad -3.2 left, +0.4 right) lowers the AUC to 0.80-0.81 and the share of stroke
  knees caught at 95% specificity from 70% to 54-58%.
- Leg-to-leg swing-peak difference: healthy (Fukuchi overground slow, 41 people) median 3.2 deg, 95th percentile 7.2;
  stroke median 13.7, 64% above 7.2; AUC 0.86. Our Pi left-right bias relative to the iPad is 3.5 deg and has to be
  calibrated out before this marker is used.

**Where the gap is (v3.7.3, ours minus healthy mean):** early stance +4.7 to +8.7 deg (iPad), +8.8 to +14.1 (Pi);
around the swing peak -9.5 to -12.4 (iPad), -12.1 to -18.5 (Pi), against Moreira 0.56, Lencioni below 0.8 m/s and
Van Criekinge. Range of motion 4-19 deg smaller than healthy across the four equivalence references.

## Limits

Plausibility against other people, not accuracy on the same person. Published data are walking without a walker;
leaning on a walker at 0.55 m/s already lowers the swing peak and bends the stance knee. Our signals come from one
healthy person; stiff-knee passes are few (8 iPad, 3 per Pi knee). Van Criekinge stroke speeds are not in the release.

## Files

- `scripts/extract.py`: reads the Van Criekinge Excel files and Lencioni MAT files (python3 -I, paths as arguments).
- `scripts/analyse_ext.py`: all numbers in `results.json` and figures f9-f15 (needs `datasets_deep_20261124` outputs).
- `figures/`: f9 new references, f10 RMSE heatmap (27 conditions), f11 tolerance (7 datasets), f12 speed trend,
  f13 stroke curves, f14 per-person stroke map, f15 detection with our error and leg-to-leg difference;
  v3.7.3: f16 one panel per dataset, f17 features vs speed (Moreira, Fukuchi, Lencioni), f18 equivalence against
  four references, f19 difference curve against three references (`gap_by_ref`, `EQ4` in results.json).
- `analyse_ext.log`: console output.
