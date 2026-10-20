
#### Table VII, calibration A04 (replica / case schedule, k_test; baselines: mean ± SE over the 5 pairing groups of 20 draws; learners: over 5 seeds)

| Metric | Paper | Ours (mean ± SE, n) | Δ% | Judgment | Notes |
|---|---|---|---|---|---|
| U, Case I, ea | -24,200 | -21,040 ± 25 (n=5) | 13.1 | CLOSE | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case I, da | -19,400 | -20,507 ± 8 (n=5) | -5.7 | CLOSE | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case I, oa | -18,200 | -20,301 ± 16 (n=5) | -11.5 | CLOSE | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case I, oa_myopic | - | -20,306 ± 16 (n=5) | - | - | labelled addition; level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case I, ddpg[A04] | -19,000 | -5.09e+06 ± 5.06e+06 (n=5) | -26,673.3 | DIVERGENT | median -32,530; 1/5 seeds catastrophic (< -1e6); G = -23,190.38 [-87,421.86, 41,041.11]; sat 0.370; share 0.82 |
| U, Case I, dql[A04] | -19,600 | -24,665 ± 1,282 (n=5) | -25.8 | DIVERGENT | median -23,216; 0/5 seeds catastrophic (< -1e6); G = -20.18 [-36.48, -3.88]; sat 0.570; share 0.82 |
| U, Case II, ea | -41,000 | -36,937 ± 44 (n=5) | 9.9 | CLOSE | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case II, da | -39,000 | -34,081 ± 23 (n=5) | 12.6 | CLOSE | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case II, oa | -35,900 | -32,956 ± 18 (n=5) | 8.2 | CLOSE | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case II, oa_myopic | - | -32,981 ± 19 (n=5) | - | - | labelled addition; level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case II, ddpg[A04] | -35,400 | -41,410 ± 241 (n=5) | -17.0 | CLOSE | median -41,315; 0/5 seeds catastrophic (< -1e6); G = -6.51 [-7.11, -5.91]; sat 0.189; share 0.96 |
| U, Case II, dql[A04] | -35,900 | -36,654 ± 612 (n=5) | -2.1 | MATCH | median -36,999; 0/5 seeds catastrophic (< -1e6); G = -2.29 [-3.79, -0.78]; sat 0.693; share 0.86 |
| U, Case III (omega 0.5), ea | -13,700 | -4,629 ± 1 (n=5) | 66.2 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case III (omega 0.5), da | -7,960 | -3,826 ± 1 (n=5) | 51.9 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case III (omega 0.5), oa | -6,720 | -3,291 ± 1 (n=5) | 51.0 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case III (omega 0.5), oa_blind | - | -3,291 ± 1 (n=5) | - | - | OA without L_bm knowledge (= full-run OA); level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case III (omega 0.5), ddpg[A04] | -8,140 | -18,307 ± 59 (n=5) | -124.9 | DIVERGENT | median -18,365; 0/5 seeds catastrophic (< -1e6); G = -27.07 [-27.36, -26.78]; sat 0.0023; share 1.00 |
| U, Case III (omega 0.5), ddpg[A04](zero-shot I) | -8,140 | -8.58e+12 ± 4.56e+12 (n=5) | -1.05e+11 | DIVERGENT | median -3.63e+12; 3/5 seeds catastrophic (< -1e6); G = -1.61e+10 [-3.98e+10, 7.66e+09]; sat 0.245; share 0.80; secondary |
| U, Case III (omega 0.5), dql[A04] | -7,910 | -1.87e+13 ± 1.36e+13 (n=5) | -2.36e+11 | DIVERGENT | median -7.35e+12; 4/5 seeds catastrophic (< -1e6); G = -3.49e+10 [-1.06e+11, 3.57e+10]; sat 0.770; share 0.58 |
| U, Case III (omega 0.5), dql[A04](zero-shot I) | -7,910 | -2.62e+13 ± 1.61e+13 (n=5) | -3.31e+11 | DIVERGENT | median -4.47e+12; 3/5 seeds catastrophic (< -1e6); G = -4.89e+10 [-1.32e+11, 3.45e+10]; sat 0.612; share 0.68; secondary |
| U, Case III (omega 0.005), ea | -13,700 | -4,287 ± 1 (n=5) | 68.7 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case III (omega 0.005), da | -7,960 | -3,106 ± 1 (n=5) | 61.0 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case III (omega 0.005), oa | -6,720 | -2,718 ± 1 (n=5) | 59.6 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case III (omega 0.005), oa_blind | - | -2,718 ± 1 (n=5) | - | - | OA without L_bm knowledge (= full-run OA); level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case III (omega 0.005), ddpg[A04] | -8,140 | -17,979 ± 59 (n=5) | -120.9 | DIVERGENT | median -18,038; 0/5 seeds catastrophic (< -1e6); G = -38.34 [-38.76, -37.92]; sat 0.000303; share 1.00 |
| U, Case III (omega 0.005), ddpg[A04](zero-shot I) | -8,140 | -12,925 ± 2,730 (n=5) | -58.8 | DIVERGENT | median -15,938; 0/5 seeds catastrophic (< -1e6); G = -25.31 [-44.85, -5.77]; sat 0.296; share 0.77; secondary |
| U, Case III (omega 0.005), dql[A04] | -7,910 | -6,945 ± 2,236 (n=5) | 12.2 | CLOSE | median -4,713; 0/5 seeds catastrophic (< -1e6); G = -9.90 [-25.90, 6.11]; sat 0.719; share 0.59 |
| U, Case III (omega 0.005), dql[A04](zero-shot I) | -7,910 | -4,355 ± 366 (n=5) | 44.9 | DIVERGENT | median -4,186; 0/5 seeds catastrophic (< -1e6); G = -3.22 [-5.83, -0.61]; sat 0.601; share 0.62; secondary |
| U, Case IV, ea | -55,700 | -25,536 ± 30 (n=5) | 54.2 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case IV, da | -55,900 | -24,711 ± 2 (n=5) | 55.8 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case IV, oa | -50,500 | -24,452 ± 3 (n=5) | 51.6 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case IV, oa_blind | - | -24,744 ± 7 (n=5) | - | - | OA without region knowledge; level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case IV, ddpg[A04] | -51,100 | -1e+11 ± 6.57e+10 (n=5) | -1.97e+08 | DIVERGENT | median -35,242; 2/5 seeds catastrophic (< -1e6); G = -3.88e+08 [-1.1e+09, 3.23e+08]; sat 0.398; share 0.76 |
| U, Case IV, dql[A04] | -50,600 | -28,235 ± 1,714 (n=5) | 44.2 | DIVERGENT | median -26,655; 0/5 seeds catastrophic (< -1e6); G = -13.46 [-31.32, 4.40]; sat 0.568; share 0.85 |

#### Table VI (Case I T_init, slots), calibration A04

| Metric | Paper | Ours (mean ± SE, n) | Δ% | Judgment | Notes |
|---|---|---|---|---|---|
| T_init (slots), ea | 10.5 | 841.3 ± 4.3 (n=5) | 7,912.6 | DIVERGENT | scan 15,144 |
| T_init (slots), da | 14.67 | 841.3 ± 4.3 (n=5) | 5,635.0 | DIVERGENT | scan 15,144 |
| T_init (slots), oa | 20.33 | 841.3 ± 4.3 (n=5) | 4,038.3 | DIVERGENT | scan 15,144 |
| T_init (slots), oa_myopic | - | 841.3 ± 4.3 (n=5) | - | - | scan 15,144 |
| T_init (slots), ddpg[A04] | 6.0 | 781.4 ± 294.2 (n=5) | 12,923.7 | DIVERGENT | scan 14,159 |
| T_init (slots), dql[A04] | 7.83 | 445.6 ± 78.9 (n=5) | 5,590.5 | DIVERGENT | scan 8,020 |

#### Family H1-H9, calibration A04 (PRIMARY; paired t over 5 seed-level pairs, Holm over 9)

| id | learner (mean ± SE) | reference (mean ± SE) | diff [t95] | boot95 | rel | p | p_Holm | verdict |
|---|---|---|---|---|---|---|---|---|
| H1 | -5.09e+06 ± 5.06e+06 (n=5) | -21,040 ± 25 (n=5) | -5.07e+06 [-1.91e+07, 8.97e+06] | [-1.52e+07, -8,074.7] | -24,077.4 % | 0.8135 | 1.0000 | NOT SUPPORTED |
| H2 | -24,665 ± 1,282 (n=5) | -21,040 ± 25 (n=5) | -3,625.4 [-7,173.2, -77.5] | [-6,123.9, -1,702.0] | -17.2 % | 0.9765 | 1.0000 | NOT SUPPORTED |
| H3 | -5.09e+06 ± 5.06e+06 (n=5) | -20,507 ± 8 (n=5) | -5.07e+06 [-1.91e+07, 8.97e+06] | [-1.52e+07, -8,644.6] | -24,706.1 % | 0.8135 | 1.0000 | NOT SUPPORTED |
| H4 | -24,665 ± 1,282 (n=5) | -20,507 ± 8 (n=5) | -4,158.7 [-7,723.2, -594.2] | [-6,700.1, -2,227.0] | -20.3 % | 0.9842 | 1.0000 | NOT SUPPORTED |
| H5 | -41,410 ± 241 (n=5) | -34,081 ± 23 (n=5) | -7,328.8 [-7,995.8, -6,661.9] | [-7,807.9, -6,946.4] | -21.5 % | 1.0000 | 1.0000 | NOT SUPPORTED |
| H6 | -36,654 ± 612 (n=5) | -34,081 ± 23 (n=5) | -2,573.1 [-4,259.4, -886.8] | [-3,489.8, -1,416.6] | -7.5 % | 0.9933 | 1.0000 | NOT SUPPORTED |
| H7 | 781.4 ± 294.2 (n=5) | 841.3 ± 4.3 (n=5) | -59.9 [-868.9, 749.2] | [-575.7, 455.9] | -7.1 % | 0.4236 | 1.0000 | NOT SUPPORTED |
| H8 | -5.09e+06 ± 5.06e+06 (n=5) | -20,301 ± 16 (n=5) | -5.07e+06 [-1.91e+07, 8.97e+06] | [-1.52e+07, -8,858.7] | -24,957.0 % | 0.3729 | 1.0000 | no detectable difference |
| H9 | -41,410 ± 241 (n=5) | -32,956 ± 18 (n=5) | -8,454.0 [-9,121.0, -7,787.0] | [-8,935.4, -8,067.7] | -25.7 % | 3.89e-06 | 3.5e-05 | OA better |

#### Table VII, calibration A04a (replica / case schedule, k_test; baselines: mean ± SE over the 5 pairing groups of 20 draws; learners: over 5 seeds)

| Metric | Paper | Ours (mean ± SE, n) | Δ% | Judgment | Notes |
|---|---|---|---|---|---|
| U, Case I, ea | -24,200 | -30,833 ± 52 (n=5) | -27.4 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case I, da | -19,400 | -24,665 ± 29 (n=5) | -27.1 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case I, oa | -18,200 | -22,330 ± 18 (n=5) | -22.7 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case I, oa_myopic | - | -22,423 ± 19 (n=5) | - | - | labelled addition; level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case I, ddpg[A04a] | -19,000 | -33,328 ± 1,348 (n=5) | -75.4 | DIVERGENT | median -34,705; 0/5 seeds catastrophic (< -1e6); G = -3.72 [-5.36, -2.07]; sat 0.191; share 0.97 |
| U, Case I, dql[A04a] | -19,600 | -25,985 ± 2,353 (n=5) | -32.6 | DIVERGENT | median -23,614; 0/5 seeds catastrophic (< -1e6); G = -0.58 [-3.45, 2.28]; sat 0.564; share 0.86 |
| U, Case II, ea | -41,000 | -50,127 ± 60 (n=5) | -22.3 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case II, da | -39,000 | -43,776 ± 44 (n=5) | -12.2 | CLOSE | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case II, oa | -35,900 | -41,607 ± 43 (n=5) | -15.9 | CLOSE | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case II, oa_myopic | - | -42,005 ± 45 (n=5) | - | - | labelled addition; level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case II, ddpg[A04a] | -35,400 | -45,464 ± 1,727 (n=5) | -28.4 | DIVERGENT | median -43,740; 0/5 seeds catastrophic (< -1e6); G = -0.78 [-2.98, 1.43]; sat 0.135; share 0.99 |
| U, Case II, dql[A04a] | -35,900 | -42,776 ± 817 (n=5) | -19.2 | CLOSE | median -43,816; 0/5 seeds catastrophic (< -1e6); G = 0.46 [-0.59, 1.52]; sat 0.470; share 0.89 |
| U, Case III (omega 0.5), ea | -13,700 | -4,618 ± 0 (n=5) | 66.3 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case III (omega 0.5), da | -7,960 | -3,814 ± 1 (n=5) | 52.1 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case III (omega 0.5), oa | -6,720 | -3,279 ± 0 (n=5) | 51.2 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case III (omega 0.5), oa_blind | - | -3,279 ± 0 (n=5) | - | - | OA without L_bm knowledge (= full-run OA); level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case III (omega 0.5), ddpg[A04a] | -8,140 | -2.07e+13 ± 2.07e+13 (n=5) | -2.54e+11 | DIVERGENT | median -18,366; 1/5 seeds catastrophic (< -1e6); G = -3.87e+10 [-1.46e+11, 6.87e+10]; sat 0.082; share 0.92 |
| U, Case III (omega 0.5), ddpg[A04a](zero-shot I) | -8,140 | -15,311 ± 2,982 (n=5) | -88.1 | DIVERGENT | median -18,365; 0/5 seeds catastrophic (< -1e6); G = -21.50 [-36.98, -6.02]; sat 0.137; share 0.93; secondary |
| U, Case III (omega 0.5), dql[A04a] | -7,910 | -5.22e+12 ± 2.96e+12 (n=5) | -6.6e+10 | DIVERGENT | median -4.49e+12; 4/5 seeds catastrophic (< -1e6); G = -9.78e+09 [-2.52e+10, 5.61e+09]; sat 0.791; share 0.47 |
| U, Case III (omega 0.5), dql[A04a](zero-shot I) | -7,910 | -5.66e+13 ± 2.33e+13 (n=5) | -7.15e+11 | DIVERGENT | median -8.11e+13; 5/5 seeds catastrophic (< -1e6); G = -1.06e+11 [-2.26e+11, 1.5e+10]; sat 0.687; share 0.64; secondary |
| U, Case III (omega 0.005), ea | -13,700 | -4,276 ± 0 (n=5) | 68.8 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case III (omega 0.005), da | -7,960 | -3,094 ± 0 (n=5) | 61.1 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case III (omega 0.005), oa | -6,720 | -2,706 ± 0 (n=5) | 59.7 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case III (omega 0.005), oa_blind | - | -2,706 ± 0 (n=5) | - | - | OA without L_bm knowledge (= full-run OA); level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case III (omega 0.005), ddpg[A04a] | -8,140 | -2.08e+13 ± 2.08e+13 (n=5) | -2.56e+11 | DIVERGENT | median -18,038; 1/5 seeds catastrophic (< -1e6); G = -5.37e+10 [-2.03e+11, 9.53e+10]; sat 0.080; share 0.92 |
| U, Case III (omega 0.005), ddpg[A04a](zero-shot I) | -8,140 | -15,013 ± 2,952 (n=5) | -84.4 | DIVERGENT | median -18,038; 0/5 seeds catastrophic (< -1e6); G = -30.71 [-51.83, -9.60]; sat 0.142; share 0.93; secondary |
| U, Case III (omega 0.005), dql[A04a] | -7,910 | -3e+12 ± 3e+12 (n=5) | -3.79e+10 | DIVERGENT | median -5,033; 1/5 seeds catastrophic (< -1e6); G = -7.73e+09 [-2.92e+10, 1.37e+10]; sat 0.808; share 0.45 |
| U, Case III (omega 0.005), dql[A04a](zero-shot I) | -7,910 | -3,581 ± 243 (n=5) | 54.7 | DIVERGENT | median -3,269; 0/5 seeds catastrophic (< -1e6); G = -1.25 [-2.99, 0.49]; sat 0.659; share 0.64; secondary |
| U, Case IV, ea | -55,700 | -36,979 ± 54 (n=5) | 33.6 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case IV, da | -55,900 | -31,861 ± 33 (n=5) | 43.0 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case IV, oa | -50,500 | -30,815 ± 29 (n=5) | 39.0 | DIVERGENT | level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case IV, oa_blind | - | -34,032 ± 33 (n=5) | - | - | OA without region knowledge; level anchor-dependent (SNR0 fitted to Fig. 8) |
| U, Case IV, ddpg[A04a] | -51,100 | -37,010 ± 817 (n=5) | 27.6 | DIVERGENT | median -37,766; 0/5 seeds catastrophic (< -1e6); G = -4.93 [-7.18, -2.69]; sat 0.193; share 0.97 |
| U, Case IV, dql[A04a] | -50,600 | -3.68e+06 ± 3.64e+06 (n=5) | -7,163.2 | DIVERGENT | median -31,881; 1/5 seeds catastrophic (< -1e6); G = -3,534.54 [-13,348.31, 6,279.24]; sat 0.520; share 0.88 |

#### Table VI (Case I T_init, slots), calibration A04a

| Metric | Paper | Ours (mean ± SE, n) | Δ% | Judgment | Notes |
|---|---|---|---|---|---|
| T_init (slots), ea | 10.5 | 17.3 ± 2.1 (n=5) | 64.3 | DIVERGENT (calibrated) | scan 311; CALIBRATED (A-04a): S_ref fitted to this table's pooled EA/DA/OA value, not a test |
| T_init (slots), da | 14.67 | 17.3 ± 2.1 (n=5) | 17.6 | CLOSE (calibrated) | scan 311; CALIBRATED (A-04a): S_ref fitted to this table's pooled EA/DA/OA value, not a test |
| T_init (slots), oa | 20.33 | 17.3 ± 2.1 (n=5) | -15.1 | CLOSE (calibrated) | scan 311; CALIBRATED (A-04a): S_ref fitted to this table's pooled EA/DA/OA value, not a test |
| T_init (slots), oa_myopic | - | 17.3 ± 2.1 (n=5) | - | - (calibrated) | scan 311; CALIBRATED (A-04a): S_ref fitted to this table's pooled EA/DA/OA value, not a test |
| T_init (slots), ddpg[A04a] | 6.0 | 1,003.4 ± 224.8 (n=5) | 16,624.2 | DIVERGENT | scan 18,648 |
| T_init (slots), dql[A04a] | 7.83 | 207.0 ± 116.8 (n=5) | 2,543.6 | DIVERGENT | scan 3,726 |

#### Family H1-H9, calibration A04a (sensitivity; paired t over 5 seed-level pairs, Holm over 9)

| id | learner (mean ± SE) | reference (mean ± SE) | diff [t95] | boot95 | rel | p | p_Holm | verdict |
|---|---|---|---|---|---|---|---|---|
| H1 | -33,328 ± 1,348 (n=5) | -30,833 ± 52 (n=5) | -2,494.4 [-6,332.3, 1,343.6] | [-3,964.5, 277.8] | -8.1 % | 0.9273 | 1.0000 | NOT SUPPORTED |
| H2 | -25,985 ± 2,353 (n=5) | -30,833 ± 52 (n=5) | 4,849.0 [-1,813.1, 11,511.1] | [-133.6, 7,786.4] | 15.7 % | 0.0567 | 0.4536 | NOT SUPPORTED |
| H3 | -33,328 ± 1,348 (n=5) | -24,665 ± 29 (n=5) | -8,662.8 [-12,458.5, -4,867.1] | [-10,099.3, -5,913.1] | -35.1 % | 0.9984 | 1.0000 | NOT SUPPORTED |
| H4 | -25,985 ± 2,353 (n=5) | -24,665 ± 29 (n=5) | -1,319.4 [-7,924.7, 5,285.8] | [-6,262.2, 1,589.3] | -5.3 % | 0.6956 | 1.0000 | NOT SUPPORTED |
| H5 | -45,464 ± 1,727 (n=5) | -43,776 ± 44 (n=5) | -1,688.0 [-6,468.5, 3,092.5] | [-5,159.8, 156.0] | -3.9 % | 0.8088 | 1.0000 | NOT SUPPORTED |
| H6 | -42,776 ± 817 (n=5) | -43,776 ± 44 (n=5) | 1,000.5 [-1,289.6, 3,290.5] | [-382.6, 2,492.7] | 2.3 % | 0.1459 | 0.8756 | NOT SUPPORTED |
| H7 | 1,003.4 ± 224.8 (n=5) | 17.3 ± 2.1 (n=5) | 986.2 [365.8, 1,606.6] | [538.7, 1,220.4] | 5,716.5 % | 0.9942 | 1.0000 | NOT SUPPORTED |
| H8 | -33,328 ± 1,348 (n=5) | -22,330 ± 18 (n=5) | -10,998.2 [-14,774.6, -7,221.8] | [-12,421.9, -8,259.0] | -49.3 % | 0.00127 | 0.0114 | OA better |
| H9 | -45,464 ± 1,727 (n=5) | -41,607 ± 43 (n=5) | -3,857.3 [-8,637.6, 922.9] | [-7,328.9, -2,015.8] | -9.3 % | 0.0886 | 0.6201 | no detectable difference |

#### Comparable claims, A04

- C3_ddpg_vs_oa_I: rel -24,957.0 %, boot95 [-74,816.8, -43.7] % -> CONTRADICTED
- C3_dql_vs_oa_I: rel -21.5 %, boot95 [-34.1, -11.9] % -> CONTRADICTED
- C6_ddpg_vs_dql_case1: rel -20,523.7 %, boot95 [-56,767.5, -23.9] % -> CONTRADICTED
- C6_ddpg_vs_dql_case2: rel -13.0 %, boot95 [-16.8, -9.6] % -> CONTRADICTED
- C6_ddpg_vs_dql_case3f: rel 100.0 %, boot95 [100.0, 100.0] % -> CONTRADICTED
- C6_ddpg_vs_dql_case4: rel -3.56e+08 %, boot95 [-8.14e+08, -25.2] % -> CONTRADICTED
- C5_oa_minus_ddpg_spawn: diff 2.21e+10 t95 [-2.34e+10, 6.76e+10] p 0.2485
- C5_oa_minus_dql_spawn: diff 1,834 t95 [328, 3,340] p 0.0277
- C5_test_type: ONE-SAMPLE t of the 5 learner seed means against OA's mean over the 50 shared spawn episodes (oa_sp is a constant vector; OA's episode variance is not in the test)

#### Predictions P1-P6, calibration A04

{
 "P1": {
  "identical": true,
  "verdict": "GO"
 },
 "P2": {
  "scan_share_of_U": {
   "ea": 0.7197788622074774,
   "da": 0.7384703814726582,
   "oa": 0.7459325098050534,
   "oa_myopic": 0.7457753540783222,
   "ddpg": 0.4345600119223234,
   "dql": 0.331215951530521
  },
  "verdict": "KILL"
 },
 "P3": {
  "c_ea": 5896.330035529495,
  "c_da": 5363.006424139155,
  "c_oa": 5157.737035762935,
  "ordering_holds": true,
  "verdict": "GO"
 },
 "P4": {
  "ddpg": {
   "g_mean": -0.18835386644185578,
   "viol_near_frac": 0.033541609102285284,
   "verdict": "KILL"
  },
  "dql": {
   "g_mean": -0.1823799,
   "viol_near_frac": 0.017118496277982655,
   "verdict": "KILL"
  }
 },
 "P5": {
  "ddpg": {
   "lam_multi": 411514.38783820777,
   "verdict": "KILL"
  },
  "dql": {
   "lam_multi": 40368.212568173745,
   "verdict": "GO"
  }
 },
 "P6": {
  "ddpg": {
   "c": 5072772.468272134,
   "c_oa_myopic": 5162.018665348888,
   "se_myopic": 61.41630711196545,
   "track_share": 0.8155360470791285,
   "verdict": "GO"
  },
  "dql": {
   "c": 16645.177336632845,
   "c_oa_myopic": 5162.018665348888,
   "se_myopic": 61.41630711196545,
   "track_share": 0.8216297897308122,
   "verdict": "GO"
  }
 }
}

#### Comparable claims, A04a

- C3_ddpg_vs_oa_I: rel -49.3 %, boot95 [-55.7, -37.0] % -> CONTRADICTED
- C3_dql_vs_oa_I: rel -16.4 %, boot95 [-37.4, -3.4] % -> UNDETERMINED
- C6_ddpg_vs_dql_case1: rel -28.3 %, boot95 [-45.2, -10.0] % -> CONTRADICTED
- C6_ddpg_vs_dql_case2: rel -6.3 %, boot95 [-12.8, -0.5] % -> UNDETERMINED
- C6_ddpg_vs_dql_case3f: rel -295.4 %, boot95 [-2,106.9, 100.0] % -> UNDETERMINED
- C6_ddpg_vs_dql_case4: rel 99.0 %, boot95 [-19.4, 99.7] % -> UNDETERMINED
- C5_oa_minus_ddpg_spawn: diff 4.29e+09 t95 [-7.62e+09, 1.62e+10] p 0.3739
- C5_oa_minus_dql_spawn: diff 1.44e+07 t95 [-2.56e+07, 5.45e+07] p 0.3739
- C5_test_type: ONE-SAMPLE t of the 5 learner seed means against OA's mean over the 50 shared spawn episodes (oa_sp is a constant vector; OA's episode variance is not in the test)

#### Predictions P1-P6, calibration A04a

{
 "P1": {
  "identical": true,
  "verdict": "GO"
 },
 "P2": {
  "scan_share_of_U": {
   "ea": 0.010079469424073714,
   "da": 0.012597035747308537,
   "oa": 0.013912158097644975,
   "oa_myopic": 0.013854102137308377,
   "ddpg": 0.5404114055096876,
   "dql": 0.12385735606417592
  },
  "verdict": "KILL"
 },
 "P3": {
  "c_ea": 30522.95174469792,
  "c_da": 24354.531088706437,
  "c_oa": 22019.10285686306,
  "ordering_holds": true,
  "verdict": "GO"
 },
 "P4": {
  "ddpg": {
   "g_mean": -0.05039270593748605,
   "viol_near_frac": 0.044086472574023906,
   "verdict": "KILL"
  },
  "dql": {
   "g_mean": -0.1489603,
   "viol_near_frac": 0.022681519796842645,
   "verdict": "KILL"
  }
 },
 "P5": {
  "ddpg": {
   "lam_multi": 1504164.7471675943,
   "verdict": "KILL"
  },
  "dql": {
   "lam_multi": 69109.20468739858,
   "verdict": "GO"
  }
 },
 "P6": {
  "ddpg": {
   "c": 14680.046484797014,
   "c_oa_myopic": 22112.87562381302,
   "se_myopic": 56.21727548774868,
   "track_share": 0.9732056765086512,
   "verdict": "GO"
  },
  "dql": {
   "c": 22258.652725209657,
   "c_oa_myopic": 22112.87562381302,
   "se_myopic": 56.21727548774868,
   "track_share": 0.8598109490324543,
   "verdict": "GO"
  }
 }
}

#### Ablation / debug

{
 "ddpg": {
  "fixlam": {
   "U": "-25,679.303 \u00b1 1,564.561",
   "sat": "0.448 \u00b1 0.062",
   "g": "-0.179 \u00b1 0.024",
   "lam_max": "5,000.000 \u00b1 0.000",
   "share": "0.825 \u00b1 0.028",
   "tinit": "457.163 \u00b1 152.679",
   "c": "17,450.363 \u00b1 1,522.167",
   "G": "-25.492 \u00b1 7.911",
   "sat_lower_p": 0.7107759236709259
  },
  "nolam": {
   "U": "-24,600.471 \u00b1 924.755",
   "sat": "0.432 \u00b1 0.044",
   "g": "-0.177 \u00b1 0.021",
   "lam_max": "0.000 \u00b1 0.000",
   "share": "0.828 \u00b1 0.024",
   "tinit": "444.115 \u00b1 119.806",
   "c": "16,606.401 \u00b1 2,993.142",
   "G": "-19.940 \u00b1 4.650",
   "sat_lower_p": 0.6853157000213153
  },
  "rule": "DOES NOT SUPPORT",
  "A04": {
   "U": "-5.09e+06 \u00b1 5.06e+06",
   "sat": "0.370 \u00b1 0.100",
   "g": "-0.188 \u00b1 0.068",
   "lam_max": "1.13e+06 \u00b1 543,995.518",
   "share": "0.816 \u00b1 0.079",
   "tinit": "781.424 \u00b1 294.199",
   "c": "5.07e+06 \u00b1 5.06e+06",
   "G": "-23,190.375 \u00b1 23,134.433"
  }
 },
 "dql": {
  "fixlam": {
   "U": "-23,197.967 \u00b1 765.978",
   "sat": "0.685 \u00b1 0.062",
   "g": "-0.189 \u00b1 0.019",
   "lam_max": "5,000.000 \u00b1 0.000",
   "share": "0.814 \u00b1 0.022",
   "tinit": "464.552 \u00b1 99.396",
   "c": "14,836.037 \u00b1 2,506.117",
   "G": "-12.861 \u00b1 3.580",
   "sat_lower_p": 0.9407831210139748
  },
  "nolam": {
   "U": "-25,070.400 \u00b1 2,091.662",
   "sat": "0.434 \u00b1 0.073",
   "g": "-0.117 \u00b1 0.016",
   "lam_max": "0.000 \u00b1 0.000",
   "share": "0.897 \u00b1 0.018",
   "tinit": "683.990 \u00b1 134.165",
   "c": "12,758.580 \u00b1 1,264.400",
   "G": "-22.438 \u00b1 10.638",
   "sat_lower_p": 0.05604324999980255
  },
  "rule": "DOES NOT SUPPORT",
  "A04": {
   "U": "-24,665.437 \u00b1 1,282.020",
   "sat": "0.570 \u00b1 0.024",
   "g": "-0.182 \u00b1 0.013",
   "lam_max": "116,580.000 \u00b1 19,720.084",
   "share": "0.822 \u00b1 0.015",
   "tinit": "445.570 \u00b1 78.855",
   "c": "16,645.177 \u00b1 2,164.709",
   "G": "-20.177 \u00b1 5.872"
  }
 },
 "debug_ddpg": {
  "dual50": {
   "ref": "A04",
   "U": "-27,390.392 \u00b1 1,379.249",
   "sat": "0.294 \u00b1 0.046",
   "g": "-0.136 \u00b1 0.033",
   "lam_max": "15,930.320 \u00b1 3,541.855",
   "share": "0.875 \u00b1 0.038",
   "tinit": "712.921 \u00b1 184.759",
   "c": "14,432.252 \u00b1 1,992.408",
   "G": "-32.909 \u00b1 5.347",
   "p_one_sided": 0.18677933419315204,
   "G_world": "-32.909 \u00b1 5.347",
   "verdict": "REFUTED"
  },
  "raw": {
   "ref": "A04",
   "U": "-2.41e+10 \u00b1 1.55e+10",
   "sat": "0.143 \u00b1 0.00902",
   "g": "-0.043 \u00b1 0.00902",
   "lam_max": "2.44e+06 \u00b1 4,967.541",
   "share": "0.983 \u00b1 0.010",
   "tinit": "1,472.596 \u00b1 18.246",
   "c": "2.41e+10 \u00b1 1.55e+10",
   "G": "-1.15e+08 \u00b1 7.15e+07",
   "p_one_sided": 0.9021993704842378,
   "G_world": "-1.15e+08 \u00b1 7.15e+07",
   "verdict": "REFUTED"
  },
  "A04a_dual50": {
   "ref": "A04a",
   "U": "-30,137.271 \u00b1 3,512.891",
   "sat": "0.306 \u00b1 0.016",
   "g": "-0.141 \u00b1 0.017",
   "lam_max": "15,026.218 \u00b1 570.438",
   "share": "0.869 \u00b1 0.020",
   "tinit": "353.828 \u00b1 122.294",
   "c": "23,768.361 \u00b1 5,237.662",
   "G": "-48.395 \u00b1 19.372",
   "p_one_sided": 0.19634652178980205,
   "G_world": "-2.344 \u00b1 1.500",
   "verdict": "REFUTED"
  },
  "A04a": {
   "ref": "A04",
   "U": "-33,327.866 \u00b1 1,348.124",
   "sat": "0.191 \u00b1 0.064",
   "g": "-0.050 \u00b1 0.023",
   "lam_max": "2.02e+06 \u00b1 472,551.779",
   "share": "0.973 \u00b1 0.027",
   "tinit": "1,003.450 \u00b1 224.839",
   "c": "14,680.046 \u00b1 2,846.478",
   "G": "-62.746 \u00b1 7.011",
   "p_one_sided": 0.18701155959627336,
   "G_world": "-3.717 \u00b1 0.592",
   "verdict": "n/a (different world: compare G, not U)"
  }
 },
 "debug_dql": {
  "dual50": {
   "ref": "A04",
   "U": "-22,562.098 \u00b1 791.299",
   "sat": "0.516 \u00b1 0.042",
   "g": "-0.134 \u00b1 0.00831",
   "lam_max": "9,564.400 \u00b1 406.546",
   "share": "0.878 \u00b1 0.00965",
   "tinit": "555.406 \u00b1 78.144",
   "c": "12,419.458 \u00b1 1,399.351",
   "G": "-10.487 \u00b1 4.302",
   "p_one_sided": 0.10428980418391001,
   "G_world": "-10.487 \u00b1 4.302",
   "verdict": "REFUTED"
  },
  "A04a_dual50": {
   "ref": "A04a",
   "U": "-22,207.845 \u00b1 649.947",
   "sat": "0.407 \u00b1 0.024",
   "g": "-0.090 \u00b1 0.00868",
   "lam_max": "15,943.900 \u00b1 1,182.401",
   "share": "0.927 \u00b1 0.00994",
   "tinit": "347.715 \u00b1 130.945",
   "c": "15,948.975 \u00b1 2,804.481",
   "G": "-8.447 \u00b1 3.484",
   "p_one_sided": 0.10904565706019931,
   "G_world": "1.054 \u00b1 0.275",
   "verdict": "REFUTED"
  },
  "A04a": {
   "ref": "A04",
   "U": "-25,984.503 \u00b1 2,353.303",
   "sat": "0.564 \u00b1 0.041",
   "g": "-0.149 \u00b1 0.011",
   "lam_max": "136,860.000 \u00b1 17,071.075",
   "share": "0.860 \u00b1 0.013",
   "tinit": "206.992 \u00b1 116.777",
   "c": "22,258.653 \u00b1 494.233",
   "G": "-25.908 \u00b1 10.540",
   "p_one_sided": 0.8031608049921184,
   "G_world": "-0.582 \u00b1 1.032",
   "verdict": "n/a (different world: compare G, not U)"
  }
 }
}

#### Curves (A-18 plateau, lambda)

- ddpg_case1[A04]: plateau 3/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['2.17e+07', '5.91e+06', '1.27e+08', '1.05e+08', '2.97e+06']
- ddpg_case1[A04a]: plateau 4/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['8.86e+07', '187,593', '1.29e+08', '2.86e+07', '1e+08']
- ddpg_case1[dual50]: plateau 1/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['411,782', '467,021', '267,634', '291,053', '1.13e+06']
- ddpg_case1[A04a_dual50]: plateau 1/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = True, max chunk-mean lambda per seed ['88,564', '552,474', '464,118', '157,605', '48,125']
- ddpg_case1[fixlam]: plateau 1/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = True, max chunk-mean lambda per seed ['5,000', '5,000', '5,000', '5,000', '5,000']
- ddpg_case1[nolam]: plateau 2/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = True, max chunk-mean lambda per seed ['0', '0', '0', '0', '0']
- ddpg_case1[raw]: plateau 5/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['1.02e+08', '1.07e+08', '1.2e+08', '3.92e+07', '6.55e+07']
- ddpg_case2[A04]: plateau 4/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['1.09e+08', '6.06e+06', '1.1e+08', '1.21e+08', '1.11e+08']
- ddpg_case2[A04a]: plateau 4/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['1.13e+08', '1.17e+08', '1.12e+08', '1.24e+08', '6.12e+07']
- ddpg_case3[A04]: plateau 5/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['1.38e+08', '1.39e+08', '1.36e+08', '1.31e+08', '1.31e+08']
- ddpg_case3[A04a]: plateau 5/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['8.04e+07', '1.4e+08', '1.38e+08', '1.18e+08', '2.8e+07']
- dql_case1[A04]: plateau 2/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = True, max chunk-mean lambda per seed ['258,648', '169,871', '182,457', '216,154', '336,086']
- dql_case1[A04a]: plateau 2/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = True, max chunk-mean lambda per seed ['319,130', '191,330', '167,255', '246,652', '363,705']
- dql_case1[dual50]: plateau 1/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = True, max chunk-mean lambda per seed ['19,272', '34,589', '23,382', '44,549', '38,812']
- dql_case1[A04a_dual50]: plateau 1/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = True, max chunk-mean lambda per seed ['36,693', '33,267', '53,076', '47,157', '47,506']
- dql_case1[fixlam]: plateau 1/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = True, max chunk-mean lambda per seed ['5,000', '5,000', '5,000', '5,000', '5,000']
- dql_case1[nolam]: plateau 3/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = True, max chunk-mean lambda per seed ['0', '0', '0', '0', '0']
- dql_case2[A04]: plateau 0/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = True, max chunk-mean lambda per seed ['448,457', '404,570', '292,831', '350,664', '633,649']
- dql_case2[A04a]: plateau 2/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = True, max chunk-mean lambda per seed ['315,977', '409,850', '315,881', '421,327', '798,676']
- dql_case3[A04]: plateau 0/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['303,551', '713,900', '1.32e+06', '4.77e+06', '2.24e+06']
- dql_case3[A04a]: plateau 1/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['389,713', '6.09e+06', '1.32e+06', '4.93e+06', '2.27e+06']
- recheck_ddpg_case1[ctl]: plateau 2/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['2.17e+07', '5.91e+06', '1.27e+08', '1.05e+08', '2.97e+06']
- recheck_ddpg_case1[req]: plateau 3/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['1.27e+07', '4.11e+06', '3.93e+08', '7.59e+06', '1.87e+07']
- recheck_ddpg_case1[lit]: plateau 2/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['1.27e+07', '4.11e+06', '6.59e+06', '7.59e+06', '1.24e+09']
- recheck_ddpg_case1[n0zero]: plateau 2/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = True, max chunk-mean lambda per seed ['147,089', '650,102', '430,300', '491,076', '163,277']
- recheck_ddpg_case3[w_fast]: plateau 5/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['6.42e+06', '1.35e+08', '1.44e+06', '1.09e+08', '1.36e+06']
- recheck_ddpg_case3[w_slow]: plateau 3/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['1.31e+08', '1.36e+08', '1.36e+08', '1.15e+06', '8.5e+06']
- recheck_dql_case1[ctl]: plateau 3/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['273,160', '475,065', '519,180', '2.52e+06', '2.79e+06']
- recheck_dql_case1[req]: plateau 4/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['2.96e+07', '9.38e+07', '2.8e+07', '9.22e+07', '9e+07']
- recheck_dql_case1[lit]: plateau 3/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['2.96e+07', '9.19e+07', '2.8e+07', '9.33e+07', '8.92e+07']
- recheck_dql_case1[n0zero]: plateau 1/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = True, max chunk-mean lambda per seed ['50,118', '43,126', '11,066', '32,659', '83,276']
- recheck_dql_case1[eps01]: plateau 2/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = True, max chunk-mean lambda per seed ['553,382', '204,785', '193,086', '202,612', '169,348']
- recheck_dql_case3[w_fast]: plateau 0/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['712,431', '1.75e+06', '3.41e+06', '5.95e+06', '6.66e+06']
- recheck_dql_case3[w_slow]: plateau 0/5, windup fixed (max chunk-mean lambda < 1e6 in all seeds) = False, max chunk-mean lambda per seed ['288,017', '531,311', '1.37e+06', '2.99e+06', '2.14e+06']

#### C8

- case1|ddpg_case1[A04]: mean 0.059 t95 [-0.756, 0.873] -> undetermined
- case1|ddpg_case1[A04a]: mean 0.374 t95 [nan, nan] -> undetermined
- case1|dql_case1[A04]: mean 0.535 t95 [0.222, 0.847] -> positive
- case1|dql_case1[A04a]: mean 0.740 t95 [0.589, 0.891] -> positive
- case1|ea[A04]: mean - t95 [nan, nan] -> undetermined
- case1|ea[A04a]: mean 0.244 t95 [0.237, 0.251] -> positive
- case1|da[A04]: mean 0.820 t95 [0.820, 0.820] -> positive
- case1|da[A04a]: mean 0.881 t95 [0.874, 0.888] -> positive
- case1|oa[A04]: mean 0.820 t95 [0.820, 0.820] -> positive
- case1|oa[A04a]: mean 0.908 t95 [0.903, 0.913] -> positive
- case2|ddpg_case2[A04]: mean -0.028 t95 [nan, nan] -> undetermined
- case2|ddpg_case2[A04a]: mean -0.298 t95 [nan, nan] -> undetermined
- case2|dql_case2[A04]: mean -0.110 t95 [-0.245, 0.026] -> undetermined
- case2|dql_case2[A04a]: mean -0.339 t95 [-0.481, -0.198] -> negative
- case2|ea[A04]: mean - t95 [nan, nan] -> undetermined
- case2|ea[A04a]: mean -0.349 t95 [-0.352, -0.347] -> negative
- case2|da[A04]: mean -0.044 t95 [-0.044, -0.043] -> negative
- case2|da[A04a]: mean -0.771 t95 [-0.773, -0.770] -> negative
- case2|oa[A04]: mean -0.042 t95 [-0.043, -0.042] -> negative
- case2|oa[A04a]: mean -0.525 t95 [-0.531, -0.519] -> negative
- case4|ddpg_case1[A04]: mean -0.037 t95 [-0.293, 0.220] -> undetermined
- case4|ddpg_case1[A04a]: mean 0.124 t95 [-0.151, 0.399] -> undetermined
- case4|dql_case1[A04]: mean -0.161 t95 [-0.393, 0.071] -> undetermined
- case4|dql_case1[A04a]: mean -0.382 t95 [-0.584, -0.181] -> negative
- case4|ea[A04]: mean -0.167 t95 [-0.168, -0.166] -> negative
- case4|ea[A04a]: mean -0.084 t95 [-0.086, -0.082] -> negative
- case4|da[A04]: mean -0.391 t95 [-0.392, -0.390] -> negative
- case4|da[A04a]: mean -0.535 t95 [-0.537, -0.534] -> negative
- case4|oa[A04]: mean -0.367 t95 [-0.367, -0.366] -> negative
- case4|oa[A04a]: mean -0.564 t95 [-0.571, -0.558] -> negative
- case4|oa_blind[A04]: mean -0.616 t95 [-0.617, -0.616] -> negative
- case4|oa_blind[A04a]: mean -0.926 t95 [-0.927, -0.924] -> negative
- case3|ddpg_case3[A04]: mean 0.000 t95 [0.000, 0.000] -> undetermined
- case3|ddpg_case3[A04a]: mean 0.000 t95 [0.000, 0.000] -> undetermined
- case3|dql_case3[A04]: mean -0.054 t95 [-0.287, 0.179] -> undetermined
- case3|dql_case3[A04a]: mean -0.147 t95 [-0.562, 0.269] -> undetermined
- case3|ea[A04]: mean -0.00135 t95 [-0.00135, -0.00135] -> negative
- case3|ea[A04a]: mean -0.00135 t95 [-0.00135, -0.00135] -> negative
- case3|da[A04]: mean 0.177 t95 [0.175, 0.178] -> positive
- case3|da[A04a]: mean 0.177 t95 [0.175, 0.178] -> positive
- case3|oa[A04]: mean 0.320 t95 [0.318, 0.322] -> positive
- case3|oa[A04a]: mean 0.320 t95 [0.318, 0.322] -> positive
- case3|oa_blind[A04]: mean 0.320 t95 [0.318, 0.322] -> positive
- case3|oa_blind[A04a]: mean 0.320 t95 [0.318, 0.322] -> positive
