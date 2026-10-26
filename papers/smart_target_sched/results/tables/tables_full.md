tag `full` variant `primary`: n per cell = [100]; Δ% = (ours − paper)/paper; judgment MATCH <= 5 %, CLOSE <= 20 %; authors' published output = the paper's table (TeX-typeset, identical to the paper numbers, 81/81 by `tests/test_official_parity.py`); z = (ours − paper)/SE in standard errors of our mean.

| Metric (cell, policy) | Paper | Ours (mean ± SE, n) | Δ% | z | Judgment | Notes |
|---|---|---|---|---|---|---|
| t1_q2 K=1, Whittle/MP | 823.19 | 823.00 ± 1.71 (n=100) | -0.02 | -0.1 | MATCH | impr. over best baseline 5.24 % (paper 5.24 %) |
| t1_q2 K=1, myopic | 868.71 | 868.54 ± 2.17 (n=100) | -0.02 | -0.1 | MATCH |  |
| t1_q2 K=1, TEV/TEC | 871.19 | 872.03 ± 2.12 (n=100) | +0.10 | +0.4 | MATCH |  |
| t1_q2 K=2, Whittle/MP | 400.53 | 399.75 ± 0.69 (n=100) | -0.19 | -1.1 | MATCH | impr. over best baseline 1.27 % (paper 1.31 %) |
| t1_q2 K=2, myopic | 405.85 | 404.90 ± 0.75 (n=100) | -0.23 | -1.3 | MATCH |  |
| t1_q2 K=2, TEV/TEC | 406.17 | 405.05 ± 0.76 (n=100) | -0.28 | -1.5 | MATCH |  |
| t1_q2 K=3, Whittle/MP | 284.65 | 285.36 ± 0.53 (n=100) | +0.25 | +1.3 | MATCH | impr. over best baseline 3.08 % (paper 3.09 %) |
| t1_q2 K=3, myopic | 293.80 | 294.47 ± 0.54 (n=100) | +0.23 | +1.2 | MATCH |  |
| t1_q2 K=3, TEV/TEC | 293.72 | 294.43 ± 0.55 (n=100) | +0.24 | +1.3 | MATCH |  |
| t1_q29 K=1, Whittle/MP | 961.25 | 961.67 ± 1.82 (n=100) | +0.04 | +0.2 | MATCH | impr. over best baseline 3.49 % (paper 3.29 %) |
| t1_q29 K=1, myopic | 993.93 | 996.40 ± 2.00 (n=100) | +0.25 | +1.2 | MATCH |  |
| t1_q29 K=1, TEV/TEC | 1009.15 | 1012.95 ± 2.37 (n=100) | +0.38 | +1.6 | MATCH |  |
| t1_q29 K=2, Whittle/MP | 458.49 | 457.86 ± 0.75 (n=100) | -0.14 | -0.8 | MATCH | impr. over best baseline 1.32 % (paper 1.28 %) |
| t1_q29 K=2, myopic | 464.43 | 463.99 ± 0.80 (n=100) | -0.10 | -0.6 | MATCH |  |
| t1_q29 K=2, TEV/TEC | 465.12 | 464.32 ± 0.73 (n=100) | -0.17 | -1.1 | MATCH |  |
| t1_q29 K=3, Whittle/MP | 319.84 | 319.87 ± 0.59 (n=100) | +0.01 | +0.1 | MATCH | impr. over best baseline 1.87 % (paper 1.90 %) |
| t1_q29 K=3, myopic | 326.04 | 325.99 ± 0.64 (n=100) | -0.02 | -0.1 | MATCH |  |
| t1_q29 K=3, TEV/TEC | 334.06 | 333.75 ± 0.62 (n=100) | -0.09 | -0.5 | MATCH |  |
| t2_q2 K=1, Whittle/MP | 750.91 | 750.00 ± 1.72 (n=100) | -0.12 | -0.5 | MATCH | impr. over best baseline 4.91 % (paper 5.00 %) |
| t2_q2 K=1, myopic | 790.61 | 788.75 ± 2.25 (n=100) | -0.24 | -0.8 | MATCH |  |
| t2_q2 K=1, TEV/TEC | 790.40 | 790.22 ± 2.19 (n=100) | -0.02 | -0.1 | MATCH |  |
| t2_q2 K=2, Whittle/MP | 377.36 | 375.40 ± 0.66 (n=100) | -0.52 | -3.0 | MATCH | impr. over best baseline 1.11 % (paper 1.19 %) |
| t2_q2 K=2, myopic | 381.92 | 379.63 ± 0.66 (n=100) | -0.60 | -3.5 | MATCH |  |
| t2_q2 K=2, TEV/TEC | 384.06 | 381.97 ± 0.70 (n=100) | -0.54 | -3.0 | MATCH |  |
| t2_q2 K=3, Whittle/MP | 268.30 | 266.88 ± 0.61 (n=100) | -0.53 | -2.3 | MATCH | impr. over best baseline 2.70 % (paper 2.70 %) |
| t2_q2 K=3, myopic | 275.81 | 274.38 ± 0.62 (n=100) | -0.52 | -2.3 | MATCH |  |
| t2_q2 K=3, TEV/TEC | 275.75 | 274.30 ± 0.63 (n=100) | -0.53 | -2.3 | MATCH |  |
| t2_q29 K=1, Whittle/MP | 817.23 | 816.22 ± 1.51 (n=100) | -0.12 | -0.7 | MATCH | impr. over best baseline 4.06 % (paper 3.85 %) |
| t2_q29 K=1, myopic | 849.91 | 850.80 ± 1.76 (n=100) | +0.10 | +0.5 | MATCH |  |
| t2_q29 K=1, TEV/TEC | 861.88 | 860.97 ± 2.02 (n=100) | -0.11 | -0.4 | MATCH |  |
| t2_q29 K=2, Whittle/MP | 406.26 | 406.69 ± 0.78 (n=100) | +0.11 | +0.6 | MATCH | impr. over best baseline 0.92 % (paper 0.88 %) |
| t2_q29 K=2, myopic | 409.88 | 410.48 ± 0.76 (n=100) | +0.15 | +0.8 | MATCH |  |
| t2_q29 K=2, TEV/TEC | 410.56 | 410.98 ± 0.73 (n=100) | +0.10 | +0.6 | MATCH |  |
| t2_q29 K=3, Whittle/MP | 285.67 | 286.25 ± 0.57 (n=100) | +0.20 | +1.0 | MATCH | impr. over best baseline 3.55 % (paper 3.55 %) |
| t2_q29 K=3, myopic | 296.35 | 296.95 ± 0.59 (n=100) | +0.20 | +1.0 | MATCH |  |
| t2_q29 K=3, TEV/TEC | 296.19 | 296.78 ± 0.59 (n=100) | +0.20 | +1.0 | MATCH |  |
| t3_q2 K=1, Whittle/MP | 1554.35 | 1561.32 ± 2.89 (n=100) | +0.45 | +2.4 | MATCH | impr. over best baseline 3.78 % (paper 3.72 %) |
| t3_q2 K=1, myopic | 1614.41 | 1622.60 ± 3.09 (n=100) | +0.51 | +2.6 | MATCH |  |
| t3_q2 K=1, TEV/TEC | 1622.97 | 1630.42 ± 3.08 (n=100) | +0.46 | +2.4 | MATCH |  |
| t3_q2 K=2, Whittle/MP | 772.19 | 773.56 ± 1.48 (n=100) | +0.18 | +0.9 | MATCH | impr. over best baseline 4.33 % (paper 4.35 %) |
| t3_q2 K=2, myopic | 807.35 | 808.61 ± 1.66 (n=100) | +0.16 | +0.8 | MATCH |  |
| t3_q2 K=2, TEV/TEC | 808.89 | 810.85 ± 1.58 (n=100) | +0.24 | +1.2 | MATCH |  |
| t3_q2 K=3, Whittle/MP | 547.38 | 547.60 ± 1.14 (n=100) | +0.04 | +0.2 | MATCH | impr. over best baseline 3.53 % (paper 3.48 %) |
| t3_q2 K=3, myopic | 567.14 | 567.64 ± 1.14 (n=100) | +0.09 | +0.4 | MATCH |  |
| t3_q2 K=3, TEV/TEC | 567.94 | 568.79 ± 1.14 (n=100) | +0.15 | +0.7 | MATCH |  |
| t3_q2345 K=1, Whittle/MP | 1664.83 | 1667.75 ± 3.09 (n=100) | +0.18 | +0.9 | MATCH | impr. over best baseline 3.95 % (paper 3.86 %) |
| t3_q2345 K=1, myopic | 1731.69 | 1736.35 ± 3.14 (n=100) | +0.27 | +1.5 | MATCH |  |
| t3_q2345 K=1, TEV/TEC | 1733.95 | 1738.68 ± 3.15 (n=100) | +0.27 | +1.5 | MATCH |  |
| t3_q2345 K=2, Whittle/MP | 821.05 | 819.87 ± 1.57 (n=100) | -0.14 | -0.8 | MATCH | impr. over best baseline 4.35 % (paper 4.42 %) |
| t3_q2345 K=2, myopic | 859.02 | 857.15 ± 1.71 (n=100) | -0.22 | -1.1 | MATCH |  |
| t3_q2345 K=2, TEV/TEC | 860.42 | 859.18 ± 1.70 (n=100) | -0.14 | -0.7 | MATCH |  |
| t3_q2345 K=3, Whittle/MP | 581.47 | 579.95 ± 0.98 (n=100) | -0.26 | -1.6 | MATCH | impr. over best baseline 3.86 % (paper 4.01 %) |
| t3_q2345 K=3, myopic | 605.75 | 603.26 ± 1.04 (n=100) | -0.41 | -2.4 | MATCH |  |
| t3_q2345 K=3, TEV/TEC | 605.73 | 603.51 ± 1.10 (n=100) | -0.37 | -2.0 | MATCH |  |
| t4_reckless K=1, Whittle/MP | 4364.18 | 4369.49 ± 5.33 (n=100) | +0.12 | +1.0 | MATCH | impr. over best baseline 1.43 % (paper 1.63 %) |
| t4_reckless K=1, myopic | 4480.28 | 4488.46 ± 6.38 (n=100) | +0.18 | +1.3 | MATCH |  |
| t4_reckless K=1, TEV/TEC | 4436.27 | 4432.98 ± 6.05 (n=100) | -0.07 | -0.5 | MATCH |  |
| t4_reckless K=2, Whittle/MP | 1142.87 | 1143.76 ± 1.02 (n=100) | +0.08 | +0.9 | MATCH | impr. over best baseline 0.81 % (paper 0.89 %) |
| t4_reckless K=2, myopic | 1153.19 | 1153.07 ± 1.34 (n=100) | -0.01 | -0.1 | MATCH |  |
| t4_reckless K=2, TEV/TEC | 1153.43 | 1153.44 ± 1.20 (n=100) | +0.00 | +0.0 | MATCH |  |
| t4_reckless K=3, Whittle/MP | 610.10 | 610.33 ± 0.39 (n=100) | +0.04 | +0.6 | MATCH | impr. over best baseline 0.36 % (paper 0.50 %) |
| t4_reckless K=3, myopic | 613.15 | 612.52 ± 0.72 (n=100) | -0.10 | -0.9 | MATCH |  |
| t4_reckless K=3, TEV/TEC | 633.80 | 633.74 ± 0.41 (n=100) | -0.01 | -0.1 | MATCH |  |
| t4_cautious K=1, Whittle/MP | 3468.00 | 3472.17 ± 4.29 (n=100) | +0.12 | +1.0 | MATCH | impr. over best baseline 2.01 % (paper 1.88 %) |
| t4_cautious K=1, myopic | 3584.63 | 3591.60 ± 5.95 (n=100) | +0.19 | +1.2 | MATCH |  |
| t4_cautious K=1, TEV/TEC | 3534.40 | 3543.25 ± 5.02 (n=100) | +0.25 | +1.8 | MATCH |  |
| t4_cautious K=2, Whittle/MP | 902.06 | 903.78 ± 0.84 (n=100) | +0.19 | +2.0 | MATCH | impr. over best baseline 1.68 % (paper 1.66 %) |
| t4_cautious K=2, myopic | 931.63 | 933.54 ± 1.18 (n=100) | +0.21 | +1.6 | MATCH |  |
| t4_cautious K=2, TEV/TEC | 917.27 | 919.21 ± 1.09 (n=100) | +0.21 | +1.8 | MATCH |  |
| t4_cautious K=3, Whittle/MP | 492.69 | 492.40 ± 0.39 (n=100) | -0.06 | -0.7 | MATCH | impr. over best baseline 1.52 % (paper 1.54 %) |
| t4_cautious K=3, myopic | 500.40 | 500.01 ± 0.47 (n=100) | -0.08 | -0.8 | MATCH |  |
| t4_cautious K=3, TEV/TEC | 504.08 | 503.56 ± 0.40 (n=100) | -0.10 | -1.3 | MATCH |  |
| t4_mixed K=1, Whittle/MP | 6777.79 | 6765.75 ± 11.37 (n=100) | -0.18 | -1.1 | MATCH | impr. over best baseline 1.78 % (paper 1.48 %) |
| t4_mixed K=1, myopic | 7014.92 | 7016.23 ± 10.61 (n=100) | +0.02 | +0.1 | MATCH |  |
| t4_mixed K=1, TEV/TEC | 6879.65 | 6888.09 ± 9.17 (n=100) | +0.12 | +0.9 | MATCH |  |
| t4_mixed K=2, Whittle/MP | 1824.24 | 1819.30 ± 2.57 (n=100) | -0.27 | -1.9 | MATCH | impr. over best baseline 2.24 % (paper 1.93 %) |
| t4_mixed K=2, myopic | 1895.25 | 1897.60 ± 2.87 (n=100) | +0.12 | +0.8 | MATCH |  |
| t4_mixed K=2, TEV/TEC | 1860.21 | 1860.96 ± 2.52 (n=100) | +0.04 | +0.3 | MATCH |  |
| t4_mixed K=3, Whittle/MP | 990.12 | 988.88 ± 1.01 (n=100) | -0.12 | -1.2 | MATCH | impr. over best baseline 3.15 % (paper 3.16 %) |
| t4_mixed K=3, myopic | 1022.45 | 1021.07 ± 1.27 (n=100) | -0.14 | -1.1 | MATCH |  |
| t4_mixed K=3, TEV/TEC | 1040.56 | 1036.79 ± 1.61 (n=100) | -0.36 | -2.3 | MATCH |  |

**Summary**: 81 MATCH / 0 CLOSE / 0 DIVERGENT of 81; max |Δ| = 0.60 %, median |Δ| = 0.16 %; mean signed Δ of the index arm = -0.03 %; |z| ≤ 2 in 68 of 81; ordering index < min(myopic, TEV) holds in 27 of 27 cells (paper: 27 of 27); relative improvement over the best baseline: ours 0.36-5.24 % vs paper 0.50-5.24 %, correlation 0.996.
