tag `full_s1` variant `primary`: n per cell = [100]; Δ% = (ours − paper)/paper; judgment MATCH <= 5 %, CLOSE <= 20 %; authors' published output = the paper's table (TeX-typeset, identical to the paper numbers, 81/81 by `tests/test_official_parity.py`); z = (ours − paper)/SE in standard errors of our mean.

| Metric (cell, policy) | Paper | Ours (mean ± SE, n) | Δ% | z | Judgment | Notes |
|---|---|---|---|---|---|---|
| t1_q2 K=1, Whittle/MP | 823.19 | 826.79 ± 2.00 (n=100) | +0.44 | +1.8 | MATCH | impr. over best baseline 5.43 % (paper 5.24 %) |
| t1_q2 K=1, myopic | 868.71 | 874.24 ± 2.35 (n=100) | +0.64 | +2.4 | MATCH |  |
| t1_q2 K=1, TEV/TEC | 871.19 | 876.07 ± 2.29 (n=100) | +0.56 | +2.1 | MATCH |  |
| t1_q2 K=2, Whittle/MP | 400.53 | 401.78 ± 0.75 (n=100) | +0.31 | +1.7 | MATCH | impr. over best baseline 1.25 % (paper 1.31 %) |
| t1_q2 K=2, myopic | 405.85 | 406.85 ± 0.78 (n=100) | +0.25 | +1.3 | MATCH |  |
| t1_q2 K=2, TEV/TEC | 406.17 | 407.43 ± 0.83 (n=100) | +0.31 | +1.5 | MATCH |  |
| t1_q2 K=3, Whittle/MP | 284.65 | 284.26 ± 0.59 (n=100) | -0.14 | -0.7 | MATCH | impr. over best baseline 3.09 % (paper 3.09 %) |
| t1_q2 K=3, myopic | 293.80 | 293.35 ± 0.60 (n=100) | -0.15 | -0.8 | MATCH |  |
| t1_q2 K=3, TEV/TEC | 293.72 | 293.31 ± 0.60 (n=100) | -0.14 | -0.7 | MATCH |  |
| t1_q29 K=1, Whittle/MP | 961.25 | 966.09 ± 1.62 (n=100) | +0.50 | +3.0 | MATCH | impr. over best baseline 3.35 % (paper 3.29 %) |
| t1_q29 K=1, myopic | 993.93 | 999.55 ± 1.80 (n=100) | +0.57 | +3.1 | MATCH |  |
| t1_q29 K=1, TEV/TEC | 1009.15 | 1015.05 ± 2.13 (n=100) | +0.58 | +2.8 | MATCH |  |
| t1_q29 K=2, Whittle/MP | 458.49 | 457.95 ± 0.88 (n=100) | -0.12 | -0.6 | MATCH | impr. over best baseline 1.33 % (paper 1.28 %) |
| t1_q29 K=2, myopic | 464.43 | 464.11 ± 0.94 (n=100) | -0.07 | -0.3 | MATCH |  |
| t1_q29 K=2, TEV/TEC | 465.12 | 464.40 ± 0.89 (n=100) | -0.16 | -0.8 | MATCH |  |
| t1_q29 K=3, Whittle/MP | 319.84 | 318.50 ± 0.60 (n=100) | -0.42 | -2.2 | MATCH | impr. over best baseline 1.83 % (paper 1.90 %) |
| t1_q29 K=3, myopic | 326.04 | 324.43 ± 0.64 (n=100) | -0.49 | -2.5 | MATCH |  |
| t1_q29 K=3, TEV/TEC | 334.06 | 332.55 ± 0.66 (n=100) | -0.45 | -2.3 | MATCH |  |
| t2_q2 K=1, Whittle/MP | 750.91 | 751.29 ± 1.78 (n=100) | +0.05 | +0.2 | MATCH | impr. over best baseline 5.05 % (paper 5.00 %) |
| t2_q2 K=1, myopic | 790.61 | 791.22 ± 2.21 (n=100) | +0.08 | +0.3 | MATCH |  |
| t2_q2 K=1, TEV/TEC | 790.40 | 791.30 ± 2.23 (n=100) | +0.11 | +0.4 | MATCH |  |
| t2_q2 K=2, Whittle/MP | 377.36 | 375.04 ± 0.81 (n=100) | -0.62 | -2.9 | MATCH | impr. over best baseline 1.14 % (paper 1.19 %) |
| t2_q2 K=2, myopic | 381.92 | 379.35 ± 0.83 (n=100) | -0.67 | -3.1 | MATCH |  |
| t2_q2 K=2, TEV/TEC | 384.06 | 381.42 ± 0.85 (n=100) | -0.69 | -3.1 | MATCH |  |
| t2_q2 K=3, Whittle/MP | 268.30 | 268.68 ± 0.63 (n=100) | +0.14 | +0.6 | MATCH | impr. over best baseline 2.70 % (paper 2.70 %) |
| t2_q2 K=3, myopic | 275.81 | 276.19 ± 0.64 (n=100) | +0.14 | +0.6 | MATCH |  |
| t2_q2 K=3, TEV/TEC | 275.75 | 276.15 ± 0.65 (n=100) | +0.15 | +0.6 | MATCH |  |
| t2_q29 K=1, Whittle/MP | 817.23 | 814.82 ± 1.42 (n=100) | -0.29 | -1.7 | MATCH | impr. over best baseline 4.03 % (paper 3.85 %) |
| t2_q29 K=1, myopic | 849.91 | 849.00 ± 1.84 (n=100) | -0.11 | -0.5 | MATCH |  |
| t2_q29 K=1, TEV/TEC | 861.88 | 859.85 ± 1.83 (n=100) | -0.24 | -1.1 | MATCH |  |
| t2_q29 K=2, Whittle/MP | 406.26 | 403.63 ± 0.71 (n=100) | -0.65 | -3.7 | MATCH | impr. over best baseline 1.03 % (paper 0.88 %) |
| t2_q29 K=2, myopic | 409.88 | 407.84 ± 0.69 (n=100) | -0.50 | -3.0 | MATCH |  |
| t2_q29 K=2, TEV/TEC | 410.56 | 408.25 ± 0.67 (n=100) | -0.56 | -3.5 | MATCH |  |
| t2_q29 K=3, Whittle/MP | 285.67 | 285.08 ± 0.56 (n=100) | -0.21 | -1.1 | MATCH | impr. over best baseline 3.54 % (paper 3.55 %) |
| t2_q29 K=3, myopic | 296.35 | 295.76 ± 0.57 (n=100) | -0.20 | -1.0 | MATCH |  |
| t2_q29 K=3, TEV/TEC | 296.19 | 295.54 ± 0.58 (n=100) | -0.22 | -1.1 | MATCH |  |
| t3_q2 K=1, Whittle/MP | 1554.35 | 1550.68 ± 2.92 (n=100) | -0.24 | -1.3 | MATCH | impr. over best baseline 3.75 % (paper 3.72 %) |
| t3_q2 K=1, myopic | 1614.41 | 1611.11 ± 3.15 (n=100) | -0.20 | -1.0 | MATCH |  |
| t3_q2 K=1, TEV/TEC | 1622.97 | 1618.17 ± 3.23 (n=100) | -0.30 | -1.5 | MATCH |  |
| t3_q2 K=2, Whittle/MP | 772.19 | 773.49 ± 1.62 (n=100) | +0.17 | +0.8 | MATCH | impr. over best baseline 4.28 % (paper 4.35 %) |
| t3_q2 K=2, myopic | 807.35 | 808.07 ± 1.82 (n=100) | +0.09 | +0.4 | MATCH |  |
| t3_q2 K=2, TEV/TEC | 808.89 | 810.26 ± 1.73 (n=100) | +0.17 | +0.8 | MATCH |  |
| t3_q2 K=3, Whittle/MP | 547.38 | 548.92 ± 1.23 (n=100) | +0.28 | +1.3 | MATCH | impr. over best baseline 3.46 % (paper 3.48 %) |
| t3_q2 K=3, myopic | 567.14 | 568.60 ± 1.26 (n=100) | +0.26 | +1.2 | MATCH |  |
| t3_q2 K=3, TEV/TEC | 567.94 | 569.47 ± 1.28 (n=100) | +0.27 | +1.2 | MATCH |  |
| t3_q2345 K=1, Whittle/MP | 1664.83 | 1665.78 ± 3.41 (n=100) | +0.06 | +0.3 | MATCH | impr. over best baseline 3.96 % (paper 3.86 %) |
| t3_q2345 K=1, myopic | 1731.69 | 1734.41 ± 3.64 (n=100) | +0.16 | +0.7 | MATCH |  |
| t3_q2345 K=1, TEV/TEC | 1733.95 | 1736.20 ± 3.64 (n=100) | +0.13 | +0.6 | MATCH |  |
| t3_q2345 K=2, Whittle/MP | 821.05 | 817.28 ± 1.63 (n=100) | -0.46 | -2.3 | MATCH | impr. over best baseline 4.37 % (paper 4.42 %) |
| t3_q2345 K=2, myopic | 859.02 | 854.66 ± 1.80 (n=100) | -0.51 | -2.4 | MATCH |  |
| t3_q2345 K=2, TEV/TEC | 860.42 | 856.62 ± 1.84 (n=100) | -0.44 | -2.1 | MATCH |  |
| t3_q2345 K=3, Whittle/MP | 581.47 | 579.65 ± 0.97 (n=100) | -0.31 | -1.9 | MATCH | impr. over best baseline 3.86 % (paper 4.01 %) |
| t3_q2345 K=3, myopic | 605.75 | 603.21 ± 1.06 (n=100) | -0.42 | -2.4 | MATCH |  |
| t3_q2345 K=3, TEV/TEC | 605.73 | 602.90 ± 1.06 (n=100) | -0.47 | -2.7 | MATCH |  |
| t4_reckless K=1, Whittle/MP | 4364.18 | 4372.56 ± 5.31 (n=100) | +0.19 | +1.6 | MATCH | impr. over best baseline 1.66 % (paper 1.63 %) |
| t4_reckless K=1, myopic | 4480.28 | 4493.65 ± 6.66 (n=100) | +0.30 | +2.0 | MATCH |  |
| t4_reckless K=1, TEV/TEC | 4436.27 | 4446.45 ± 5.62 (n=100) | +0.23 | +1.8 | MATCH |  |
| t4_reckless K=2, Whittle/MP | 1142.87 | 1145.06 ± 1.06 (n=100) | +0.19 | +2.1 | MATCH | impr. over best baseline 0.81 % (paper 0.89 %) |
| t4_reckless K=2, myopic | 1153.19 | 1155.11 ± 1.22 (n=100) | +0.17 | +1.6 | MATCH |  |
| t4_reckless K=2, TEV/TEC | 1153.43 | 1154.37 ± 1.26 (n=100) | +0.08 | +0.7 | MATCH |  |
| t4_reckless K=3, Whittle/MP | 610.10 | 610.33 ± 0.46 (n=100) | +0.04 | +0.5 | MATCH | impr. over best baseline 0.26 % (paper 0.50 %) |
| t4_reckless K=3, myopic | 613.15 | 611.92 ± 0.56 (n=100) | -0.20 | -2.2 | MATCH |  |
| t4_reckless K=3, TEV/TEC | 633.80 | 634.08 ± 0.51 (n=100) | +0.04 | +0.5 | MATCH |  |
| t4_cautious K=1, Whittle/MP | 3468.00 | 3463.70 ± 4.33 (n=100) | -0.12 | -1.0 | MATCH | impr. over best baseline 2.11 % (paper 1.88 %) |
| t4_cautious K=1, myopic | 3584.63 | 3586.47 ± 6.08 (n=100) | +0.05 | +0.3 | MATCH |  |
| t4_cautious K=1, TEV/TEC | 3534.40 | 3538.42 ± 5.38 (n=100) | +0.11 | +0.7 | MATCH |  |
| t4_cautious K=2, Whittle/MP | 902.06 | 903.21 ± 0.79 (n=100) | +0.13 | +1.4 | MATCH | impr. over best baseline 1.67 % (paper 1.66 %) |
| t4_cautious K=2, myopic | 931.63 | 932.92 ± 1.16 (n=100) | +0.14 | +1.1 | MATCH |  |
| t4_cautious K=2, TEV/TEC | 917.27 | 918.54 ± 1.06 (n=100) | +0.14 | +1.2 | MATCH |  |
| t4_cautious K=3, Whittle/MP | 492.69 | 492.32 ± 0.39 (n=100) | -0.08 | -1.0 | MATCH | impr. over best baseline 1.47 % (paper 1.54 %) |
| t4_cautious K=3, myopic | 500.40 | 499.65 ± 0.54 (n=100) | -0.15 | -1.4 | MATCH |  |
| t4_cautious K=3, TEV/TEC | 504.08 | 503.66 ± 0.40 (n=100) | -0.08 | -1.0 | MATCH |  |
| t4_mixed K=1, Whittle/MP | 6777.79 | 6752.98 ± 10.12 (n=100) | -0.37 | -2.5 | MATCH | impr. over best baseline 1.69 % (paper 1.48 %) |
| t4_mixed K=1, myopic | 7014.92 | 7000.87 ± 10.64 (n=100) | -0.20 | -1.3 | MATCH |  |
| t4_mixed K=1, TEV/TEC | 6879.65 | 6869.38 ± 9.47 (n=100) | -0.15 | -1.1 | MATCH |  |
| t4_mixed K=2, Whittle/MP | 1824.24 | 1821.26 ± 2.74 (n=100) | -0.16 | -1.1 | MATCH | impr. over best baseline 2.25 % (paper 1.93 %) |
| t4_mixed K=2, myopic | 1895.25 | 1893.91 ± 2.91 (n=100) | -0.07 | -0.5 | MATCH |  |
| t4_mixed K=2, TEV/TEC | 1860.21 | 1863.11 ± 2.53 (n=100) | +0.16 | +1.1 | MATCH |  |
| t4_mixed K=3, Whittle/MP | 990.12 | 989.74 ± 1.16 (n=100) | -0.04 | -0.3 | MATCH | impr. over best baseline 3.08 % (paper 3.16 %) |
| t4_mixed K=3, myopic | 1022.45 | 1021.14 ± 1.43 (n=100) | -0.13 | -0.9 | MATCH |  |
| t4_mixed K=3, TEV/TEC | 1040.56 | 1039.46 ± 1.75 (n=100) | -0.11 | -0.6 | MATCH |  |

**Summary**: 81 MATCH / 0 CLOSE / 0 DIVERGENT of 81; max |Δ| = 0.69 %, median |Δ| = 0.19 %; mean signed Δ of the index arm = -0.06 %; |z| ≤ 2 in 58 of 81; ordering index < min(myopic, TEV) holds in 27 of 27 cells (paper: 27 of 27); relative improvement over the best baseline: ours 0.26-5.43 % vs paper 0.50-5.24 %, correlation 0.996.
