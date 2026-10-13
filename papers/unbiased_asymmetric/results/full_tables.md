| Metric | Paper (20 runs, mean ± s.e.m.) | Ours (mean ± SE, n) | Δ% | Judgment | Informative? | Notes |
|---|---|---|---|---|---|---|
| HH-3 a2c_asym_hs avg100 at the COMMON step t∈(1.50M, 2.00M] (stopped early, budget) | +0.914 ± 0.051 | +0.846 ± 0.140 (5) | -7% | CLOSE | yes | z vs paper -0.45; paper @T: +1; data: full; @2M t∈(1.9,2.0]M: authors' code +1.000 ± 0.000 (n=3) vs ours +0.908 ± 0.091 (n=5), Welch p 0.37 (no difference detected; underpowered, not equivalence) |
| HH-3 a2c_asym_s avg100 at the COMMON step t∈(3.66M, 4.16M] (stopped early, budget) | +0.138 ± 0.019 | +0.085 ± 0.041 (5) | -38% | DIVERGENT | yes | z vs paper -1.15; paper @T: +0.16; data: full; @2M t∈(1.9,2.0]M: authors' code +0.143 ± 0.016 (n=3) vs ours +0.092 ± 0.045 (n=5), Welch p 0.33 (no difference detected; underpowered, not equivalence) |
| HH-3 a2c avg100 at the COMMON step t∈(1.85M, 2.35M] (stopped early, budget) | +0.043 ± 0.053 | +0.005 ± 0.009 (5) | abs rule | MATCH | NO (start-of-training +0.030 also MATCHes) | z vs paper -0.71; paper @T: +0.1; data: full |
| HH-4 a2c_asym_hs avg100 at the COMMON step t∈(2.81M, 3.31M] (stopped early, budget) | +0.277 ± 0.097 | +0.118 ± 0.113 (5) | -57% | DIVERGENT | yes | z vs paper -1.07; paper @T: +0.79; data: full |
| HH-4 a2c_asym_s avg100 at the COMMON step t∈(4.71M, 5.21M] (stopped early, budget) | +0.039 ± 0.013 | +0.053 ± 0.020 (5) | abs rule | MATCH | yes | z vs paper +0.58; paper @T: +0.04; data: full |
| HH-4 a2c avg100 at the COMMON step t∈(2.55M, 3.05M] (stopped early, budget) | +0.006 ± 0.012 | +0.002 ± 0.003 (5) | abs rule | MATCH | NO (start-of-training -0.004 also MATCHes) | z vs paper -0.29; paper @T: +0.01; data: full |
| Shopping-5 a2c_asym_hs avg100 over t∈[1.90M, 1.99M] (final, T) | +3.501 ± 0.100 | +3.616 ± 0.050 (5) | +3% | MATCH | yes | z vs paper +1.02; paper @T: +3; data: small; @2M t∈(1.9,2.0]M: authors' code +3.610 ± 0.019 (n=3) vs ours +3.616 ± 0.050 (n=5), Welch p 0.92 (no difference detected; underpowered, not equivalence) |
| Shopping-5 a2c_asym_s avg100 over t∈[1.91M, 2.00M] (final, T) | -5.403 ± 0.226 | -5.460 ± 0.129 (5) | -1% | MATCH | yes | z vs paper -0.22; paper @T: -6; data: small; @2M t∈(1.9,2.0]M: authors' code -5.548 ± 0.137 (n=3) vs ours -5.460 ± 0.129 (n=5), Welch p 0.66 (no difference detected; underpowered, not equivalence) |
| Shopping-5 a2c avg100 over t∈[1.91M, 2.00M] (final, T) | +3.586 ± 0.060 | +3.600 ± 0.036 (5) | +0% | MATCH | yes | z vs paper +0.21; paper @T: +3; data: small; @2M t∈(1.9,2.0]M: authors' code +3.642 ± 0.038 (n=3) vs ours +3.600 ± 0.036 (n=5), Welch p 0.46 (no difference detected; underpowered, not equivalence) |
| Shopping-5 a2c_react_2 avg100 over t∈[1.90M, 1.99M] (final, T) | -17.462 ± 4.072 | -14.180 ± 3.135 (5) | +19% | MATCH | yes | z vs paper +0.64; paper @T: -17; data: small |
| Shopping-5 a2c_react_4 avg100 over t∈[1.91M, 2.00M] (final, T) | +2.079 ± 0.235 | +2.463 ± 0.015 (5) | +18% | CLOSE | yes | z vs paper +1.63; paper @T: +2; data: small |
| Shopping-6 a2c_asym_hs avg100 at the COMMON step t∈(0.96M, 1.06M] (stopped early, budget) | -2.740 ± 0.366 | -2.918 ± 0.177 (5) | -6% | MATCH | yes | z vs paper -0.44; paper @T: +2; data: full |
| Shopping-6 a2c_asym_s avg100 at the COMMON step t∈(1.36M, 1.46M] (stopped early, budget) | -14.432 ± 0.404 | -14.543 ± 0.282 (5) | -1% | MATCH | yes | z vs paper -0.22; paper @T: -7; data: full |
| Shopping-6 a2c avg100 at the COMMON step t∈(1.24M, 1.34M] (stopped early, budget) | -36.355 ± 6.260 | -35.221 ± 15.680 (5) | +3% | MATCH | yes | z vs paper +0.07; paper @T: +2; data: full |
| Shopping-6 a2c_react_2 avg100 over t∈[1.91M, 2.00M] (final, T) | -100.472 ± 0.122 | -100.370 ± 0.094 (5) | +0% | MATCH | yes | z vs paper +0.66; paper @T: -100; data: full |
| Shopping-6 a2c_react_4 avg100 at the COMMON step t∈(1.51M, 1.61M] (stopped early, budget) | -126.315 ± 1.658 | -125.977 ± 2.477 (5) | +0% | MATCH | yes | z vs paper +0.11; paper @T: -83; data: full |

SENSITIVITY (truncated cells)
| Cell | t_stop per run (M) | Spearman(t_stop, final) | (A) per-run finals vs paper over mean window | (B) per-run finals vs paper over each run's window | (C) common step (primary) |
|---|---|---|---|---|---|
| HH-3 a2c_asym_hs | 2.10, 2.00, 2.45, 2.05, 2.30 | -0.50 | +0.978 vs +0.957: MATCH | +0.978 vs +0.952: MATCH | (1.50, 2.00]M +0.846 vs +0.914: CLOSE |
| HH-3 a2c_asym_s | 4.46, 4.21, 4.31, 4.16, 4.46 | -1.00 | +0.077 vs +0.137: DIVERGENT | +0.077 vs +0.136: DIVERGENT | (3.66, 4.16]M +0.085 vs +0.138: DIVERGENT |
| HH-3 a2c | 2.75, 2.65, 2.70, 2.65, 2.35 | +0.10 | +0.010 vs +0.036: MATCH | +0.010 vs +0.037: MATCH | (1.85, 2.35]M +0.005 vs +0.043: MATCH |
| HH-4 a2c_asym_hs | 3.76, 3.31, 3.50, 3.56, 3.61 | -0.70 | +0.127 vs +0.275: DIVERGENT | +0.127 vs +0.272: DIVERGENT | (2.81, 3.31]M +0.118 vs +0.277: DIVERGENT |
| HH-4 a2c_asym_s | 6.11, 5.21, 6.06, 5.81, 5.36 | -0.70 | +0.047 vs +0.039: MATCH | +0.047 vs +0.035: MATCH | (4.71, 5.21]M +0.053 vs +0.039: MATCH |
| HH-4 a2c | 3.16, 3.05, 3.11, 3.21, 3.11 | +0.90 | +0.010 vs +0.006: MATCH | +0.010 vs +0.005: MATCH | (2.55, 3.05]M +0.002 vs +0.006: MATCH |
| Shopping-6 a2c_asym_hs | 1.06, 1.08, 1.13, 1.15, 1.17 | +0.50 | -3.106 vs -2.772: MATCH | -3.106 vs -2.745: CLOSE | (0.96, 1.06]M -2.918 vs -2.740: MATCH |
| Shopping-6 a2c_asym_s | 1.51, 1.46, 1.50, 1.49, 1.49 | +0.90 | -14.275 vs -13.827: MATCH | -14.275 vs -13.822: MATCH | (1.36, 1.46]M -14.543 vs -14.432: MATCH |
| Shopping-6 a2c | 1.44, 1.68, 1.52, 1.34, 1.49 | -0.30 | -9.140 vs -13.124: MATCH | -9.140 vs -15.833: DIVERGENT | (1.24, 1.34]M -35.221 vs -36.355: MATCH |
| Shopping-6 a2c_react_4 | 1.61, 1.65, 1.61, 1.62, 1.64 | -0.60 | -125.513 vs -125.577: MATCH | -125.513 vs -125.493: MATCH | (1.51, 1.61]M -125.977 vs -126.315: MATCH |

SPEED
| Metric (first t where the mean curve reaches thr) | Paper | Authors' code (mean curve, n) | Ours (mean curve; per-run) | Δ% vs paper | Judgment | Δ% vs authors' code |
|---|---|---|---|---|---|---|
| HH-3 a2c_asym_hs thr0.9 | 1.666M | 1.153M (n=3) | 1.779M (0.973M, 0.736M, 1.948M, 0.890M, 1.557M) | +6.8% | CLOSE | +54.3% |
| HH-3 a2c_asym_hs thr0.5 | 1.088M | 0.904M (n=3) | 0.896M (0.870M, 0.667M, 1.773M, 0.837M, 1.381M) | -17.7% | CLOSE | -0.9% |
| Shopping-5 a2c_asym_hs thr-5 | 0.373M | 0.358M (n=3) | 0.357M (0.357M, 0.346M, 0.359M, 0.354M, 0.361M) | -4.4% | MATCH | -0.4% |
| Shopping-5 a2c_asym_s thr-5 | 1.931M | 1.929M (n=3) | 1.885M (1.821M, 1.815M, 1.835M, 1.763M, 1.814M) | -2.4% | MATCH | -2.3% |
| Shopping-5 a2c thr-5 | 0.648M | 0.570M (n=3) | 0.593M (0.618M, 0.576M, 0.503M, 0.575M, 0.590M) | -8.5% | CLOSE | +4.0% |
| Shopping-5 a2c_react_4 thr-5 | 1.294M | 0.913M (n=3) | 1.047M (0.972M, 0.966M, 0.914M, 1.000M, 1.075M) | -19.0% | CLOSE | +14.7% |
| Shopping-6 a2c_asym_hs thr-5 | 0.829M | — | 0.765M (0.613M, 0.642M, 0.735M, 0.746M, 0.781M) | -7.8% | CLOSE | — |
| Shopping-6 a2c thr-5 | 1.562M | — | never (1.316M, 1.618M, 1.431M, 1.249M, 1.392M) | — | n/a | — |

PRIMARY TESTS
| Test | mean diff | paired-t 95% CI | bootstrap 95% CI | wins/n | p (paired t) | p Holm | p (Welch) | p Holm (Welch) |
|---|---|---|---|---|---|---|---|---|
| HH-3 hs-A2C common step t∈(1.50M, 2.00M] | +0.855 | [+0.458, +1.251] | [+0.567, +1.016] | 5/5 | 0.0039 | 0.016 | 0.0036 | 0.014 |
| HH-3 hs-asym_s common step t∈(1.50M, 2.00M] | +0.769 | [+0.343, +1.195] | [+0.456, +0.960] | 5/5 | 0.0074 | 0.022 | 0.0044 | 0.014 |
| HH-4 hs-A2C common step t∈(2.55M, 3.05M] | +0.077 | [-0.109, +0.263] | [-0.015, +0.213] | 2/5 | 0.31 | 0.63 | 0.31 | 0.62 |
| HH-4 hs-asym_s common step t∈(2.81M, 3.31M] | +0.094 | [-0.199, +0.388] | [-0.044, +0.310] | 2/5 | 0.42 | 0.63 | 0.45 | 0.62 |
| Shopping-5 hs-asym_s final (small data) | +9.076 | [+8.785, +9.368] | [+8.901, +9.268] | 5/5 | 1.1e-07 | 8.6e-07 | 9.2e-09 | 7.4e-08 |
| Shopping-6 hs-asym_s common step t∈(0.96M, 1.06M] | +11.707 | [+9.990, +13.424] | [+10.666, +12.785] | 5/5 | 4.6e-05 | 0.00032 | 2.1e-05 | 0.00015 |
| Shopping-5 hs-A2C auc (small data) | +10.944 | [+8.447, +13.441] | [+9.136, +12.049] | 5/5 | 0.00026 | 0.0016 | 0.00021 | 0.0011 |
| Shopping-6 hs-A2C auc (t <= 1.06M, common window) | +44.846 | [+30.726, +58.965] | [+35.935, +53.756] | 5/5 | 0.00091 | 0.0046 | 0.00014 | 0.00086 |

SECONDARY TESTS
| Test | mean diff | paired-t 95% CI | bootstrap 95% CI | wins/n | p (paired t) | p Holm | p (Welch) | p Holm (Welch) |
|---|---|---|---|---|---|---|---|---|

A1
| Test | mean diff | paired-t 95% CI | bootstrap 95% CI | wins/n | p (paired t) | p Holm | p (Welch) | p Holm (Welch) |
|---|---|---|---|---|---|---|---|---|
| Shopping-5 asym_s MC - asym_s TD0 final | -21.678 | [-89.599, +46.243] | [-71.167, +4.917] | 3/5 | 0.43 | 0.43 | — | — (CRN-paired) |
| Shopping-5 asym_s MC - asym_s TD0 auc | -64.083 | [-89.996, -38.169] | [-82.536, -49.251] | 0/5 | 0.0024 | 0.0047 | — | — (CRN-paired) |
