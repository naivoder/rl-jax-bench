REC: 5285 ± 270 (n=5; per seed [5047, 6174, 4523, 5238, 5441])
| arm (claim) | paper ρ [spread] {boot95} clamp | ours ρ (mean ± SE, n) t95 | diff ± SE; t p; Holm p | Δ% | pre-declared | generic | interval reading |
|---|---|---|---|---|---|---|---|
| A1_obsnorm_none (Q1) | 0.518 [0.47, 0.65] {0.49, 0.60} - | 0.562 (2969 ± 300, 5) [0.274, 0.849] | -2315 ± 547; 0.0134; 0.1203 | +8.5 | MATCH (inside paper spread) | MATCH (inside paper spread) | REPRODUCED-DIRECTION |
| A2a_lastlayer_1 (Q2a) | 0.812 [0.69, 1.07] {0.72, 0.99} - | 0.984 (5199 ± 575, 5) [0.702, 1.265] | -85 ± 536; 0.8812; 1.0000 | +21.2 | CONFLICT (MATCH clause (inside spread, signs agree) and DIVERGENT clause (gap > 20 %) both hold) | MATCH (inside paper spread; gap > 20 %) | INCONCLUSIVE |
| A2b_initstd_1 (Q2b) | 0.565 [0.49, 0.65] {0.52, 0.61} - | 1.164 (6152 ± 372, 5) [0.995, 1.333] | +867 ± 321; 0.0540; 0.4321 | +106.2 | DIVERGENT (gap > 20 %) | DIVERGENT (gap > 20 %, outside spread) | INCONSISTENT; paper boot excludes ours |
| A3a_shared_bc1 (Q3) | 1.008 [0.79, 1.30] {0.81, 1.22} - | 0.901 (4763 ± 463, 5) [0.653, 1.150] | -521 ± 473; 0.3318; 1.0000; TOST p 0.494 | -10.6 | MATCH (inside paper spread) | MATCH (inside paper spread) | INCONCLUSIVE |
| A4_lambda_0p95 (Q4) | 0.802 [0.50, 1.28] {0.61, 1.13} - | 1.009 (5334 ± 180, 5) [0.840, 1.178] | +49 ± 322; 0.8857; 1.0000 | +25.8 | DIVERGENT (gap > 20 %) | MATCH (inside paper spread; gap > 20 %) | INCONSISTENT |
| A5a_lr_1em4 (Q5) | 0.798 [0.69, 0.83] {0.73, 0.83} - | 1.038 (5483 ± 197, 5) [0.945, 1.130] | +199 ± 176; 0.3228; 1.0000 | +30.1 | DIVERGENT (gap > 20 %) | DIVERGENT (gap > 20 %, outside spread) | INCONSISTENT; paper boot excludes ours |
| A6_entropy_1em3 (Q6) | 1.171 [0.82, 1.62] {0.71, 1.55} arm | 0.984 (5198 ± 150, 5) [0.867, 1.101] | -86 ± 223; 0.7186; 1.0000; TOST p 0.059 | -16.0 | UNCLASSIFIED (no clause applies: gap <= 20 %, point-estimate signs differ, not Holm-significant) | MATCH (inside paper spread; paper CI clamped at the max (not counted as reproduction)) | INCONSISTENT |
| A7_valuenorm_none (Q7) | 0.723 [0.62, 0.95] {0.65, 0.88} - | 0.965 (5101 ± 180, 5) [0.817, 1.113] | -184 ± 282; 0.5496; 1.0000 | +33.5 | DIVERGENT (gap > 20 %) | DIVERGENT (gap > 20 %, outside spread) | INCONSISTENT; paper boot excludes ours |
| A8_advnorm_true (Q8) | 0.947 [0.75, 1.16] {0.81, 1.10} - | 1.019 (5387 ± 249, 5) [0.811, 1.228] | +102 ± 396; 0.8087; 1.0000; TOST p 0.171 | +7.7 | UNCLASSIFIED (no clause applies: gap <= 20 %, point-estimate signs differ, not Holm-significant) | MATCH (inside paper spread) | INCONCLUSIVE |
tally: {"predeclared": {"CONFLICT": ["A2a_lastlayer_1"], "DIVERGENT": ["A2b_initstd_1", "A4_lambda_0p95", "A5a_lr_1em4", "A7_valuenorm_none"], "MATCH": ["A1_obsnorm_none", "A3a_shared_bc1"], "UNCLASSIFIED": ["A6_entropy_1em3", "A8_advnorm_true"]}, "generic": {"DIVERGENT": ["A2b_initstd_1", "A5a_lr_1em4", "A7_valuenorm_none"], "MATCH": ["A1_obsnorm_none", "A2a_lastlayer_1", "A3a_shared_bc1", "A4_lambda_0p95", "A6_entropy_1em3", "A8_advnorm_true"]}, "ci": {"INCONCLUSIVE": ["A2a_lastlayer_1", "A3a_shared_bc1", "A8_advnorm_true"], "INCONSISTENT": ["A2b_initstd_1", "A4_lambda_0p95", "A5a_lr_1em4", "A6_entropy_1em3", "A7_valuenorm_none"], "REPRODUCED-DIRECTION": ["A1_obsnorm_none"]}}
{"base_mean": 1648.9119873046875, "base_se": 206.3278043145867, "n_base": 3, "rho": 0.3120229125598783, "welch_t": -10.70861477264461, "welch_p": 4.067577234912635e-05}
[small-hc side-by-side, HEAD semantics, n=3] A1_obsnorm_none rho=0.696 diff=-1944±247
[small-hc side-by-side, HEAD semantics, n=3] A2a_lastlayer_1 rho=0.797 diff=-1298±332
[small-hc side-by-side, HEAD semantics, n=3] A2b_initstd_1 rho=1.005 diff=+32±216
[small-hc side-by-side, HEAD semantics, n=3] A3a_shared_bc1 rho=0.854 diff=-930±588
[small-hc side-by-side, HEAD semantics, n=3] A4_lambda_0p95 rho=0.924 diff=-486±107
[small-hc side-by-side, HEAD semantics, n=3] A5a_lr_1em4 rho=0.921 diff=-504±69
[small-hc side-by-side, HEAD semantics, n=3] A6_entropy_1em3 rho=0.882 diff=-754±26
[small-hc side-by-side, HEAD semantics, n=3] A7_valuenorm_none rho=0.666 diff=-2134±157
[small-hc side-by-side, HEAD semantics, n=3] A8_advnorm_true rho=0.964 diff=-228±364
[quarantined] walker2d A1_obsnorm_none rho=0.707 paper=0.500 n_finite=3
[quarantined] walker2d A2a_lastlayer_1 rho=0.424 paper=0.902 n_finite=3
[quarantined] walker2d A2b_initstd_1 rho=0.869 paper=0.885 n_finite=3
[quarantined] walker2d A3a_shared_bc1 rho=1.098 paper=1.387 n_finite=3
[quarantined] walker2d A4_lambda_0p95 rho=0.859 paper=0.890 n_finite=3
[quarantined] walker2d A5a_lr_1em4 rho=0.239 paper=0.652 n_finite=3
[quarantined] walker2d A6_entropy_1em3 rho=0.950 paper=0.907 n_finite=3
[quarantined] walker2d A7_valuenorm_none rho=1.437 paper=1.716 n_finite=3
[quarantined] walker2d A8_advnorm_true rho=0.914 paper=0.964 n_finite=3
[quarantined] walker2d A5b_lr_1em3 rho=0.965 paper=0.983 n_finite=3
[quarantined] hopper A1_obsnorm_none rho=0.541 paper=1.002 n_finite=3
[quarantined] hopper A2a_lastlayer_1 rho=0.815 paper=0.974 n_finite=3
[quarantined] hopper A2b_initstd_1 rho=1.385 paper=1.108 n_finite=3
[quarantined] hopper A3a_shared_bc1 rho=1.064 paper=0.981 n_finite=3
[quarantined] hopper A4_lambda_0p95 rho=1.063 paper=1.001 n_finite=3
[quarantined] hopper A5a_lr_1em4 rho=0.938 paper=0.890 n_finite=3
[quarantined] hopper A6_entropy_1em3 rho=0.896 paper=0.781 n_finite=3
[quarantined] hopper A7_valuenorm_none rho=0.893 paper=0.956 n_finite=3
[quarantined] hopper A8_advnorm_true rho=0.732 paper=1.028 n_finite=3
[quarantined] hopper A5b_lr_1em3 rho=0.703 paper=1.020 n_finite=3
eval noise: {'median_score_se_from_episodes': 86.48215547232417, 'max_score_se_from_episodes': 162.33510089870174, 'rec_seed_sd': 602.8905290556452, 'variance_share_median': 0.02057671715237027} nonfinite: 
