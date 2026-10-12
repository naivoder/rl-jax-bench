| Metric | Paper | Authors' code | Ours (mean ± SE, n) | Δ% | Judgment | Notes |
|---|---|---|---|---|---|---|
| E0 final reward hopper_ppo (its 468–487) | 2450 (1964–3024) | 2590 ± 312 (3) | 2404 ± 136 (5) | -2% | MATCH | proxy env; spread rule |
| E0 final reward hopper_trpo (its 468–487) | 2408 (1538–2982) | 2255 ± 45.556 (3) | 2152 ± 308 (5) | -11% | MATCH | proxy env; spread rule |
| E0 final reward walker2d_ppo (its 468–487) | 2396 (1526–3661) | 2621 ± 369 (3) | 3344 ± 170 (5) | +40% | MATCH | proxy env; spread rule |
| E0 final reward walker2d_trpo (its 468–487) | 2892 (2293–3315) | 3578 ± 295 (3) | 3315 ± 140 (5) | +15% | MATCH | proxy env; spread rule |
| E1 to_true(n=2000) hopper_ppo it 0 | 0.472 (0.471..0.473) interp. | 0.727 ± 0.078 (3; ref 1e5) | 0.763 ± 0.063 (5) | +62% | DIVERGENT | ref 1e6 |
| E1 to_true(n=2000) hopper_ppo it 150 | 0.155 (0.018..0.270) interp. | – | 0.322 ± 0.091 (5) | +108% | DIVERGENT | ref 1e6 |
| E1 to_true(n=2000) hopper_ppo it 450 | 0.024 (0.009..0.043) interp. | – | 0.066 ± 0.031 (5) | +171% | DIVERGENT | ref 1e6 |
| E1 pairwise(n=2000) hopper_ppo it 450 | 0.014 (-0.00085..0.036) interp. | – | 0.020 ± 0.014 (5) | +43% | MATCH | ref 1e6 |
| E1 to_true(n=10000) hopper_ppo it 450 | 0.041 (-0.0048..0.075) | – | 0.028 ± 0.045 (5) | -33% | MATCH | ref 1e6 |
| E1 to_true(n=100000) hopper_ppo it 450 | 0.103 (0.079..0.149) | – | 0.167 ± 0.039 (3) | +62% | DIVERGENT | 3 seeds |
| E1 to_true(n=2000) hopper_trpo it 0 | 0.783 (0.777..0.792) interp. | 0.877 ± 0.016 (3; ref 1e5) | 0.863 ± 0.019 (5) | +10% | CLOSE | ref 1e6 |
| E1 to_true(n=2000) hopper_trpo it 150 | 0.129 (0.073..0.192) interp. | – | 0.200 ± 0.022 (5) | +55% | DIVERGENT | ref 1e6 |
| E1 to_true(n=2000) hopper_trpo it 450 | 0.015 (-0.0098..0.053) interp. | – | 0.140 ± 0.086 (5) | +808% | DIVERGENT | ref 1e6 |
| E1 pairwise(n=2000) hopper_trpo it 450 | 0.032 (0.0097..0.045) interp. | – | 0.048 ± 0.024 (5) | +53% | CLOSE | ref 1e6 |
| E1 to_true(n=10000) hopper_trpo it 450 | 0.050 (0.0086..0.123) | – | 0.171 ± 0.086 (5) | +242% | DIVERGENT | ref 1e6 |
| E1 to_true(n=100000) hopper_trpo it 450 | 0.186 (0.063..0.259) | – | 0.565 ± 0.153 (3) | +204% | DIVERGENT | 3 seeds |
| E1 to_true(n=2000) walker2d_ppo it 0 | 0.794 (0.793..0.796) interp. | 0.786 ± 0.017 (3; ref 1e5) | 0.777 ± 0.016 (5) | -2% | MATCH | ref 1e6 |
| E1 to_true(n=2000) walker2d_ppo it 150 | 0.182 (0.121..0.218) interp. | – | 0.148 ± 0.045 (5) | -18% | MATCH | ref 1e6 |
| E1 to_true(n=2000) walker2d_ppo it 450 | 0.056 (0.029..0.110) interp. | – | 0.061 ± 0.024 (5) | +9% | MATCH | ref 1e6 |
| E1 pairwise(n=2000) walker2d_ppo it 450 | 0.013 (-0.011..0.031) interp. | – | -0.0081 ± 0.016 (5) | -161% | MATCH | ref 1e6 |
| E1 to_true(n=10000) walker2d_ppo it 450 | 0.098 (0.051..0.186) | – | 0.028 ± 0.014 (5) | -71% | DIVERGENT | ref 1e6 |
| E1 to_true(n=100000) walker2d_ppo it 450 | 0.315 (0.209..0.504) | – | 0.205 ± 0.049 (3) | -35% | DIVERGENT | 3 seeds |
| E1 to_true(n=2000) walker2d_trpo it 0 | 0.799 (0.797..0.802) interp. | 0.780 ± 0.033 (3; ref 1e5) | 0.764 ± 0.041 (5) | -4% | MATCH | ref 1e6 |
| E1 to_true(n=2000) walker2d_trpo it 150 | 0.107 (0.092..0.116) interp. | – | 0.172 ± 0.026 (5) | +61% | DIVERGENT | ref 1e6 |
| E1 to_true(n=2000) walker2d_trpo it 450 | 0.116 (0.104..0.133) interp. | – | 0.075 ± 0.010 (5) | -35% | DIVERGENT | ref 1e6 |
| E1 pairwise(n=2000) walker2d_trpo it 450 | 0.022 (0.016..0.031) interp. | – | 0.0046 ± 0.0094 (5) | -79% | CLOSE | ref 1e6 |
| E1 to_true(n=10000) walker2d_trpo it 450 | 0.192 (0.147..0.231) | – | 0.124 ± 0.037 (5) | -35% | DIVERGENT | ref 1e6 |
| E1 to_true(n=100000) walker2d_trpo it 450 | 0.598 (0.507..0.695) | – | 0.405 ± 0.018 (3) | -32% | DIVERGENT | 3 seeds |
| E2 held-out Returns-MRE hopper_ppo it 450 | 0.347 [0.075, 0.454] | 0.725 | median 0.438; 0.425 ± 0.114 (5) | +26% | MATCH | median of 5 (paper: median of 5) |
| E2 held-out GAE-MRE (code form) hopper_ppo it 450 | 0.029 [0.001, 0.060] | 0.084 | median 0.050; 0.047 ± 0.021 (5) | +69% | MATCH | median of 5 (paper: median of 5) |
| E2 held-out GAE-MRE (|·|/|·|) hopper_ppo it 450 | 0.029 [0.001, 0.060] | 0.056 | median 0.032; 0.035 ± 0.011 (5) | +9% | MATCH (H2 reading) | median of 5 (paper: median of 5) |
| E2 held-out Returns-MRE hopper_trpo it 450 | 0.278 [0.151, 0.382] | 0.418 | median 0.215; 0.279 ± 0.096 (5) | -23% | MATCH | median of 5 (paper: median of 5) |
| E2 held-out GAE-MRE (code form) hopper_trpo it 450 | 0.030 [0.00063, 0.048] | 5.588 | median 0.132; 2.533 ± 1.496 (5) | +342% | DIVERGENT | median of 5 (paper: median of 5); TRPO code-form not reproducible by authors' code |
| E2 held-out GAE-MRE (|·|/|·|) hopper_trpo it 450 | 0.030 [0.00063, 0.048] | 0.041 | median 0.016; 0.028 ± 0.0092 (5) | -47% | MATCH (H2 reading) | median of 5 (paper: median of 5) |
| E2 held-out Returns-MRE walker2d_ppo it 450 | 0.301 [0.264, 0.397] | 0.397 | median 0.436; 0.407 ± 0.056 (5) | +45% | DIVERGENT | median of 5 (paper: median of 5) |
| E2 held-out GAE-MRE (code form) walker2d_ppo it 450 | 0.043 [0.034, 0.056] | 0.050 | median 0.045; 0.034 ± 0.0096 (5) | +6% | MATCH | median of 5 (paper: median of 5) |
| E2 held-out GAE-MRE (|·|/|·|) walker2d_ppo it 450 | 0.043 [0.034, 0.056] | 0.046 | median 0.045; 0.045 ± 0.0054 (5) | +6% | MATCH (H2 reading) | median of 5 (paper: median of 5) |
| E2 held-out Returns-MRE walker2d_trpo it 450 | 0.414 [0.211, 0.437] | 0.544 | median 0.680; 0.617 ± 0.093 (5) | +64% | DIVERGENT | median of 5 (paper: median of 5) |
| E2 held-out GAE-MRE (code form) walker2d_trpo it 450 | 0.051 [0.018, 0.058] | 11.378 | median 13.897; 11.191 ± 2.939 (5) | +27121% | DIVERGENT | median of 5 (paper: median of 5); TRPO code-form not reproducible by authors' code |
| E2 held-out GAE-MRE (|·|/|·|) walker2d_trpo it 450 | 0.051 [0.018, 0.058] | 0.070 | median 0.068; 0.061 ± 0.0051 (5) | +33% | DIVERGENT (H2 reading) | median of 5 (paper: median of 5) |
| E3 Fig 4 pairwise(1e4) hopper it 150 zero (batch-mean centred) | 0.037 | – | 0.0045 ± 0.0039 (5) | -88% | DIVERGENT (A3: CLOSE) | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) hopper it 150 zero (un-centred; post-review) | 0.037 | – | -0.014 ± 0.011 (5) | -138% | DIVERGENT | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) hopper it 150 gae | 0.174 | – | 0.278 ± 0.123 (5) | +60% | DIVERGENT | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) hopper it 150 vstar | 0.368 | – | 0.393 ± 0.123 (5) | +7% | CLOSE | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) hopper it 300 zero (batch-mean centred) | 0.032 | – | 0.010 ± 0.014 (5) | -68% | DIVERGENT (A3: CLOSE) | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) hopper it 300 zero (un-centred; post-review) | 0.032 | – | -0.0018 ± 0.0072 (5) | -106% | DIVERGENT (A3: CLOSE) | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) hopper it 300 gae | 0.105 | – | 0.0037 ± 0.022 (5) | -96% | DIVERGENT | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) hopper it 300 vstar | 0.211 | – | 0.041 ± 0.032 (5) | -81% | DIVERGENT | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) hopper it 450 zero (batch-mean centred) | 0.028 | – | 0.00086 ± 0.011 (5) | -97% | DIVERGENT (A3: CLOSE) | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) hopper it 450 zero (un-centred; post-review) | 0.028 | – | 0.0021 ± 0.012 (5) | -93% | DIVERGENT (A3: CLOSE) | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) hopper it 450 gae | 0.179 | – | 0.029 ± 0.028 (5) | -84% | DIVERGENT | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) hopper it 450 vstar | 0.360 | – | 0.032 ± 0.032 (5) | -91% | DIVERGENT | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) walker2d it 150 zero (batch-mean centred) | 0.024 | – | -0.013 ± 0.0092 (5) | -155% | DIVERGENT (A3: CLOSE) | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) walker2d it 150 zero (un-centred; post-review) | 0.024 | – | 0.0013 ± 0.014 (5) | -95% | DIVERGENT (A3: CLOSE) | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) walker2d it 150 gae | 0.123 | – | 0.119 ± 0.029 (5) | -3% | MATCH | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) walker2d it 150 vstar | 0.234 | – | 0.120 ± 0.023 (5) | -49% | DIVERGENT | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) walker2d it 300 zero (batch-mean centred) | 0.024 | – | 0.0045 ± 0.004 (5) | -81% | CLOSE (A3: MATCH) | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) walker2d it 300 zero (un-centred; post-review) | 0.024 | – | -0.010 ± 0.0081 (5) | -143% | DIVERGENT (A3: CLOSE) | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) walker2d it 300 gae | 0.053 | – | 0.064 ± 0.036 (5) | +21% | DIVERGENT | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) walker2d it 300 vstar | 0.211 | – | 0.093 ± 0.037 (5) | -56% | DIVERGENT | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) walker2d it 450 zero (batch-mean centred) | 0.023 | – | 0.013 ± 0.0062 (5) | -44% | CLOSE (A3: MATCH) | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) walker2d it 450 zero (un-centred; post-review) | 0.023 | – | 0.0071 ± 0.0089 (5) | -70% | CLOSE (A3: MATCH) | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) walker2d it 450 gae | 0.039 | – | -0.0045 ± 0.012 (5) | -111% | DIVERGENT | PPO (assumed); V* on 1e6 pairs; declared rule |
| E3 Fig 4 pairwise(1e4) walker2d it 450 vstar | 0.142 | – | 0.0058 ± 0.011 (5) | -96% | DIVERGENT | PPO (assumed); V* on 1e6 pairs; declared rule |
| E5 reward ratio PPO / no-baseline hopper | ≈10× ('almost an order of magnitude') | – | 1.85× (2404 ± 136 (5) / 1301 ± 293 (5)) | – | FAILS | pre-declared arm keeps adv-norm centring |
| E5 reward ratio PPO / no-baseline walker2d | ≈10× ('almost an order of magnitude') | – | 3.90× (3344 ± 170 (5) / 857 ± 216 (5)) | – | PARTIAL | pre-declared arm keeps adv-norm centring |
| E5 diag hopper: nobaseline (A:=G_t, centred adv-norm; pre-declared arm) | ≈10× | – | 1.36× (1719 ± 218 (3)) | – | FAILS | 3 seeds, diagnostic |
| E5 diag hopper: nobaseline-nocenter (A:=G_t, scale-only adv-norm) | ≈10× | – | 14.50× (161 ± 36 (3)) | – | HOLDS | 3 seeds, diagnostic |
| E5 diag hopper: zerogae (GAE with V==0, centred) | ≈10× | – | 6.82× (342 ± 10 (3)) | – | HOLDS | 3 seeds, diagnostic |
| E5 reward ratio PPO / no-baseline un-centred hopper (post-review) | ≈10× | – | 16.75× (2404 ± 136 (5) / 144 ± 22 (5)) | – | HOLDS | 5 seeds, labelled post-review |
| E5 reward ratio PPO / no-baseline un-centred walker2d (post-review) | ≈10× | – | 10.67× (3344 ± 170 (5) / 313 ± 8 (5)) | – | HOLDS | 5 seeds, labelled post-review |
