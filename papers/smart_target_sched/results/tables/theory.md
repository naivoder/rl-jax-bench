#### Theory claim 1: PCLI1 and PCLI2 (the paper shows them on a few plots; here a fine sweep)

78 curves = {reckless, cautious} x Q^CT = 2, 3, ..., 40; each on the paper's grid P = 0.01, ..., 15 (1500 points), index at z = P (the policy's own evaluation point). PCLI1: min over all curves of g(P,P) = 0.1123; violations (g <= 1e-12): 0 at z = P and 0 at the fixed thresholds z in {1, 2, 4, 10} (Figs 2-3 style). PCLI2: decreases of mp*(P) larger than 1e-9 over the grid: 0 (worst relative drop 0.00e+00); mp* >= 0.1119 > 0 (bounded below); largest relative change between neighbouring grid points (continuity proxy) 0.083 (at the smallest P, where mp* itself is small and the grid relative step is 100 %). **PCLI1 and PCLI2: REPRODUCED (0 violations in 117000 index evaluations).** (The 4-D case is different: PCLI1 was flagged 8 times in 1.7e6 index evaluations in the small-scale run, all in the 4-D runs; the paper itself says there is no theory there.)

#### Theory claim 2 (extension): the rollout MP index equals Whittle's index obtained by an independent route (grid value iteration, indifference lambda) and the problem is Whittle-indexable

| type | Q^CT | max rel. difference rollout vs VI-Whittle over P in {0.5,1,2,3,5,8,10,15} | states (of 2000) where the passive set is NOT nested in lambda (160 lambdas) |
|---|---|---|---|
| reckless | 2 | 2.69e-04 | 0 |
| reckless | 4 | 1.10e-04 | 0 |
| reckless | 10 | 1.50e-04 | 0 |
| reckless | 40 | 6.84e-05 | 0 |
| cautious | 2 | 3.45e-04 | 0 |
| cautious | 4 | 1.35e-04 | 0 |
| cautious | 10 | 1.71e-04 | 0 |
| cautious | 40 | 8.32e-05 | 0 |

Worst relative difference 3.45e-04 (declared tolerance 1e-3, limited by the linear-interpolation error of the 0.01 grid); all 8 (type, Q^CT) problems are Whittle-indexable on the grid (passive sets nested in lambda): **the two routes agree and indexability holds numerically.**

#### Theory claim 3 (extension): PCLI3, the third condition the paper never checks, per jump

| type | Q^CT | P0 | tau | jumps | resolved jumps | max per-jump relative residual | median per-jump | aggregate relative residual |
|---|---|---|---|---|---|---|---|---|
| reckless | 4 | 0.5 | 100 | 734 | 733 | 6.98e-01 | 5.65e-05 | 4.01e-05 |
| reckless | 4 | 2.0 | 100 | 743 | 739 | 7.43e-01 | 3.42e-05 | 3.69e-05 |
| reckless | 4 | 8.0 | 100 | 755 | 749 | 6.97e-01 | 4.49e-05 | 3.72e-05 |
| reckless | 10 | 0.5 | 100 | 539 | 538 | 6.11e-01 | 7.83e-05 | 2.54e-05 |
| reckless | 10 | 2.0 | 100 | 535 | 532 | 6.07e-01 | 7.22e-05 | 2.69e-05 |
| reckless | 10 | 8.0 | 100 | 547 | 543 | 7.00e-01 | 6.48e-05 | 2.10e-05 |
| cautious | 4 | 0.5 | 100 | 885 | 883 | 7.87e-01 | 4.59e-05 | 3.35e-05 |
| cautious | 4 | 2.0 | 100 | 908 | 902 | 7.44e-01 | 3.52e-05 | 9.90e-06 |
| cautious | 4 | 8.0 | 100 | 902 | 900 | 7.87e-01 | 4.28e-05 | 9.54e-06 |
| cautious | 10 | 0.5 | 100 | 764 | 762 | 6.09e-01 | 3.94e-05 | 5.31e-05 |
| cautious | 10 | 2.0 | 100 | 761 | 759 | 6.97e-01 | 2.83e-05 | 3.54e-05 |
| cautious | 10 | 8.0 | 100 | 762 | 757 | 6.09e-01 | 4.94e-05 | 3.52e-05 |
| reckless | 4 | 0.5 | 400 | 803 | 787 | 9.60e-09 | 1.16e-12 | 1.19e-15 |
| reckless | 4 | 8.0 | 400 | 828 | 798 | 4.16e-09 | 1.48e-12 | 1.04e-15 |
| reckless | 10 | 2.0 | 400 | 573 | 565 | 7.58e-09 | 6.67e-13 | 6.42e-16 |
| cautious | 4 | 0.5 | 400 | 1031 | 975 | 8.14e-09 | 1.78e-12 | 0.00e+00 |
| cautious | 4 | 8.0 | 400 | 1006 | 964 | 8.39e-09 | 2.13e-12 | 8.58e-16 |
| cautious | 10 | 2.0 | 400 | 818 | 804 | 8.76e-09 | 1.27e-12 | 2.52e-15 |

At tau = 100 the identity F(P,z2) - F(P,z1) = sum_j mp*(z_j) (G(z_j-) - G(z_j)) holds in aggregate to 5.3e-05 (the tau truncation 0.9^100 = 2.7e-5) but single late jumps, whose size is below the truncation, show relative residuals up to 0.79. With tau = 400 (truncation 5e-19) the residual is **9.6e-09 per jump and 2.5e-15 in aggregate over 5059 jumps**: PCLI3 holds to machine-level precision, so the MP index IS a marginal productivity index in the paper's sense (with the strict reading of the identity, not just the plots): **PCL-indexability REPRODUCED (all three conditions), beyond what the paper shows.**

#### Extension: the index policy against the EXACT optimum on small instances (open-loop enumeration, deterministic dynamics => open loop is optimal; short horizon T, so the index's infinite-horizon pricing suffers an end-of-horizon effect that Tables I-IV (T = 100, beta^T = 2.7e-5) do not have)

| case | N, K, T | replications | J* (mean ± SE) | regret % vs J*: index | myopic | TEV | PASSIVE-CONT(revisit) | ROLLOUT-1 | index exactly optimal | index vs myopic (paired, rel, p) |
|---|---|---|---|---|---|---|---|---|---|---|
| n2_k1_T10 | 2, 1, 10 | 40 | 75.13 ± 0.75 | 1.50 ± 0.29 | 4.82 ± 0.38 | 4.70 ± 0.33 | 4.82 ± 0.38 | 0.59 ± 0.13 | 17/40 | -3.11 % (p=1.8e-15) |
| n3_k1_T8 | 3, 1, 8 | 40 | 93.00 ± 0.99 | 1.39 ± 0.21 | 4.83 ± 0.41 | 5.41 ± 0.46 | 4.83 ± 0.41 | 2.21 ± 0.29 | 8/40 | -3.28 % (p=1.0e-12) |
| n4_k2_T6 | 4, 2, 6 | 40 | 98.55 ± 0.89 | 0.52 ± 0.10 | 4.63 ± 0.26 | 4.56 ± 0.29 | 4.63 ± 0.26 | 2.39 ± 0.19 | 15/40 | -3.95 % (p=6.6e-18) |
