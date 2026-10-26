#### Compute parity: wall clock and analytic cost per method (CPU, 1 pinned core per worker, nice 19, host load ~20-25 so absolute seconds are pessimistic)

Steady-state seconds for one cell = 100 replications x 100 slots x N = 8 targets, median over the 3 (4-D) or 6 (scalar) table cells that ran a first-call timing (the second run of a signature is the steady state; the first call also includes jit compilation).  per decision = / 10 000 decisions.

| kind | policy | K=1 s | K=2 s | K=3 s | per decision µs (K=1 / K=2 / K=3) | compile+first-call overhead s (median) |
|---|---|---|---|---|---|---|
| scalar | myopic | 0.068 | 0.045 | 0.050 | 6.8 / 4.5 / 5.0 | 0.28 |
| scalar | myopic_blind0 | 0.062 | 0.062 | 0.071 | 6.2 / 6.2 / 7.1 | 0.73 |
| scalar | myopic_blind1 | 0.058 | 0.056 | 0.077 | 5.8 / 5.6 / 7.7 | 0.79 |
| scalar | passive_cont | 0.110 | 0.116 | 0.127 | 11.0 / 11.6 / 12.7 | 0.26 |
| scalar | passive_cont_h10 | 0.131 | 0.074 | 0.071 | 13.1 / 7.4 / 7.1 | 0.32 |
| scalar | passive_cont_h2 | 0.088 | 0.072 | 0.083 | 8.8 / 7.2 / 8.3 | 0.78 |
| scalar | passive_cont_h20 | 0.061 | 0.055 | 0.072 | 6.1 / 5.5 / 7.2 | 0.79 |
| scalar | passive_cont_h3 | 0.074 | 0.089 | 0.086 | 7.4 / 8.9 / 8.6 | 0.25 |
| scalar | passive_cont_h5 | 0.056 | 0.072 | 0.079 | 5.6 / 7.2 / 7.9 | 0.90 |
| scalar | passive_cont_h50 | 0.083 | 0.110 | 0.128 | 8.3 / 11.0 / 12.8 | 0.75 |
| scalar | passive_cont_revisit | 0.126 | 0.052 | 0.056 | 12.6 / 5.2 / 5.6 | 0.81 |
| scalar | random | 0.111 | 0.090 | 0.050 | 11.1 / 9.0 / 5.0 | 0.28 |
| scalar | rollout1 | 2.533 | 11.750 | 21.320 | 253.3 / 1175.0 / 2132.0 | 1.02 |
| scalar | tev | 0.044 | 0.043 | 0.054 | 4.4 / 4.3 / 5.4 | 0.17 |
| scalar | whittle | 0.428 | 0.308 | 0.353 | 42.8 / 30.8 / 35.3 | 0.97 |
| scalar | whittle_blind0 | 0.343 | 0.459 | 0.382 | 34.3 / 45.9 / 38.2 | 1.06 |
| scalar | whittle_blind1 | 0.361 | 0.383 | 0.412 | 36.1 / 38.3 / 41.2 | 0.96 |
| matrix | myopic | 0.189 | 0.181 | 0.224 | 18.9 / 18.1 / 22.4 | 0.29 |
| matrix | myopic_blind0 | 0.125 | 0.300 | 0.426 | 12.5 / 30.0 / 42.6 | 2.07 |
| matrix | myopic_blind1 | 0.131 | 0.485 | 0.243 | 13.1 / 48.5 / 24.3 | 2.23 |
| matrix | passive_cont | 9.108 | 9.314 | 8.711 | 910.8 / 931.4 / 871.1 | 1.53 |
| matrix | passive_cont_h10 | 1.076 | 1.102 | 1.005 | 107.6 / 110.2 / 100.5 | 2.20 |
| matrix | passive_cont_h2 | 0.316 | 0.389 | 0.447 | 31.6 / 38.9 / 44.7 | 1.26 |
| matrix | passive_cont_h20 | 1.698 | 1.735 | 2.224 | 169.8 / 173.5 / 222.4 | 1.00 |
| matrix | passive_cont_h3 | 0.431 | 0.381 | 0.559 | 43.1 / 38.1 / 55.9 | 0.49 |
| matrix | passive_cont_h5 | 0.586 | 0.530 | 0.741 | 58.6 / 53.0 / 74.1 | 1.06 |
| matrix | passive_cont_h50 | 4.321 | 3.600 | 5.204 | 432.1 / 360.0 / 520.4 | 1.99 |
| matrix | passive_cont_revisit | 0.946 | 0.579 | 0.479 | 94.6 / 57.9 / 47.9 | 2.13 |
| matrix | random | 0.158 | 0.148 | 0.196 | 15.8 / 14.8 / 19.6 | 1.89 |
| matrix | rollout1 | 27.217 | 176.369 | 518.676 | 2721.7 / 17636.9 / 51867.6 | 0.00 |
| matrix | tev | 0.161 | 0.195 | 0.167 | 16.1 / 19.5 / 16.7 | 0.20 |
| matrix | whittle | 9.575 | 7.640 | 9.184 | 957.5 / 764.0 / 918.4 | 1.27 |
| matrix | whittle_blind0 | 7.779 | 9.172 | 10.442 | 777.9 / 917.2 / 1044.2 | 2.30 |
| matrix | whittle_blind1 | 7.597 | 11.414 | 10.210 | 759.7 / 1141.4 / 1021.0 | 3.57 |

Analytic cost per scheduling decision at N = 8 (phi-steps = one TEC update of one target; ROLLOUT-1 phi-steps use the myopic base policy: 3N per base slot + N for the candidate action). FLOP estimates: scalar phi-step ≈ 25 flops, 4-D phi-step ≈ 1100 flops (two 4×4 mode updates incl. the 2×2 inverse), stat/cost ≈ 1 flop; these are order-of-magnitude constants, not profiler output.

| policy | phi-steps per decision (N=8) | scalar flops | 4-D flops | tunables searched |
|---|---|---|---|---|
| whittle | 1600 | 4.00e+04 | 1.76e+06 | tau (8 values) |
| passive_cont | 1600 | 4.00e+04 | 1.76e+06 | horizon (8 values) |
| passive_cont_h10 | 160 | 4.00e+03 | 1.76e+05 | - |
| myopic | 16 | 4.00e+02 | 1.76e+04 | none |
| tev | 0 | 0.00e+00 | 0.00e+00 | none |
| rollout1, K=1 | 5824 | 1.46e+05 | 6.41e+06 | base length (4 values) |
| rollout1, K=2 | 20384 | 5.10e+05 | 2.24e+07 | - |
| rollout1, K=3 | 40768 | 1.02e+06 | 4.48e+07 | - |
