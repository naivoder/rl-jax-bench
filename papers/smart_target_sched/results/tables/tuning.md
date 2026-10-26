#### Tuning parity: each arm's one knob is searched on the independent resample `full_s1` (selection by mean J per cell) and evaluated on `full` (the binding set)

Search spaces: index tau ∈ {2,3,5,10,20,50,100,200}; PASSIVE-CONT horizon ∈ {2,3,5,10,20,50,100} ∪ {revisit = ceil(N/K)}; ROLLOUT-1 base length ∈ {1,2,3,5,10,30,100} (scalar and 4-D; widened after review, the original grid {5,10,30,100} / {5,10,30} stopped at its best values); myopic and TEV have no knob. The paper itself tunes nothing.

| cell | tau* (tune) | index(tau*) | index(100) | h* | PASSIVE(h*) | b* | ROLLOUT-1(b*) | myopic | TEV | index(tau*) − PASSIVE(h*) (rel, p) | index(tau*) − ROLLOUT-1(b*) (rel, p) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t1_q2 K=1 | 5 | 819.4 | 823.0 | 2 | 868.5 | 5 | 811.7 | 868.5 | 872.0 | -5.66 % (Holm p=5.1e-70) | +0.95 % (Holm p=1.9e-17) |
| t1_q2 K=2 | 20 | 399.4 | 399.8 | 2 | 404.9 | 3 | 399.0 | 404.9 | 405.0 | -1.36 % (Holm p=1.3e-60) | +0.09 % (Holm p=3e-05) |
| t1_q2 K=3 | 50 | 285.4 | 285.4 | 2 | 294.5 | 2 | 285.3 | 294.5 | 294.4 | -3.09 % (Holm p=7.8e-176) | +0.01 % (Holm p=0.13) |
| t1_q29 K=1 | 5 | 961.4 | 961.7 | 2 | 996.4 | 5 | 952.2 | 996.4 | 1012.9 | -3.51 % (Holm p=2.8e-46) | +0.96 % (Holm p=5.7e-13) |
| t1_q29 K=2 | 100 | 457.9 | 457.9 | 2 | 464.0 | 3 | 457.9 | 464.0 | 464.3 | -1.32 % (Holm p=6.9e-64) | -0.00 % (Holm p=1) |
| t1_q29 K=3 | 100 | 319.9 | 319.9 | 2 | 326.0 | 2 | 320.1 | 326.0 | 333.8 | -1.87 % (Holm p=5.5e-60) | -0.08 % (Holm p=3e-05) |
| t2_q2 K=1 | 5 | 747.5 | 750.0 | 2 | 788.7 | 5 | 741.2 | 788.7 | 790.2 | -5.24 % (Holm p=1.6e-63) | +0.84 % (Holm p=1.5e-20) |
| t2_q2 K=2 | 20 | 375.2 | 375.4 | 2 | 379.6 | 2 | 375.1 | 379.6 | 382.0 | -1.18 % (Holm p=1.3e-49) | +0.00 % (Holm p=1) |
| t2_q2 K=3 | 50 | 266.9 | 266.9 | 2 | 274.4 | 2 | 266.9 | 274.4 | 274.3 | -2.73 % (Holm p=2e-152) | +0.01 % (Holm p=1) |
| t2_q29 K=1 | 5 | 814.4 | 816.2 | 2 | 850.8 | 5 | 806.6 | 850.8 | 861.0 | -4.28 % (Holm p=7.3e-53) | +0.97 % (Holm p=5.7e-22) |
| t2_q29 K=2 | 50 | 406.6 | 406.7 | 2 | 410.5 | 2 | 406.1 | 410.5 | 411.0 | -0.94 % (Holm p=6e-38) | +0.13 % (Holm p=5.8e-08) |
| t2_q29 K=3 | 50 | 286.3 | 286.3 | 2 | 296.9 | 2 | 286.2 | 296.9 | 296.8 | -3.60 % (Holm p=3.5e-145) | +0.02 % (Holm p=0.13) |
| t3_q2 K=1 | 100 | 1561.3 | 1561.3 | 2 | 1622.6 | 5 | 1543.1 | 1622.6 | 1630.4 | -3.78 % (Holm p=3e-90) | +1.18 % (Holm p=6.9e-49) |
| t3_q2 K=2 | 50 | 773.4 | 773.6 | 2 | 808.6 | 3 | 769.0 | 808.6 | 810.9 | -4.36 % (Holm p=2.2e-92) | +0.57 % (Holm p=8e-23) |
| t3_q2 K=3 | 100 | 547.6 | 547.6 | 2 | 567.6 | 2 | 547.1 | 567.6 | 568.8 | -3.53 % (Holm p=4.2e-89) | +0.08 % (Holm p=9.2e-05) |
| t3_q2345 K=1 | 100 | 1667.7 | 1667.7 | 2 | 1736.3 | 5 | 1646.3 | 1736.3 | 1738.7 | -3.95 % (Holm p=1.3e-73) | +1.30 % (Holm p=1e-41) |
| t3_q2345 K=2 | 50 | 819.6 | 819.9 | 2 | 857.1 | 3 | 815.5 | 857.1 | 859.2 | -4.38 % (Holm p=1.2e-82) | +0.50 % (Holm p=3e-19) |
| t3_q2345 K=3 | 50 | 579.9 | 579.9 | 2 | 603.3 | 2 | 579.4 | 603.3 | 603.5 | -3.87 % (Holm p=1e-85) | +0.09 % (Holm p=0.00015) |
| t4_reckless K=1 | 50 | 4369.9 | 4369.5 | 2 | 4488.5 | 30 | 4343.0 | 4488.5 | 4433.0 | -2.64 % (Holm p=4.7e-45) | +0.62 % (Holm p=1.4e-11) |
| t4_reckless K=2 | 20 | 1141.8 | 1143.8 | 2 | 1153.1 | 2 | 1135.9 | 1153.1 | 1153.4 | -0.98 % (Holm p=5e-19) | +0.52 % (Holm p=1.8e-09) |
| t4_reckless K=3 | 2 | 609.6 | 610.3 | 2 | 612.5 | 30 | 608.6 | 612.5 | 633.7 | -0.48 % (Holm p=0.00015) | +0.16 % (Holm p=4.7e-08) |
| t4_cautious K=1 | 100 | 3472.2 | 3472.2 | 2 | 3591.6 | 5 | 3433.8 | 3591.6 | 3543.2 | -3.33 % (Holm p=6e-42) | +1.12 % (Holm p=4.1e-34) |
| t4_cautious K=2 | 20 | 903.0 | 903.8 | 2 | 933.5 | 2 | 901.8 | 933.5 | 919.2 | -3.27 % (Holm p=2e-64) | +0.14 % (Holm p=0.00043) |
| t4_cautious K=3 | 100 | 492.4 | 492.4 | 2 | 500.0 | 2 | 492.1 | 500.0 | 503.6 | -1.52 % (Holm p=9.5e-40) | +0.07 % (Holm p=0.00017) |
| t4_mixed K=1 | 5 | 6729.4 | 6765.8 | 2 | 7016.2 | 100 | 6637.1 | 7016.2 | 6888.1 | -4.09 % (Holm p=1.7e-35) | +1.39 % (Holm p=5.5e-13) |
| t4_mixed K=2 | 3 | 1792.3 | 1819.3 | 2 | 1897.6 | 3 | 1792.2 | 1897.6 | 1861.0 | -5.55 % (Holm p=1.7e-62) | +0.00 % (Holm p=1) |
| t4_mixed K=3 | 2 | 980.0 | 988.9 | 2 | 1021.1 | 2 | 979.7 | 1021.1 | 1036.8 | -4.02 % (Holm p=2.1e-56) | +0.03 % (Holm p=1) |

**S7** (54 tests, Holm): tuned index cheaper than tuned PASSIVE-CONT in 27/27 cells; cheaper than tuned ROLLOUT-1 in 1/27 cells and DEARER than tuned ROLLOUT-1 in 19/27 cells (Holm p < 0.05). Mean relative advantage of the tuned index: over PASSIVE-CONT 3.13 %, over ROLLOUT-1 -0.43 % (positive = index cheaper). Tuned index vs index(tau=100): mean -0.169 %.

ROLLOUT-1 base lengths NOT run (and so not searched): none -- the grid is complete in all 27 cells on both seeds.
