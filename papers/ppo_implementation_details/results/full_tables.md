
### Headline: ours vs authors' code vs paper (final-window return, mean ± s.e.)

| Metric | Paper | Ours (mean ± SE, n) | Δ% vs paper | Judgment vs paper | Authors' code: CleanRL / post / baselines | Δ% vs CleanRL (Welch p; 95 % CI of Δ) | Judgment vs CleanRL (one-SD equivalence rule) | Judgment vs post | Notes |
|---|---|---|---|---|---|---|---|---|---|
| CartPole-v1 `separate` @500k | 490.1 ± 3.4 (raw, n=3; plot read ~490); Welch p=0.47, q=0.733 | 486.5 ± 3.3 (n=10) | -0.7 % | MATCH | 489.0 ± 2.9 (n=10) / 483.9 ± 4.9 (n=5) / 489.7 ± 1.9 (n=3) | -0.5 % (p=0.57; [-10.8, 5.5]) | MATCH | MATCH | |
| CartPole-v1 `shared` @500k | 259.4 ± 7.7 (raw, n=3; plot read ~277); Welch p=0.28, q=0.992 | 248.2 ± 3.4 (n=10) | -4.3 % | MATCH | 235.6 ± 4.5 (n=10) / 250.0 ± 5.7 (n=5) / 244.2 ± 3.4 (n=3) | +5.3 % (p=0.04; [1.7, 22.6]) | MATCH | MATCH | |
| Acrobot-v1 `separate` @500k | -83.8 ± 0.3 (raw, n=3; plot read ~-85); Welch p=0.34, q=0.867 | -125.9 ± 41.6 (n=10) | -50.1 % | DIVERGENT | -125.8 ± 41.6 (n=10) / -83.7 ± 0.2 (n=5) / -83.4 ± 0.4 (n=3) | -0.0 % (p=1.00; [-124.4, 124.4]) | CLOSE | DIVERGENT | |
| Acrobot-v1 `shared` @500k | -377.0 ± 82.5 (raw, n=3; plot read ~-380); Welch p=0.45, q=0.083 | -298.4 ± 36.4 (n=10) | +20.9 % | DIVERGENT | -280.8 ± 40.9 (n=10) / -385.8 ± 30.1 (n=5) / -331.8 ± 71.8 (n=3) | -6.3 % (p=0.75; [-118.4, 84.0]) | CLOSE | DIVERGENT | |
| MountainCar-v0 `separate` @500k | -200.0 ± 0.0 (raw, n=3; plot read ~-200); Welch p=–, q=0.500 | -200.0 ± 0.0 (n=10) | +0.0 % | MATCH | – / -200.0 ± 0.0 (n=3) / – | – | MATCH (all −200, identical) | MATCH (identical floor) | |
| MountainCar-v0 `shared` @500k | -200.0 ± 0.0 (raw, n=3; plot read ~-200); Welch p=–, q=0.500 | -200.0 ± 0.0 (n=10) | +0.0 % | MATCH | – / -200.0 ± 0.0 (n=3) / – | – | MATCH (all −200, identical) | MATCH (identical floor) | |
| Pendulum-v1 `gymnasium029` @1000k | none (proxy) | -992.9 ± 60.9 (n=10) | – | – | -1099.4 ± 37.6 (n=5) / – / – | +9.7 % (p=0.16; [-27.3, 234.8]) | CLOSE | – | |
| Pendulum-v1 `gym021` @1000k | none (proxy) | -1008.2 ± 54.9 (n=10) | – | – | – / -1107.9 ± 66.1 (n=5) / – | – | – | CLOSE | |
| Pendulum-v1 `default` @1000k | none (proxy) | -1049.5 ± 48.1 (n=10) | – | – | – / – / – | – | – | – | |
| CartPole-v1 masked LSTM `gymnasium029` @500k | none (proxy) | 381.1 ± 15.8 (n=10) | – | – | 403.4 ± 7.3 (n=5) / – / – | -5.5 % (p=0.22; [-55.8, 8.5]) | CLOSE | – | |
| CartPole-v1 masked LSTM `lstm` @500k | none (proxy) | 377.5 ± 11.4 (n=10) | – | – | – / – / – | – | – | – | |

### Secondary metrics: AUC (50-bin mean) and steps to threshold (10-bin running mean)

| Setting | ours AUC | CleanRL AUC | post AUC | Δ% AUC vs CleanRL (Welch p) | ours steps-to-thr (median; reached) | CleanRL | post |
|---|---|---|---|---|---|---|---|
| CartPole-v1 `separate` | 414.5 ± 2.5 | 415.7 ± 1.2 | 412.5 ± 3.8 | -0.3 % (p=0.67) → MATCH | 310k; 10/10 | 305k; 10/10 | 300k; 5/5 |
| CartPole-v1 `shared` | 156.7 ± 1.0 | 154.4 ± 1.7 | 155.5 ± 3.0 | +1.5 % (p=0.26) → MATCH | never; 0/10 | never; 0/10 | never; 0/5 |
| Acrobot-v1 `separate` | -152.8 ± 38.7 | -150.6 ± 38.8 | -113.1 ± 3.1 | -1.5 % (p=0.97) → CLOSE | 180k; 9/10 | 190k; 9/10 | 190k; 5/5 |
| Acrobot-v1 `shared` | -449.2 ± 10.7 | -441.4 ± 12.8 | -472.1 ± 6.6 | -1.8 % (p=0.65) → MATCH | never; 0/10 | never; 0/10 | never; 0/5 |

### Primary family: separate > shared, ours, paired by seed, Holm over 2

| Claim | Δ (sep − shared) mean ± SE | 95 % t-CI | bootstrap CI | paired t p | Holm p | wins | supported? |
|---|---|---|---|---|---|---|---|
| separate > shared, CartPole-v1 | +238.3 ± 3.9 | [229.4, 247.2] | [230.8, 245.5] | 4.5e-13 | 9.1e-13 | 10/10 | YES |
| separate > shared, Acrobot-v1 | +172.5 ± 52.4 | [54.1, 291.0] | [71.0, 263.6] | 0.0093 | 0.0093 | 9/10 | YES |

### ablation (paired by seed, Holm over the family)

| Test | A | B | Δ (A − B) ± SE | rel. | bootstrap CI | paired t p | Holm p | wins |
|---|---|---|---|---|---|---|---|---|
| AB1 adv-norm per-mb vs per-batch, CartPole | 486.5 ± 3.3 | 487.0 ± 4.1 | -0.5 ± 5.6 | -0.1 % | [-10.7, 10.3] | 0.93 | 1 | 4/10 |
| AB2 adv-norm per-mb vs per-batch, Acrobot | -125.9 ± 41.6 | -83.8 ± 0.3 | -42.0 ± 41.6 | -50.2 % | [-125.4, 0.0] | 0.34 | 1 | 3/10 |
| AB3 value clip on vs off, CartPole | 486.5 ± 3.3 | 441.2 ± 30.8 | +45.3 ± 29.2 | +10.3 % | [-1.3, 106.3] | 0.16 | 0.78 | 7/10 |
| AB4 value clip on vs off, Acrobot | -125.9 ± 41.6 | -84.7 ± 0.5 | -41.2 ± 41.7 | -48.6 % | [-125.0, 1.5] | 0.35 | 1 | 4/10 |
| AB5 value clip on vs off, Pendulum | -1049.5 ± 48.1 | -983.0 ± 60.9 | -66.6 ± 58.2 | -6.8 % | [-171.2, 41.6] | 0.28 | 1 | 4/10 |
| AB6 reward scaling on vs off, Pendulum | -1049.5 ± 48.1 | -441.9 ± 58.7 | -607.6 ± 88.4 | -137.5 % | [-771.6, -444.1] | 7.3e-05 | 0.00058 | 0/10 |
| AB7 truncation bootstrap correct vs official, CartPole | 500.0 ± 0.0 | 486.5 ± 3.3 | +13.5 ± 3.3 | +2.8 % | [7.8, 19.8] | 0.0025 | 0.017 | 10/10 |
| AB8 truncation bootstrap correct vs official, Pendulum | -152.9 ± 2.7 | -1049.5 ± 48.1 | +896.6 ± 49.5 | +85.4 % | [807.7, 991.3] | 2.2e-08 | 2e-07 | 10/10 |
| AB9 LSTM reset on vs off, masked CartPole | 377.5 ± 11.4 | 420.0 ± 6.2 | -42.5 ± 15.0 | -10.1 % | [-69.6, -15.1] | 0.019 | 0.12 | 2/10 |

### exploratory (paired by seed, no correction)

| Test | A | B | Δ (A − B) ± SE | rel. | bootstrap CI | paired t p | Holm p | wins |
|---|---|---|---|---|---|---|---|---|
| E-x1 L5 broken (zero-state training) vs correct | 409.9 ± 12.1 | 377.5 ± 11.4 | +32.4 ± 14.6 | +8.6 % | [4.8, 58.8] | 0.054 | – | 7/10 |
| E-x2 LSTM vs memoryless ff_masked | 377.5 ± 11.4 | 49.7 ± 0.3 | +327.8 ± 11.3 | +659.9 % | [306.7, 347.8] | 3.5e-10 | – | 10/10 |
| E-x4 truncation bootstrap, Acrobot | -84.2 ± 0.3 | -125.9 ± 41.6 | +41.7 ± 41.7 | +33.1 % | [-0.7, 125.4] | 0.34 | – | 5/10 |
| E-x5a ddof 0 vs 1, CartPole | 489.6 ± 2.5 | 486.5 ± 3.3 | +3.1 ± 4.4 | +0.6 % | [-4.6, 11.6] | 0.5 | – | 6/10 |
| E-x5b ddof 0 vs 1, Acrobot | -125.4 ± 41.6 | -125.9 ± 41.6 | +0.5 ± 0.5 | +0.4 % | [-0.5, 1.3] | 0.36 | – | 6/10 |
| E-x6a Pendulum gym021 vs baselines wrappers | -1008.2 ± 54.9 | -1049.5 ± 48.1 | +41.3 ± 39.2 | +3.9 % | [-30.3, 114.1] | 0.32 | – | 5/10 |
| E-x6b Pendulum gymnasium029 vs baselines wrappers | -992.9 ± 60.9 | -1049.5 ± 48.1 | +56.6 ± 56.7 | +5.4 % | [-48.0, 158.3] | 0.34 | – | 6/10 |
| E-x6c LSTM gymnasium029 vs baselines wrappers | 381.1 ± 15.8 | 377.5 ± 11.4 | +3.6 ± 22.3 | +1.0 % | [-38.5, 45.1] | 0.87 | – | 7/10 |
| E-x8 per-batch adv-norm, Pendulum | -1049.5 ± 48.1 | -1029.1 ± 60.9 | -20.4 ± 70.5 | -2.0 % | [-146.9, 115.6] | 0.78 | – | 4/10 |

E-x7 CleanRL CartPole-v1: no-clip-vloss 451.2 ± 26.8 vs clip 494.0 ± 3.4 (seeds 1–5): Δ -42.8 (-8.7 %), Welch p=0.19, CI [-93.2, 1.6]

E-x7 CleanRL Acrobot-v1: no-clip-vloss -84.1 ± 0.4 vs clip -84.1 ± 0.4 (seeds 1–5): Δ -0.1 (-0.1 %), Welch p=0.89, CI [-0.9, 0.9]

### Properties T1 (max|ρ−1| first minibatch, all updates) and T2 (fraction of updates with k3 > 0.02)

| Setting | ours max\|ρ−1\| | ours frac k3>0.02 | ours mean k3 (last mb) | CleanRL frac k3>0.02 | CleanRL mean k3 | ours clipfrac | ours expl. var | ours frac k1<0 |
|---|---|---|---|---|---|---|---|---|
| CartPole-v1/separate | 1.2e-06 | 0.0000 | 2.33e-04 | 0.0000 | 2.37e-04 | 0.000 | 0.19 | 0.426 |
| CartPole-v1/shared | 1.7e-06 | 0.0002 | 9.61e-04 | 0.0000 | 9.61e-04 | 0.001 | 0.43 | 0.276 |
| CartPole-v1/advnorm_batch | 1.2e-06 | 0.0000 | 2.38e-04 | – | – | 0.000 | 0.20 | 0.427 |
| CartPole-v1/advnorm_ddof0 | 1.2e-06 | 0.0000 | 2.39e-04 | – | – | 0.000 | 0.18 | 0.426 |
| CartPole-v1/no_vclip | 9.5e-07 | 0.0000 | 5.06e-04 | – | – | 0.001 | 0.10 | 0.394 |
| CartPole-v1/trunc_bootstrap | 2.4e-06 | 0.0001 | 3.51e-04 | – | – | 0.000 | -0.27 | 0.396 |
| Acrobot-v1/separate | 2.4e-06 | 0.0000 | 6.60e-05 | 0.0000 | 6.76e-05 | 0.000 | 0.71 | 0.456 |
| Acrobot-v1/shared | 9.5e-07 | 0.0000 | 3.57e-04 | 0.0002 | 3.45e-04 | 0.000 | 0.33 | 0.458 |
| Acrobot-v1/advnorm_batch | 2.4e-06 | 0.0000 | 6.48e-05 | – | – | 0.000 | 0.78 | 0.448 |
| Acrobot-v1/advnorm_ddof0 | 2.9e-06 | 0.0000 | 7.23e-05 | – | – | 0.000 | 0.72 | 0.456 |
| Acrobot-v1/no_vclip | 2.9e-06 | 0.0000 | 1.55e-04 | – | – | 0.000 | 0.79 | 0.411 |
| Acrobot-v1/trunc_bootstrap | 2.4e-06 | 0.0000 | 6.01e-05 | – | – | 0.000 | 0.78 | 0.460 |
| MountainCar-v0/separate | 4.8e-07 | 0.0000 | 3.49e-05 | – | – | 0.000 | -0.00 | 0.487 |
| MountainCar-v0/shared | 9.5e-07 | 0.0002 | 4.34e-04 | – | – | 0.000 | -0.02 | 0.444 |
| Pendulum-v1/default | 5.7e-05 | 0.0934 | 9.69e-03 | – | – | 0.103 | 0.40 | 0.004 |
| Pendulum-v1/gym021 | 5.7e-05 | 0.0982 | 9.94e-03 | – | – | 0.104 | 0.46 | 0.004 |
| Pendulum-v1/gymnasium029 | 6.9e-05 | 0.0822 | 9.39e-03 | 0.0799 | 9.36e-03 | 0.099 | 0.45 | 0.006 |
| Pendulum-v1/trunc_bootstrap | 1.3e-05 | 0.0221 | 6.34e-03 | – | – | 0.056 | 0.97 | 0.013 |
| Pendulum-v1/no_rscale | 4.8e-06 | 0.0000 | 3.44e-03 | – | – | 0.025 | 0.42 | 0.041 |
| Pendulum-v1/no_vclip | 6.5e-05 | 0.1072 | 1.05e-02 | – | – | 0.106 | 0.46 | 0.007 |
| Pendulum-v1/advnorm_batch | 4.5e-05 | 0.0525 | 8.46e-03 | – | – | 0.091 | 0.41 | 0.005 |
| CartPole-v1/lstm (LSTM) | 7.4e-06 | 0.0047 | 4.54e-03 | – | – | 0.029 | 0.53 | 0.104 |
| CartPole-v1/gymnasium029 (LSTM) | 5.2e-06 | 0.0035 | 4.44e-03 | 0.0025 | 4.55e-03 | 0.027 | 0.54 | 0.117 |
| CartPole-v1/lstm_noreset (LSTM) | 2.9e-06 | 0.0020 | 3.77e-03 | – | – | 0.018 | 0.47 | 0.160 |
| CartPole-v1/ff_masked (LSTM) | 1.4e-06 | 0.0000 | 4.85e-04 | – | – | 0.000 | 0.09 | 0.397 |
| CartPole-v1/ff_full (LSTM) | 1.4e-06 | 0.0000 | 6.57e-04 | – | – | 0.001 | 0.25 | 0.373 |
| CartPole-v1/lstm_zero_train (LSTM) | 3.2e+01 | 0.2693 | 1.55e-02 | – | – | 0.058 | 0.49 | 0.038 |

### T3: remove the post's one tested mechanism (separate → shared): relative drop

CartPole-v1: ours -49 % vs paper-implied -47 %
Acrobot-v1: ours -137 % vs paper-implied -350 %

### Compute (per (env, arm); ours: one GPU process, seeds vmapped; oracle: CPU, one process per seed)

| Setting | ours compile s | ours steady s (all seeds) | ours steady env-steps/s/seed | oracle wall s/seed (mean) | analytic FLOPs per env step |
|---|---|---|---|---|---|
| CartPole-v1/separate | 9.0 | 17.0 | 29,367 | 103 | 231,296 |
| CartPole-v1/shared | 8.8 | 15.4 | 32,366 | 98 | 118,144 |
| CartPole-v1/advnorm_batch | 8.5 | 17.2 | 29,056 | – | 231,296 |
| CartPole-v1/advnorm_ddof0 | 8.7 | 17.3 | 28,982 | – | 231,296 |
| CartPole-v1/no_vclip | 8.9 | 17.3 | 28,918 | – | 231,296 |
| CartPole-v1/trunc_bootstrap | 11.7 | 19.8 | 25,231 | – | 231,296 |
| Acrobot-v1/separate | 10.4 | 19.8 | 25,209 | 143 | 239,616 |
| Acrobot-v1/shared | 9.6 | 17.8 | 28,021 | 127 | 123,136 |
| Acrobot-v1/advnorm_batch | 10.6 | 19.4 | 25,745 | – | 239,616 |
| Acrobot-v1/advnorm_ddof0 | 10.4 | 19.8 | 25,275 | – | 239,616 |
| Acrobot-v1/no_vclip | 11.4 | 22.0 | 22,765 | – | 239,616 |
| Acrobot-v1/trunc_bootstrap | 10.3 | 19.9 | 25,087 | – | 239,616 |
| MountainCar-v0/separate | 9.1 | 17.7 | 28,214 | – | 226,304 |
| MountainCar-v0/shared | 6.8 | 14.4 | 34,791 | – | 116,480 |
| Pendulum-v1/default | 8.5 | 63.7 | 15,700 | – | 539,648 |
| Pendulum-v1/gym021 | 7.9 | 62.5 | 15,990 | – | 539,648 |
| Pendulum-v1/gymnasium029 | 8.9 | 65.8 | 15,207 | 915 | 539,648 |
| Pendulum-v1/trunc_bootstrap | 10.9 | 71.0 | 14,075 | – | 539,648 |
| Pendulum-v1/no_rscale | 10.5 | 67.6 | 14,799 | – | 539,648 |
| Pendulum-v1/no_vclip | 10.6 | 66.6 | 15,013 | – | 539,648 |
| Pendulum-v1/advnorm_batch | 7.7 | 64.3 | 15,546 | – | 539,648 |
| CartPole-v1/lstm (LSTM) | 8.6 | 85.0 | 5,885 | – | 2,572,544 |
| CartPole-v1/gymnasium029 (LSTM) | 10.7 | 85.1 | 5,878 | 590 | 2,572,544 |
| CartPole-v1/lstm_noreset (LSTM) | 11.0 | 81.9 | 6,107 | – | 2,572,544 |
| CartPole-v1/ff_masked (LSTM) | 10.3 | 10.3 | 48,313 | – | 231,296 |
| CartPole-v1/ff_full (LSTM) | 10.0 | 10.7 | 46,624 | – | 231,296 |
| CartPole-v1/lstm_zero_train (LSTM) | 10.1 | 84.7 | 5,901 | – | 2,572,544 |

### Investigations I-1..I-4 (exploratory)

I-1 ours Acrobot separate, seeds 1–50: -92.4 ± 8.3 (median -84.2); stuck 1/50; vs paper -10.3 % → CLOSE
I-1 ours Acrobot shared, seeds 1–50: -350.5 ± 14.8 (median -381.6); stuck 19/50; vs paper +7.0 % → CLOSE
I-1 CleanRL Acrobot separate, 30 seeds: -98.0 ± 13.9; stuck 1/30; all authors' code stuck 1/38; Fisher (ours [1, 49] vs authors [1, 37]) p=1.00; ours(50) vs CleanRL Welch Δ +5.6 (+5.7 %) p=0.73 → CLOSE
I-1 Acrobot shared: ours (50) vs pooled authors' code (18: -318.4 ± 27.8): Δ -32.1 (-10.1 %), Welch p=0.32, CI [-93.9, 28.2] → CLOSE
I-2 random policy: gymnax 0.0145 ± 0.0019 (return -499.14) vs gymnasium 0.0152 ± 0.0019 (return -498.87); z=-0.28
I-3 max|rho-1| rollout path vs training path: {"log_std=0.0": {"float32": 1.9669532775878906e-06, "float64": 3.552713678800501e-15}, "log_std=-2.0": {"float32": 1.4483928680419922e-05, "float64": 2.6201263381153694e-14}, "log_std=-3.0": {"float32": 3.981590270996094e-05, "float64": 7.083222897108499e-14}, "log_std=-4.0": {"float32": 0.00010770559310913086, "float64": 1.9317880628477724e-13}, "log_std=-4.5": {"float32": 0.0001780986785888672, "float64": 3.1796787425264483e-13}, "device": "cuda:0"}
I-4 CleanRL LSTM L3 off 404.2 ± 12.3 vs on 403.4 ± 7.3 (n=5/5): Δ +0.8 (+0.2 %), Welch p=0.96, CI [-25.6, 23.8]
