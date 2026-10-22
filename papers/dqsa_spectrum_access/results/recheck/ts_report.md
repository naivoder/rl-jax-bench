
#### E2proxy (n_exp 200, DP tuning worlds 100); plateau slots 60-99, Mbps

| T_s (ms) | rho | aloha | OCA  | DP tuned AT T_s (p_update, tau) | DP literal tau 0.5 | OCA same-slot (genie ceiling) | strongest legitimate comparator |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.999 | 41.1 | 69.4 | 59.1 (0.5, 0.1) | 8.8 | 71.6 | 69.4 |
| 1 | 0.904 | 41.0 | 55.2 | 55.0 (0.3, 0.2) | 8.8 | 72.0 | 55.2 |
| 3 | 0.291 | 41.0 | 44.2 | 54.5 (0.3, 0.2) | 8.9 | 72.1 | 54.5 |
| 10 | 0.220 | 41.1 | 44.0 | 54.6 (0.3, 0.2) | 8.9 | 72.2 | 54.6 |

log-rate order of the four non-genie baseline arms under the 7 conventions (t = 50 and 100, every T_s): 2 cell(s) change order: [('Ts=0.0001', 50, {('aloha', 'oca', 'dp_tuned_at_Ts', 'dp_literal_tau0.5'): ['db_bit', 'ln_bit', 'ln_mbpssum', 'log2_bit'], ('oca', 'aloha', 'dp_tuned_at_Ts', 'dp_literal_tau0.5'): ['db_mbit', 'ln_mbit', 'log2_mbit']}), ('Ts=0.003', 50, {('aloha', 'oca', 'dp_tuned_at_Ts', 'dp_literal_tau0.5'): ['db_bit', 'ln_bit', 'log2_bit'], ('oca', 'aloha', 'dp_tuned_at_Ts', 'dp_literal_tau0.5'): ['db_mbit', 'ln_mbit', 'ln_mbpssum', 'log2_mbit']})]

#### E3proxy (n_exp 200, DP tuning worlds 100); plateau slots 60-99, Mbps

| T_s (ms) | rho | aloha | OCA  | DP tuned AT T_s (p_update, tau) | DP literal tau 0.5 | OCA same-slot (genie ceiling) | strongest legitimate comparator |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.999 | 20.6 | 26.0 | 29.6 (0.5, 0.1) | 4.5 | 36.2 | 29.6 |
| 1 | 0.904 | 20.6 | 22.8 | 28.0 (0.3, 0.2) | 4.4 | 36.4 | 28.0 |
| 3 | 0.291 | 20.6 | 20.6 | 27.7 (0.3, 0.2) | 4.5 | 36.5 | 27.7 |
| 10 | 0.220 | 20.6 | 20.6 | 27.5 (0.3, 0.2) | 4.4 | 36.6 | 27.5 |

log-rate order of the four non-genie baseline arms under the 7 conventions (t = 50 and 100, every T_s): 1 cell(s) change order: [('Ts=0.001', 100, {('aloha', 'oca', 'dp_tuned_at_Ts', 'dp_literal_tau0.5'): ['db_bit', 'ln_bit', 'log2_bit'], ('oca', 'aloha', 'dp_tuned_at_Ts', 'dp_literal_tau0.5'): ['db_mbit', 'ln_mbit', 'ln_mbpssum', 'log2_mbit']})]

#### E2 (n_exp 50, DP tuning worlds 12); plateau slots 60-99, Mbps

| T_s (ms) | rho | aloha | OCA  | DP tuned AT T_s (p_update, tau) | DP literal tau 0.5 | OCA same-slot (genie ceiling) | strongest legitimate comparator |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.999 | 40.0 | 72.2 | 75.8 (0.2, 0.05) | 8.7 | 75.2 | 75.8 |
| 1 | 0.904 | 40.0 | 54.1 | 63.3 (0.1, 0.1) | 8.8 | 75.1 | 63.3 |
| 3 | 0.291 | 40.1 | 44.7 | 61.8 (0.1, 0.1) | 8.7 | 75.1 | 61.8 |
| 10 | 0.220 | 40.1 | 44.5 | 62.7 (0.1, 0.1) | 8.6 | 75.0 | 62.7 |

log-rate order of the four non-genie baseline arms under the 7 conventions (t = 50 and 100, every T_s): 1 cell(s) change order: [('Ts=0.001', 50, {('aloha', 'oca', 'dp_tuned_at_Ts', 'dp_literal_tau0.5'): ['db_bit', 'ln_bit', 'log2_bit'], ('oca', 'aloha', 'dp_tuned_at_Ts', 'dp_literal_tau0.5'): ['db_mbit', 'ln_mbit', 'ln_mbpssum', 'log2_mbit']})]

#### E3 (n_exp 50, DP tuning worlds 12); plateau slots 60-99, Mbps

| T_s (ms) | rho | aloha | OCA  | DP tuned AT T_s (p_update, tau) | DP literal tau 0.5 | OCA same-slot (genie ceiling) | strongest legitimate comparator |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.999 | 19.9 | 26.7 | 38.0 (0.2, 0.05) | 4.4 | 37.5 | 38.0 |
| 1 | 0.904 | 19.9 | 23.1 | 31.7 (0.1, 0.1) | 4.4 | 37.5 | 31.7 |
| 3 | 0.291 | 20.0 | 20.7 | 30.6 (0.1, 0.1) | 4.4 | 37.3 | 30.6 |
| 10 | 0.220 | 20.0 | 20.7 | 31.4 (0.1, 0.1) | 4.3 | 37.4 | 31.4 |

log-rate order of the four non-genie baseline arms under the 7 conventions (t = 50 and 100, every T_s): 1 cell(s) change order: [('Ts=0.0001', 100, {('aloha', 'oca', 'dp_tuned_at_Ts', 'dp_literal_tau0.5'): ['db_bit', 'ln_bit', 'ln_mbpssum', 'log2_bit'], ('oca', 'aloha', 'dp_tuned_at_Ts', 'dp_literal_tau0.5'): ['db_mbit', 'ln_mbit', 'log2_mbit']})]
