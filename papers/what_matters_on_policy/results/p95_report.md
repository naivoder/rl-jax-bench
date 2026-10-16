| N | official CI: empirical (MC SE) | exact (hypergeom.) | exact (binomial) | l/h CI: empirical | exact (hypergeom.) | P(same order) mean | P(same best) mean |
|---|---|---|---|---|---|---|---|
| 32 | 0.3265 (0.0012) | 0.3254 | 0.3200 | 0.3997 | 0.3985 | 0.552 | 0.623 |
| 64 | 0.5723 (0.0013) | 0.5730 | 0.5624 | 0.6250 | 0.6248 | 0.593 | 0.659 |
| 128 | 0.8084 (0.0010) | 0.8066 | 0.7914 | 0.8376 | 0.8366 | 0.631 | 0.698 |
| 256 | 0.9349 (0.0006) | 0.9353 | 0.9181 | 0.9446 | 0.9442 | 0.670 | 0.740 |

cells within 3 MC SE of the exact hypergeometric coverage: 200/200; largest |z|: 2.77 (gradient_clipping, Ant-v1, N=128, official); 2.40 (reward_normalization, HalfCheetah-v1, N=256, official); 2.37 (gradient_clipping, Walker2d-v1, N=256, official)

| choice | env | k | P(same order) N=32 / 64 / 128 / 256 | P(same best) N=32 / 64 / 128 / 256 |
|---|---|---|---|---|
| input_normalization | Hopper-v1 | 2 | 0.49 / 0.50 / 0.52 / 0.56 | 0.49 / 0.50 / 0.52 / 0.56 |
| input_normalization | Humanoid-v1 | 2 | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 1.00 |
| input_normalization | Walker2d-v1 | 2 | 0.98 / 1.00 / 1.00 / 1.00 | 0.98 / 1.00 / 1.00 / 1.00 |
| input_normalization | HalfCheetah-v1 | 2 | 0.94 / 0.97 / 0.99 / 1.00 | 0.94 / 0.97 / 0.99 / 1.00 |
| input_normalization | Ant-v1 | 2 | 0.90 / 0.96 / 1.00 / 1.00 | 0.90 / 0.96 / 1.00 / 1.00 |
| reward_normalization | Hopper-v1 | 2 | 0.67 / 0.73 / 0.83 / 0.91 | 0.67 / 0.73 / 0.83 / 0.91 |
| reward_normalization | Humanoid-v1 | 2 | 0.93 / 0.98 / 1.00 / 1.00 | 0.93 / 0.98 / 1.00 / 1.00 |
| reward_normalization | Walker2d-v1 | 2 | 0.93 / 0.98 / 1.00 / 1.00 | 0.93 / 0.98 / 1.00 / 1.00 |
| reward_normalization | HalfCheetah-v1 | 2 | 0.70 / 0.77 / 0.85 / 0.94 | 0.70 / 0.77 / 0.85 / 0.94 |
| reward_normalization | Ant-v1 | 2 | 0.70 / 0.72 / 0.78 / 0.85 | 0.70 / 0.72 / 0.78 / 0.85 |
| normalize_advantages | Hopper-v1 | 2 | 0.65 / 0.68 / 0.75 / 0.81 | 0.65 / 0.68 / 0.75 / 0.81 |
| normalize_advantages | Humanoid-v1 | 2 | 0.53 / 0.59 / 0.64 / 0.70 | 0.53 / 0.59 / 0.64 / 0.70 |
| normalize_advantages | Walker2d-v1 | 2 | 0.52 / 0.51 / 0.54 / 0.57 | 0.52 / 0.51 / 0.54 / 0.57 |
| normalize_advantages | HalfCheetah-v1 | 2 | 0.50 / 0.53 / 0.56 / 0.62 | 0.50 / 0.53 / 0.56 / 0.62 |
| normalize_advantages | Ant-v1 | 2 | 0.52 / 0.57 / 0.60 / 0.65 | 0.52 / 0.57 / 0.60 / 0.65 |
| gradient_clipping | Hopper-v1 | 5 | 0.02 / 0.01 / 0.02 / 0.02 | 0.23 / 0.25 / 0.25 / 0.28 |
| gradient_clipping | Humanoid-v1 | 5 | 0.01 / 0.01 / 0.01 / 0.01 | 0.26 / 0.25 / 0.28 / 0.28 |
| gradient_clipping | Walker2d-v1 | 5 | 0.01 / 0.01 / 0.01 / 0.03 | 0.24 / 0.25 / 0.29 / 0.34 |
| gradient_clipping | HalfCheetah-v1 | 5 | 0.02 / 0.02 / 0.03 / 0.04 | 0.28 / 0.33 / 0.37 / 0.45 |
| gradient_clipping | Ant-v1 | 5 | 0.01 / 0.01 / 0.01 / 0.01 | 0.23 / 0.24 / 0.26 / 0.27 |
| learning_rate | Hopper-v1 | 4 | 0.53 / 0.59 / 0.63 / 0.67 | 0.59 / 0.60 / 0.63 / 0.67 |
| learning_rate | Humanoid-v1 | 4 | 0.46 / 0.63 / 0.71 / 0.77 | 0.86 / 0.96 / 0.99 / 1.00 |
| learning_rate | Walker2d-v1 | 4 | 0.51 / 0.59 / 0.68 / 0.79 | 0.57 / 0.62 / 0.69 / 0.79 |
| learning_rate | HalfCheetah-v1 | 4 | 0.51 / 0.58 / 0.69 / 0.80 | 0.57 / 0.61 / 0.70 / 0.80 |
| learning_rate | Ant-v1 | 4 | 0.79 / 0.86 / 0.94 / 0.99 | 0.79 / 0.86 / 0.94 / 0.99 |
