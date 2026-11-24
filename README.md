# rl-jax-bench

Reinforcement learning papers reimplemented in JAX, each checked against the published results and, where the
authors released code, against that code.

## Setup

```bash
uv sync                # CPU
uv sync --extra cuda   # NVIDIA GPU (CUDA 12)
```

Every paper lives in `papers/<name>/` with its own README (paper link, official code link, what was reproduced,
how to run it). All papers share this one environment and the small helper package in `src/rl_jax_bench/`
(GPU setup, logging, seed-level statistics, a recurrent double-DQN core).

```bash
cd papers/<name>
uv run pytest          # that paper's tests (run from the paper folder: each paper has its own `src` package)
```

`uv run pytest` at the repository root runs the tests of the shared package only.

Tests that compare against the authors' code skip until that code is present. Where a paper has such tests,
`sh scripts/official/setup.sh` in its folder fetches the authors' repository at the pinned commit and builds the
environment it needs; the paper's README says what was and was not verified about that script.

## Papers

| Paper | Folder | Official code |
|---|---|---|
| A Closer Look at Deep Policy Gradients ([ICLR 2020](https://arxiv.org/abs/1811.02553)) | [papers/closer_look_pg](papers/closer_look_pg) | none released for this paper (companion code [MadryLab/implementation-matters](https://github.com/MadryLab/implementation-matters) used as oracle) |
| Bridging the Imitation Gap by Adaptive Insubordination ([NeurIPS 2021](https://arxiv.org/abs/2007.12173)) | [papers/advisor_imitation_gap](papers/advisor_imitation_gap) | [allenai/advisor](https://github.com/allenai/advisor) |
| Constrained Deep Reinforcement Learning for Cognitive Radar Resource Management ([arXiv 2026](https://arxiv.org/abs/2606.05526)) | [papers/constrained_radar_rrm](papers/constrained_radar_rrm) | none released |
| Deep Multi-User Reinforcement Learning for Distributed Dynamic Spectrum Access ([IEEE TWC 2019](https://arxiv.org/abs/1704.02613)) | [papers/dqsa_spectrum_access](papers/dqsa_spectrum_access) | none released |
| Deep Q-learning from Demonstrations ([AAAI 2018](https://arxiv.org/abs/1704.03732)) | [papers/dqfd](papers/dqfd) | none released |
| Deep Reinforcement Learning at the Edge of the Statistical Precipice ([NeurIPS 2021](https://arxiv.org/abs/2108.13264)) | [papers/statistical_precipice](papers/statistical_precipice) | [google-research/rliable](https://github.com/google-research/rliable) |
| Deep Reinforcement Learning Control for Radar Detection and Tracking in Congested Spectral Environments ([IEEE TCCN 2020](https://arxiv.org/abs/2006.13173)) | [papers/radar_drqn_coexistence](papers/radar_drqn_coexistence) | none released |
| GOSPA-Driven Non-Myopic Multi-Sensor Management with Multi-Bernoulli Filtering ([arXiv 2025](https://arxiv.org/abs/2511.01045)) | [papers/gospa_sensor_mgmt](papers/gospa_sensor_mgmt) | [sggjon5/GOSPA-Driven-Non-Myopic-Multi-Sensor-Management-with-Multi-Bernoulli-Filtering](https://github.com/sggjon5/GOSPA-Driven-Non-Myopic-Multi-Sensor-Management-with-Multi-Bernoulli-Filtering) |
| Hindsight Credit Assignment ([NeurIPS 2019](https://arxiv.org/abs/1912.02503)) | [papers/hindsight_credit](papers/hindsight_credit) | [hca-neurips2019/hca](https://github.com/hca-neurips2019/hca) |
| Jump-Start Reinforcement Learning ([ICML 2023](https://arxiv.org/abs/2204.02372)) | [papers/jsrl](papers/jsrl) | none released (third-party oracle: [steventango/jumpstart-rl](https://github.com/steventango/jumpstart-rl)) |
| Maintaining Plasticity in Deep Continual Learning ([arXiv 2023](https://arxiv.org/abs/2306.13812); Nature 2024) | [papers/loss_of_plasticity](papers/loss_of_plasticity) | [shibhansh/loss-of-plasticity](https://github.com/shibhansh/loss-of-plasticity) |
| Multi-Target Radar Search and Track Using Sequence-Capable Deep Reinforcement Learning ([arXiv 2025](https://arxiv.org/abs/2502.13584)) | [papers/radar_search_track_seq](papers/radar_search_track_seq) | none released |
| Non-myopic Beam Scheduling for Multiple Smart Target Tracking in Phased Array Radar Network ([arXiv 2023](https://arxiv.org/abs/2312.07858)) | [papers/smart_target_sched](papers/smart_target_sched) | none released |
| On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems ([arXiv 2008](https://arxiv.org/abs/0805.3415)) | [papers/nonstationary_ucb](papers/nonstationary_ucb) | none released |
| Poisson multi-Bernoulli mixture filter: direct derivation and implementation ([IEEE TAES 2018](https://arxiv.org/abs/1703.04264)) | [papers/pmbm_tracker](papers/pmbm_tracker) | [Agarciafernandez/MTT](https://github.com/Agarciafernandez/MTT) |
| Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs ([ICML 2022](https://arxiv.org/abs/2110.05038)) | [papers/recurrent_pomdp_baselines](papers/recurrent_pomdp_baselines) | [twni2016/pomdp-baselines](https://github.com/twni2016/pomdp-baselines) |
| Robust Asymmetric Learning in POMDPs ([ICML 2021](https://arxiv.org/abs/2012.15566)) | [papers/robust_asymmetric_a2d](papers/robust_asymmetric_a2d) | [plai-group/a2d](https://github.com/plai-group/a2d) |
| Streaming Deep Reinforcement Learning Finally Works ([arXiv 2024](https://arxiv.org/abs/2410.14606)) | [papers/streaming_deep_rl](papers/streaming_deep_rl) | [mohmdelsayed/streaming-drl](https://github.com/mohmdelsayed/streaming-drl) |
| The 37 Implementation Details of Proximal Policy Optimization ([ICLR Blog Track 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)) | [papers/ppo_implementation_details](papers/ppo_implementation_details) | [vwxyzjn/ppo-implementation-details](https://github.com/vwxyzjn/ppo-implementation-details) |
| Time Limits in Reinforcement Learning ([ICML 2018](https://arxiv.org/abs/1712.00378)) | [papers/time_limits_rl](papers/time_limits_rl) | none released (oracles: [Tonic](https://github.com/fabiopardo/tonic), OpenAI Baselines) |
| Unbiased Asymmetric Reinforcement Learning under Partial Observability ([AAMAS 2022](https://arxiv.org/abs/2105.11674)) | [papers/unbiased_asymmetric](papers/unbiased_asymmetric) | [abaisero/asym-rlpo](https://github.com/abaisero/asym-rlpo) |
| Variance Reduction for Reinforcement Learning in Input-Driven Environments ([ICLR 2019](https://arxiv.org/abs/1807.02264)) | [papers/input_driven_baselines](papers/input_driven_baselines) | [hongzimao/input_driven_rl_example](https://github.com/hongzimao/input_driven_rl_example) |
| What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study ([ICLR 2021](https://arxiv.org/abs/2006.05990)) | [papers/what_matters_on_policy](papers/what_matters_on_policy) | [google-research/seed_rl](https://github.com/google-research/seed_rl) |

Two entries are not reinforcement learning: `pmbm_tracker` is a multi-target tracking filter and
`gospa_sensor_mgmt` is sensor management by tree search over such a filter.

## Conventions

- Each README reports the paper's numbers beside this reproduction's (mean and standard error over seeds), with a
  verdict per quantity, and lists what was not reproduced or not run.
- Uncertainty is the standard error over seeds; a claim that one method beats another uses a paired test, and
  families of tests are Holm-corrected (`rl_jax_bench.stats`).
- Stage names used across papers: `toys` (exactly computable checks), `small` (a reduced-scale run of the whole
  protocol), `full` (paper scale), `recheck` (a second pass that re-examined the first verdicts).
- Outputs go to `runs/` inside each paper folder (git-ignored). `results/` holds the small final tables the READMEs cite.
