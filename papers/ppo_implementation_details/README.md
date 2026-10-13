# The 37 Implementation Details of Proximal Policy Optimization
Shengyi Huang, Rousslan Fernand Julien Dossa, Antonin Raffin, Anssi Kanervisto, Weixun Wang, ICLR 2022 Blog Track.
Paper: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/. Official code:
[vwxyzjn/ppo-implementation-details](https://github.com/vwxyzjn/ppo-implementation-details) (commit `fbef824`, see
"Checked against"); [CleanRL](https://github.com/vwxyzjn/cleanrl) (`fe8d8a0`) and
[openai/baselines](https://github.com/openai/baselines) (`ea25b9e`) were used as further oracles.

## What this is
The post is a reproduction and catalogue, not a new algorithm. It takes openai/baselines ppo2 at `ea25b9e` as the
reference PPO, lists 37 implementation details of it, and shows that short PyTorch scripts implementing them give
learning curves that match ppo2's. Its one comparative claim is that separate actor and critic networks clearly
beat a shared torso on simple environments. This folder reimplements the post's PPO variants (`ppo.py`,
`ppo_shared.py`, `ppo_continuous_action.py`, the logic of `ppo_atari_lstm.py`) in JAX (flax.linen + optax + gymnax)
as one jitted update, vmapped over seeds. It then compares them with the paper's own runs and with the authors' code
run on the same machine.

Labels used in code and tables:
- **Details.** C1-C13 core, A1-A9 Atari, K1-K9 continuous, L1-L5 LSTM, M1 MultiDiscrete, X1-X4 auxiliary, numbered
  in the post's order.
- **Experiments.** E1 is classic control (CartPole-v1, Acrobot-v1, MountainCar-v0). E3 is the MuJoCo proxy
  (gymnax Pendulum-v1). E4 is the LSTM proxy (CartPole with the velocities masked). E5/E6 (Gym-muRTS) are not run.
- **Claims.** Q-A: the implementation matches the reference. Q-B: separate beats shared. T1-T3 are the post's
  stated properties: ratio = 1 at the first minibatch, approx_kl mostly below 0.02, and the effect of removing
  separate networks.
- **Our additions.** AB1-AB9 are pre-declared ablations, our extension. I-1..I-5 and E-x1..E-x8 are exploratory
  investigations.
- **Run stages.** "small" is 1/5 of the paper's horizon, "full" is the paper's horizon, "official_check" is the
  authors' code at E1 scale, "review" and "recheck" are later re-examinations. Tags under `runs/` use these names.
- **Verdict rule.** MATCH means |Δ| ≤ 5 %. CLOSE means ≤ 20 %, or a larger Δ that is within one SD of the oracle
  (the one-SD equivalence rule). DIVERGENT means anything else.

## Results
Metric: the mean raw return of training episodes that end in the last 10 % of steps ("final window"), mean ± SE over
seeds. The same `src/eval.summarize` is applied to ours, to the authors' code, and to the paper's own W&B runs
(n = 3 per row, `data/paper_raw.json`). Horizon: 500k steps (E1, E4) and 1M steps (E3). Ours uses 10 seeds unless
stated.

| Quantity | Paper (raw runs, n = 3) | Authors' code on our machine | This reproduction (mean ± SE, n) | Verdict |
|---|---|---|---|---|
| CartPole separate | 490.1 ± 3.4 | CleanRL 489.0 ± 2.9 (10), post's code 486.2 ± 2.7 (10) | 486.5 ± 3.3 (10) | MATCH (−0.7 %; vs CleanRL −0.5 %, p = 0.57) |
| CartPole shared | 259.4 ± 7.7 | CleanRL 235.6 ± 4.5 (10), post's code 250.3 ± 5.1 (10) | 248.2 ± 3.4 (10); 243.9 pooled (30) | CLOSE (−6.0 % pooled, Welch p = 0.17); ours vs the post's code −0.9 % |
| Acrobot separate, 10 seeds | −83.8 ± 0.3 | CleanRL −125.8 ± 41.6 (10) | −125.9 ± 41.6 (10) | DIVERGENT by Δ % (−50 %): one seed never learns; Welch p = 0.34 vs the paper |
| Acrobot separate, 50 seeds | −83.8 | CleanRL −98.0 ± 13.9 (30) | −92.4 ± 8.3 (50; median −84.2) | CLOSE (−10.3 %) |
| Acrobot shared, 10 seeds | −377.0 ± 82.5 | CleanRL −280.8 ± 40.9 (10), post's code −320.8 ± 39.0 (10) | −298.4 ± 36.4 (10) | DIVERGENT by Δ % (+20.9 %); Welch p = 0.45 vs the paper |
| Acrobot shared, 50 seeds | −377.0 | pooled authors' code −318.4 ± 27.8 (18) | −350.5 ± 14.8 (50) | CLOSE (+7.0 %) |
| MountainCar separate / shared | −200 | post's code −200 (3) | −200.0 ± 0 / −200.0 ± 0 | MATCH (nobody learns) |
| Q-A, loss half (policy / value loss vs the paper's logs) | - | - | 16 loss rows, all within ±16 % | SUPPORTED |
| Q-B separate > shared, CartPole | "clearly outperforms" | - | Δ +238.3 ± 3.9, 10/10 seeds, Holm p 9e-13 | SUPPORTED at vf_coef 0.5 |
| Q-B separate > shared, Acrobot | "clearly outperforms" | - | Δ +172.5 ± 52.4, 9/10 seeds, Holm p 0.0093 | SUPPORTED |
| T1 max\|ρ−1\| at the first minibatch | 1 | - | ≤ 7.4e-6 (classic control, LSTM); 5.7e-5 to 6.9e-5 (Pendulum) | holds up to float32 rounding (see caveats) |
| T2 fraction of updates with approx_kl (k3) > 0.02 | "generally below" | CleanRL 0-0.02 % / 8.0 % | 0-0.02 % classic control, 0.2-0.5 % LSTM, 8.2 % Pendulum | holds on discrete tasks; about 90 % of updates on Pendulum, same as CleanRL |
| T3 CartPole drop, separate to shared | −47 % | CleanRL −52 % | −49 % | MATCH |
| E3 Pendulum proxy, gymnasium029 wrappers | none | CleanRL −1099.4 ± 37.6 (5) | −992.9 ± 60.9 | CLOSE (+9.7 %, p = 0.16) |
| E3 Pendulum proxy, the post's gym 0.21 wrappers | none | post's code −1107.9 ± 66.1 (5) | −1008.2 ± 54.9 | CLOSE (+9.0 %, p = 0.28) |
| E3 Pendulum proxy, baselines wrappers | none | baselines ppo2 on identical v1 dynamics | −1049.5 ± 48.1 | MATCH (0.0 %) |
| E4 masked-CartPole LSTM proxy (modified recipe) | none | CleanRL logic 397.9 ± 5.9 (20) | 393.1 ± 6.9 (30) | MATCH (−1.2 %, p = 0.60; exploratory n) |

Ablations (our extension, paired by seed, Holm over 9 tests). Three survive correction:
- **AB8 correct truncation bootstrap, Pendulum:** −152.9 ± 2.7 vs −1049.5 ± 48.1 (+85 %, Holm p 2e-7). This is the
  largest effect measured.
- **AB7 the same, CartPole:** 500.0 vs 486.5 (+2.8 %, Holm 0.017). The post keeps the no-bootstrap handling on
  purpose, for fidelity.
- **AB6 reward scaling, Pendulum:** turning it off improves the return (−441.9 vs −1049.5, Holm 6e-4). The cited
  expectation is that scaling helps. The authors' code shows the same reversal (−1099 to −463).

Not significant after Holm: per-minibatch vs per-batch advantage normalization (AB1, AB2), value clipping
(AB3-AB5), and the LSTM state reset L3 (AB9, Holm 0.12; the authors' code also shows no effect). The memory matters:
LSTM 377.5 vs memoryless 49.7 on the masked observation, and 499.2 for the memoryless net on the full observation.
`results/full_tables.md` and `results/full_report.json` hold every number.

Caveats:
- **Acrobot DIVERGENT rows.** These are per-seed bimodality, not an implementation gap. CleanRL at n = 10 is
  identical (−125.8, with one stuck seed). The paper's own three shared seeds are −470 / −213 / −448. Stuck-seed
  rates are 1/50 for ours and 1/30 for CleanRL. Env branches, the update loop, init and the stuck rate were all
  tested on our side.
- **CartPole shared.** It was MATCH at the declared n = 10 (−4.3 %) but not confirmed on 20 fresh seeds (−6.8 %),
  so it is reported as CLOSE. The authors' own code lands in the same place.
- **Q-B depends on the value-loss weight** (exploratory). On CartPole, at vf_coef 0.01 the shared torso reaches
  495.1 and beats separate (434.3, 0/10 wins). Over a symmetric vf_coef grid the best arms are not separated (495.1
  vs 491.3). On Acrobot separate wins at every vf_coef tested, by less as vf_coef falls.
- **T1 on Pendulum** exceeds the declared 1e-5 tolerance. The rollout path ((1, d) forward) and the training path
  ((64, d) forward) differ in float32 by an amount that scales with 1/σ. At σ ≈ 0.012 this reaches the observed
  level, and it is ≤ 3e-13 in float64. It is not stale data: early in training it stays ≤ 2e-6.
- **E4 recipe.** The LSTM proxy runs the post's L1-L5 under a **modified recipe** (added K8 reward scaling, clip
  0.2). The post's own recipe (clip 0.1, no reward normalization) does not learn on this proxy, in ours (22.9) or in
  the authors' code (23.1).
- **Files predating revisions.** `results/full_report.json` / `full_tables.md` were regenerated after the recheck and
  use the paper's raw runs. `results/small_report.json`, `small_table.txt`, `official_reference_report.json` and
  `full_budget_ledger.json` predate it. Their paper values are figure plot reads and some of their labels were
  revised later; the table above is final. The fresh-seed, vf_coef-grid and larger-n numbers come from
  `scripts/recheck/analysis.py` and are not in `results/`.

## What differs from the paper / known gaps
- **Not reproduced (budget or environment):** Atari, MuJoCo-v2, Atari-LSTM, Gym-μRTS (E5/E6), procgen, envpool and
  the Brax hopper proxy. MultiDiscrete (M1) and invalid-action masking (X4) are only checked in closed form
  (`tests/test_dist_closed_form.py`).
- **Proxies.** gymnax environments replace gym classic control; their dynamics are parity-tested, including the
  terminal, goal and speed-clip branches. Pendulum-v1 replaces MuJoCo, and masked-velocity CartPole replaces
  Atari-LSTM. Proxy rows have no paper number.
- **Wrapper modes.** `reward_scaling_mode` selects between three wrapper semantics:
  - `baselines` (default): VecNormalize.
  - `gymnasium029`: current CleanRL.
  - `gym021`: the post's own `ppo_continuous_action.py` on gym 0.21.
- **Precision.** float32 training with `jax_default_matmul_precision="highest"` (TF32 broke the ratio = 1 check at
  about 1e-3). The unit tests run in float64.
- **Truncation.** The default treats truncation as termination, as the official code does. The correct bootstrap is
  the `trunc_bootstrap` arm.
- **Post's code stack.** The authors'-code venv used torch 1.13.1, not the post's lockfile stack (torch 1.10.1). Its
  trajectories therefore differ from the paper's.
- **Small residual on Acrobot separate.** Among learned seeds, ours (−84.3 ± 0.2, n = 20) vs the post's code
  (−83.3 ± 0.2) differs by 1.0 (p = 0.009, unplanned). This is most likely a gym-0.21 vs gymnax/gymnasium level
  difference. It is untested and moves no label.
- **Not run in this folder's tests:** GPU vs CPU float32 trajectories diverge per seed (seen on Acrobot seed 9).
  This is float32 chaos, not a code path.

## Layout
- `src/model.py`: separate and shared actor-critic MLPs, the LSTM agent in torch parameter layout, and the
  distributions.
- `src/algorithm.py`: GAE with an optional truncation bootstrap, advantage normalization, clipped surrogate,
  clipped value loss, KL estimators, and the optimizer (global-norm clip, Adam with ε = 1e-5, lr annealing).
- `src/data.py`: gymnax / Brax environments, vectorized auto-reset, and VecNormalize-style obs and reward
  normalization in the three wrapper modes.
- `src/train.py`: the PPO update as one jitted step, with `lax.scan` over updates and `vmap` over seeds. `ARMS`
  lists each arm's config overrides. Also handles checkpoints and the NaN guard.
- `src/eval.py`: the paper's metric (final window, AUC, steps to threshold), a greedy diagnostic eval, and loaders.
- `src/ppo_debug.py`: CleanRL's debug variables (k1/k2/k3, clipfrac, explained variance, max|ρ−1|) and a pre-flight
  check, with an optional mask.
- `src/configs/`: `classic_control`, `pendulum`, `lstm_masked_cartpole` and `brax_hopper` configs. `sources.yaml`
  gives the source of every value (paper, official code, or ASSUMED).
- `data/paper_raw.json`: the paper's own W&B runs (E1), reduced with our metric.
- `scripts/small/`:
  - `run_small.sh`: small-scale runs.
  - `run_difftests.sh`: the differential tests against the authors' code; also rebuilds the fixtures.
  - `difftest_export.py`, `wrapper_export.py`, `env_export.py`: our side of the differential tests.
  - `difftest_compare.py`, `make_parity_fixtures.py`: compare and pack.
  - `report.py`: small-scale report.
- `scripts/official_check/`:
  - `metric_export.py`: metric differential test, our side.
  - `reference_report.py`: report on the authors' code at E1 scale.
- `scripts/full/`:
  - `run_full.py`: E1/E3/E4 launcher, 10 seeds per arm.
  - `run_investigation.sh`: Acrobot seeds 11-50 and the random-policy check.
  - `random_policy_acrobot.py`, `ratio_numerics.py`: investigations I-2 and I-3.
  - `report.py`: headline tables, test families and T1-T3. `--paper-ref plot` uses figure reads instead.
- `scripts/review/`:
  - `rerun.sh`, `rerun_diff.py`: bit-for-bit rerun check.
  - `run_resolution.sh`, `resolution_report.py`: the post's own LSTM recipe.
  - `lstm_multiupdate_export.py`, `lstm_multiupdate_compare.py`, `make_multiupdate_fixture.py`: multi-update LSTM
    differential test.
  - `flags_check.py`, `config_diff.py`, `recompute_from_raw.py`: consistency checks.
- `scripts/recheck/`:
  - `run_fix_cpu.sh`: before/after, vf_coef grid and fresh-seed reruns.
  - `analysis.py`: their read-outs.
  - `paper_raw.py`: builds `data/paper_raw.json` from downloaded W&B histories.
  - `ff_multiupdate_export.py`, `make_ff_multi_fixture.py`, `make_env_branch_fixture.py`: fixture builders.
  - `launch_post_s6_10.py`: the post's code, seeds 6-10.
- `scripts/official/`: our harness around the authors' code. It runs under `$OFFICIAL_ROOT`'s own venvs.
  - `oracle_*.py`: the oracle side of each differential test.
  - `run_cleanrl.py`, `run_post.py`, `run_baselines*.py`: run one script unmodified.
  - `launch_*.py`: CPU pools for the reference, full and follow-up oracle runs.
  - `make_cleanrl_*.py`: write the derived CleanRL variants into `$OFFICIAL_ROOT/harness/`.
- `scripts/check_resume.py`: checks that a resumed run reproduces the uninterrupted one exactly.
- `scripts/gather_seeds.py`, `scripts/repro_commands.py`: per-seed collection and the exact command of every
  recorded run.

## Run it
From this folder:
```sh
uv run python -m pytest -q                                            # about 2 min on CPU
uv run python -m src.train --config src/configs/classic_control.yaml \
    --envs CartPole-v1,Acrobot-v1 --arms separate,shared --tag e1      # 10 seeds vmapped per arm
uv run python -m scripts.full.run_full E1 E3 E4                        # full-scale runs (about 26 min on an RTX 3090)
sh scripts/full/run_investigation.sh
uv run python -m scripts.full.report                                   # needs $OFFICIAL_ROOT/oracle_runs for the oracle columns
sh scripts/small/run_difftests.sh                                      # needs $OFFICIAL_ROOT (see below)
```
Outputs go to `runs/<tag>/<env>/<arm>/seed_<n>/metrics.json`. Each file holds the episodes as
`[global_step, return_sum, count, length_sum]`, the per-update debug variables, a summary and the timing. The whole
original study used about 1.7 GPU-h of process wall time on an RTX 3090 (reproduction 1.08, recheck 0.66), plus
about 2.5 h of CPU wall time for the authors' code. The original suite (59 passed, 1 skipped) took 70 s. Shell drivers use
`PY=${PY:-"uv run python"}`.

## Tests
- `test_grads.py`: worked examples W1-W12 in float64 (GAE, normalization, clipped losses, Adam, wrappers), a bandit
  z-test and `check_grads`.
- `test_smoke.py`: a short training run.
- `test_oracle_parity.py`: one minibatch of loss, gradients, Adam step, GAE and lr, for each agent type, plus
  wrappers, env dynamics and the metric. Checked against the authors' code outputs stored in
  `tests/fixtures/official/` (numbers only, never their code). Tolerance 1e-5.
- `test_ff_multiupdate_parity.py`, `test_lstm_multiupdate_parity.py`: several whole updates replayed against
  CleanRL's update blocks.
- `test_env_branch_parity.py`: gymnax vs gymnasium 0.29.1 / gym 0.15.7 on terminal, goal and speed-clip branches,
  with power checks. One case skips by design: gym 0.15.7 Pendulum-v0 integrates in the other order, and a power
  check covers that.
- `test_dist_closed_form.py`: M1 and X4 in closed form.
- `test_ppo_debug.py`: `src/ppo_debug.py` against `src/algorithm.py` and CleanRL's formulas.

All tests run on CPU from committed fixtures. None needs external data.

## Checked against
- **Authors' code.** The post's repo `vwxyzjn/ppo-implementation-details@fbef824` (2022-07-01), CleanRL `fe8d8a0`,
  openai/baselines `ea25b9e`, and the authors' baselines fork `e2cb1c9`.
  - `fbef824` is a revision after the post (2022-03-25). The post's pinned `eb40cbe` no longer exists. The PPO files
    at `fbef824` are byte-identical to the pre-publication `339df2e`.
  - W&B code snapshots of the paper's runs are algorithmically identical to `fbef824` (AST diff).
- **Tolerance.** The differential tests use |ours − theirs| ≤ 1e-5 + 1e-5·|theirs|. Max errors: 5.1e-6 for a single
  minibatch, 1.7e-5 absolute (7.6e-6 relative) for whole feed-forward updates, 1.9e-6 for whole LSTM updates,
  3.8e-6 for wrappers, and 4.8e-7 for env streams.
- **External setup.** The authors' clones, their venvs (`.venv` with torch 2.4.1 / gymnasium 0.29.1, `.venv_post`
  with gym 0.21.0 / torch 1.13.1, `.conda_py37` for baselines) and their reference outputs (`oracle_runs/`) live
  under `OFFICIAL_ROOT`. The default is `external/official` (git-ignored). The original oracle run folders were
  named with older stage names, so rename them to `small`, `full` and `review` before pointing the reports at them.
  `harness/review_ppo_continuous_action_norscale.py` (CleanRL's continuous script with the reward normalization
  removed) is a hand-edited copy of authors' code and is not shipped.
