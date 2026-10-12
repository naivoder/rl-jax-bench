# A Closer Look at Deep Policy Gradients
Andrew Ilyas, Logan Engstrom, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, Aleksander Madry,
ICLR 2020. Paper: https://arxiv.org/abs/1811.02553 (v4). Official code: none released for this paper; the authors'
companion repository [MadryLab/implementation-matters](https://github.com/MadryLab/implementation-matters) (commit
`5ee6ecb`) was used as a reference oracle.

## What this is
The paper audits PPO and TRPO on MuJoCo locomotion. Its main tools are measurements. It reports that the policy
gradient estimated from one training batch agrees poorly with the true gradient. It also finds that the learned value
function fits returns poorly and is only a weak baseline. Finally, it shows that the surrogate objective's landscape
differs from the true reward's landscape. This folder reimplements those measurements as jittable JAX functions,
together with the PPO and TRPO trainers they audit. It reruns them on Hopper and Walker2d. Humanoid was not run.

Experiment labels used in code and tables: E0 training reward (Fig 8), E1 gradient quality (Figs 9/10), E2 value
prediction error (Figs 3/11/12), E3 baseline variance (Fig 4), E4 objective landscapes (Figs 5-7, 13-20), E5
no-baseline training. P1-P4 are the four pre-declared paired tests (Holm over 14 tests, alpha 0.05). C1-C9 are the
paper's qualitative claims.

## Results
5 training seeds per arm, 488 iterations x 2048 steps. Environments are brax MJX proxies, see "What differs". The
judgment rule was fixed before the runs: MATCH means within 5 % or inside the paper's per-agent spread, CLOSE means
within 20 %, and DIVERGENT means anything else. The paper's numbers are read off its vector figures.
"Authors' code" is the companion repository run unmodified on gymnasium v4 environments (3 seeds).

| Quantity | Paper | Authors' code | This reproduction (mean ± SE, n) | Verdict |
|---|---|---|---|---|
| Final reward Hopper PPO (its 468-487) | 2450 (1964-3024) | 2590 ± 312 (3) | 2404 ± 136 (5) | MATCH |
| Final reward Hopper TRPO | 2408 (1538-2982) | 2255 ± 46 (3) | 2152 ± 308 (5) | MATCH |
| Final reward Walker2d PPO | 2396 (1526-3661) | 2621 ± 369 (3) | 3344 ± 170 (5) | MATCH by spread only (+40 %) |
| Final reward Walker2d TRPO | 2892 (2293-3315) | 3578 ± 295 (3) | 3315 ± 140 (5) | MATCH |
| Gradient cosine to true, n = 2000, it 0 (4 arms) | 0.47-0.80 | 0.73-0.88 (ref 1e5) | 0.76-0.86 (5) | 3 of 4 MATCH/CLOSE; Hopper PPO DIVERGENT (+62 %) |
| Gradient cosine to true, n = 2000, it 150-450 (8 cells) | 0.015-0.18 | - | 0.06-0.32 (5) | quality is poor after it 0, as claimed; Hopper values higher than the paper's |
| P1: cosine falls from it 0 to it 450 | yes | - | +0.69 to +0.72 drop, 5/5 seeds in every arm | HOLDS 4/4 |
| P2: held-out Returns-MRE > GAE-MRE at it 450 | yes | - | 5/5 seeds in every arm | HOLDS Walker2d, PARTIAL Hopper |
| P3: learned baseline beats no baseline (pairwise, 1e4) | yes | - | Walker2d +0.058 ± 0.006, Hopper +0.098 ± 0.041 | HOLDS Walker2d, PARTIAL Hopper |
| E5: PPO vs no-baseline PPO reward ratio | "almost an order of magnitude" | - | literal no-baseline arm: 16.8x Hopper, 10.7x Walker2d (5) | HOLDS |
| E5 with the pre-declared arm (official per-minibatch advantage centring kept) | same | - | 1.9x Hopper, 3.9x Walker2d (5) | FAILS / PARTIAL |
| C9a: true landscape jagged at 2e3 pairs, smoother at 1e5 | yes | - | roughness falls 1.5-13.5x in 24/24 cells (seed 0) | HOLDS |
| C9b: surrogate tracks true reward early, less later | yes | - | TRPO yes; PPO's clipped surface does not agree even at it 0 | PARTIAL |
| C9c: PPO's many-sample step "consistently" lowers true reward | yes | - | 0 of 6 cells decrease significantly (1 seed, 1e5 pairs) | FAILS |

Overall, the pre-declared family gives 8 HOLDS, 6 PARTIAL and 0 FAILS. With n = 5 a PARTIAL is a low-power
non-rejection: 80 % power needs a standardized effect of at least 1.7. The per-cell comparison gives E1 74 MATCH /
32 CLOSE / 54 DIVERGENT of 160 cells, E2 7 MATCH / 17 DIVERGENT, and E3 1 MATCH / 11 CLOSE / 24 DIVERGENT.
`results/headline.md` and `results/tables.txt` hold every cell.

Caveats:
- Value error (C6): the paper's "about 50 % off" Returns-MRE reproduces (median 0.29-0.42). It is a heavy-tail
  artifact, though: 1.4-1.9 % of states with |G| near 0 supply about half of the mean. The median relative error is
  0.03-0.10.
- Value error vs the true value (C7): the paper's claim holds. A same-size V* fit on 1e6 pairs does no better than the
  agent's V. A V* that also sees the episode step cuts Returns-MSE to 4-36 % (Hopper) and 22-29 % (Walker2d) of the
  agent's. The gap is time-limit aliasing in the value input, not return noise. See `results/vstar_time_*.json`.
- E3 (Fig 4) at iterations 300/450 diverges strongly. The paper's Fig 4 is inconsistent with its own Fig 9 at those
  iterations.
- Our two gradient instruments agree with each other (SNR-model residual -0.005 ± 0.005). The paper's own curves do
  not satisfy the same identity (residual -0.034 to -0.078).
- The gradient cosine at n = 1e6 was not measured. The C3 "large n recovers quality" claim was extrapolated from n
  up to 1e5 and is not tested here.

## What differs from the paper / known gaps
- **Environments are proxies.** We use brax 0.14.2 MJX `hopper` / `walker2d` (MuJoCo 3.15 physics). The paper used
  mujoco-py Hopper-v2 / Walker2d-v2. Brax's XML uses Euler instead of RK4, 4 solver iterations, its own contact
  settings and a reduced collision set. Observation, reward and termination definitions match gymnasium v4 exactly
  (`scripts/env_parity.py`). The one-step physics differs: the median next-observation error under random actions
  is 0.15 % for Hopper and 1.0 % for Walker2d. Absolute rewards are therefore not directly comparable.
- **Environment fix after the main runs.** brax's `State.x` lagged one substep behind gym's post-step position. The
  fix (`env_post_step_kinematics=True`, now the default) moves fixed-policy returns by +2 to +7 % and changes no
  verdict. The full-scale checkpoints were trained in the legacy env. `src.eval` follows each run's stored config.
- **Not run.** Humanoid (Figs 1, 2, 6, 7, 13, 14, 19, 20) was not run because of the budget. The paper's
  step-quality and learning-rate claims could not be reproduced because their material is absent from arXiv v4.
  Gradient estimates at n = 1e6 were not run.
- **Unresolved divergences.**
  - Hopper TRPO late-training to_true is +0.12 ± 0.05 above the official code run on our env at it 450. This is
    p 0.034 uncorrected and not significant after Holm.
  - Hopper TRPO learns about 25 % slower than the official code over its 40-70. This is not significant after Holm
    and is not located by the lockstep test, which covers the update only and not the data path.
  - The Hopper PPO it-0 cosines diverge. The authors' code agrees with us, so the cause is the paper or v2 physics,
    which we cannot test.
  - Walker2d Returns-MRE is above the paper's value and equal to the authors' code within agent spread.
- **Scaled settings.** The V* fit uses 1e6 pairs (paper: 5e6). The landscape true reward and the many-sample step
  use 1e5 pairs (paper: 1e6). Landscapes use one agent (seed 0) on a 7x7 grid. PPO is assumed for Fig 4, because
  the paper does not state the algorithm. Every value and its source is in `src/configs/sources.yaml`. Values tagged
  "official code" come from the companion repository, not from this paper.
- **TRPO code-form GAE-MRE** (squared form used in the official code) is not reproducible by the authors' own code
  either (5.6 / 11.4 against the paper's 0.03 / 0.05).
- **Results tables** in `results/` are the reducer output of the full-scale runs plus the post-review reruns. A
  second checking pass later revised several cell verdicts in both directions: E1 at it 0 like-for-like
  interpolation, E2 on a window median, Fig 8 smoothing. Those revisions are summarized above and are not
  re-tabulated. `scripts/recheck_report.py` regenerates them.

## Layout
- `src/model.py`: flax policy / value MLPs, Gaussian log-likelihood and KL.
- `src/data.py`: brax MJX env wrapper, running normalizers, single-actor rollouts.
- `src/algorithm.py`: GAE, PPO and TRPO updates, and the instruments (gradient estimate, cosine / SNR statistics,
  value errors, V* fit, landscape grid).
- `src/train.py`: training (E0, E2, E5). Writes `runs/<exp>/<env>_<alg>/seed_<n>/metrics.json` and orbax
  checkpoints, and resumes automatically.
- `src/eval.py`: instrument runner (E1-E4) on saved checkpoints.
- `src/toys.py`: exactly computable toy problems that exercise the instrument code paths.
- `src/common.py`: GPU bootstrap, logging, PRNG key tree, git info.
- `src/configs/default.py`, `sources.yaml`: every setting, tagged with its source. `paper_figdata.json` holds the
  numbers extracted from the paper's vector figures.
- `scripts/toy_checks.py`: toys at full size.
- `scripts/small_scale.sh`, `scripts/small_scale_report.py`: a 10x-smaller run of the whole protocol, and its report.
- `scripts/train_all.sh`: full-scale training, 4 arms plus 2 no-baseline arms, 5 seeds each.
- `scripts/instruments_all.sh`: every instrument and diagnostic command behind `results/`.
- `scripts/landscape_rerun.sh`, `scripts/e3_uncentered.sh`: the landscape rerun with the whole-pair-budget true
  reward, and the un-centered no-baseline E3 arm. Both are called by `instruments_all.sh`.
- `scripts/report.py`, `scripts/headline.py`: full-scale tables, pre-declared tests and theory checks, and the
  compact headline table.
- `scripts/recheck_report.py`: before/after tables of the second checking pass.
- `scripts/aggregate_seeds.py`: collects per-seed results into `runs/seed_<n>/metrics.json`.
- `scripts/value_mre_variants.py`: E2 metric-construction check (median, complete episodes only).
- `scripts/vstar_time.py`: time-aware V* refit (the C7 test).
- `scripts/mc_true_value.py`: Monte-Carlo true value from cloned simulator states.
- `scripts/it0_norm_check.py`: E1 at it 0 under two normalizer states.
- `scripts/ref_reliability.py`: split-half reliability of the 1e6-pair reference gradient.
- `scripts/env_parity.py`: env definitions and one-step dynamics against recorded gymnasium v4 transitions.
- `scripts/env_fix_eval.py`: before/after of the env fix on fixed policies.
- `scripts/oracle_diff.py`: official PyTorch primitives against ours on identical batches. Also records
  `tests/fixtures/official/`.
- `scripts/lockstep_diff.py`: multi-iteration lockstep of our update against the official update.
- `scripts/official_ckpt_grad.py`: our gradient instrument on the official code's checkpoints.
- `scripts/compare_official.py`, `scripts/official_reference_report.py`: paper / authors' code / ours tables.
- `scripts/official/`: `setup.sh` fetches the authors' code and builds its venv (see Checked against).
  Harness scripts that run inside the official-code venv. They are `oracle_harness.py`,
  `official_ref_run.py`, `official_grad_ref.py`, `lockstep_replay.py`, `record_env.py` and
  `make_gradstats_fixture.py`.

## Run it
From this folder, with the root environment (`uv sync` at the repo root):

    uv run python scripts/toy_checks.py                    # toys, a few minutes on CPU
    sh scripts/small_scale.sh                              # whole protocol at 1/10 scale, ~1 h
    sh scripts/train_all.sh                                # 6 CPU processes, ~30-40 min each
    sh scripts/instruments_all.sh                          # all instruments, many CPU-hours (see below)
    uv run python scripts/report.py && uv run python scripts/headline.py
    uv run python -m src.train --env hopper --alg ppo --seeds 0 1 2 3 4 --iters 488 --out runs/e0   # one arm

Outputs go to `runs/`, which is git-ignored. Training runs on CPU, because single-environment rollouts are
latency-bound. Measured speeds were 238 µs/step on CPU and 3.8 ms/step on an RTX 3090. Original costs:
- E1: about 2.5-5 min per agent-checkpoint.
- E3 + V*: about 5 min per agent-checkpoint.
- Landscape true reward: 12-17 min per Hopper agent and 1.1-1.6 h per Walker2d agent on CPU.
- GPU use: about 2 GPU-hours in total on one RTX 3090, mostly for landscape true-reward batches.

Float32 throughout, and host-side reductions run in float64. XLA deterministic GPU ops are off because MJX scatter is
about 15x slower with them. CPU runs are deterministic per seed and batch layout.

## Tests
`uv run pytest` (about 3 min on CPU):
- `test_smoke.py`: one rollout, update and instrument pass. The normalizers freeze during instruments.
- `test_grads.py`: estimator correctness on exactly computable toys, the surrogate-landscape identity, the SNR cosine
  model, and the GAE / lambda-return identity.
- `test_instruments.py`: the landscape true reward uses every completed episode. The un-centered zero arm has no
  baseline. Also checks the per-minibatch PPO surface, the run-name parser and the E5 value target.
- `test_official_parity.py`: replays `tests/fixtures/official/*.npz`, which were recorded from the official code.
  Covers GAE, returns, MRE forms, log-likelihood, KL, surrogate and gradient, CG, Fisher-vector products,
  normalizers, env definitions and gradient statistics. No external data is needed.
- `test_oracle.py`: a live differential test against the official code (~35 s). It needs the official-code
  environment and is skipped when `$OFFICIAL_ROOT/.venv` is absent; `sh scripts/official/setup.sh` creates it.

## Checked against
The authors' companion code, MadryLab/implementation-matters at commit
`5ee6ecb12545365d9178135e65576adfc0d82f52`, run unmodified on CPU in its own venv. The pinned versions were Python
3.12, torch 2.5.1+cpu, gymnasium 1.0.0, mujoco 3.15.0 and numpy 2.5.4. `sh scripts/official/setup.sh` clones the
repository at that commit (unpatched) into `$OFFICIAL_ROOT/implementation-matters/` and builds that venv, with those
versions pinned, as `$OFFICIAL_ROOT/.venv/`. The default `OFFICIAL_ROOT` is `external/official` inside this folder.
The script has not been run end to end in this form; the pinned commit was checked to exist. Reference-run outputs (`ref/`, `ref_full/`, `grad_ref*/`, `env_ref/`) are written there too.
- Primitives agree with ours to float32 tolerance on identical batches (`results/oracle_diff.json`).
- A 10-iteration lockstep of the full training update, from init and from its 150 / 450, agrees to float-chaos level.
- Our gradient instrument agrees with the authors' on the authors' own checkpoints.
- Where we differ from the paper, the authors' code mostly reproduces our numbers, not the paper's. See
  `results/official_reference_tables.txt`.
