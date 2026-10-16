# Streaming Deep Reinforcement Learning Finally Works
Mohamed Elsayed, Gautham Vasan, A. Rupam Mahmood, arXiv preprint 2024 (v2). Paper: https://arxiv.org/abs/2410.14606v2.
Official code: [mohmdelsayed/streaming-drl](https://github.com/mohmdelsayed/streaming-drl), branch `2024`, commit
`407dca7`. The repository's default branch belongs to a later 2026 revision of the paper, with a different optimizer
and a different SparseInit, and was not used.

## What this is
The paper claims that deep RL can learn from a stream: one transition at a time, with no replay buffer, no batches and
no target network. Its "stream-x" recipe has five parts. **ObGD** is an eligibility-trace update whose step size is
bounded so that it cannot overshoot. It also uses traces with lambda = 0.8, **SparseInit**, LayerNorm without
parameters, and online observation normalization and reward scaling. This folder reimplements stream Q(lambda),
stream AC(lambda) and stream TD(lambda) in JAX. It also reimplements the classic and DQN1 baselines and the Figure 7/8
ablation arms. All arms and seeds of one experiment run as lanes of one vmapped `lax.scan` program and differ only
through a per-lane config (`src/train.py`, `ARMS`).

Labels used in code and tables:
- Experiments: E3 is MinAtar Breakout (Fig. 4). E6/E7 are the Hopper component and trace ablations (Figs. 7, 8). E8
  is stream TD on ETTm2 (Fig. 10). E2 is a stream AC vs PPO comparison on the same simulator.
- Hypotheses: H1/H2 say stream Q beats classic Q on MinAtar. H3-H8 are the ablation orderings of Figs. 7/8 (no ObGD,
  no normalization, no LayerNorm, no SparseInit, classic AC, lambda = 0). H9 says stream TD beats classic TD on ETTm2.
- Baseline constructions the paper does not release, so we had to assume them:
  - B1: the classic baselines remove every stream-x component (no LayerNorm, default dense init, no normalization),
    with Adam on the trace.
  - B3: the "no ObGD" ablation arm is Adam on the same trace update, with LayerNorm, SparseInit and normalization
    kept.
  - B7: DQN1 uses the CleanRL `dqn.py` defaults.

## Results
Final verdicts, after a review pass and a later recheck. The judgment rule was fixed before the runs. MATCH means
within 5 % of the paper's value at the same step. CLOSE means within 20 %. DIVERGENT means anything else. Paper values
are vector-digitized from the PDF figures (`src/configs/paper_digitized.json`) and read at a grid point of the figure's
own bins: 125k steps for Fig. 4 and 50k for Fig. 7. "Authors' code" means their code, unmodified, run on our hardware.

| Quantity | Paper | Authors' code | This reproduction (mean ± SE, n seeds) | Verdict |
|---|---|---|---|---|
| Breakout stream Q(0.8), 125k bin at 1.0M | 5.058 | not run (~17 h/seed on CPU) | 5.139 ± 0.05 (30) | MATCH (+1.6 %) |
| ... at 0.875M / 0.75M | 3.319 / 2.344 | - | 3.373 ± 0.02 / 2.361 ± 0.01 (30) | MATCH (+1.6 / +0.7 %) |
| ... at 0.125M / 0.25M | 0.461 / 0.632 | 0.462 ± 0.005 / 0.624 ± 0.003 (3) | 0.459 ± 0.001 / 0.625 ± 0.002 (30) | MATCH |
| Breakout classic Q(0.8) (B1) at 1.0M | 4.015 | not released | 3.916 ± 0.04 (30) | MATCH (-2.5 %) |
| H1: stream Q > classic Q at 1.0M | +1.04 | - | +1.22, 30/30 seeds | HOLDS |
| Breakout DQN1 (B7) at 1.0M | 1.247 | not released | 1.02 ± 0.01 (6) | CLOSE (-18 %), secondary |
| Stream-Q overshoot rate, peak per 50k (0.2-0.25M) | not reported | 0.078 (3) | 0.070 (30) | same as authors' code |
| Hopper-v4 stream AC(0.8) at 3.5e5 (version-matched env) | 1033 | 1067 ± 158 (5) | 1011 ± 172 (5) | MATCH (-2.2 %) |
| ... at 1e5 / 2e5 | 419 / 736 | 409 ± 73 / 614 ± 125 (5) | 391 ± 43 / 608 ± 165 (5) | CLOSE (-7 / -17 %) |
| Hopper-v4 without obs/reward normalization at 3.5e5 | 116 | not released | 118 ± 13 (3) | MATCH (+1.7 %) |
| Hopper-v4 no-ObGD arm (B3), Adam 1e-7, at 2e5 | 18.6 (flat) | Adam 1e-7 in their loop: 42 / 53 at 1e5 (2) | 84.8 ± 4.0 (5) | DIVERGENT (shape) |
| Hopper-v4 frozen initial policy, (0, 25k] | ~18.6-20.3 | 18.5 ± 1.8 (8) | 19.0 ± 1.7 (8) | MATCH |
| ETTm2 stream TD(0.8), last-5000-step MSE | qualitative only | 71.59 ± 0.68 (3) | 71.98 ± 0.35 (30) | MATCH with authors' code |
| H9: stream TD beats classic TD (B1) at the end | yes | - | 18.2 better, 30/30 seeds, p = 5.7e-28 | HOLDS, conditional on B1 |

Overall verdict: **PARTIAL reproduction.** The headline result holds. Breakout stream Q is a MATCH at every grid
point checked, and real-Hopper stream AC is a MATCH at 3.5e5 steps. One DIVERGENT remains: the paper's flat "no ObGD"
curve. That arm's construction was never released.

Caveats:
- **The stream-over-classic gap depends on assumption B1.** We built classic arms that keep LayerNorm, SparseInit and
  normalization and swap only ObGD for Adam. On Breakout this arm reaches 5.26 ± 0.19 (n = 3), the same as stream Q
  (paired difference -0.06, p 0.84; at n = 3 the test can only exclude an ObGD effect larger than about 1.4). On
  ETTm2 the same swap leaves stream TD better by 5.09 ± 0.36 instead of 18.2 (30/30 seeds). So most of the paper's
  stream-vs-classic gap comes from the components the classic baseline drops, not from ObGD.
- **The no-ObGD arm.** App. F.4 gives Adam 1e-7 for the ablation. Read that way, the ordering matches the paper:
  no-ObGD learns far less than stream AC. The paper's curve stays flat at the frozen-policy level, though, while ours
  learns slowly (84.8 at 2e5). The authors' own training loop with our construction behaves like ours. The slow
  learning therefore comes from the construction, not from our JAX code. At step sizes tuned for return (1e-5,
  selected on separate seeds), Adam learns well: 1268 ± 151 at 3.5e5 (3). The paper's flat curve stays unexplained.
- **Fig. 4 digitization.** Early versions of our digitizer calibrated the axes on tick-label centers, which shifted
  every paper value up by a constant per panel (Breakout +0.156). The axes are now fitted to the tick marks
  (`tests/test_digitize_calib.py`). Seven Breakout cells moved from CLOSE or DIVERGENT to MATCH or CLOSE as a result.
- **Small n.** The real-Hopper comparisons use 3-5 seeds. At 3.5e5 the smallest difference detectable at 80 % power
  is about 745, roughly 70 % of the mean. "No detectable gap" to the authors' code is therefore weak evidence.

## What differs from the paper / known gaps
- **Scale.** Breakout ran to 1M of the paper's 5M steps. The epsilon schedule is still that of the 5M run, so the
  result is a truncated view compared at the same step. Real Hopper ran to 3.5e5 of 2M steps. The paper's 20M-step
  MuJoCo/DMC and 200M-frame Atari results (Figs. 2, 5, 6, 11, 12, 14) were not attempted. Asterix, Seaquest and
  SpaceInvaders were not run. Stream SARSA (Fig. 13) is not implemented. Adaptive ObGD and backtracking (App. B,
  Alg. 2) are unit-tested only.
- **Environments.** MinAtar uses gymnax 1.0.0 with our sticky-action wrapper (p 0.1) and the time limit lifted. It is
  replay-exact against MinAtar 1.0.15 (0 observation / reward / done mismatches in 241 192 steps). Real Hopper-v4 is
  stepped in the authors' versions (gymnasium 0.29.1 + mujoco 2.3.7) through `scripts/official/hopper_env_server.py`.
  ETTm2 is the public CSV (sha256 pinned in `src/data.py`).
- **The Brax hopper results are struck.** The E6/E7 ablations and the E2 PPO comparison first ran on a Brax
  `positional` hopper. That simulator fails a passive-policy check: under zero torque it stands for 1000 steps, while
  Hopper-v4 falls after about 155 (`scripts/hopper_passive_gate.py`). Every Brax-proxy row is withdrawn, so H3-H8 and
  the no-LayerNorm, no-SparseInit and lambda = 0 ablations are **not tested on a valid environment**. `src/train.py`
  now refuses that backend unless `allow_failed_proxy_gate` is set, and only the original configs set it. The PPO
  arm used the code in `papers/ppo_implementation_details` and is not part of this folder.
- **Unreleased baselines.** Every classic, DQN1 and ablation verdict depends on B1, B3 and B7. DQN1 stays CLOSE.
  CleanRL's Adam epsilon (1e-8 vs 1e-4) does not explain the gap (difference -0.003), so the exploration fraction and
  target period of B7 are the open candidates.
- **Watkins cut.** The authors' code cuts the trace after the update, while the paper's Alg. 8 cuts before it. Both
  give the same result at 1M (5.32 ± 0.14 vs 5.14 ± 0.05, p 0.32).
- **Results files.** `results/small_report.json` predates the review. `results/full_report.json` was regenerated
  after the digitizer fix. Both still contain the Brax-proxy rows that were later struck, and the pre-recheck
  verdict fields. `results/review_report.json`, `results/no_obgd_report.json` and
  `results/frozen_policy_report.json` hold the later numbers. The verdicts above are the final ones.

## Layout
- `src/model.py`: Q / actor / critic networks with an optional parameter-free LayerNorm and SparseInit, dense, or
  torch-default init.
- `src/algorithm.py`: Welford statistics (float32 with Kahan compensation), observation normalization, reward
  scaling, ObGD, Adam on the trace, and Adaptive ObGD.
- `src/agents.py`: per-step stream Q, AC and TD agents.
- `src/data.py`: gymnax MinAtar with sticky actions, the Brax hopper, and ETTm2.
- `src/train.py`: the vmapped multi-arm driver, with orbax checkpoint/resume and NaN handling. Also holds the arm
  table and the PRNG key tree.
- `src/eval.py`: official 50k-step binning, figure-stride re-binning and grid reads, and ETTm2 metrics.
- `src/toys.py`: toy problems with exactly computable answers (ObGD on a linear toy, Welford / ScaleReward, and
  others).
- `src/configs/*.yaml`: one config per run: `e3_minatar`, `e6_hopper`, `e8_ett` for the small-scale run;
  `full_e3_breakout`, `full_e6_hopper` for full scale; `review_e3_cpu`, `review_e8_sens` for the review pass.
- `src/configs/sources.yaml`: every setting with its source.
- `src/configs/paper_digitized.json`: Figs. 4/7/8 digitized from the PDF.
- `scripts/toy_checks.py`: every toy check plus the differential and gradient tests. Writes `runs/toys/toys.json`.
- `scripts/small_scale.sh`, `scripts/small_report.py`: the small-scale run (E3 5e5 steps, Brax E6 2e5 steps, E8)
  and its report.
- `scripts/throughput_probe.sh`, `scripts/train_all.sh`, `scripts/report.py`: the full-scale throughput probe, run
  and report.
- `scripts/review_all.sh`, `scripts/review_e8.sh`, `scripts/review_report.py`: the review pass and the recheck runs.
- `scripts/ours_on_mujoco.py`, `scripts/ours_on_mujoco_vm.py`, `scripts/real_hopper_lanes.py`: our AC agent on real
  Hopper-v4. The first uses the root environment's gymnasium/mujoco. The other two step the authors' versions, and
  `real_hopper_lanes.py` runs several arms in one program.
- `scripts/no_obgd_report.py`, `scripts/frozen_policy_report.py`: recheck reports for the no-ObGD arm and the frozen
  policy.
- `scripts/hopper_passive.py`, `scripts/hopper_passive_mujoco.py`, `scripts/hopper_passive_gate.py`: the
  passive-policy simulator check. Its measurements are in `tests/fixtures/passive_gate/`.
- `scripts/digitize_figs.py`: figure digitizer. Needs `uv sync --extra pdf` and the PDF.
- `scripts/minatar_parity.py`, `scripts/ckpt_view.py`, `scripts/collect_seed_metrics.py`: MinAtar parity, a view of a
  checkpoint cut short by a kill, and per-seed end-points.
- `scripts/official/`: runs with the authors' code.
  - `setup.sh` fetches it.
  - `make_fixtures.py` and `make_closed_loop_fixtures.py` record the test fixtures.
  - `run_official_q.sh`, `run_official_q_eps.sh` and `run_official_ac.sh` produce reference curves, and
    `official_curves.py` bins them.
  - `run_official_td.py`, `td_same_init_official.py` and `td_same_init.py` are the ETTm2 checks.
  - `hopper_env_server.py` is the Hopper-v4 env server.
  - `minatar_rollouts.py` records MinAtar rollouts.
  - `official_frozen.py` and `run_official_trace_adam.py` are the recheck probes in the authors' loop.

## Run it
From this folder, with the root environment (`uv sync` at the repo root):

    uv run python scripts/toy_checks.py                 # toys + differential tests, a few minutes on CPU
    sh scripts/small_scale.sh                           # small-scale run (GPU for E3/E6)
    sh scripts/train_all.sh                             # full scale: E3 Breakout 1M x 30 seeds, E8, real Hopper
    sh scripts/official/setup.sh && sh scripts/review_all.sh   # review pass + recheck, CPU, many hours
    uv run python -m src.train --config src/configs/e3_minatar.yaml --tag mytag [--set key=value ...]

Outputs go to `runs/<tag>/<exp>_<env>/<arm>/seed_<n>/metrics.json` (git-ignored). Original cost was about 2.1
process-wall GPU-hours on one shared RTX 3090: about 0.8 for the full-scale E3/E6 programs and 1.4 for the
reproduction in total. The full Breakout program runs at about 1.3e3 steps/s per lane with 60 lanes. The review's
Breakout CPU program took 5.2 h for 9 lanes to 1M steps. Real-Hopper host loops take 45-100 min per program. Everything
is float32 with matmul precision `highest` (no TF32, which the 1e-5 differential tests need). Toy checks and gradient
checks run in float64.

## Tests
`uv run pytest` takes about 75 s on one CPU core:
- `test_smoke.py`: shapes, and one step of every agent and of the vmapped multi-arm program. Must finish in under
  30 s.
- `test_toys.py`: every check in `src/toys.py`, including the 5e6-sample float32 Welford precision test.
- `test_grads.py`: closed-form LayerNorm backward, the LN Q-network output and the AC policy objective, against
  autodiff.
- `test_official_parity.py`: replays `tests/fixtures/*.npz`, recorded from the authors' code. Covers the ObGD step,
  SparseInit statistics, network forwards, wrappers, full stream Q / AC update steps, 600-step closed loops on
  Breakout and Hopper, and the plotting metric. Tolerance is 1e-5, compared with the authors' own sensitivity to
  float32 rounding. No external data is needed.
- `test_driver.py`, `test_review_fixes.py`, `test_figure_strides.py`, `test_passive_gate.py`: driver features,
  regression tests for review fixes, figure-stride reading, and the passive-policy simulator check.
- `test_digitize_calib.py`: tick-mark calibration of the digitizer. It needs the paper PDF and pymupdf, so run
  `sh scripts/official/setup.sh` and `uv sync --extra pdf` first. Otherwise it is skipped, with a message that names
  them.

## Checked against
The authors' code: github.com/mohmdelsayed/streaming-drl, branch `2024`, commit
`407dca7a8b584c1c20bc649053557f66e270b1e6`, run unmodified on CPU. The venv uses Python 3.11, torch 2.3.0+cpu,
gymnasium 0.29.1, MinAtar 1.0.15, mujoco 2.3.7 and numpy 1.26.4. `sh scripts/official/setup.sh` recreates it under
`$OFFICIAL_ROOT`, by default `external/official`. Our code contains nothing copied from theirs.
- The fixtures in `tests/fixtures/` (open loop and 600-step closed loop) agree to float32 rounding. Parameters agree
  to 3e-8 (Q) and 8.4e-8 (actor), and all 63 Breakout episode returns are identical.
- A 60 000-step replay of our AC update on the authors' recorded Hopper stream stays within their own
  perturbation sensitivity. The run covers 453 episodes, and the TD-error correlation is 1.0000 in every 1k window.
  A 3 400-step replay through three time-limit truncations also matches.
- MinAtar Breakout replays exactly against MinAtar 1.0.15. Return distributions under a competent policy match
  (N = 5000, p 0.46 / 0.68).
- The reference runs at our budgets agree with ours: Breakout to 2.5e5, Hopper-v4 to 3.5e5, ETTm2 at 30 vs 3 seeds.
  Their outputs are summarized in `results/official_*.json`. Raw run directories are not shipped.
- The probe-level evidence from the recheck is not shipped: environment, measurement and learner-strength probes,
  long replays and paper-text extracts. It was one-off evidence for the verdicts above.
