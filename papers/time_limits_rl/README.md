# Time Limits in Reinforcement Learning
Fabio Pardo, Arash Tavakoli, Vitaly Levdik, Petar Kormushev, ICML 2018. Paper: https://arxiv.org/abs/1712.00378 (v4).
Official code: none released for this paper. The first author's later library
[Tonic](https://github.com/fabiopardo/tonic) (commit `0e20c89`) and
[OpenAI Baselines](https://github.com/openai/baselines) at the paper's date (commit `9fa8e1b`, 2018-01-25) were used
as reference oracles.

## What this is
The paper separates two kinds of time-limited training. If the task itself is time-limited, the remaining time is
part of the state, and the agent should see it (**time-awareness, TA**). If the time limit only resets training and
the task is time-unlimited, the agent should bootstrap from V(s') at a time-limit truncation and cut its return only
at real terminations (**partial-episode bootstrapping, PEB**). This folder reimplements both mechanisms in JAX,
the paper's toy tasks (re-implemented from the text) and its PPO experiments. Every comparison runs the arms
in one vmapped process with shared PRNG keys and identical initial parameters, so arms differ only in the
mechanism.

Labels used in code and tables: E1 Last Moment, E2 Two-Goal Gridworld (Fig 1), E3 Queue of Cars (Fig 2), E4
InvertedPendulum TA-PPO vs PPO (Fig 3), E5 critic surfaces (Fig 4), E7 Hopper PEB-PPO (Fig 6), E9 Difficult Gridworld
with Buffer-Q (Fig 7). P1-P8 are the pre-declared paired tests (Holm, alpha 0.05). "ppo2-2018" and "ppo1-2018" are
our two readings of the paper's "OpenAI Baselines PPO": Baselines ppo2 or ppo1 as of January 2018 (raw actions, the
2018 VecNormalize, the time input normalized with the observation).

## Results
Judgment rule fixed before the runs: MATCH within 5 %, CLOSE within 20 %, DIVERGENT otherwise. Paper numbers were
read from the PDF's vector paths (`scripts/review/figure_vector_readoff.py`). These are the final verdicts, after a
second checking pass (the "recheck") that corrected three configuration choices of the first PPO runs.

| Quantity | Paper | This reproduction (mean ± SE, n) | Verdict |
|---|---|---|---|
| E1 TA learner value at T, every (gamma, T) | gamma^(T-1) | exact, 10 cases x 10 seeds | MATCH |
| E2 standard-agent values next to goals | 49 / 19 | 49 / 19 (exact fixed point) | MATCH |
| E2 TA / PEB beat standard (P2 / P1) | qualitative | +1.43 every seed / +7.22 ± 0.17 (10) | SUPPORTED |
| E3 TA-PPO - PPO, exact value (P3) | qualitative | +0.1268 ± 0.0018 (10/10) | SUPPORTED |
| E3 Fig 2 TA heat map, cells within ±0.1 | figure | 9/19 cells | DIVERGENT |
| E4 IP levels (4 arms, gym-v4 port) | 922-1000 | within 1.5 % under ppo1-2018; TA gamma=1 919.1 (-5.6 %) under ppo2-2018 | MATCH (TA gamma=1 CLOSE on the ppo2 primary) |
| E4 IP TA - PPO, gamma = 1 (P4) | +51 | ppo1-2018 +54.4 ± 12.0 (30), fresh seeds +50.8 ± 12.9 (30); ppo2-2018 +7.0 ± 23.6 (30) | implementation-dependent: MATCH under ppo1, not reproduced under ppo2 |
| E4 IP TA - PPO, gamma = 0.99 (P5) | +38.5 | ppo2-2018 +34.0 ± 13.8 (p 0.02); ppo1-2018 +22.3 ± 5.1 (30) | MATCH in direction (ppo2 fresh-seed confirmation failed, p 0.054) |
| E5 Fig 4, gamma = 1 panel: TA critic learns first | yes | TA first 10/10 | MATCH |
| E5 Fig 4, gamma = 0.99 panel: TA critic learns first | yes | depends on the PPO reading (reversed under ppo1-2018) | DIVERGENT |
| E7 Hopper PEB - PPO(T=300) (P6), Brax proxy | +43.3 | +33.4 ± 4.9 at ~1e6 steps (10), ppo2-2018 | PROXY: direction and significance hold, size 77 %; DIVERGENT on size |
| Hopper TA - PPO, gamma = 1, Brax proxy | positive | +2330 ± 312 (10/10) | PROXY: direction holds; time-unaware PPO level collapses (46.6 vs 811) |
| E9 PEB level, all buffers | -59.29 | -60.15 ± 0.01 (30 each) | MATCH |
| E9 PEB - no-PEB at B = 1e6 / 100 (P7 / P8) | +140.6 / +125.9 | +139.83 ± 0.02 / +133.3 ± 1.0 (30) | MATCH / CLOSE |
| E9 no-PEB B = 60 / 100 | -94.7 / -185.2 | -112.3 ± 9.3 / -193.5 ± 1.0 (30) | CLOSE (bimodal) / MATCH |

In short, both mechanisms are exactly right where they can be computed (toys P1, P2, P3, P7 and P8 supported, Holm
≤ 0.0078). The InvertedPendulum time-awareness gain depends on the PPO implementation, and the official Baselines code
behaves the same way: Baselines ppo1 @9fa8e1b gives +26.7 ± 14.7 at gamma = 1, and Baselines ppo2 @9fa8e1b gives no
gain (-38.9 ± 16.8 at 60 seeds). Hopper was only run on a Brax proxy.

Caveats:
- **First PPO runs were wrong in configuration, not in code.** The first full-scale E4 run gave TA - PPO = +131.9 at
  gamma = 1 (2.6x the paper). That came from Tonic's x3 action rescaler, the post-2018 VecNormalize, and the time
  input left raw after normalization. With those reverted the gap falls by 124.9 ± 33.0 (p 7e-4). The flags default
  to the original behavior. `results/full_report.json` predates the recheck: use `results/recheck_report.json` for
  E4, E5 and E7.
- The gamma = 0.99 verdict leans on a ppo1 reading that was revised after its first result (a TF1-numerics switch,
  `ppo1_numerics`, justified by the official code alone). As first declared it gave +12.3, p 0.069.
- Ours (ppo2-2018) and the official ppo2 differ at the gamma = 1 gap level (+9.1 vs -38.9, Welch p 0.038). This is
  unexplained, but both are below the paper.
- E9: the paper's -59.29 plateau implies a goal-entering reward of 0 and a fixed tie order. We did not adopt that
  convention, so our -60.15 matches the level but not the mechanism behind it.
- The paper's "gap larger at gamma = 1 than at 0.99" is not shown on IP, either by us or by the paper's own panels
  (+51 vs +38.5).

## What differs from the paper / known gaps
- **Environments.** InvertedPendulum runs on an MJX port of gym InvertedPendulum-v4 that uses gym's own model file
  (exact to 1e-6 before termination). The paper used -v1 under mujoco-py 0.5. Hopper runs on Brax `hopper` as a
  PROXY. An exact Hopper-v4 port exists (`src/data.py`), but it ran at 1.0e3 env-steps/s on CPU, too slow for the
  budget. Absolute Hopper returns are not comparable.
- **Not run.** Walker2d and the other Fig 3 / Fig 6 tasks; the 3e6-step E7 budget under ppo2-2018; ppo1 on Hopper;
  a decomposition of the Hopper PPO collapse.
- **Not tested.** The E5 critic dynamics were not compared against the official Baselines code (its runs validate
  returns only). Mujoco-v1 physics and any unreleased Baselines changes by the authors could not be tested. Where
  the paper's code inserts the time input is an assumption.
- **Small changes.** E4 uses the mean training return over the last 10 % of the env steps actually run
  (`src/eval.py::final_window_return`). E7 is the discounted (0.99) evaluation return over a 3000-step horizon, which
  stands in for the paper's 1e6. Every setting and its source is in `src/configs/sources.yaml`.
- **Baselines oracle harness not shipped.** The ppo1 / ppo2 @9fa8e1b reference runs used a TF1 / py3.7 venv and a
  harness that lived with the run outputs, and neither is in this folder. `scripts/recheck/analyze_recheck.py` reads
  their outputs from `runs/recheck/official-oracle/` when present.

## Layout
- `src/data.py`: environments: Last Moment, Two-Goal and Difficult Gridworld (tables), Queue of Cars, Brax IP /
  Hopper proxies, MJX ports of gym IP-v4 / Hopper-v4 (`src/assets/gym_v4/`), runner-side time limit and time feature.
- `src/algorithm.py`: `gae_peb` (GAE with separate termination and reset flags), tabular Q-learning
  (standard / TA / PEB), REINFORCE on Last Moment, and Buffer-Q.
- `src/exact.py`: exact DP and fixed points, used as ground truth for the toys.
- `src/model.py`: actor-critic with a paired-init time input.
- `src/ppo8/`: vendored PPO losses, optimizer, networks and normalizers from `papers/ppo_implementation_details`
  (code SHA-pinned by `tests/test_vendor.py`).
- `src/train.py`: the PPO trainer (`lax.scan` over updates, `vmap` over arm x seed lanes, NaN guard, orbax resume)
  and the tabular drivers.
- `src/eval.py`: the per-experiment metrics and paired statistics.
- `src/timelimits.py`: float64 numpy reference GAE plus aliasing and boundary statistics.
- `src/configs/`: one YAML per experiment plus `sources.yaml`.
- `scripts/toys/run_toys.py`: E1/E2 learners against exact values. `make_tonic_fixtures.py` records the Tonic fixtures.
- `scripts/small/run_small.sh`, `report.py`, `run_e9.py`: small-scale pass, its report, and the E9 driver.
- `scripts/full/run_full.sh`, `report.py`: full-scale runs and the pre-declared Holm family.
- `scripts/review/run_review.sh`, `analyze_review.py`, `window_diff.py`: E5 dynamics to 1e6, the ppo1 readings, the
  Hopper TA proxy, and the E4 window correction.
- `scripts/review/figure_vector_readoff.py`: extracts the paper's figure values from the PDF (needs `pymupdf`,
  `uv sync --extra pdf`; PDF at `$PAPER_PDF`, default `external/2018-time-limits-in-reinforcement-learning.pdf`).
- `scripts/recheck/run_recheck.sh`, `analyze_recheck.py`: re-runs under the paper-era PPO readings and their analysis.
- `scripts/review_checks/`: independent recomputation, lane-coupling and determinism checks.
- `scripts/exact_solvers/`: standalone numpy reference solvers (`grid.py` is used by `tests/test_grads.py`).
- `scripts/oracle/make_env_fixtures.py`, `ref_report.py`: gym-v4 env fixtures via Tonic, and the paper / Tonic / ours
  table.
- `scripts/official/`: harness scripts that run inside the oracle venvs: `run_tonic.py`, `env_trace.py`,
  `ref_run.py` and `launch_ref.sh`, plus pip freezes of the two venvs.
- `scripts/collect_seed_metrics.py`: per-seed index `runs/seed_<n>/metrics.json`.

## Run it
From this folder, with the root environment (`uv sync` at the repo root):

    uv run python scripts/toys/run_toys.py                 # E1/E2 vs exact values, CPU, minutes
    sh scripts/small/run_small.sh                          # all arms at small scale, ~15 CPU-min
    sh scripts/full/run_full.sh                            # full scale, ~7 CPU-h + a few GPU-min (E3)
    sh scripts/review/run_review.sh                        # ~3 CPU-h
    sh scripts/recheck/run_recheck.sh                      # ~30 CPU-h (paper-era PPO readings)
    uv run python -m src.train --config src/configs/e3_queue_of_cars.yaml --tag mytag [--set key=value]

Outputs go to `runs/<tag>/<exp>/<arm>/seed_<n>/metrics.json` (git-ignored). Results reproduce per (config, arm set,
seed set), because the lanes of one process share a batch. Original hardware was an RTX 3090 and a 16-core CPU.
Brax proxies and tabular scans were faster on CPU, and the whole reproduction used about 0.17 GPU-hours and about
45 CPU-hours. Learners use float32 with `jax_default_matmul_precision="highest"`, and exact solvers use float64.

## Tests
`uv run pytest` (CPU, see the report for runtime):
- `test_smoke.py`: one step of every env, learner and trainer path.
- `test_grads.py`: closed forms for the toys, `check_grads` on the PPO loss, and hand-computed GAE with PEB.
- `test_official_parity.py`: our `gae_peb` and runner against Tonic's `lambda_returns` and time-limit wrappers, and
  our MJX ports against gym-v4. Uses fixtures recorded from the official code in `tests/fixtures/`.
- `test_ablation_flags.py`, `test_review_fixes.py`, `test_recheck_fixes.py`: every configuration switch, including
  the paper-era raw-action interface, the reward scaler and the ppo1 numerics. Each defaults to the original
  behavior.
- `test_timelimits.py`: `src/timelimits.py` against `gae_peb` and hand-checked windows.
- `test_vendor.py`: SHA-256 pins of `src/ppo8/`.

No test needs external data. Regenerating fixtures requires the oracle venvs under `$OFFICIAL_ROOT`.

## Checked against
- Tonic @`0e20c894ee68278ab68322de61bb2c7204a11d5f`. Mechanics fixtures used Python 3.10, numpy 1.23.5 and gym 0.23.1
  (`.venv`). Env fixtures and reference runs used torch 1.13.1+cpu, gym 0.24.1 and mujoco 2.2.0 (`.venv-ref`, with
  freezes in `scripts/official/`). Point `OFFICIAL_ROOT` (default `external/official`) at a directory holding
  `tonic/`, `.venv/`, `.venv-ref/`. Reference runs are written to `$OFFICIAL_ROOT/ref_runs/`. Tonic shows no IP TA
  gain because it z-scores the time input. It is the only official reference for Hopper.
- OpenAI Baselines @`9fa8e1baf1d1f975b87b369a8082122eac812eb1`, run unmodified in a TF 1.15 venv on a numpy IP-v4
  port that is bit-identical to gym. With the corrected flags, our ppo1-2018 and ppo2-2018 readings agree with
  Baselines ppo1 / ppo2 arm by arm (Welch p ≥ 0.14 for ppo1). The one exception is the ppo2 gap-level residual
  noted above.
