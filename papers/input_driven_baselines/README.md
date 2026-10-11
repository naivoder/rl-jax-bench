# Variance Reduction for Reinforcement Learning in Input-Driven Environments
Hongzi Mao, Shaileshh Bojja Venkatakrishnan, Malte Schwarzkopf, Mohammad Alizadeh, ICLR 2019.
Paper: https://arxiv.org/abs/1807.02264 (v3). Official code:
[hongzimao/input_driven_rl_example](https://github.com/hongzimao/input_driven_rl_example) (commit `6c9cc23`; load
balancing only: environment, heuristic agents, A2C trainers for the state and 10-value arms).

## What this is
In input-driven environments an exogenous input process (here, a job arrival stream) drives the state and the
reward. The paper argues that a state-only baseline cannot remove the variance that this input adds to the policy
gradient. It proposes input-dependent baselines that also condition on the input sequence. They come as one value
network per fixed input sequence ("multi-value", 10 networks) or as a meta-learned baseline (MAML). The paper reports
25-33 % better test reward than the state baseline on load balancing (Fig. 5). This folder reimplements in JAX the
App. A toy, the 10-server load-balancing simulator, the A2C learner and all four arms: `none`, `state`, `multi`
(10 value networks) and `maml`, plus two mechanism ablations. ABR, MuJoCo and the two-server toy were not run.

Labels used below and in the code:
- toys: the App. A one-dimensional toy, where the gradient is known in closed form.
- small scale: a 10x-smaller load-balancing proxy, 250 iterations.
- full scale: 2,500 iterations x 16 rollouts x 500 jobs (2e7 timesteps under the 16 x 500 reading of Fig. 5's x axis).
- tuned: a non-paper config chosen on validation inputs with the same 8-point grid for every arm (critic / meta lr
  1e-2, entropy decay over 1,000 iterations).
- long budget: paper config resumed to 10,000 iterations. This is the x = 0.5e7 point of Fig. 5 if the x axis counts
  500 timesteps per iteration (see "What differs").

## Results
Load balancing, 10 seeds per arm unless stated, test reward x 1e-4 on 100 fixed unseen input sequences, last-5
checkpoint mean. Paper values are digitized from Fig. 5 (`data/fig5_digitized.json`). The labels were fixed before
the runs: MATCH means within 5 % or inside the paper's shaded spread. CLOSE means the paper value is inside our 95 %
CI, or inside a range the paper states. DIVERGENT means anything else. "Authors' code" is the official trainer, run
unmodified in this setup (1 seed).

| Quantity | Paper | Authors' code | This reproduction (mean ± SE, n) | Verdict |
|---|---|---|---|---|
| Toy: unbiasedness and closed-form variance (App. A) | V1/V2 closed form | - | ratio within 0.4 %, all 8 cells unbiased (\|z\| < 3) | MATCH |
| Heuristic line (App. J: JSQ) | -54.0 | SPT -60.2, LeastWork -63.8 | `jsq_first` -52.1 ± 2.7 (test); -55.6 ± 1.2 (2,000 seqs) | MATCH under the first-index-tie reading only |
| State arm, 2,500 iterations | -60.4 (end of curve) | -119.7 | -112.8 ± 4.8 (10) | DIVERGENT from the paper; matches the authors' code |
| 10-value arm, 2,500 iterations | -42.0 | -97.3 | -98.8 ± 6.7 (10) | DIVERGENT from the paper; matches the authors' code |
| MAML arm, 2,500 iterations | -39.6 | no code | -104.8 ± 5.5 (10) | DIVERGENT |
| State arm, x = 0.5e7 (long budget) | -69.4 | not run | -63.7 ± 3.2 (5) | CLOSE |
| 10-value arm, x = 0.5e7 | -50.2 | not run | -48.4 ± 1.9 (5); -47.6 ± 1.9 with the authors' critic schedule | MATCH / CLOSE (+5.1 %, boundary) |
| MAML arm, x = 0.5e7 | -50.4 | no code | -57.7 ± 1.0 (3) | DIVERGENT |
| Endpoint x = 2e7 (40,000 iterations) | -39.6 / -42.0 / -60.4 | not run | not run | not tested |
| 10-value gain over state, paper config, 2,500 its | 25-33 % | - | +12.4 % (Holm p 0.20) | DIVERGENT (n.s.) |
| 10-value gain over state, paper config, long budget | 25-33 % | - | +24.1 % [+5.5, +42.7], p 0.023 (5); +25.3 % with the authors' schedule | CLOSE (exploratory) |
| 10-value gain, tuned config, fresh seeds 10-19 | +30.4 % (digitized) | - | +31.5 % [+22.8, +40.2], Holm p 3.8e-5, 10/10 | MATCH (non-paper config) |
| MAML gain over state | 25-33 % | - | +7.1 % paper config; +14.6 % tuned (Holm p 0.057); +11.0 % long budget (n 3) | DIVERGENT |
| MAML is the best arm | yes | - | MAML - 10-value -7.3, p 0.14 (3) at long budget | DIVERGENT |
| Input-dependent arm beats the heuristic | yes | - | 10-value vs `jsq_first` +3.8, p 0.11 (5) at x = 0.5e7; paper's own gap there is also +3.8 | DIVERGENT (n.s.); 2e7 not run |
| State arm below the heuristic | yes | - | -11.6, p 0.02 | MATCH (qualitative) |
| Variance mechanism: Var(PG) vs fitted state critic | lower | - | own-sequence net 0.19x (paper config), 0.10x (tuned), 10/10 policies; mismatched net 1.05x | MATCH (direction) |
| Ablations attribute the gain to the baseline | implied -44 % / -52 % | - | every ablation t-interval contains 0 | DIVERGENT |

Caveats:
- **The x-axis reading decides most labels.** Fig. 5's x axis is in timesteps, but the paper does not say how many
  timesteps one iteration counts. Under 16 x 500 = 8,000 per iteration, 2e7 is 2,500 iterations. There every
  absolute row is DIVERGENT, and the authors' own trainer lands where ours does. Under 500 per iteration, 0.5e7 is
  10,000 iterations, the end of App. K's entropy decay. There state is CLOSE and 10-value MATCH or CLOSE. Two facts
  argue against the second reading. The authors' code decays entropy by 1e-3 per iteration and reaches its floor
  near 1,000 iterations. OpenAI A2C, which the paper says it extends, counts timesteps over all environments. Every
  x = 0.5e7 verdict is therefore conditional and exploratory: seeds 0-2 at 10k were seen before the plan was
  written, and no fresh-seed confirmation was run.
- The tuned-config 10-value gain is confirmed on pre-declared fresh seeds. It is a non-paper config, and the gain is
  window-sensitive (+28.8 % over the last 10 % of training).
- The arms differ in more than the baseline. They differ in how many distinct input sequences one update averages
  over (16 / 4 / 1 for state / multi / maml), in batch layout and in critic capacity. A return difference is
  therefore not attributable to the baseline alone, and the ablations do not isolate it.
- The heuristic verdict depends on the tie rule. JSQ with ties to the faster server gives -41.4, and the authors'
  SPT agent gives -59.6. Both are DIVERGENT.
- Fig. 5's shading (std over episodes or over seeds) is unresolved. The authors' code says episodes; the band widths
  suggest otherwise.

## What differs from the paper / known gaps
- **MAML is not App. K's MAML.** The arm run is first-order MAML: inner SGD 1e-2 with clip 10 on a mean loss, Adam
  outer, k = 16. App. K's literal summed inner loss at 1e-4 diverges by curvature at every reward scale
  (`src/configs/maml_literal.yaml`; NaN at iteration 12). App. K's meta-baseline is therefore unreproduced. The
  authors released no MAML code, and no selection of the inner loop was run.
- **Multi-critic schedule.** By default (`mv_schedule: groups`) we train 4 nets x 4 rollouts with one Adam step per
  net. The authors draw a net index per agent with replacement and take one Adam step per agent
  (`mv_schedule: official`, matches their update to 9e-6). This choice changes returns little: -1.3 ± 4.4 at 2,500
  iterations (n.s.) and +0.74 at 10k (p 0.034).
- **Observation tie rule.** The learned runs use `obs_tie: strict`. It differs from the authors' observation on 1.2 %
  of steps, where an arrival coincides with a completion. Its effect on learned policies was not measured.
  `obs_tie: official` matches the authors' environment.
- **Assumed values.** The paper gives no seed count, Adam epsilon, evaluation cadence, multi-value sampling or reward
  scale. Every value and its source is in `src/configs/sources.yaml`.
- **Not run.** ABR, MuJoCo, the two-server toy (config only), the x = 2e7 endpoint at 40,000 iterations, the
  authors' trainer past 2,500 iterations and at its own defaults for 5k+ iterations, the state arm with the authors'
  non-causal `wall_time/final_time` critic feature, and ablations at 10k iterations.
- **README-config discrepancy resolved.** Our 10-value arm first looked 1.6-1.95x worse than the authors' published
  README curve. Against the authors' unmodified trainer at 3 seeds it is 1.02x at 500 and 1,000 iterations, and
  0.84-0.89x with their schedule. The authors' published curve beats all three of their own seeds. At that config the
  authors' 10-value arm does not beat their state arm either (+0.7 at 1,000 iterations).
- **Results files predate later revisions.** `results/full_report.json` was produced before the recheck, and its
  labels are pre-recheck where they differ from the table above. `results/recheck_report.json` was produced before
  the README-config differential, which moved the 10-value x = 0.5e7 label from MATCH to CLOSE under the authors'
  schedule. The table above gives the final verdicts.

## Layout
- `src/data.py`: exogenous input processes (job streams, toy inputs) as pure functions of a PRNG key, and the key tree.
- `src/envs.py`: load-balancing simulator (closed-form FIFO recursion with exact integer times), heuristics, and an
  independent event-driven numpy reference simulator.
- `src/model.py`: flax MLP actor and critic (Glorot init, zero bias).
- `src/algorithm.py`: returns, A2C actor loss, critic loss, inner SGD adaptation, cross-fitted MAML, per-rollout PG
  variance.
- `src/train.py`: jitted train step, evaluation, orbax checkpointing with bit-identical resume.
- `src/eval.py`: heuristic table, matched-policy gradient-variance probe, toy driver (`--toys`).
- `src/toys.py`: App. A toy: closed forms, exact enumeration, negative controls, learned baselines, learning.
- `src/common.py`: logging re-exports, git provenance, NaN guard.
- `src/configs/`: `load_balance.yaml` (paper App. J/K), `small.yaml`, `tuned.yaml`, `maml_literal.yaml`,
  `two_server.yaml`, and `sources.yaml` (provenance of every value).
- `data/fig5_digitized.json`: Fig. 5 (left) curves extracted from the PDF's vector graphics.
- `scripts/small_scale.sh`: small-scale run of all arms, heuristics and probe.
- `scripts/train_all.sh`: full-scale training (4 arms x 10 seeds, 2 ablations x 5 seeds), probes, overlap check.
- `scripts/tune.sh`: equal-budget tuning grid, then the tuned config on seeds 0-9 and its ablations.
- `scripts/confirm_runs.sh`: tuned config on fresh seeds 10-19, observation sensitivity, literal MAML inner loss.
- `scripts/probe_maml_k16.sh`: MAML variance probes at the training k = 16.
- `scripts/long_budget.sh`: resume paper-config runs to 10,000 iterations, the authors' critic schedule, recheck report.
- `scripts/report.py`: full-scale aggregation (paired tests, Holm, bootstrap CIs, ablations, probes, tuning).
- `scripts/summarize.py`: per-experiment summary and per-seed merge.
- `scripts/recheck_report.py`: long-budget and critic-schedule aggregation.
- `scripts/heuristics_recheck.py`: heuristic table under both tie rules, optionally against the authors' agents.
- `scripts/bench.py`: throughput and FLOPs per arm.
- `scripts/check_overlap.py`: hashes every training sequence and checks that none equals a test sequence.
- `scripts/review_checks.py`: heuristic numbers quoted in review, on CPU.
- `scripts/review_probes.py`: determinism, resume, causality, unbiasedness and budget probes.
- `scripts/readme_config_run.py`: our env and learner at the authors' README configuration.
- `scripts/official_env_diff.py`, `scripts/official_grad_diff.py`, `scripts/heuristics_cross.py`,
  `scripts/critic_update_diff.py`: our side of the differential tests against the authors' code. They read the
  `.npz` dumps written by the scripts below.
- `scripts/official/`: `setup.sh` fetches the authors' code and builds its venv (see Checked against).
  Scripts that run in the authors' venv. `run_official.py` runs an unmodified official script
  under TF 2.13 `tf.compat.v1` (labeled shim: TF 1.2.1 / Python 3.6 could not be installed). `official_train.sh`
  launches an official trainer. `off_env_dump.py` and `off_grad_dump.py` dump the authors' environment and
  actor/critic losses and gradients. `critic_update_official.py` replays the authors' multi-critic update.

## Run it
From this folder, with the root environment (`uv sync` at the repo root):

    uv run python -m src.eval --toys --out runs/toys          # App. A toy, ~20 s on GPU
    sh scripts/small_scale.sh                                 # 10x-smaller proxy, all arms, 3 seeds
    sh scripts/train_all.sh                                   # full scale, 55 runs + probes
    sh scripts/tune.sh && sh scripts/confirm_runs.sh          # tuning grid, tuned config, fresh-seed confirmation
    sh scripts/long_budget.sh                                 # 10,000-iteration recheck
    uv run python scripts/report.py                           # runs/full/report.json
    uv run python -m src.train --arm multi --seeds 0,1,2 --out runs/full --set key=value   # one arm, any override

Outputs go to `runs/<exp>/<arm>/seed_<n>/metrics.json` (git-ignored), with orbax checkpoints in `ckpt/`. Rerunning a
command resumes from the latest checkpoint and reproduces the uninterrupted run, because every key is
`fold_in(branch, iteration)`. A resume refuses a checkpoint written under a different config. One full-scale run
(2,500 iterations) takes about 25 ms per iteration on an RTX 3090, about 1 min. The whole study used about 6.7 GPU-hours.
Networks, returns and losses are float32. The simulator keeps event times and job sizes in int32, and service times
are exact integer floors.

## Tests
`uv run pytest` (about 80 s on CPU):
- `test_smoke.py`: env shapes, exact input replay, one train step per arm, the multi-value update touching only the
  sampled nets, the shuffled ablation head, baselines from the pre-update critic, the authors' critic schedule.
- `test_grads.py`: `check_grads` on the actor and value losses, the second-order MAML meta-gradient, the
  cross-fitted baseline being independent of its own half's returns, discounted returns, per-rollout gradients.
- `test_env_reference.py`: the closed-form simulator equals the independent event-driven numpy simulator.
- `test_review_fixes.py`: config-guarded resume, metrics flushed at checkpoints, observation layouts, heuristic rules,
  the start-up hash check, the k = 16 probe.
- `test_official_env.py`: replays identical job sequences and actions through the authors' environment and ours.
  It needs the authors' clone and is skipped when it is absent; `sh scripts/official/setup.sh` fetches it.
  `N_EPS=50 python -m tests.test_official_env` writes the full measurement.

## Checked against
The authors' code, hongzimao/input_driven_rl_example at commit `6c9cc23174651e3890721a3f6420ca98a90eb99f`, run
unmodified in its own venv: Python 3.8, tensorflow-cpu 2.13.1, numpy 1.24.3. `sh scripts/official/setup.sh` clones
the repository at that commit (unpatched) into `$OFFICIAL_ROOT/input_driven_rl_example/` and builds that venv, with
those versions pinned, as `$OFFICIAL_ROOT/.venv/`. The default `OFFICIAL_ROOT` is `external/official` inside this
folder. The script has not been run end to end in this form; the pinned commit was checked to exist.
- Simulator: rewards identical on 99,999 of 100,000 replayed steps and done flags identical, once the authors'
  float64 `int(size/rate)` is replaced by the exact floor (`results/official_env_diff.json`).
- The authors' SPT agent and our `spt_official` give identical returns in 50 of 50 episodes. Heuristic levels on our
  test inputs agree (`results/heuristics_cross.json`, `results/heuristics_recheck.json`).
- Actor and critic gradients agree to 2.3e-5 relative on identical batches (`results/official_grad_diff.json`).
- The multi-critic update with `mv_schedule: official` matches the authors' per-agent update to 9e-6
  (`results/critic_update_diff.json`).
- Learner end to end, paper hyperparameters, 2,500 iterations: the authors' trainer gives state -119.7 and 10-value
  -97.3 (seed 42, `results/official_reference.json`). Ours gives -112.8 ± 4.8 and -98.8 ± 6.7. At the authors'
  README config, see `results/readme_config_report.json`.
- The authors' code has no MAML, ABR, MuJoCo or two-server code, so those parts have no oracle.
