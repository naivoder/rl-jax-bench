# Deep Reinforcement Learning Control for Radar Detection and Tracking in Congested Spectral Environments
Charles E. Thornton, Mark A. Kozy, R. Michael Buehrer, Anthony F. Martone, Kelly D. Sherbondy, IEEE Transactions on
Cognitive Communications and Networking (TCCN) 2020. Paper: https://arxiv.org/abs/2006.13173 (v3). Official code: none
released for this paper. CleanRL `dqn_jax.py`, pomdp-baselines and an independent TF 2.2 Keras learner were used as
oracles (see "Checked against").

## What this is
A pulsed radar picks a contiguous run of 20 MHz sub-bands (15 actions over a 100 MHz channel) for every pulse while
a communications interferer occupies some of the sub-bands. The paper claims that Deep Q-learning (DQN) beats policy
iteration (PI) and sense-and-avoid (SAA), and that a Double Deep Recurrent Q-Network (DDRQN: double-Q plus an LSTM
over 10-step traces) does better still, with lower variability. This folder reimplements the environment and the
exact optimal policy pi* (a positive control), plus the SAA, random, constant-action and PI baselines. It also
implements DQN, a double-Q MLP ablation (DDQN), a DDQN on 10-step traces and DDRQN. Every arm runs as one jitted,
vmapped program over lanes = (experiment, reward reading, seed).

Labels used in code and tables:
- **Experiments.** E1 is the Markov interferer with p = 0.4 (Fig 3 left) and E2 the same with p = 0.2 (Fig 3 right).
  E3 is the p schedule of Fig 4 (segments 15/15/15/15/10 CPIs), E4 the deterministic frequency-hopping sweep (FH,
  Table II), E6 the 2.45 GHz recording and E7 the SDRadar hardware. A CPI is 1000 pulses.
- **Reward readings.** `eq3` is Eq. 3 as printed (primary). `lin` is a linear reading identified from the
  learner-free levels of Figs 3/4 (secondary). Under `eq3`, values above about 0.71 at p = 0.4 cannot be reached, so
  some of the paper's plotted levels cannot be matched by construction.
- **Claims.** P1-P8 are the pre-declared paired tests (one-sided paired t, Holm over the 8 per reading, the seed is the
  unit). P1/P7 test DDRQN > PI at p = 0.4/0.2, P2/P8 DQN > PI, P3 DDRQN > SAA, P4 DQN > SAA, P5 DDRQN less variable
  than DQN, and P6 DDRQN >= DQN. H-a ("approaches pi*") holds iff the upper 95 % t bound of J* - J_conv is <= 0.02.
- **Metrics.** J is the mean per-pulse reward over the 70 evaluation CPIs. J_conv is the same over CPIs 11-70, and
  sigma_within is the within-seed SD of the per-CPI mean.
- **DDRQN variants.** T-A is a sliding-window LSTM, T-B uses hard_sigmoid gates, and T-C trains for the paper's 500
  training CPIs.
- **Stages.** "toys" are exact toy checks. "small" is a 10x-smaller proxy run, and "full" is the full protocol at the
  declared 100-CPI training fallback. "review" covers PI and learners at 500 CPIs, no act-time reset and an lr sweep.
  "recheck" covers plain SGD, the Keras loss scale and a per-trace update cadence.

## Results
"Full" setting: 100 training CPIs (paper: 500; a fallback declared before the run) plus 70 evaluation CPIs, Table I
hyperparameters, Adam at lr 1e-3, 10 seeds, values are mean ± SE. Paper values are read off Figs 3/4 (±0.01), which are
on the `lin` scale. MATCH means within 5 % (or within 0.01 + our 95 % half-width), CLOSE within 20 %, and DIVERGENT
anything else.

| Quantity | Paper | This reproduction (mean ± SE, n = 10) | Verdict |
|---|---|---|---|
| Fig 3L pi*, `lin` | 0.845 | 0.8335 ± 0.0002 (exact 0.8333) | MATCH (the level that identifies `lin`, so not evidence) |
| Fig 3L DQN, `lin` | 0.75 (dips to 0.50) | 0.8331 ± 0.0002 (DQN plays pi*) | CLOSE, ours above |
| Fig 3L DDRQN, `lin` | 0.835 | J_conv 0.7962 ± 0.0018 | MATCH on the window-matched re-grade, but P6 REVERSED |
| Fig 3L PI, `eq3`, 500 CPIs | 0.71 | 0.6908 | MATCH: the plotted "PI" curve is pi* scored under Eq. 3 |
| Fig 3R DDRQN / DQN, `lin` | 0.86 / 0.82 | 0.8137 ± 0.0026 / 0.8634 ± 0.0005 | CLOSE / CLOSE |
| Fig 4 PI segments 1-5, `lin`, 500 CPIs | .945/.87/.74/.62/.47 | .933/.867/.733/.599/.468 | MATCH x5 |
| Table II FH, SAA (BW / coll / missed / adapt) | 64.0 / 60 / 40 / 100 | 64.00 / 60.00 / 40.00 / 100.00 | MATCH, exact |
| Table II M.C. DQN, BW / missed | 60.0 / 48.9 | 60.60 ± 0.19 / 49.18 ± 0.29 | MATCH |
| Table II M.C. DDRQN, coll / adapt | 0.25 / 11.4 | 9.49 ± 0.39 / 58.43 ± 0.94 | DIVERGENT |
| Primary family `eq3` (P1-P8), 100 and 500 CPIs | all hold | 3 GO (every DQN claim) / 5 REVERSED (every DDRQN claim) | PARTIAL |
| Secondary family `lin`, 100 / 500 CPIs | all hold | 5 GO / 3 REVERSED; 4 GO / 4 REVERSED | PARTIAL |
| P6 DDRQN - DQN, E1 `eq3`, Adam lr 1e-3 | DDRQN ahead | -0.108 ± 0.004, 0/10 seeds | REVERSED |
| P6 DDRQN - DQN, E1 `eq3`, plain SGD (Eq. 9), fresh seed pool | DDRQN ahead | +0.0001 ± 0.0001, 4/10 | tie at pi* (not supported, not reversed) |
| H-a DDRQN, E1 `eq3`, plain SGD | approaches pi* | gap 0.0001 [-0.0000, +0.0002] | holds |
| E6 (2.45 GHz recording), E7 (hardware) | printed | - | not reproducible: data / hardware not released |

**Final verdict: PARTIAL.** Every DQN claim reproduces under every reading. The paper's DDRQN *advantage* (P5, P6)
is reproduced under no reading. With Adam at the Table I learning rate (our assumption: the paper does not name the
optimizer), DDRQN is clearly below DQN. TF 2.2 Keras on our streams shows the same deficit (-0.115 vs our -0.112, n =
3). Anything that cuts the effective step size removes most of the deficit:
- under Eq. 9's plain SGD, DDRQN ties DQN at pi* (P5, P6 not supported, P1/P3/P7 GO, H-a holds);
- lr 1e-4 shrinks the gap from -0.112 to -0.007;
- one update per stored trace shrinks it to -0.004.

DQN already sits at pi*, and pi* is memoryless here, so a tie is the ceiling for any learner in this environment.

Caveats:
- **E1 cannot separate reactive from non-reactive learners.** pi*(p = 0.4) is the constant action 00111 (0 %
  adaptations), so "DQN closes the SAA-to-pi* gap" means DQN learned to hold one action. The informative cells are
  E2 (pi* = SAA, 20 % adaptations) and the E3 segments.
- **PI.** With 100 training CPIs, both learners beat PI by 0.06-0.09. That was an artifact of the fallback: at the
  paper's 500 CPIs, PI is within 0.002-0.018 of pi* (exactly pi* on FH). DQN still beats PI by +0.002 to +0.017 (10/10
  seeds), which is statistically clean but negligible in practice. The Fig 3R (p = 0.2) PI gap reproduces only under a
  long-episode reading of the unspecified target model.
- **Ablation** (E1, paired, n = 10). Adding the LSTM on identical 32 x 10 traces lowers J_conv by 0.104 (`eq3`) /
  0.037 (`lin`) and raises variability, with 0/10 seeds the other way. Double-Q has no effect (-0.0003). The paper
  implies the opposite sign for the LSTM.
- **Variants** (`results/variants_tests.json`, `results/long_train_tests.json`). T-A recovers +0.042 of the `eq3`
  deficit, T-B nothing. With T-C (500 CPIs), DDRQN rises by +0.040 / +0.025 and reaches 0.822 under `lin` ("> 0.8"),
  but it stays below DQN. T-C is post hoc and descriptive.
- **Paper inconsistencies.** The FH 80 MHz bandwidth entries in Table II contradict the paper's own 0 % missed
  opportunities, which require 64.0 MHz. The M.C. SAA "21.2 % adaptations" contradicts its 22.58 % collisions (the
  exact SAA values are 20/20/40 %).
- **Reproducibility scope.** Results are bit-reproducible on the same device and backend only. CPU and GPU learner
  trajectories differ from CPI 0 (float32 reduction order) but stay inside the seed spread
  (`results/cpu_rerun_compare.json`). Every reported number comes from one RTX 3090 with deterministic XLA ops.
- **Files that predate a revision.** `results/full_tests.json` holds the full-scale grading as first run. Its PI
  comparisons, figure judgments and DDRQN attribution were revised later by `results/review_report.json`,
  `results/regrade_figs.json` and `results/recheck_report.json`. The verdicts above are the final ones.

## What differs from the paper / known gaps
- **Unstated details were assumed.** Each assumption is tagged `ASSUMED` in `src/configs/sources.yaml`. They cover the
  optimizer (Adam with Keras defaults), the update cadence (one step per pulse) and the target trajectory model (the
  paper says only "randomly defined"). They also cover the state encoding, the reading of "batch 32" for DDRQN (32
  traces of 10), zero-start aligned traces with an act-time LSTM reset, and PI's treatment of unseen state-action
  pairs. The reward reading is ambiguous, so both readings are graded, as separate families.
- **Training length.** The declared families were graded at 100 training CPIs. PI, the E2/E3 learners and the E1
  learners were rerun at 500 (review, T-C), but the SGD and per-trace-cadence readings were run at 100 CPIs, on E1/E2
  and `eq3` only. They were not run at 500 CPIs, under `lin` or on E3/E4.
- **Not run:** a "batch 32 = 32 transitions" reading for DDRQN, the literal output ReLU of §III-C at 500 CPIs, raw
  observations at 10 seeds, 100-pulse CPIs for Fig 4, a larger replay buffer for the LSTM, and the SINR column
  (Eq. 15 constants are not given). E6 and E7 cannot be reproduced.
- **Table D1** in `tests/test_grads.py` is our own exact solution of the paper's MDP (value iteration on the latent
  chain). It is not a table from the paper.

## Layout
- `src/data.py`: the environment as pure `jax.lax` functions: action masks, a*(theta), reward tables (both readings),
  interference chains, target model, `env_step`. There is no dataset.
- `src/algorithm.py`: re-exports `rl_jax_bench.drqn`, the shared recurrent double-DQN core. That core started as this
  paper's file. The shared copy only adds options that are off by default; at the defaults, the AST matches this
  paper's original.
- `src/model.py`: per-arm configs. The arms differ only in (LSTM, trace length, double), plus the variant arms.
- `src/baselines.py`: exact value iteration (pi*, Q*, stationary J), SAA, random, and certainty-equivalent PI over
  16,000 states (scipy).
- `src/train.py`: the driver: lanes, PRNG key tree, chunked train/eval scan, frozen evaluation, baselines on the same
  evaluation streams, and orbax resume.
- `src/eval.py`: per-seed `metrics.json`, the seed-level summary and the stream-identity check.
- `src/toys.py`: toy checks (exact solver, FH-SAA row, Monte-Carlo J, tabular and network cores on 2-state toys).
- `src/configs/*.yaml`: `default` (paper scale), `small`, `full`, `full_500cpi` (T-C), `review_*`, `recheck`.
  `sources.yaml` gives the source of every value.
- `data/fig{3,4}_vector.json`: curves extracted from the paper's vector figures.
- `scripts/small_scale.sh`, `scripts/train_all.sh`: every run behind `results/`, in order, with its grader.
- `scripts/full_report.py`, `variants_report.py`, `long_train_report.py`, `review_report.py`, `recheck_report.py`,
  `regrade_figs.py`: graders (Holm families, H-a, gap fraction, figure and Table II judgments).
- `scripts/toy_diag.py`, `scripts/init_diag.py`: toy diagnostics (convergence control, initializer sensitivity).
- `scripts/flops.py`, `scripts/bench.py`, `scripts/bench_update.py`: compute accounting and micro-benchmarks.
- `scripts/cartpole_reference.py`, `scripts/official_ref_compare.py`: our core on CleanRL's CartPole task, compared
  with CleanRL's own logs.
- `scripts/official/`: `setup.sh` (oracle clones and venvs), `run_cleanrl_ref.sh`, and the fixture recorders
  `cleanrl_harness.py`, `pomdp_harness.py` and `parity_fixtures.py`. `keras/` holds the independent TF 2.2 Keras
  learner (`keras_ref.py`), the stream dumper and the comparison scripts.

## Run it
From this folder, with the root environment (`uv sync` at the repo root):

    uv run python -m pytest -q                       # ~3 min on CPU
    sh scripts/small_scale.sh                        # toys + 10x-smaller proxy
    sh scripts/train_all.sh                          # every full-scale, variant, review and recheck run + graders
    uv run python src/train.py --config src/configs/default.yaml --out runs/paper_scale   # paper scale (500 CPIs) in one run

Outputs go to `runs/`. Original costs on one RTX 3090:
- full-scale run: 26 min;
- T-A/T-B: 10 min; T-C: 17 min;
- review runs: about 55 min in total; recheck: 14 min.

The whole paper took about 2.8 GPU-hours. The paper-scale single run (500 CPIs for every arm) was projected at 1.3
GPU-hours and was never run as one launch. Training, environment and evaluation run in float32, and the exact
solver, PI and statistics in float64. `check_grads` and the pomdp-baselines differential use float64.

## Tests
`uv run python -m pytest -q`: 27 tests, about 2.5 min on CPU, no external data:
- `test_smoke.py`: env shapes, one jitted pulse and one update for every arm, one baseline CPI, PI on a tiny
  transition set, and the stream identity across methods.
- `test_grads.py`: `check_grads`, the exact MDP against Table D1, double-Q and max targets by hand, and burn-in and
  stored-state equivalences. Also checks trace-writer alignment.
- `test_official_parity.py`: replays fixtures recorded from oracle code (`tests/fixtures/`, sha256 in
  `SHA256SUMS`). Covers the CleanRL DQN update with max and double targets, the pomdp-baselines recurrent sequence
  target and gradients, the CleanRL environment and episodic-return path, and torch `nn.LSTM` against our
  Keras-convention cell.
- `test_variant_arms.py`, `test_recheck_arms.py`: T-A/T-B, no-reset, lr and SGD arms, the Keras loss scale, the
  update cadence and PI's strict-improvement guard.
- `test_resume.py`: a run resumed from a mid-run checkpoint reproduces the uninterrupted run's per-CPI outputs.

## Checked against
No author code was released for this paper. Searches of arXiv ancillary files, the paper text, PapersWithCode, GitHub
and the first author's dissertation found none. Labelled second-best oracles were used instead.
- **CleanRL `dqn_jax.py` @ `fe8d8a0`**, unmodified, in its own venv (jax 0.4.8, flax 0.6.8, gymnasium 0.29.1).
  - The update (loss, parameters after 249 Adam steps, target syncs, epsilon) matches with max and double targets.
  - The environment and return path match too.
  - Reference run on CartPole-v1, 10 seeds: with our Keras-default glorot init, our core is DIVERGENT (444 ± 21 vs
    494 ± 3). With lecun init it is at PARITY (495 ± 4), so the gap comes from the initializer
    (`results/cartpole_compare.json`). The radar runs keep glorot, the paper framework's default.
- **pomdp-baselines @ `e7c19c3`** (torch 1.13.1, float64): the recurrent sequence target, its 16 gradient tensors and
  the LSTM cell all match to 1e-10.
- **TF 2.2 tf.keras**, the paper's framework, hosts an independent learner with its own buffer, writer and targets,
  fed our exogenous streams (`scripts/official/keras/`).
  - Keras defaults match ours (`results/keras_defaults.json`).
  - Per-update losses match to 1e-3 relative for 500 updates (`results/keras_diffA_compare.json`).
  - Under Adam, Keras reproduces the DDRQN deficit at E1: -0.115 vs our -0.112, n = 3
    (`results/keras_refB_compare.json`).
  - Under plain SGD at E2, Keras matches our learners to about 1e-5 J_conv per seed
    (`results/keras_sgdE2_compare.json`).
  - Greedy actions during online evaluation agree on only 94 % (DQN) and 44-56 % (DDRQN) of pulses. Keras float32 vs
    float64 diverges earlier than ours vs Keras, so we attribute this to float chaos. Frozen-network acting agrees on
    1000/1000 pulses.

To repeat these checks, run `sh scripts/official/setup.sh`. It clones both repositories at the pinned commits into
`$OFFICIAL_ROOT` (default `external/official`) and builds the three venvs. Then run the recorders in
`scripts/official/`, `scripts/official/run_cleanrl_ref.sh` with `scripts/official_ref_compare.py`, and
`scripts/official/keras/run_*.sh`. `setup.sh` has not been run end to end in this repository.
