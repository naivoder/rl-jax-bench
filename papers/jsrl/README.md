# Jump-Start Reinforcement Learning
Ikechukwu Uchendu, Ted Xiao, Yao Lu, Banghua Zhu, Mengyuan Yan, Joséphine Simon, Matthew Bennice, Chuyuan Fu,
Cong Ma, Jiantao Jiao, Sergey Levine, Karol Hausman, ICML 2023 (PMLR 202). Paper:
https://arxiv.org/abs/2204.02372 (v2). Official code: none released for this paper. The first author's
`implicit_q_learning` fork is identical to upstream IQL and contains no JSRL code. The third-party SB3 implementation
[steventango/jumpstart-rl](https://github.com/steventango/jumpstart-rl) (commit `e5f690fe`) was used as the reference oracle.

## What this is
JSRL lets a fixed guide policy act for the first `h` steps of each training episode. A learned exploration policy
acts for the rest of the episode, and all the data trains the learner. `h` then shrinks along a curriculum, either on
a performance threshold or drawn at random per episode. The paper claims that this beats both learning from scratch
and policy initialization, that the student ends above its guide, and that the sample cost is polynomial rather than
exponential in the horizon (Theorem 4.1, Theorem A.3, Algorithm 2).

This folder reimplements the mechanism in pure JAX at reduced scale on gymnax MinAtar. The exploration policy is a
PER dueling double DQN with 10-step returns ("PDD", the paper's own "DQN with eps-greedy" example). The guides are
weak PPO snapshots. The combination-lock theory is reproduced exactly on a CPU toy.
The paper's D4RL AntMaze/Adroit (IQL) and vision-grasping (QT-Opt) experiments were not run: the budget was too
small and the grasping simulator is proprietary.

Arms differ only in traced hyper-parameters and share one vmapped program:
`JSRL` (curriculum), `JSRL-Random`, `PDD` (from scratch, which equals JSRL with h = 0 bit for bit), `DQfD`,
`BCinit-FT` (a cross-entropy pre-trained Q net then pure TD, a DQN analogue of the paper's naive policy
initialization), and `IMIT` (the clone itself, never updated online, used as the floor).

Labels used in code and tables:
- R(j) is the mean raw return of 32 pi_e-ALONE episodes at eps 0.01, at evaluation point j = 0..100.
- EARLY is the mean of R(1..10) and FINAL is the mean of R(98..100). mu_g is the guide's true mean.
- H1a (strict) is EARLY(JSRL) - EARLY(PDD) with pi_e alone. H1c uses JSRL's combined policy instead (guide plus
  tail, the policy the paper plots). H1m / H1p are stage-matched: pi_e alone at, and over the 10 evaluations from,
  JSRL's first h = 0 point.
- H2 is FINAL(JSRL) - mu_g. H3 is FINAL(JSRL) - FINAL(DQfD).
- The paper's claims are C2 (naive initialization collapses), C3 (JSRL beats from-scratch early and final), C4
  (curriculum vs random) and C5 (the student exceeds the guide). Secondary tests are S-C4early, S-C4final (TOST),
  S-C2, S-BC and S-IMIT.
- T1-T4 are the theory fits. Lock-a..Lock-g are the combination-lock checks in `tests/test_lock.py`.
  Small-1..Small-6 are the pass criteria of the small-scale run, which were declared before it.
- Amendments A1/A2 were written before any full-scale number. A1 grades C3-early on the combined policy (H1c) and
  keeps pi_e alone as the strict family. A2 sets the S-C4final margin to 0.1 x the pooled FINAL.

## Results
Breakout only, 200 k env steps (1/5 of the paper's 1 M), 10 seeds, `src/configs/default.yaml` unless marked "amended".
"Amended" is `src/configs/learner_minatar.yaml`, run on the same seeds. It sets lr 2.5e-4 and target period 1,000,
the MinAtar reference DQN constants. The theory rows are CPU Monte Carlo. Verdicts are the final ones.

| Quantity | Paper | This reproduction (mean ± SE, n) | Verdict |
|---|---|---|---|
| Lock P(final policy optimal), H=10, A=2, m=10 | (1-q)^H from Thm A.3 | 0.995187 ± 0.000049 MC vs closed 0.995128 (2 M traj); exact enumeration error 5.6e-17; scratch at most 0.0931 | MATCH |
| T1 slope of ln(1-P) in m, A=2 / A=3 | -0.6931 / -0.4055 | -0.6736 ± 0.0009 / -0.3810 ± 0.0004 | MATCH / CLOSE (6.0 %, linearization) |
| T2 exponent of 1-P in H | 1 | 1.0015 ± 0.0077 | MATCH |
| T3 slope of ln E[episodes to first reward] in H (scratch) | ln 2 = 0.6931 | 0.6938 ± 0.0006 | MATCH |
| T4 coverage, monotone in C | grows with C | monotone; MC vs closed form max \|z\| 0.52 | MATCH as an implementation check only |
| Algorithm 2 as written, slope of ln S in ln T / in ln H | ≤ -1/3 / ≤ 7/3 | -0.3328 / 1.333 (derived -1/3, 4/3) | T-rate MATCH but near-tautological (an exponent-1/2 oracle gives -0.4972); H-rate consistent |
| Real learner on the lock, steps to 0.9 vs H | polynomial (JSRL) vs exponential (scratch) | log-log slope 0.33 [0.15, 0.55]; scratch unsolved at H ≥ 16 (5 seeds) | CONSISTENT (descriptive) |
| H1a early, pi_e alone, JSRL - PDD | not stated for pi_e alone | -1.797 ± 0.163 (0/10); fresh seeds -1.570 ± 0.153 (0/10); amended -2.383 ± 0.100 (0/10) | DIVERGENT (opposite sign) |
| H1c early, combined policy - PDD | JSRL ahead early | +4.911 ± 0.110 (10/10) | MATCH in direction, by construction (guide plus tail) |
| H1p stage-matched pi_e from first h = 0 | - | +0.384 ± 0.787 (8/10); amended +5.188 ± 0.935 (10/10, Holm 5.4e-4) | not separated / GO amended |
| C3-final, FINAL(JSRL) - FINAL(PDD) | 8/8 rows ≥ from-scratch | +3.29 ± 0.83 (9/10); amended +8.75 ± 3.02 (9/10); sticky actions 0.1: +0.10 ± 0.46 (5/10) | REPRODUCED in direction on deterministic MinAtar only; DIVERGENT under sticky actions |
| H2 / C5, FINAL(JSRL) - mu_g | 8/8 rows | +10.54 ± 1.01 (10/10, Holm 3.8e-6); amended +18.36 | GO, but PDD also beats this weak guide (+7.25 ± 1.01, 10/10) |
| H3, FINAL(JSRL) - FINAL(DQfD) | never compared | +10.57 ± 1.01 (10/10); PDD - DQfD +7.29 | GO, not specific to the roll-in |
| C4 early, curriculum - random, pi_e alone | curriculum earlier | -2.13 ± 0.16 (0/10); amended -3.64 | measurement-dependent (combined-policy reading: 10/10 in the paper's direction) |
| C4 final, TOST at 0.1 x pooled FINAL | comparable | +0.33 ± 1.18 (p 0.135); amended +5.79 ± 2.56 (8/10) | UNDETERMINED |
| C2 naive init collapses (BCinit-FT) | Fig. 2: ~40 -> ~0, persistent to 1 M | dip -81 % of R(0) (amended -40 %), then recovery to FINAL - R(0) = +3.9 (+6.8) | transient dip only; persistent collapse NOT reproduced |
| Four-game 1 M primary family (Breakout, Asterix, Seaquest, SpaceInvaders) | - | not run | UNREPRODUCIBLE here (budget) |

Small-scale run per arm (pi_e alone, default config): FINAL JSRL 17.32 ± 1.01, JSRL-Random 16.99 ± 0.93,
PDD 14.04 ± 1.01, BCinit-FT 10.56 ± 0.74, DQfD 6.75 ± 0.04, IMIT 6.64 ± 0.05. The guide's mu_g is
6.784 ± 0.029. The curriculum reached h = 0 at evaluation points 52-83 (median 58).

Caveats:
- Early deficit mechanism. In early training ~97 % of the replay is single-policy guide data, so Q for non-guide
  actions is unconstrained extrapolation. A 6,000-step guide-off warm-up in our code removes the deficit
  (+2.0 ± 0.29 vs JSRL, 5/5, n = 5; probe P-B). The third-party code shows the same deficit once its warm-up issue is
  removed (see "Checked against"). These are probe results at n = 5, not an established cause. Neither tau, lr,
  Huber loss, eps-on-own-steps nor 1-step returns removes the deficit, and tau, lr and Huber widen it.
- The paper's plotted quantity is not identified. If its curves show the combined policy, H1c is the matching reading
  and holds trivially.
- Sticky actions 0.1 (the original MinAtar default) make the guide weaker (4.83 vs 6.78) and shorter. The roll-in
  advantage at FINAL then disappears, but guide strength and env stochasticity are confounded. The sticky run was
  rerun bit for bit.
- The amended JSRL arm hits the 1,000-step cap in 5/10 runs, so its FINAL is a lower bound.
- H3 compares DQfD at constant eps 0.01 with JSRL annealing eps from 1, so it confounds exploration schedules.
- Files in `results/small/` predate the recheck. `small_check.json` Small-5 uses the paper ratio 0.39; the corrected
  reading of Fig. 4 is 0.56, and the band [0.056, 5.6] still passes. `analysis_*.json` there report the pre-amendment
  verdict wording.

## What differs from the paper / known gaps
- MinAtar Breakout instead of D4RL / robotic grasping, a DQN-family learner instead of IQL / QT-Opt, and 200 k
  instead of 1 M steps. The other three games only passed a pipeline smoke run (`scripts/smoke_games.sh`).
- Not run: the 12-test four-game family at 1 M (stopped at the GPU budget, no truncated window graded), tolerance
  0.15 (Table 11), FALCON+ / Corollary A.5, a conservative or BC-regularized learner, the separate-actor /
  fresh-critic variant of C2, and the warm-up arm of Figs. 7/8.
- Stage list: h_0 = round(H1) with step round(H1)/n, n + 1 values (Sec. 4.2 / Alg. 1). The n-value reading of
  App. A.4 is not run. Tolerance 0.05, MA window 5, n = 10 and the per-stage window reset are our assumptions where
  the paper is silent.
- Env: gymnax 1.0.0 MinAtar with two SpaceInvaders repairs and an Asterix spawn repair (on by default), each checked
  against the original numpy MinAtar. Sticky actions are off unless `--sticky` is passed.
- The learner constants come from our DQfD reproduction (`papers/dqfd`). They were not tuned for any arm, and the
  inherited constants are weak for PDD on Breakout, which is why the amended config exists.

## Layout
- `src/envs.py`: MinAtar wrapper (bool obs, terminated/truncated, Asterix/SpaceInvaders repairs, sticky actions).
- `src/model.py`: dueling Q network and the PPO actor-critic used as guide.
- `src/replay.py`: exact prioritized replay with a demo region and n-step windows.
- `src/data.py`: demo collection and loading.
- `src/guide.py`: loads the PPO guide snapshot.
- `src/curriculum.py`: stage list and the beta threshold state machine.
- `src/algorithm.py`: losses, arms, the switching rule, the interaction scan and evaluation.
- `src/lock.py`: the combination lock (exact, Monte Carlo, ETC oracle, Algorithm 2 literal) as numpy and as a jittable env.
- `src/train.py`: entry point.
- `src/eval.py`: the pre-declared statistics.
- `src/configs/`: `default.yaml` (original runs) and `learner_minatar.yaml` (amended learner).
- `data/guides/`: four PPO guide snapshots (3.2 MB, sha256 in `SHA256.txt`).
- `data/demos/`: the demo sets (0.8 MB) and the guide statistics / stage lists per game. They are shipped because the
  script that trained the guides belongs to the DQfD reproduction (`papers/dqfd`) and is not part of this folder;
  `scripts/guide_stats.py` regenerates the `*_guide*.json` files and the repaired Asterix demo set.
- `scripts/_common.py`: entry-point preamble.
- `scripts/guide_stats.py`: guide true mean, H1 and stage lists.
- `scripts/toys_report.py`: lock closed form vs exact enumeration vs Monte Carlo.
- `scripts/toys_learner.py`: the real learner on the lock (Lock-g).
- `scripts/theory_fits.py`: T1-T4.
- `scripts/alg2.py`, `scripts/alg2_control.py`: Algorithm 2 rates and the oracle-exponent control.
- `scripts/small_check.py`: Small-1..Small-5.
- `scripts/analyze_amendments.py`: A1/A2 statistics for a run directory.
- `scripts/fresh_seeds_analyze.py`: the fresh-seed run (seeds 10..19, eps_own and 1-step variants).
- `scripts/recheck_compare.py`, `scripts/recheck_sticky.py`: default vs amended learner; the sticky-actions run.
- `scripts/identity_checks.py`: JSRL forced to h = 0 equals PDD bit for bit. An optional reference from the
  DQfD reproduction is not shipped.
- `scripts/flops_cost.py`: XLA FLOPs per method.
- `scripts/write_seed_summaries.py`: per-seed headline scalars.
- `scripts/smoke_games.sh`: smoke run of the other three games.
- `scripts/official/`: third-party and original-MinAtar harness, run in a separate venv.
  - `setup.sh`: fetches and builds everything below.
  - `record_stevtango_fixtures.py`, `ref_r10_probe.py`, `ref_dqn_eps_probe.py`: write the test fixtures.
  - `ref_pointmaze.py`, `ref_analyze.py`: their own PointMaze example.
  - `record_official_env.py`, `policy_parity.py`, `run_parity.sh`: env parity against the original MinAtar.
  - `on_our_task/`: their JSRL wrapper around SB3 DQN on our Breakout with a numpy port of our guide, plus probes;
    `run_all.sh` runs it.

## Run it
Run everything from this folder with the root environment. The original timings are from one shared RTX 3090, so
expect them to be slower on CPU.
```
uv run python scripts/guide_stats.py Breakout Seaquest Asterix SpaceInvaders
uv run python -m src.train --game Breakout --seeds 10 --steps 200000 --out runs/small/Breakout --ckpt-every 0      # ~0.55 GPU-h, 6 arms
uv run python scripts/small_check.py runs/small/Breakout Breakout
uv run python -m src.eval --out runs/small/Breakout --game Breakout --json runs/small/analysis_Breakout.json
uv run python -m src.train --game Breakout --seeds 10 --steps 2e5 --arms PDD,JSRL,JSRL+reset_on_adv=0,JSRL+tol=0 --out runs/verify/Breakout --ckpt-every 0   # tolerance/window run, ~950 s
uv run python -m src.train --game Breakout --seeds 10 --seed0 10 --top-seed 0 --steps 2e5 --arms 'PDD,JSRL,JSRL+eps_own=1,PDD+lam_n=0,JSRL+lam_n=0,JSRL+eps_own=1+lam_n=0' --pretrain 0 --out runs/verify/Breakout_r3 --ckpt-every 0
uv run python scripts/fresh_seeds_analyze.py runs/verify/Breakout_r3
uv run python scripts/analyze_amendments.py runs/verify Breakout
uv run python -m src.train --game Breakout --seeds 10 --steps 2e5 --config src/configs/learner_minatar.yaml --out runs/recheck/fix/Breakout --ckpt-every 0   # amended learner, ~4,400 s shared GPU
uv run python -m src.train --game Breakout --seeds 10 --seed0 20 --top-seed 0 --steps 2e5 --arms PDD,JSRL --pretrain 0 --sticky 0.1 --out runs/recheck/fix_sticky/Breakout --ckpt-every 0
uv run python scripts/recheck_sticky.py runs/recheck/fix_sticky/Breakout runs/recheck/sticky_analysis.json
uv run python scripts/toys_report.py; uv run python scripts/theory_fits.py; uv run python scripts/alg2.py   # CPU, alg2 ~10 min
```
The `--steps 1e6` four-game runs use the same command with `--game` set. On a GPU the original runs set
`XLA_FLAGS=--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN,WHILE`. Do not add `CONDITIONAL`
(it crashes Asterix), and do not use `WHILE` for Asterix at 50 runs (SIGSEGV). Run units sequentially, because
concurrent processes time-slice one GPU.

## Tests
`uv run pytest -q` (CPU, ~90 s, 78 tests, none skipped and none need external code):
- smoke: shapes, one interaction chunk with the guide, a curriculum step.
- grads: TD-loss gradients in float64, targets are stop-gradient.
- lock (Lock-a..Lock-g): closed form = exact enumeration, Monte Carlo within 4 SE, scratch A^H, coverage, stage-list
  rounding, the beta state machine vs a python reference and vs a transcription of steventango's rolling window, the
  switching rule through the production code, Algorithm 2 vectorized vs reference.
- amendments: eps_own counting, stage-matched indexing, TOST margin.
- official parity: 11 tests against fixtures recorded by running steventango's real classes.

## Checked against
- steventango/jumpstart-rl `e5f690fe` (SB3, third-party). Our beta / stage / switch code equals their real classes
  on 1,597 of 1,600 cases; the other 3 are documented float ties. The fixtures cover their issues:
  - their stage list is wrong when n does not divide H1;
  - their switching rule is offset by one step;
  - (issue 10) their EvalCallback evaluates the combined policy on training counters;
  - (issue 14) their DQN wrapper drops eps-greedy.
- Their code on our task (`scripts/official/on_our_task/`, CPU, n = 5), with the guide port checked to 2.6e-6 logit
  error and guide mean 6.796 ± 0.031 vs our 6.784 ± 0.029:
  - n-step targets equal SB3 2.7.0's `NStepReplayBuffer` to 9.5e-7.
  - (issue 16) their per-episode counter is wrong in 98.6 % of training calls.
  - (issue 17) the training observation goes stale after each evaluation.
  - (issue 18) the guide never acts before `learning_starts`.
  - As-is their early pi_e is not behind scratch (OR-1: +0.283 ± 0.221, p 0.27). With the roll-in from step 0
    (P-A) it is, -1.663 ± 0.153 (0/5, p 4.1e-4), matching ours.
  - Their FINAL is +4.59 ± 1.50 over scratch (5/5).
- Their PointMaze TD3 example (3 seeds): no early pi_e deficit there. Issue 18 does not apply at `learning_starts`
  100, so this remains a task / learner difference.
- Original numpy MinAtar `4241f246` (`run_parity.sh`), driven by the guide:
  - repaired Asterix, 135 k steps: 0 observation / reward / terminal mismatches;
  - unrepaired control: diverges at the predicted step 926;
  - Breakout, Seaquest and SpaceInvaders: 0 mismatches.
- Learner parity: our PDD equals the DQfD reproduction (`papers/dqfd`) PDD to 2.98e-8 on CPU. The reference script lives
  in that project and is not shipped.
- Removed test: one test compared the first-h0 index stored in the small-scale run's per-seed `metrics.json` against
  `src/eval.first_h0_point`. It passed on the original run outputs, which are not shipped.

`sh scripts/official/setup.sh` clones the two repositories at the pinned commits into `$OFFICIAL_ROOT` (default
`external/official`), builds the reference venv and installs SB3 2.7.0. The on-our-task scripts run with a work
directory as cwd; `sh scripts/official/on_our_task/run_all.sh` runs them in order (a few CPU-hours). Its results are
in `results/official/on_our_task.json` and `nstep_parity.json`. setup.sh has not been run in this folder; only the
pinned commits were checked to exist.
