# What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study
Marcin Andrychowicz, Anton Raichuk, Piotr Stańczyk, Manu Orsini, Sertan Girgin, Raphael Marinier, Léonard Hussenot,
Matthieu Geist, Olivier Pietquin, Marcin Michalski, Sylvain Gelly, Olivier Bachem, ICLR 2021 (read as arXiv v1,
"What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study"). Paper:
https://arxiv.org/abs/2006.05990. Official code: [google-research/seed_rl](https://github.com/google-research/seed_rl)
(commit `0e1e0ac`; the paper itself links no code, the seed_rl README names the study). The authors also released
the raw study data (about 130 k configurations x 3 seeds), which we use as the paper's numbers.

## What this is
The paper trains about 250 000 PPO-style agents on five MuJoCo tasks while randomly varying more than 50 design
choices. It then recommends settings per choice, judged by the conditional 95th percentile of the score over all
configurations that use a given value. This folder reimplements the study's unified on-policy agent in JAX (brax
MJX environments, all lanes vmapped in one program), builds the recommended configuration REC, and tests the paper's
per-choice claims as one-factor contrasts around REC. It also reruns the paper's statistic on the authors' own data.

Labels used in code and tables: REC = the recommended configuration; BASE = Table 2's base configuration; A1-A8 =
the one-factor arms (A1 obs-norm off, A2a last policy layer x1.0, A2b initial std 1.0, A3a shared trunk, A4 GAE
lambda 0.95, A5a lr 1e-4, A6 entropy 1e-3, A7 value normalization off, A8 per-minibatch advantage normalization);
Q1-Q8 = the paper's claim for the matching arm; X1-X6 = diagnostic and sensitivity arms (`src/configs/agent.yaml`);
T1-T9 = the closed-form toy checks in `tests/test_grads.py`; C## = the paper's numbered choices; EXP-P95(a) = the
paper's statistic re-run on its own data. "Small-scale" = 3 envs x 3 seeds, "full-scale" = halfcheetah x 5 seeds,
"recheck" = a later pass that re-tested the divergent cells (faithful env, lr horizon, power).

## Results
Score = mean of the 10 evaluation points 100 k ... 1 M env steps (20 stochastic episodes each; paper: 100). Paper
ratio = conditional-95th-percentile ratio arm/REC-value from the authors' raw data, [spread] = [lo_arm/hi_rec,
hi_arm/lo_rec], {bootstrap 95 % CI over the raw rows}. Ours = rho = mean(arm)/mean(REC), paired seeds, [paired t95 of
rho]. REC on the halfcheetah proxy: 5 285 ± 270 (n = 5). The judgment rule (fixed before the run): MATCH = inside the
paper's spread with agreeing sign; DIVERGENT = gap > 20 % or opposite significant sign; CONFLICT = both clauses hold;
UNCLASSIFIED = no clause applies. Verdicts are the FINAL ones, after the recheck.

| Quantity | Paper | This reproduction (mean ± SE, n) | Verdict |
|---|---|---|---|
| A1 obs-norm off (Q1) | 0.518 [0.47, 0.65] {0.49, 0.60} | 0.562 [0.27, 0.85] (2 969 ± 300, 5) | MATCH under every reading (Holm p 0.12, n.s.) |
| A2a last layer x1.0 (Q2a) | 0.812 [0.69, 1.07] {0.72, 0.99} | 0.984 [0.70, 1.27] (5 199 ± 575, 5) | CONFLICT (inside spread, gap 21 %); at our orthogonal initializer the paper's own ratio is 1.064 {0.65, 1.52}, no detectable effect |
| A2b initial std 1.0 (Q2b) | 0.565 [0.49, 0.65] {0.52, 0.61} | proxy 1.164 [0.995, 1.333] (6 152 ± 372, 5); gymnasium v4 1.135 (5) | DIVERGENT, robust; the authors' code on v4 also gives 0.966 (5) |
| A3a shared trunk (Q3) | 1.008 [0.79, 1.30] {0.81, 1.22} | 0.901 [0.65, 1.15] (4 763 ± 463, 5) | MATCH, knife-edge (0.9013 vs the 0.900 margin; not TOST-equivalent) |
| A4 GAE lambda 0.95 (Q4) | 0.802 [0.50, 1.28] {0.61, 1.13} | 1.009 [0.84, 1.18] (5 334 ± 180, 5) | DIVERGENT by the 20 % gap clause (MATCH under the generic spread rule); not rechecked |
| A5a lr 1e-4 (Q5) | 0.798 [0.69, 0.83] {0.73, 0.83} | proxy 1.038 (5); after the lr-horizon fix 0.983 [0.73, 1.24] (5); v4 0.989 (5), post-fix 0.899 (3) | DIVERGENT, recheck incomplete; the authors' code on v4 gives 0.864, and the paper's own ratio ranges 0.45-1.05 across its experiment groups |
| A6 entropy 1e-3 (Q6) | 1.171 [0.82, 1.62] {0.71, 1.55} (CI clamped) | 0.984 [0.87, 1.10] (5 198 ± 150, 5) | UNCLASSIFIED; consistent with the text "no regularizer helps significantly" (TOST p 0.059, not equivalent) |
| A7 value-norm off (Q7) | 0.723 [0.62, 0.95] {0.65, 0.88} | proxy 0.965 (5); gymnasium v4 0.846 [0.70, 1.00] (10) | MATCH on the faithful env (post hoc env choice; Holm p 0.13); the proxy n = 5 cell is an underpowered draw. Authors' code on v4: 0.656 (10) |
| A8 adv-norm on (Q8) | 0.947 [0.75, 1.16] {0.81, 1.10} | 1.019 [0.81, 1.23] (5 387 ± 249, 5) | UNCLASSIFIED; consistent with "matters little" (TOST p 0.17) |
| REC vs Table-2 BASE, halfcheetah | recommendations should help | REC 3.2x BASE (BASE 1 649 ± 206, n = 3; Welch p 4e-5) | HOLDS (authors' code, Hopper-v4: 1.38x) |
| Table 5 quantiles (90/95/99/max, 5 envs) recomputed from raw data | Table 5 | 20 of 20 numbers equal | MATCH |
| Coverage of the paper's binomial CI vs exact theory | nominal 0.95 | 200/200 cells within 3 MC SE; 0.950 at n = 1 000, 0.33 at N = 32 | MATCH; the CI under-covers below about 100 configurations |
| BASE, Hopper | 1 560 ± 21 (40 study rows) | authors' code on v4: 1 250 ± 44 (3); ours (proxy): 1 335 ± 102 (3) | CLOSE: authors' configuration score (median of 3) is -17.1 %, the 5th percentile of the rows; ours -7.9 % |
| BASE, HalfCheetah | 1 069 ± 11 (35 rows) | 1 649 ± 206 (3, proxy) | +54 %, across simulators; curves agree with the authors' code to 400 k |

Tally over the 9 primary cells after the recheck (pre-declared rule, best-evidence env per cell): MATCH 3 (A1, A7 on
the faithful env, A3a knife-edge), CONFLICT 1 (A2a), DIVERGENT 3 (A2b robust; A5a and A4 with the recheck
incomplete), UNCLASSIFIED 2 (A6, A8). No contrast is Holm-significant at n = 5 (REC seed CV 11 %, so n = 5 has
power only for effects of about 25 % or more).

Caveats:
- **Different estimand.** The paper's ratio is a conditional 95th percentile over about 1 000 random configurations
  per value, most of them far from REC. Ours is a one-factor seed-mean contrast around one strong configuration.
  Restricting the raw rows to REC-like configurations (`scripts/paper_conditional.py`) moves several paper ratios
  toward ours (A7 0.947 {0.71, 1.29}); for A5a the paper's effect survives the conditioning (0.781 {0.58, 0.90}).
- **A2b is the one robust divergence**, and the authors' own modules share it on the same simulator, so it is not
  our learner. What remains (estimand, gym v1 / MuJoCo 2.0, the asynchronous SEED actors) is untested.
- **The A7 and A2b "after recheck" verdicts rest on the 2 M lr-horizon v4 runs**, which is no longer the code
  default. The post-fix v4 rows have n = 3 only (A7 0.639 [0.07, 1.21], A2b 1.031).
- `results/full_report.{json,md}` were generated before the recheck: their A2b/A5a/A7 judgments are the proxy-only
  ones in the table's "proxy" entries. `results/recheck_report.json` holds the revised judgments. `results/small_*`
  are the small-scale lanes, which ran with the released-HEAD action semantics that the study never used. Their
  walker2d/hopper rows carry no judgment and halfcheetah is reported side by side only.

## What differs from the paper / known gaps
- **Environments are proxies.** Training uses brax 0.14.2 MJX `halfcheetah`/`walker2d`/`hopper`. The paper used gym
  `*-v1` on MuJoCo 2.0. HalfCheetah's reward, observation, reset and termination are identical to gym's; the only
  differences are torso/shin floor contact and 4 solver iterations, and policy-transfer checks keep the
  A2b/REC ordering under gym dynamics. Walker2d deviates more. The faithful-env path `scripts/train_gym_v4.py` runs
  our learner on gymnasium HalfCheetah-v4 for the recheck.
- **Action semantics.** The released seed_rl HEAD stores post-squash actions; the study ran an older distribution
  (seed_rl@68c15aa) that stores the pre-squash sample. We follow the study-era version (`logp_presquash: true`).
  With the HEAD clip semantics the authors' own BASE collapses to NaN on 3/3 seeds.
- **lr horizon.** The full-scale run decayed lr over 2 M frames (as the official learner derives it), so lr was
  about lr0/2 at the 1 M scoring end. The default is now the training length (lr -> 0 at 1 M); `--lr_horizon_frames
  2000000` and the `lr2M_*` arms reproduce the old schedule. The A2b/A4/A7 post-fix proxy re-run was killed at its
  time limit and has no result.
- **A3a confound.** The shared-trunk arm also shrinks the value net from 256 to 64 wide. The control arm `A3x_sep_v64`
  shows value width alone has no measurable effect (A3x/REC 1.020); the isolated sharing contrast is 0.884 (p 0.26).
- **Assumed settings** (not fixed by the paper): entropy coefficient 1e-3, value width 256, shared-trunk width 64,
  one extra iteration to finish at 1 001 472 steps. Every key is traced to its source in `src/configs/sources.yaml`.
- **Not run (budget):** walker2d/hopper at full scale (so the env-specific claims Q1-on-Hopper and Q7-on-Walker2d
  are untested), Ant, Humanoid, the secondary arms at full scale, A4 on the faithful env, and the SEED distributed
  system itself. Evaluation used 20 episodes per point instead of 100 (adds about 2 % to the seed variance).

## Layout
- `src/algorithm.py` collection, GAE, PPO update (PopArt, advantage recompute, keras-style Adam), evaluation.
- `src/model.py` policy/value networks, initializers, tanh/clip action distributions (study-era and HEAD semantics).
- `src/data.py` brax MJX env wrapper, observation normalizer, env fingerprint.
- `src/train.py` vmapped multi-arm, multi-seed trainer with checkpoint/resume (`python -m src.train`).
- `src/eval.py` per-seed aggregation, paired tests, the paper's percentile CI (`python -m src.eval --tag <tag>`).
- `src/configs/agent.yaml` REC, BASE and every arm; `sources.yaml` per-key provenance; `paper_raw_p95.txt` the
  paper's conditional percentiles recomputed from the authors' raw data (the paper-number source).
- `scripts/train_all.sh` every training run behind the tables, sequential.
- `scripts/report.py` full-scale table and judgments; `small_report.py` small-scale table; `recheck_report.py`
  recheck tables (faithful env, lr horizon); `write_run_config.py` writes a run's config from its own lanes.
- `scripts/train_gym_v4.py` our learner on gymnasium HalfCheetah-v4; `recheck_v4.sh` the v4 comparison runs.
- `scripts/p95_subsample.py`, `p95_report.py` EXP-P95(a); `paper_conditional.py` paper ratios on REC-like subsets.
- `scripts/official_cause_test.py`, `official_loop_summary.py` summaries of the authors'-module runs.
- `scripts/brax_ppo_reference.py` Brax's own PPO as a second-best reference curve; `bench_env.py` env throughput;
  `base_logratio_probe.py` per-epoch log-ratio diagnostic for BASE.
- `scripts/review/` bootstrap CIs of the paper ratios (`paper_ratio_boot.py`, `c1_*_boot.py`, `a2b_reclike_boot.py`),
  `recompute_full.py` (recompute the full-scale scores from curves), `study_score_column.py`, `determinism.sh`.
- `scripts/official/` our harness around the authors' modules (runs in their TF venv): `official_loop.py` /
  `official_loop_v4.py` synchronous training loops, `difftest_*.py` fixture generators, `official_dump.py` +
  `make_slim_replay.py` replay fixtures, `metric_oracle.py` notebook metric fixtures, `studyera_dist.py` study-era
  distribution adapter, `convert_data.py` / `recompute.py` / `base_rows.py` raw-data tools, `official_reference.sh`.

## Run it
From this folder (`PY` defaults to `uv run python`):

    uv run python -m src.train --env halfcheetah --arms REC,A1_obsnorm_none --seeds 0-4 --tag full --eval_episodes 20 --det_final 0
    uv run python -m src.eval --tag full && uv run python scripts/report.py
    sh scripts/train_all.sh            # everything; about 1.5 GPU-hours on one RTX 3090 (shared)

The full-scale run (50 lanes, 1 M steps each) took 15 min on an RTX 3090 including 111 s of compile. On CPU the
trainer runs at about 300 env-steps/s, so a 1 M-step lane needs about 1 h. `scripts/train_all.sh` passes
`--lr_horizon_frames 2000000` where the original runs used the old schedule. The authors'-module and v4 runs are CPU
only (7-36 min per 1 M-step run): `sh scripts/official/official_reference.sh`, `sh scripts/recheck_v4.sh`.
The raw-data scripts (`p95_subsample.py` about 55 min, `paper_conditional.py` 5 min, `report.py`) need the
authors' data under `$OFFICIAL_ROOT/data` (see below).

## Tests
`nice -n 19 ../../.venv/bin/python -m pytest -q` (CPU). `test_grads.py` checks T1-T9 against closed forms and exact
enumeration (GAE, the PPO 1/(1+eps) bound, log-probs at clipped/squashed actions, std offsets, normalizer merge,
exact CI coverage, scores). `test_smoke.py` runs the trainer, checks that each arm changes exactly its keys, the
shared-trunk capacity, observation-stat ordering and the lr horizon. `test_env_semantics.py` pins brax time-limit,
termination and reset semantics. `test_official_parity.py` and `test_replay_parity.py` compare our update with the
authors' modules on committed fixtures (`tests/fixtures/`, 4.6 MB): single steps, multi-epoch loops, metric code and
two real HalfCheetah-v4 iterations. `test_reference_ppo8.py` compares shared estimators with the sibling
reproduction `papers/ppo_implementation_details` (located by `PPO_REFERENCE`, default
`../ppo_implementation_details/src/algorithm.py`; skipped when absent). No test needs the authors' checkout.

## Checked against
The authors' modules, google-research/seed_rl@`0e1e0ac`, plus the study-era distribution file from commit `68c15aa`,
run unmodified in their own TF venv (Python 3.11, tensorflow-cpu 2.15.1, tensorflow-probability 0.23.0, gymnasium
0.29.1, mujoco 3.1.6). Only the SEED gRPC actor/learner plumbing is replaced by our synchronous loop. Set it up under
`OFFICIAL_ROOT` (default `external/official` in this folder, git-ignored):

    git clone https://github.com/google-research/seed_rl $OFFICIAL_ROOT/seed_rl && git -C $OFFICIAL_ROOT/seed_rl checkout 0e1e0ac
    git -C $OFFICIAL_ROOT/seed_rl show 68c15aa:common/parametric_distribution.py > $OFFICIAL_ROOT/pd_68c15aa.py
    curl -o $OFFICIAL_ROOT/data/mujoco123_data.json https://storage.googleapis.com/seed_rl_external_data_release/mujoco123_data.json
    # sha256 ce5f423c1d693ca69c00365045287ef9513621f367012c58b2ad201571203d24 (387 MB); then
    python scripts/official/convert_data.py      # -> $OFFICIAL_ROOT/data/mujoco123.pkl

Agreement (identical inputs): single training steps (7 cases), the multi-epoch `ppo_training_step` and the actor path
match to float32 tolerance; replayed real-data iterations 0-11 agree to 6e-7 relative per iteration; late
iterations 300-311 and an episode-boundary iteration agree at the float32 clip-flip floor. On the same simulator
(HalfCheetah-v4, 1 M steps) REC scores 5 938 ± 197 (theirs, n = 10) vs 5 707 ± 221 (ours, n = 10), p 0.45, and the
arm x code interactions are not significant (A7 p 0.077, A2b 0.22, A5a 0.14). The paper's Table 5 is reproduced from
the raw data to the unit. Not reproducible here: the SEED distributed system and gym `*-v1` on MuJoCo 2.0.
