# Deep Reinforcement Learning at the Edge of the Statistical Precipice
Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare, NeurIPS 2021.
Paper: https://arxiv.org/abs/2108.13264 (v1). Official code:
[google-research/rliable](https://github.com/google-research/rliable) (commit `3ccd9f4`).

## What this is
The paper is a methodology paper. It argues that deep-RL results reported as point estimates from 3-10 runs are
dominated by statistical uncertainty, and that the sample median of task means is biased. It proposes three tools:
stratified-bootstrap interval estimates, run-score performance profiles, and robust aggregates (interquartile mean,
optimality gap, average probability of improvement). It then re-analyses published Atari, DM Control and Procgen
results. This folder reimplements the whole toolkit in float64 JAX and differential-tests it against rliable. It
reruns every paper number whose input data are still public. It also adds our own 100-seed PPO pool for the few-run
study, because the paper's Atari-100k runs are no longer available.

Labels used in code and tables:
- Experiments: E0 the estimators and intervals, E1 the Atari-100k case study, E2 Atari-100k aggregates, E3 Atari 200M,
  E4 DM Control, E5 Procgen, E6 our own few-run study (not in the paper).
- E6 primary tests (declared before the run, Holm over both): H1 is `separate` vs `shared` PPO (expected large
  effect), H2 is `separate` vs `advnorm_batch` (expected small effect).
- Paper claims: Q1-Q4 are the qualitative ALE claims of §5.1. M1-M4 are the methodological claims (IQM is less
  biased / tighter than the median, percentile is the best interval method from about 10 runs). T1-T3 are
  theoretical predictions we derived: T1 CI width vs N, T2 percentile coverage vs N, T3 IQM bias is O(1/N).
  A1-A3 are ablations (trimming, stratification, run pooling).
- Run names: `toys` (exact toy checks), `small` (20-seed run), `full` (100-seed run), `recheck` (a later checking
  pass that re-ran several analyses).

## Results
Judgment rule, fixed before the runs: MATCH means within 5 % or inside the paper's drawn 95 % CI, CLOSE means within
20 %, and DIVERGENT means anything else. Paper values are read from the vector drawings in the PDF. For ALE rows, "ours"
is one 5-run x 55-game dataset (the public Dopamine runs), so the uncertainty is a 95 % stratified-bootstrap CI, not an
SE over seeds. On that same data, the authors' own pipeline (rliable plus the Colab cells) gives our point estimates to
1.8e-15 or better.

| Quantity | Paper | This reproduction | Verdict |
|---|---|---|---|
| E0 estimators vs rliable, 200 synthetic inputs | - | points within 1e-12; bootstrap replays exact; intervals equal in distribution | MATCH |
| E3/E4 estimator rows (C51, IQN, Rainbow on Fig 9 / A.20 / A.27 / A.28; 6 DM Control studentized rows) | 45 numbers | 44 MATCH, 1 CLOSE (Rainbow; the paper's 200th iteration is not public, we use 199M) | MATCH |
| Fig 10 right, IQM vs frames, C51 / IQN / Rainbow, 1-175M | 27 points + 27 bands | within 3e-4 of the paper | 54 / 54 MATCH |
| Rows involving DQN (Nature) or DQN (Adam) | 36 numbers | 4 of 7 point statistics per DQN agent lie outside the paper's own CI; authors' code gives our number | DIVERGENT (data, see gaps) |
| Q2, Q3, Q4 (metric changes the ordering; OG ranks; IQN vs Rainbow crossing near tau = 2) | yes | same orderings; last crossing at tau = 2.1 | MATCH |
| Q1 (DQN (Adam) and C51 intervals overlap) | all four overlap | median / mean / OG overlap, IQM does not | DIVERGENT (DQN data) |
| E1, E2, E5, DM Control per-run methods | many | not run: the release bucket returns HTTP 404 and there is no mirror | NOT REPRODUCIBLE |
| E6 H1: IQM(separate) - IQM(shared), n = 100 + 100 | - | +0.425 [+0.407, +0.450], P(sep > shared) 0.995 | holds under CleanRL defaults; null with scaled returns |
| E6 H2: IQM(separate) - IQM(advnorm_batch) | - | +0.0011 [-0.0026, +0.0047], Welch p 0.45 | no detectable difference |
| Few runs mislead (null pair H2, 3 runs per arm, 10,000 studies) | 3 runs are too few | 13 % of studies give a CI excluding 0 (2.6x nominal); 3-run percentile CIs cover 63-75 % | SUPPORTED |
| IQM less biased than the median | yes | 2 / 3 E6 arms, 5 / 5 ALE agents | SUPPORTED |
| M3: percentile is the best interval method from about 10 runs | yes | not on our 2-task bounded pools (0 / 12 cells); yes on a 26-task semi-synthetic pool (8 / 8) | MIXED |
| T1: CI width ~ sqrt(N-1)/N, beta = 0.449 on the paper's grid | - | paper's Fig 6 IQM widths give beta 0.452 | SUPPORTED |
| T2: percentile coverage formula | - | predicts the paper's Fig 6 IQM coverage within 2.3 pp (with finite-population correction) | SUPPORTED |
| T3: IQM bias O(1/N) at finite N | - | paper's A.17 fits 0 / 5 algorithms (free kappa 1.23-1.36), ours 0 / 3 | NOT SUPPORTED |
| Tukey trade-off: trimmed mean wins on heavy tails, loses on light tails | yes | MSE(mean) = 12.1x MSE(IQM) on the mixture arm, 0.46x on the light-tailed arm | SUPPORTED |

E6 pool: 100 seeds x 3 arms x 2 tasks (gymnax CartPole-v1, Acrobot-v1), 500k steps, score = mean of 100 sampled-action
evaluation episodes after training, min-max normalized. Raw returns (mean ± SE, n = 100): separate 490.50 ± 0.80 /
-88.43 ± 4.16, shared 251.58 ± 2.04 / -316.68 ± 11.53, advnorm_batch 490.07 ± 0.89 / -84.73 ± 0.30.

Caveats:
- The H1 effect is specific to the CleanRL classic-control defaults (unscaled returns, one trunk for `shared`). With
  returns scaled for every arm, H1 is null on the same seeds and on 100 fresh seeds (`results/full/recheck_rs.json`).
- H2 is an interval statement. It excludes IQM differences outside [-0.003, +0.005], but it does not show a zero effect.
  On the same runs the mean difference has the opposite sign (-0.0033) because of one failed Acrobot seed. This is
  the paper's outlier argument for the IQM.
- Every E6 claim about interval methods rests on M = 2 bounded tasks, which is far from the paper's 26 games.
- Fig 11a's overlap reading at 100k changes under the corrected studentized interval.
- The verdicts above are the final ones. Several files in `results/full/` (`ale.json`, `e6.json`, `e6_declared.json`)
  were written before the recheck pass. That pass relabeled T3 (PARTIAL to NOT SUPPORTED), split M3 by pool, made H1
  configuration-specific and added the Fig 10 right rows. The `recheck_*.json` files hold the revised analyses.

## What differs from the paper / known gaps
- **Data.** The paper's per-run scores (`gs://rl-benchmark-data`) are gone, so Atari 100k (Figs 2-7, A.13-A.19, A.23,
  A.26, A.29), Procgen and the per-run DM Control methods cannot be reproduced. The ALE rows use Dopamine's public
  `baselines/atari/data/<game>.json` at commit `5873f54`, which holds 199 iterations, not 200.
- **DQN rows unresolved.** The public DQN (Nature) and DQN (Adam) runs are not the paper's input. The authors' own
  pipeline gives our numbers on them, and the data version, final iteration, agent mapping, read-off and MC error were
  each ruled out. The alternative sources (Dopamine 200-iteration pickles, DQN-Replay) return HTTP 403. The mapping
  of "DQN (Adam)" to Dopamine's "DQN (Adam + MSE in JAX)" is an assumption.
- **E6 is ours.** E6 is our own substitute for the Atari-100k case study. It runs the paper's procedures on data we
  can audit, and it is not a reproduction of a paper number. Its PPO is the one from this repository's
  `ppo_implementation_details` reproduction, vendored in `src/ppo8/`.
- **Reproducibility of training.** E6 training is bit-reproducible only for the same seed batch on the same backend.
  Rerunning seeds 1-100 as one vmapped CPU batch reproduces the stored pool 100 / 100. A seed trained in a different
  batch, or on GPU, follows a different trajectory (float re-association, 1 ulp after one update).
- **Fixes made during the work.** `exact_div` was not correctly rounded under vmap on CPU, so bc / bca endpoints were one
  atom off. The toy-check key used Python's salted string hash. Both were fixed before the reported results. The bca
  jackknife is per output by default; arch's pooled form is kept for the parity tests only.

## Layout
- `src/algorithm.py`: every estimator (median, mean, IQM modes a/b/c, OG, P(X>Y), profiles), the stratified / task /
  independent bootstraps, percentile / basic / bc / bca intervals, rank matrix, studentized CI and the few-run procedures.
- `src/eval.py`: toy ground-truth checks (exact enumeration, closed forms, identities, coverage prediction) and aggregate tables.
- `src/data.py`: normalization constants, Dopamine download and parser, E6 pool loader, synthetic generators.
- `src/train.py`: E6 training driver around the vendored PPO, with post-training evaluation; `src/ppo8/`: the vendored PPO.
- `src/model.py`: re-exports the PPO networks (the method itself has none).
- `src/crossed_stats.py`: numpy / scipy statistics for crossed seeds x worlds designs (two-way ANOVA, seed-mean t CI,
  max-T sign-flip test, Holm, Clopper-Pearson, P(X>Y), profiles). Use it instead of the stratified bootstrap when one
  trained policy is scored on many worlds.
- `src/configs/sources.yaml`: every setting with its source; `e6.yaml`: the E6 study.
- `data/`: `paper_readoff/readoff.json` (paper values from the PDF's vector drawings), `e0/dmc_reported.json` and
  `e0/ale_final.npz` (small inputs of the rliable reference run).
- `scripts/run_all.sh`: every command behind `results/`, in order.
- `scripts/make_inputs.py`, `scripts/e0_difftest.py`: E0 inputs and the differential test against the rliable output.
- `scripts/official/rliable_reference.py`, `scripts/official/reference_ale.py`: run inside the official venv; rliable
  on the E0 inputs, and the authors' ALE pipeline end to end (raw JSON, Colab normalization cells, rliable).
- `scripts/official_compare.py`: ours vs `reference_ale.py`.
- `scripts/build_fixture.py`, `scripts/build_official_fixture.py`: extract the test fixtures (numbers only).
- `scripts/toys.py`, `scripts/small_scale.py`: toy checks and the 20-seed run.
- `scripts/full_ale.py`, `scripts/full_e6.py`, `scripts/full_e6_sens.py`: ALE / DMC numbers, the 100-seed E6 study
  and its labeled sensitivities (paired analysis, greedy pool, with-replacement subsampling, IQM mode c).
- `scripts/data_provenance.py`: tests whether the public DQN data can be the paper's.
- `scripts/recheck_theory.py`, `scripts/recheck_semisynth.py`, `scripts/recheck_rs.py`: T1-T3 on the vector reads,
  the 26-task semi-synthetic coverage pool, and the reward-scaling sensitivity.
- `scripts/report.py`, `scripts/recheck_report.py`: markdown tables.
- `scripts/paper/*.py`: read the paper's figures from the PDF's vector paths.

## Run it
From this folder, with the root environment (`uv sync` at the repo root): `sh scripts/run_all.sh`, or single steps:

    uv run python scripts/toys.py                     # toy checks, minutes on CPU
    uv run python -m src.train --tag full --set "seeds=[$(seq -s, 1 100)]"   # E6, ~1.2 h on 3 CPU cores
    uv run python scripts/full_e6.py && uv run python scripts/report.py all

Outputs go to `runs/` (git-ignored); `results/` mirrors `runs/` for the files cited here. Large inputs live under
`DATA_ROOT` (default `external/data`): `e0/inputs.npz` (made by `make_inputs.py`), `e0/oracle.npz` (made by
`rliable_reference.py`, 171 MB), `official/reference_ale.npz` and `dopamine/*.json` (downloaded by
`src.data.fetch_dopamine()`, 25 MB). The paper PDF is located by `PAPER_PDF`. Statistics run in float64, training in
float32, never in the same process. Originally the whole study used about 1 GPU-hour (wall-clock upper bound),
and the full run used none.

## Tests
`uv run pytest` (about 3 min on CPU), no external data needed:
- `test_smoke.py`: estimator shapes, one bootstrap, every CI method, one PPO update in a float32 subprocess.
- `test_grads.py`: statistical identities and exact toys (the method has no gradient).
- `test_oracle_parity.py`: rliable outputs in `tests/fixtures/oracle_small.npz`, to 1e-12.
- `test_official_parity.py`: the authors' ALE pipeline (`tests/fixtures/official_ale.npz`). Constants and normalization
  match exactly, points to 1e-12, CI construction replayed on identical replicates to 1e-9.
- `test_procedures.py`: coverage accounting, rank direction and ties, trimmed-MSE and difference distributions.
- `test_crossed_stats.py`: `src/crossed_stats.py` against scipy and `src/algorithm.py`.

## Checked against
rliable at commit `3ccd9f4dea577a04d3d2b557f259aac08badbd81` plus the paper's Colab
(`deep_rl_precipice_colab.ipynb`), run in its own venv (Python 3.11, arch 7.0.0, numpy 1.26.4, scipy 1.13.1,
pandas 2.2.2, inflection 0.5.1). Point `OFFICIAL_ROOT` at a directory holding the clone as `rliable/` and that venv as
`.venv/` (default `external/official`). The library's own tests pass in that venv. Results: `results/e0/difftest.json`
and `results/official/compare.json` (normalized matrices identical, points within 1.8e-15, interval endpoints within
1.3e-15 on replayed replicates).
