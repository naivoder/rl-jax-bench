# GOSPA-Driven Non-Myopic Multi-Sensor Management with Multi-Bernoulli Filtering
George Jones and Ángel F. García-Fernández, arXiv preprint 2025 (v2, 4 May 2026). Paper:
https://arxiv.org/abs/2511.01045. Official code:
[sggjon5/GOSPA-Driven-Non-Myopic-Multi-Sensor-Management-with-Multi-Bernoulli-Filtering](https://github.com/sggjon5/GOSPA-Driven-Non-Myopic-Multi-Sensor-Management-with-Multi-Bernoulli-Filtering)
(Python/NumPy, commit `b5959f6`).

## What this is
This is a sensor-management (planning) paper, not reinforcement learning. Nothing is learned. Two mobile sensors
track an unknown, time-varying number of targets with a Gaussian multi-Bernoulli filter. At every step a Monte Carlo
tree search (MCTS) picks the sensors' next moves. It minimizes a closed-form upper bound on the expected mean-square
GOSPA error over a lookahead window. The paper claims that this non-myopic, GOSPA-driven planner beats a myopic one and
a KLD (information-gain) driven planner of the same search budget, and reports this in Table III (time-averaged
RMS-GOSPA, 50 Monte Carlo runs, three clutter rates).

This folder reimplements the filter, the GOSPA metric, the bound and the MCTS as fixed-shape, jit- and vmap-able JAX
functions in float64. A whole closed-loop experiment (one configuration and algorithm, all runs and clutter rates) is
one vmapped batch. The `train.py` name follows the repository template; it is the experiment driver.

**Two configurations.** The authors' code differs from the paper text in about 30 places (table below). The
published figures were made by the code: the Figure 3 truth is reproduced bit for bit only with the code's birth
probability 0.05, not the paper's 0.03. We therefore run both:
- **CFG-O**: the authors' code behavior. Table III numbers are graded on it.
- **CFG-P2**: the method as written in the paper text. CFG-P (v1) is an earlier, superseded version of it, kept
  because some follow-up arms were flown on it. CFG-P2 adds two corrections: the planner bound uses the full 4-D
  `tr(P)` of eqs. 29/39, and the MCTS budget counts nodes added to the tree (Sec. IV-C.2), with an iteration cap of 2B.

Labels: E1 Table III, E2 Figure 5 decomposition, E3 Figure 4 snapshot, E4 Figure 3 truth, E5 Table IV runtime.
Claims: H1 (C1) non-myopic GD beats myopic GD; H2 (C2) MCTS3-GD is the best GD variant; H3 (C3) GD beats KLD;
C4 results are consistent across clutter; C5 myopic sensors get stuck behind the obstacle while non-myopic ones
navigate. T1-T3 are checks of the paper's theory. P1-P10 are the analysis rules fixed before the runs. F-A is the
ablation, F-S the one-at-a-time sensitivity flips S1-S14 / K1-K2, F-T the hyper-parameter parity grid, and S-TRUTH
reruns on ten other truths from the authors' generator.

## Results
50 runs per cell, as in the paper. G is the paper's statistic, the time mean of the per-step RMS-GOSPA over runs.
The paper prints no spread. The rule, fixed before the runs, is MATCH within 5 %, CLOSE within 20 %, DIVERGENT
otherwise. Paired tests use the per-run mean GOSPA, with a paired t, a 10 000-resample bootstrap CI and Holm over 33
tests per configuration. All compute was CPU (about 25 core-hours, 0 GPU-hours).

| Quantity | Paper | This reproduction | Verdict |
|---|---|---|---|
| Table III, CFG-O, 30 cells | 72.80 (myopic), 47.17-56.18 | 29 MATCH / 1 CLOSE / 0 DIVERGENT, median abs. deviation 0.8 %, max 5.6 % | MATCH |
| Myopic-GD / -KLD, all clutter | 72.80 | 72.80 (every run identical) | MATCH |
| MCTS3-GD, lambda_C = 0.1 / 1 / 2 | 47.17 / 47.88 / 50.53 | 45.39 [44.04, 46.47] / 48.93 [47.60, 49.96] / 49.73 [48.45, 50.63] | MATCH |
| MCTS4-KLD, lambda_C = 2 (worst cell) | 52.36 | 55.28 [53.54, 56.62] | CLOSE (+5.6 %) |
| Table III, CFG-P2 (paper text), 30 cells | as above | 6 MATCH / 18 CLOSE / 6 DIVERGENT; Myopic-GD 41.8 / 67.6 / 78.1 instead of a constant 72.80 | DIVERGENT |
| H1: MCTS-GD beats Myopic-GD | yes | CFG-O 12/12 Holm-significant, diff -21 to -32 units; CFG-P2 12/12 | REPRODUCED |
| H2: MCTS3-GD best GD variant | yes | n = 50: lowest G in 3/3 columns but 4/9 tests significant (MCTS4 not separated) | PARTIAL |
| H2 at n = 200 (extension, P10) | - | 3/3 columns, 8/9 significant | REPRODUCED (extension only) |
| H3: GD beats KLD (authors' KLD form) | yes | CFG-O GD lower in 12/12 cells, 8/12 significant, 0 against; CFG-P2 12/12 | REPRODUCED |
| H3 against the information-gain KLD (CFG-O) | yes | 8/12 in direction, 1 significant | PARTIAL |
| C4: consistent across clutter | range <= 3.5 | CFG-O max range 4.34 (MCTS3-GD) | REPRODUCED |
| C5: non-myopic sensors near the origin at t = 50 (rule P1) | yes | 58 / 82 / 92 % of runs at lambda_C = 0.1 / 1 / 2 (threshold 90 %) | NOT at lambda_C = 1 |
| F-A: removing lookahead / GD objective / search hurts | implied | 9/9 significant in both configurations | CONFIRMED |
| E5: runtime ordering Myopic < MCTS1 < MCTS2 < MCTS3 < MCTS4 | Table IV | same ordering (GD 0.003 to 0.036 s/step on one core) | MATCH (ordering) |
| T1: Lemma-1 bound dominates the true single-Bernoulli cost | yes | 36/36 with exact hypothesis probabilities; the printed eq. 38 product falls below Monte Carlo in 22/36 | HOLDS with exact p(h) |
| T3: reduced tree is pessimistic vs the full tree | implied | 95 % (eq. 38) and 100 % (exact) of 60 cases | HOLDS |

Caveats:
- The 6 myopic cells do not test the planner. HOLD (sensors never move) also gives 72.80. Of the 24 planner-sensitive
  CFG-O cells, 23 MATCH and 1 is CLOSE.
- The paper's KLD columns at lambda_C = 1 and 2 are identical in every row, which looks like a copy error. Eight
  graded KLD cells, including the single CLOSE, are graded against values that may be duplicated.
- H3 depends on the truth. On ten other truths from the authors' generator, GD is lower than KLD in 9/12 cells. With
  the truth as the unit of independence only 1/12 is significant. Those ten truths are a sparse-scene subset (truths
  with more than 9 targets or more than 5 alive at once do not fit the array shapes), not a sample of the generator.
- C5: the declared rule is a poor measure. At the final step both sensors end above the obstacle in 0 of 50 runs for
  every MCTS-GD variant and clutter rate, and both pass below y = 0 in 96-100 % of runs. Myopic sensors end above
  the obstacle in 100 % of runs. The sensors do get through; they just are not near the origin at t = 50.
- The value of search is overstated by the RANDOM control. A search-free, obstacle-aware rule
  (`scripts/geo_greedy_heuristic.py`) reaches G 47.02 / 48.31 / 50.96, against MCTS3-GD 45.39 / 48.93 / 49.73.
  MCTS3-GD is significantly better only at lambda_C = 0.1 (p = 0.006).
- T2: MCTS decision regret does not vanish with budget at the code's UCT constant eps = 0.05 (mean regret about 35
  from B = 98 to 1568). With eps = 300 it falls to about 1.
- Files in `results/` were written before the last revisions. Final verdicts are the ones in this README. In
  particular, the authors'-code z-scores in `authors_zscores.json` subtract our 50-run G from an n = 1-4 run G. G is
  biased down for small n (for example 53.40 at n = 50 vs 49.85 for a single run). Re-centered, the "KLD-side offset"
  disappears (mean z -0.24 ± 0.30). `e1_report.json` holds CFG-P v1, not CFG-P2; CFG-P2 is in `p2_report.json`.

## What differs from the paper / known gaps
- **Why the paper text diverges (CFG-P2).** Making the filter's detection probability the code's hard 40 m field of
  view (D4) is sufficient to restore a clutter-independent myopic result (73.59 at every rate) and brings MCTS3-GD to
  CLOSE. Of 16 one-at-a-time flips on CFG-O, only the detection model (D3, D4) is material. Two other readings of
  the text were found but not run as declared fixes: the clutter density outside the field of view, which is our
  assumption (a zero density makes Myopic-GD clutter-independent at about 41), and the exact eq. 48 expectation
  instead of p_D at the mean (moves cells by up to 15 units). Eq. 48 inside a 40 m field of view is a second
  text-grounded reading under which the paper's numbers are largely recovered. The CFG-P2 verdict is therefore
  "DIVERGENT, with these alternatives untested", not "explained".
- **Assumed, because the paper does not say:** the CFG-P2 KLD form, KL(predicted || node-merged Bernoulli); the
  estimator threshold 0.5; the UCT constant 0.05 (both from the code); the iteration cap 2B. Every config value and
  its source is in `src/configs/sources.yaml`.
- **Determinism.** CFG-O is bit-reproducible across batch shapes. The CFG-P KLD arms are not: far from all targets
  every root child's KLD cost is zero up to rounding, so the argmin depends on XLA's batch-dependent last bit. The
  statistics are unaffected.
- **Replicated code quirks** (CFG-O): uncapped merged existence r > 1 in the MCTS blocks (D21) caused 2 NaN planning
  steps in MCTS1-KLD over 150 lanes. It was immaterial when flipped (S3).
- **Not run:** the GPU path; Lemma 1 beyond the single-Bernoulli, no-clutter case; several CFG-P2 follow-up cells
  (listed in `scripts/run_all.sh`); the P9 divergence arms on CFG-P2 (flown on CFG-P v1 only); the KLD blocks of the
  authors' `main.py` were never replayed bit for bit (only GD blocks and Myopic-KLD).

Paper text vs the authors' code (D = difference id, used in code comments):

| D | Topic | Paper | Authors' code (= CFG-O) |
|---|---|---|---|
| D1 | birth existence r_b | 0.03 | 0.05 |
| D2 | measurement noise R | 2 I | 3 I |
| D3 | truth detection | p_D = 0.999 exp(-(d/40)^2 / 2), eq. 48 | 0.999, then measurements > 40 m from the sensor deleted |
| D4 | filter p_D | eq. 48 at the predicted mean | 0.999 if the mean is within 40 m, else 0 |
| D6, D7 | clutter | Poisson in the field-of-view disc | Poisson in the 80x80 square, cropped to the disc |
| D8, D9 | planner state | full NCV model, tr(P) | position random walk, position trace |
| D10 | first planned step | cost on the predicted density | root predicted a second time |
| D11 | p(h) for two sensors | eq. 38 product | sequential (equals the exact joint) |
| D12 | KLD objective | sum of Bernoulli KLDs | Gaussian part only; the joint tree multiplies by the sum of existences |
| D13 | UCT constant | unstated | 0.05 |
| D14 | backed-up value | eq. 45 | includes the selected node's running mean |
| D15 | budget | nodes added | MCTS iterations |
| D16 | expansion order | random | deterministic, last action first |
| D20-D22 | filter details | LBP, merge if Mahalanobis < 1 | tolerance 1e-4, merge capped at r = 1 only in myopic blocks, prune r < 1e-4 |
| D23 | estimator | r > Gamma_d | r > 0.5 |
| D24 | GOSPA base metric | not stated | Euclidean on the 4-D state, velocity included |
| D25 | Table III statistic | "RMS-GOSPA across all time steps" | mean over steps of the RMS over runs |
| D30 | LBP | run to convergence | one sweep (`mu_old = mu` aliasing in `lbp.py`) |

D5, D17-D19, D26-D29 agree or are unstated in the paper and follow the code in both configurations (initial
sensors (5, 150) and (-5, 150), independent per-sensor trees, rollout uniform over feasible actions).

## Layout
- `src/model.py` scenario constants, CFG-O / CFG-P / CFG-P2, the algorithm table (Table II), Table III values.
- `src/data.py` the authors' fixed truth (sha256-checked) and the measurement simulator.
- `src/filter.py` Gaussian multi-Bernoulli filter: predict, per-sensor update (LBP + TOMB), merge, prune, estimate.
- `src/gospa_jax.py` exact GOSPA (p = 2, alpha = 2) with its localization / missed / false split.
- `src/planner.py` reduced-tree node cost (GD bound, KLD), synthetic multi-sensor update, MCTS.
- `src/algorithm.py` lane-vmapped closed-loop step, PRNG key tree, health counters, test injection hooks.
- `src/train.py` experiment driver with orbax checkpoint / resume; `src/eval.py` Table III statistic, grading, tests.
- `src/configs/sources.yaml` every config value with its source (paper section, code, or ASSUMED).
- `data/truth_seed123546.npy` the authors' truth (9 targets x 4 x 200), from their generator with their seed 123546;
  it reproduces all 16 count changes of Figure 3. `data/truths_strut.npz` ten truths from their generator (seeds
  1-60, first ten that fit; `scripts/official/gen_truths.py` + `scripts/build_strut.py`).
- `scripts/run_all.sh` every closed-loop run and report, sequentially. `scripts/merge_metrics.py` folds per-algorithm shards.
- `scripts/report.py` all E1 / sensitivity / S-TRUTH / divergence / H2 tables; `scripts/arms.py` the F-S and F-T arms.
- `scripts/tables.py`, `scripts/tables_extra.py`, `scripts/figs.py` markdown tables and figures from the report JSON.
- `scripts/theory_t1.py`, `scripts/theory_t2_t3.py` T1-T3; `scripts/e3_snapshot.py` E3; `scripts/e5_timing.py` E5.
- `scripts/geo_greedy_heuristic.py` the search-free navigation control.
- `scripts/followup/` CFG-P2 Table III (`cfg_p2_table.py`), its sensitivity flips (`sensitivity.py`), authors'-code z-scores (`authors_zscores.py`).
- `scripts/official/` `setup.sh`, `run_reference.sh` (+ `run_main.py`, `summarize_official_reference.py`), the fixture
  recorders `record_official_fixtures.py` and `trace_main.py`, `gen_truths.py`, `official_reference.py`.
- `results/` report JSON (E1, CFG-P2, sensitivity, S-TRUTH, divergence, H2, T1-T3, E3, E5, authors' reference
  summary) and `results/figures/` (Table III, Figure 5 decomposition, Figure 4 analogue).

## Run it
From this folder, with the root environment (`uv sync` at the repo root). CPU is the intended device.

    JAX_PLATFORMS=cpu uv run python -m src.train --cfg CFG-O --algs Myopic-GD MCTS3-GD --runs 10 --clutter 0.1 1 2 --tag small
    uv run python scripts/merge_metrics.py runs/small
    uv run python -m src.eval --tag small --cfg CFG-O --runs 10 --algs Myopic-GD MCTS3-GD
    sh scripts/run_all.sh                 # everything, about 25 CPU core-hours sequentially

Outputs go to `$GOSPA_RUNS` (default `runs/`, git-ignored). Overrides for sensitivity rows: `--set R_var=2.0`,
`--set merge_cap=always`, `--set kld_mode=info_gain`, and so on. Speed on one core with one lane: 0.003 s/step
(Myopic) to 0.036 s/step (MCTS4-GD) after a 5-13 s compile. A 150-lane E1 cell (50 runs x 3 clutter rates) takes
minutes to about half an hour. Float64 throughout (`jax_enable_x64` is asserted).

## Tests
`uv run pytest` (about 5 minutes on one CPU core, 2 GB): 87 pass; without the authors' code 27 live tests skip.
- `test_toy.py`: the exactly computable toy against hand-derived numbers and an independent NumPy oracle
  (`oracle_toy.py`). One-step bound 65.820022 with the paper's p(h) vs 82.192424 exact, merged existence 0.607143 vs
  0.5, two-step reduced vs full tree 133.646856 vs 86.895230.
- `test_search_oracle.py`: MCTS against exhaustive enumeration of the reduced tree.
- `test_official_parity.py`: replays fixtures recorded from the authors' code (`tests/fixtures/official/`), no checkout
  needed. Covers node costs, legal actions, the filter step, GOSPA, 16 full MCTS decisions, the measurement
  generator, and a replay of the authors' unmodified `main.py` (seed 123546, lambda_C = 2, four blocks, 200 steps).
- `test_filter_exact.py`, `test_filter_lbp.py`: LBP against exact enumeration of the association.
- `test_data.py`: truth checksum and the 16 count changes of Figure 3 (E4).
- `test_grads.py`: finite-difference checks of the smooth building blocks (there is no loss to differentiate).
- `test_switches.py`: the follow-up switches and the checkpoint layout key; defaults leave CFG-O / CFG-P unchanged.
- `test_smoke.py`, `test_cfgp_step.py`, `test_resume.py`, `test_strut.py`: shapes, counters, resume equality, truth set.
- Live differential tests that import the authors' code: `test_planner_node_parity.py`, `test_filter_parity.py`,
  `test_gospa_parity.py` (also a Stone Soup 1.9.1 cross-check), `test_mcts_parity.py`, `test_closed_loop_parity.py`.
  They skip unless `sh scripts/official/setup.sh` has been run (or `OFFICIAL_ROOT` points to a clone).

## Checked against
The authors' code at commit `b5959f6a696a0e566e0925ec2376be361982e773`, run unmodified in its own venv (Python 3.12,
NumPy 1.26.4, SciPy 1.11.4, Matplotlib 3.8.4). `sh scripts/official/setup.sh` puts the clone in `$OFFICIAL_ROOT/repo`
(default `external/official`) and builds that venv.
- On identical inputs our JAX code matches theirs to machine precision: node costs 1.3e-13, filter step 1.8e-14,
  GOSPA 3.9e-16, MCTS decisions 4.2e-15 (same chosen action). Replaying their `main.py` run gives identical sensor
  tracks and per-step GOSPA^2 to 6.1e-15.
- Their `main.py` on our hardware (`scripts/official/run_reference.sh`, n = 3-4 runs per GD cell, 1 per KLD cell)
  reproduces Table III at 19 MATCH / 11 CLOSE / 0 DIVERGENT. With the paper's stated r_b = 0.03 and R = 2 I, their
  code gives Myopic 73.59, exactly what ours gives with the same two overrides.
- GOSPA also agrees with Stone Soup 1.9.1 to 1e-9.
