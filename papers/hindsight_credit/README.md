# Hindsight Credit Assignment
Anna Harutyunyan, Will Dabney, Thomas Mesnard, Nicolas Heess, Mohammad G. Azar, Bilal Piot, Hado van Hasselt,
Satinder Singh, Greg Wayne, Doina Precup, Rémi Munos, NeurIPS 2019. Paper: https://arxiv.org/abs/1912.02503 (v1).
Official code: [hca-neurips2019/hca](https://github.com/hca-neurips2019/hca) (commit `3a564ad`). It contains only
the per-episode estimator: one 58-line numpy file, `hca_classes.py`, with no environments, runner, learning rates,
return bins or policy-gradient baseline.

## What this is
The paper rewrites the value function in terms of a hindsight distribution h(a | x, future), which is the
probability that action a was taken given what happened afterwards. It derives two policy-gradient estimators from
this view: State HCA conditions on future states and Return HCA conditions on the return. On small tabular tasks the
paper reports that both estimators assign credit better than an actor-critic policy gradient (PG). This folder
reimplements both estimators, the PG baseline, the paper's three tasks (Shortcut, Delayed Effect, Ambiguous Bandit)
and exact dynamic-programming / enumeration oracles in float64 JAX. Every figure experiment is rerun at the paper's
scale of 100 runs per curve.

Labels used in code and tables:
- Experiments: E1 = Fig 3 left (Shortcut learning curves). E2 = Fig 3 right (counterfactual advantage estimates).
  E3 = Fig 4 left (Delayed Effect, n = 5, 3-step bootstrap). E4 = Fig 4 middle (n = 3, σ = 2). E5 = Fig 4 right
  (noise sweep σ = 0-5). E6 / E7 = Fig 5 left / middle (Ambiguous Bandit, observed / hidden). E8 = Fig 5 right
  (hidden bandit, ε sweep).
- P1-P10: the ten pre-declared paired ordering tests, each "HCA arm minus PG" in the paper's direction (Holm over
  10, 100 paired runs, seed 0; seeds 1 and 3 replicate).
- T0-T3: theory claims tested empirically. T0 / T2 = the exact estimators are unbiased. T1 = State HCA is immune to
  exogenous reward noise. T3 = the learned-h transient under bootstrapping.
- S-*: sensitivity runs (for example S-BIN return-bin layouts, S-ORDER update order, S-HFORM / S-HPRIOR / S-HLR
  hindsight-learner readings, S-E1START, S-PGPAPER, S-AVG).
- A-*: assumptions for details the paper leaves open. Each is tagged on the config value it sets in
  `src/configs/default.yaml` / `sources.yaml`. The main ones are A-BINS (return-bin centres), A-LR (which learning
  rate is which), A-INIT (uniform initial h), A-SEG (segment semantics of the 3-step return), A-UPDATE (one summed
  update per episode from the pre-update tables), A-AVG (curve mean as the "average value" metric), A-F3R (the
  Fig 3-right protocol), A-PG-LR (PG learning rate tuned on separate seeds) and A-AB-EPS (bandit crossover ε).

## Results
100 runs per curve (seed 0), with replication seeds 1 and 3. PG's learning rate was tuned per panel on a separate
seed (999) over the paper's grid {0.1, 0.2, 0.3, 0.4}. The judgment rule was declared before any run. MATCH means
normalized error |ours − paper| / (optimum − start) ≤ 5 % or within the read-off precision plus 2 SE of the paper's
band. CLOSE means ≤ 20 %, and anything else is DIVERGENT. A relative-error rule (|Δ| / |paper|) is reported beside it.
"Authors' code" is the released snippet run unmodified inside our numpy port of the unreleased parts (100 runs).

**Overall verdict: partial reproduction.** The mechanism and theory claims reproduce. Most quantitative curves do
not, and the authors' own estimator gives our numbers rather than the paper's. Five of the ten comparative claims
reverse against a tuned PG.

| Quantity | Paper | Authors' code | This reproduction (mean ± SE, n = 100) | Verdict |
|---|---|---|---|---|
| P2: E2 counterfactual estimate, State HCA closer to the oracle than PG (p = 0.9) | yes | - | +2.156 ± 0.005, 100/100 wins | REPRODUCED |
| P3: E3 State HCA − PG, final (3-step bootstrap) | yes | - | +0.888 ± 0.019, 100/100 wins | REPRODUCED |
| P4: E3 Return HCA − PG, final | yes | - | +0.979 ± 0.019, 100/100 wins | REPRODUCED (but see caveats) |
| P1: E1 State HCA − PG, ep 100 | > 0 | - | −0.014 ± 0.002 | READING-DEPENDENT |
| P5: E4 State HCA − PG, curve mean | > 0 | - | +0.005 ± 0.012 (n.s.) | READING-DEPENDENT |
| P6: E5 σ = 5 State HCA − PG, curve mean | > 0 | - | +0.129 ± 0.088 (n.s.) | READING-DEPENDENT |
| P7: E6 State HCA − PG, ep 100 | > 0 | - | −0.064 ± 0.024 | REVERSED, FRAGILE |
| P8: E6 Return HCA − PG, ep 100 | > 0 | - | −0.190 ± 0.021 | REVERSED |
| P9: E7 Return HCA − PG, ep 100 | > 0 | - | −0.186 ± 0.020 | REVERSED |
| P10: E8 ε = 0 Return HCA − PG, curve mean | > 0 | - | −0.325 ± 0.007 | REVERSED |
| E3 State HCA final (3-step) | 0.68 | 0.90-0.92 | 0.902 ± 0.007 | DIVERGENT |
| E4 State HCA final | 0.84 | 0.954 ± 0.028 | 0.994 ± 0.000 | final MATCH/CLOSE; early curve (ep 50, 100) DIVERGENT |
| E5 σ = 5 State HCA curve mean | 0.66 | 0.691 ± 0.061 | 0.680 ± 0.064 | MATCH, conditional on A-INIT |
| E5 σ = 5 Return HCA curve mean | 0.47 | 0.120 ± 0.090 | −0.044 ± 0.087 | DIVERGENT |
| E6 State / Return HCA final | 1.846 / 1.773 | 1.807 / 1.651 | 1.766 ± 0.017 / 1.640 ± 0.014 | State CLOSE/MATCH; Return DIVERGENT |
| E7 State / Return HCA final | 1.495 / 1.765 | 1.515 / 1.663 | 1.501 ± 0.025 / 1.661 ± 0.010 | State MATCH; Return DIVERGENT |
| E8 ε = 0 Return HCA curve mean | 1.97 | 1.595 ± 0.006 | 1.595 ± 0.006 | DIVERGENT |
| E2 State / Return HCA bar, p = 0.1 | 0.45 / 0.21 | 0.994 / 0.994 | 0.935 / 0.990 | DIVERGENT |
| E1, E6, E7 PG final | −0.179 / 1.734 / 1.719 | no official PG | −0.113 / 1.830 / 1.847 | PG faster than the paper's |

Value-table totals: 38 of 84 rows DIVERGENT under the declared rule (42 under the relative rule). Excluding the 9
non-comparable E1 rows, 32 of 75 are DIVERGENT. Every DIVERGENT group is still labeled "check incomplete", because
each depends on a piece the authors did not release.

What reproduces, tightly:
- The exact hindsight estimators are unbiased (T0, T2). State HCA is immune to exogenous reward noise, while PG's
  variance grows as σ^1.998 (T1).
- HCA's counterfactual advantage estimates beat PG's at every p_long (P2 plus 14 of 14 secondary comparisons, all
  p < 1e-100). PG's E2 bars match the paper's 7 of 7 under the declared rule.
- With a 3-step bootstrap through an aliased chain, State HCA learns and PG cannot (P3). The effect is the
  learned-h transient (T3), and it holds in both update orders and under sliding-window semantics.
- State HCA fails exactly when the successor state is hidden (E7). Removing the hindsight mechanism removes all
  learning (ablation).

Caveats:
- **P4 compares unlike returns.** Return HCA never bootstraps (A-RET-MC), so P4 compares a Monte-Carlo return with
  PG's 3-step bootstrap. Against an equal-horizon tuned PG, Return HCA loses (−0.0048 ± 0.0003, 0/100 wins). P4 is
  kept as REPRODUCED because the paper makes the same comparison.
- **READING-DEPENDENT (P1, P5, P6)** means the verdict depends on how h is parameterized and estimated, which the
  paper does not pin down. P1 is REVERSED with the primary logit-SGD h and not significant with maximum-likelihood
  count h (5 of 5 seeds). P5 and P6 are not significant at the primary setting and REPRODUCED under a "residual" h
  prior (h = softmax(policy logits + ψ)) on 5 of 5 seeds; P5 also under a probability-space EMA h.
- **P7 is FRAGILE.** It is reversed on only 3 of 5 seeds at the primary setting and not significant on 4 of 5 at
  α_h ≥ 0.8 or under EMA h. It turns positive on 5 of 5 seeds against a PG matched to the paper's own PG curve
  (lr 0.05, no critic).
- **P8-P10 reversals are robust.** They hold under every h variant tried (α_h 0.1-1.6, swapped learning-rate
  reading, counts, EMA, residual), 5 seeds, exact (oracle) h_z, all 5 bin layouts × 2 update orders, ε = 0 and the
  paper-matched PG. The P10 reversal persists even with exact hindsight (−0.118 ± 0.014) and is unexplained.
- **E1 absolute values are not comparable.** The paper's start level is −1.80 and ours (uniform policy) is −0.963.
  The −1.80 start corresponds to a long-action logit of +1. On the start-independent gap fraction, our State HCA
  matches the paper's progress (0.64 / 0.92 / 0.97 vs 0.69 / 0.98 / 1.00 at episodes 20 / 50 / 100). The P1
  reversal comes from a much faster PG (0.91 vs 0.55 of the gap at episode 20), not from a slower State HCA.
- **Return HCA levels depend on the bin layout,** which the paper does not give. E4 final ranges 0.75-0.92 and E5
  σ = 5 ranges −0.04 to 0.35 across 5 layouts × 2 update orders.
- **The E2 HCA bars** are about half of what the estimators converge to. The paper's HCA bars have PG-like spread,
  which no online-h variant we tried produces. The plotted statistic or h model behind Fig 3 right is unidentified.
- **No single update order fits every panel.** Primary arms use the official snippet's order (every direction from
  the pre-update tables). The paper's Algorithms 1/2 imply training h first. That order reproduces Fig 5 right's
  slight State-HCA learning but also makes E7's State HCA learn, which the paper does not show.
- **Results files and revisions.** `results/full_report*.json/txt` and `results/sensitivity_report*` predate the
  final recheck. The verdict column above is the final one; `results/recheck_tables.md` and
  `results/recheck_revised.json` hold the revisions (P1, P5, P6 to READING-DEPENDENT; P7 to FRAGILE; E8 ε = 0.4 PG
  from MATCH to CLOSE).

## What differs from the paper / known gaps
- **Unreleased pieces were reconstructed** from the paper text: environments, episode sampler, how learning rates
  apply the summed gradients, return bins, PG baseline and critic, the Fig 3-right protocol and the "average value"
  metric. Each choice is an A-* assumption in `src/configs/sources.yaml`. Every remaining divergence sits in one of
  these pieces.
- **Paper read-offs.** Fig 5 values come from the PDF's vector graphics (`data/fig5_vector.json`). Fig 3-right error
  bars are digitized as a 95 % CI over the 1000 rollouts of one repetition (`data/fig3r_eb_full.json`). Figs 3-4
  are read at ±0.03.
- **Not reproduced:** the E8 Return level of 1.97 is not reached under any metric reading or policy lr tried. Even
  exact continuous h_z gives only 1.80. The paper's PG curves are not reproduced by any learning rate on the paper's
  grid, while lr 0.05 with no critic fits Fig 5's PG to rmse 0.015 / 0.042.
- **Not shipped:** per-run curves (`curves.npz`) and per-seed metrics are regenerated by `scripts/run_all.sh`. The
  ablation and divergence-probe JSON files are large and are not in `results/`. Their numbers are summarized above.

## Layout
- `src/data.py`: Shortcut, Delayed Effect and Ambiguous Bandit, the Theorem 7 counterexample, the episode sampler
  (common random numbers across arms) and an env checksum.
- `src/model.py`: tabular policy, hindsight, value and reward tables.
- `src/algorithm.py`: per-episode State HCA (official and Algorithm 1 forms, Monte Carlo or T-step), Return HCA and
  n-step actor-critic update directions, plus the Fig 3-right counterfactual estimators.
- `src/train.py`: vmapped runs (one jitted call per arm), the PRNG key tree and metrics writers.
- `src/eval.py`: DP values (the learning-curve metric) and exact enumeration of hindsight distributions and expected
  updates.
- `src/configs/default.yaml`, `sources.yaml`: every hyperparameter with its source.
- `data/`: numbers read from the paper's Fig 5 vector graphics and the Fig 3-right error bars.
- `scripts/run_all.sh`: every run behind `results/`, in order.
- `scripts/exact_checks.py`: 16 correctness checks against exact enumeration.
- `scripts/small_scale.py`: all experiments at 20 runs (seed 2).
- `scripts/full.py`: PG lr tuning, paper-scale evaluation, ablations, theory checks, divergence probes, cost.
- `scripts/report.py`: paper / authors' code / ours tables, judgments and the pre-declared paired tests.
- `scripts/sensitivity.py`, `scripts/sensitivity_report.py`: sensitivity runs requested in review and their tables.
- `scripts/recheck.py`, `scripts/recheck_report.py`: hindsight-learner readings on 5 seeds, the paper-matched PG
  and the E1 start, and the revised judgments.
- `scripts/diff_official.py`: per-episode differential test against the authors' snippet.
- `scripts/official_lockstep.py`: end-to-end lockstep learning against the snippet. It also records the test
  fixtures.
- `scripts/official_reference_table.py`: paper / authors' code / ours at small scale.
- `scripts/probes/`: early exact-enumeration probes (Shortcut, bandit, Theorem 7 counterexample, oracle fit).
- `scripts/official/`: `setup.sh` fetches the authors' code and builds its venv (see Checked against).
  Drivers that run inside the authors' numpy venv. `run_official.py` runs the snippet on
  given episodes. `ref_driver.py` wraps the snippet in a numpy port of the unreleased parts for lockstep and
  paper-scale reference runs.

## Run it
From this folder, with the root environment (`uv sync` at the repo root):

    uv run python -m scripts.exact_checks          # ~4 min
    uv run python -m src.train --exp E1            # one experiment (E1..E8 or all), 100 runs, seed 0
    sh scripts/run_all.sh                          # everything behind results/

Outputs go to `runs/` (git-ignored). Everything is tabular. The original runs took about 27 minutes of driver wall
clock on one RTX 3090 (shared), and the largest single run was 319 s. The official reference runs took about 9.5
minutes on CPU. Float64 throughout (`jax_enable_x64` is set in `src/__init__.py`), because the exact checks and the
official parity compare to 1e-10 or tighter.

## Tests
`uv run pytest` (about 1 min on CPU): 33 pass, and 3 skip when `runs/seed_0/curves.npz` is absent. To make them
run, regenerate it with `uv run python -m scripts.full tune` and then `uv run python -m scripts.full eval --seed 0`
(about 22 s and 61 s on the original GPU; longer on CPU).
- `test_smoke.py`: env shapes, one episode, one update per arm, tiny vmapped runs, sensitivity options.
- `test_grads.py`: gradient check of the DP value, exact unbiasedness of State HCA, hand-written updates against
  autodiff, and the sliding-window estimator.
- `test_official_parity.py`: replays `tests/fixtures/official_*.npz`, which were recorded from the authors' snippet.
  It checks per-episode gradients to 1e-10 and lockstep learning curves to 1e-9. No external data is needed.
- `test_recheck_options.py`: recheck runner options. The defaults reproduce the seed-0 evaluation curves bit for bit
  (skipped without `runs/seed_0/curves.npz`, see above). The residual h equals the primary when the policy is frozen.

## Checked against
The authors' snippet at commit `3a564ad6438df2b1c68ca75a91d713d87d56453a`, run unmodified in its own numpy-only venv
(Python 3.12, numpy 2.5.4). `sh scripts/official/setup.sh` clones
hca-neurips2019/hca at that commit (unpatched) into `$OFFICIAL_ROOT/repo/` and builds that venv as
`$OFFICIAL_ROOT/venv/`; the default `OFFICIAL_ROOT` is `external/official` inside this folder. The scripts under
`scripts/official/` and `scripts/diff_official.py` / `scripts/official_lockstep.py` use them; no test does. The
setup script has not been run end to end in this form; the pinned commit was checked to exist.
- Per-episode gradients agree to 7.1e-15 (`results/diff_official.json`).
- End-to-end lockstep learning, 32 cases: curves agree to 4.1e-13 and returns are bit-identical
  (`results/official_lockstep.json`).
- The 3-step State HCA composed from the snippet agrees to 1.8e-15 per episode and 7.8e-16 on lockstep curves.
- A literal transcription of Algorithm 1 agrees to 5.3e-15.
- PG has no official code. An independent numpy port agrees to 2.1e-15.
- Wherever the authors' estimator can run, it lands on our side of the paper by about the same amount
  (`results/official_reference_table.json`).
