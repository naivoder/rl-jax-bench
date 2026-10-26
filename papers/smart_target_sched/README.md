# Non-myopic Beam Scheduling for Multiple Smart Target Tracking in Phased Array Radar Network
Yuhang Hao, Zengfu Wang, José Niño-Mora, Jing Fu, Min Yang, Quan Pan, arXiv preprint (eess.SY) 2023.
Paper: https://arxiv.org/abs/2312.07858 (v1). Official code: none released for this paper; the authors' published
output (Tables I-IV as typeset and the vector figures in the arXiv TeX source) was used as the oracle.

## What this is
The paper schedules the beams of a phased array radar network over many "smart" targets. A smart target reacts to being
illuminated: it shifts toward a maneuvering motion mode, which raises its tracking error covariance (TEC). The problem is
cast as a restless multi-armed bandit with a discounted TEC cost. The paper proposes a Whittle-type marginal-productivity
(MP) index policy computed from truncated deterministic covariance rollouts. It claims the index beats myopic and TEV/TEC
rules in every setting and is near-optimal against a Lagrangian lower bound. This folder reimplements the target model,
the index policy, the baselines and the bound in JAX (float64). Nothing is learned: every number comes from Monte-Carlo
replications over the random initial covariance.

Labels used in code and tables:
- Tables I-IV: index / myopic / TEV costs for K = 1, 2, 3 (27 cells, 81 numbers). Figs 2-4: g, f and the MP index.
  Figs 5-6: gap to the Lagrangian bound. Figs 7-8: objective per target in the 4-D (matrix) case.
- Experiment ids E3-E10 and X1-X3, G1, G2 in `src/data.py` are labels in the PRNG key tree. X1 is the PCLI3 check, which
  the paper never runs. X2 adds our comparators. X3 draws the reaction mode instead of averaging it.
- The primary family is 54 paired tests: index vs myopic and index vs TEV in every table cell, Holm-corrected. S1-S7 are
  secondary families (S1 passive-continuation, S2 one-step rollout, S3 sampled modes, S4 ablation, S5 Figs 6-8 per K,
  S6 independent resample, S7 tuning parity). T1-T5 are hypothesis tests for the Figs 5-6 gap level (`scripts/bound_tests.py`).
- Run tags: `small` (Nmc 20 / 10), `full` (Nmc = 100, seed 0, the binding set), `full_s1` (seed 1, independent resample),
  `full_figs`, `full_x4` (X3 on Figs 5-6), `full_theory`, `review` (base-length grid fill and reruns), `recheck`.

## Results
100 replications per cell (the paper's Nmc). The judgment rule was fixed before the runs: MATCH means within 5 %, CLOSE
means within 20 %, and DIVERGENT means anything else. Gaps are compared in percentage points.

| Quantity | Paper | This reproduction (n = 100 replications) | Verdict |
|---|---|---|---|
| Tables I-IV, 81 costs | as printed | max abs deviation 0.60 %, median 0.16 %; resample (seed 1) max 0.69 % | MATCH 81/81 |
| Index beats myopic and TEV/TEC in every cell (54 paired tests, Holm) | yes | 54/54, largest Holm p 1.2e-3; advantage 0.36-5.24 % (paper 0.50-5.24 %); resample 54/54 | REPRODUCED |
| Figs 2-3 (g, f) and Fig 4 (MP index) vs the authors' plotted curves | curves | g 2.7e-5, f shape 2.8e-5 of range, mp* within 0.02 % of max | MATCH |
| G(P, z) vs z (authors' figure, 8 curves) | curves | max abs 4.8e-5 / 8.8e-5 away from jumps | MATCH |
| PCLI1 and PCLI2 hold (scalar) | yes, from plots | 0 violations in 117 000 evaluations, Q^CT = 2..40; PCLI3 holds to 1e-8 per jump; MP index = Whittle index by value iteration (3.5e-4) | REPRODUCED |
| Figs 5-6 ordering of the gaps (index < myopic, TEV) | yes | same sign as the authors' points in 102/102 K-comparisons | REPRODUCED |
| Figs 5-6 differences between the policies' gaps | - | 0.02-0.03 pp (Fig 5), 0.17-0.19 pp (Fig 6) from the authors' | MATCH |
| Figs 5-6 gap level: "about 3.0 %" (Fig 6), "lower than 10.5 %" (Fig 5) | 3.0 % / < 10.5 % | 6.9 % (Fig 6); 11.0-11.6 % at K = 98 (Fig 5); +3.8 to +4.2 pp (Fig 6), +0.55 to +4.5 pp (Fig 5) | DIVERGENT, near-optimality claims NOT reproduced |
| Figs 7-8, 42 plotted points | as plotted | max abs deviation 0.13 %, median 0.03 %; 28/28 paired tests | MATCH 42/42 |

Caveats and findings beyond the paper:
- **The gap level is unresolved.** The offset does not depend on the policy and is additive in the bound (about 1.2 per
  unit weight per target). Three independent computations of our dual V^D agree. Our V^D is the optimum of the
  relaxation as printed (eqs 13-20). On small instances with a computable exact optimum it lies 7.5-13.1 % below J*,
  while the index lies 0.3-2.0 % above J*. So our bound is correct but loose. The authors' implied bound is higher and
  still a valid lower bound, but it is not the optimum of that relaxation. A coarse "ceil" state grid of step about
  0.075 fits the authors' Fig 5 values within 0.7 %. It misses Fig 6 by 0.7-0.9 % in all 6 cells (10 replications,
  paired SE 0.0009-0.0021), and no single step fits all four scenarios. That fit is post hoc and assigns no mechanism.
  At the paper's stated grid step 0.01, no mapping or cap reproduces the authors' value.
- **One-step search.** Tuned on the resample and graded on the binding set, the index beats a tuned passive-continuation
  rule in 27/27 cells. Against a one-step rollout with a tuned base length (grid {1,2,3,5,10,30,100}), the index is
  cheaper in only 1/27 cells, dearer in 19/27 and not separated in 7/27 (mean -0.43 %). The index beats the no-search
  rules, not a short one-step search. An earlier grid {5,10,30,100} had reported 16 cheaper / 11 dearer.
- **Pricing the reaction (ablation).** An index that ignores the reaction costs +0.29 % (always calm) or +0.84 % (always
  maneuvering) more, in 27/27 cells. That is 13-34 % of the index's advantage over myopic. Most of the gain comes from
  the long horizon: tau <= 10 costs +2.4 to +2.7 %.
- **X3 (mode drawn, not averaged).** The index advantage survives in 54/54 comparisons (0.97-4.67 %). The environment
  filter in X3 is mode-matched, so this holds only for a tracker that knows the drawn mode. A mode-ignorant (IMM-like)
  tracker was not built.
- **Small instances.** The index's 0.3-2.0 % regret vs the exact optimum is mostly early-decision suboptimality. It is
  not an end-of-horizon effect in 3 of 4 instances.
- PCLI1 is flagged 8 times in 1.7e6 index evaluations, all in the 4-D case, where the paper itself has no theory.

## What differs from the paper / known gaps
- Unstated choices are tagged ASSUMED in `src/configs/sources.yaml`. They include: beta, tau and slots in §IV-D;
  omega in radians; reckless-first target order; K/N = 1/4 for Figs 7-8; the plotted K grids, read from the vector
  figures; the uniform random tie-break; and the bound's state grid (0.01 to 20, geometric to 1e6).
- Eq (37) is used as printed. The self-consistent coordinated-turn matrix runs as variant `--ft standard`. The 4-D
  match supports the q^CT channel and rejects omega in degrees-as-radians. It cannot tell eq (37) with 3 deg from no
  turn at all.
- `P0 = R0' R0` sometimes has a smallest eigenvalue near 1e-10 (Fig 7, large N). It is logged and not redrawn, because
  the paper does not say whether it redraws.
- The per-slot Lagrangian bound was computed at K = 2 only. The authors' figure code for Figs 5-6 is not available,
  so a cost-side offset in their code cannot be excluded. Only ours is excluded.
- `results/tables/` holds the report output of the full-scale run plus the review reruns (`review_*`). A later
  rechecking pass re-graded the ceil-grid hypothesis in `bound.md` / `bound2.md`, using 10 instead of 3 Fig 6
  replications. The final verdict is the one above; those two files predate it. No other number changed.

## Layout
- `src/model.py`: target model, constants, the TEC map phi (eqs 3-7), scalar and 4-D banks, frozen-trajectory checksum.
- `src/algorithm.py`: index policy (Algorithm 1), myopic (eq 34), TEV/TEC (eq 33), our comparators (passive continuation,
  one-step rollout, random, exact enumeration), ablation arms, the episode simulator.
- `src/data.py`: scenario registry for Tables I-IV and Figs 5-8, and the PRNG key tree.
- `src/bound.py`: Lagrangian dual lower bound, two independent computation paths.
- `src/pcl.py`: exact PCLI3 check (X1). `src/oracle.py`: independent literal-equation numpy reference.
- `src/train.py`: experiment driver (cells resumable, writes `runs/<tag>/seed_<r>/metrics.json`). `src/eval.py`: tables vs the paper, paired tests, Holm.
- `src/configs/sources.yaml`: every setting with its source.
- `scripts/small_all.sh`, `scripts/full_all.sh`: the small-scale and full-scale run lists (sequential).
- `scripts/preflight.py`: mechanical protocol checks before a full run. `scripts/toys_report.py`: toy-check numbers.
- `scripts/small_summary.py`: comparators, variants and timing of the small run. `scripts/report.py`: all full-scale tables.
- `scripts/theory_checks.py`: PCLI1/2 sweep, value-iteration Whittle index and indexability, PCLI3 per jump, exact optimum.
- `scripts/bound_tests.py`: T1-T5 hypotheses for the gap level. `scripts/figs234.py`: Figs 2-4 vs the authors' curves.
- `scripts/recheck_perslot.py`, `scripts/recheck_ceil_paired.py`: per-slot bound on 10 replications and grid resolution; paired ceil-grid ratios.
- `scripts/review/`: `bound_validity.py` (exact optimum vs both duals), `fig6_meanof.py` (second gap definition),
  `x3pair.py` (paired X3), `ptev.py` (two more cheap rules), `exact_idle.py` (regret by slot), `rollout1_boundary.py` and
  `recompute.py` (one-step rollout grid re-grading).
- `scripts/extract_fig_vectors.py`, `scripts/make_official_fixtures.py`, `scripts/parse_tex_tables.py`: rebuild the
  authors'-output fixtures from the arXiv source (need pymupdf: `uv sync --extra pdf`). `scripts/make_reference_traj.py`: frozen checksum trajectories.
- `scripts/official/setup.sh`: fetches the arXiv TeX source into `$OFFICIAL_ROOT` (default `external/official`).

## Run it
From this folder, with the root environment (`uv sync` at the repo root). Everything runs on CPU:

    uv run python -m pytest                                  # ~2 min
    uv run python scripts/toys_report.py
    sh scripts/small_all.sh scalar && sh scripts/small_all.sh matrix && sh scripts/small_all.sh figs
    uv run python src/eval.py --tag small && uv run python scripts/small_summary.py small
    uv run python scripts/preflight.py
    sh scripts/full_all.sh                                   # about 6-7 core-hours
    uv run python scripts/theory_checks.py A B C E
    uv run python scripts/bound_tests.py T1 T2 T3 T5 T5b T5c # about 1 core-hour with theory_checks
    uv run python scripts/report.py tables family secondary s5 s5b figs56 figs78 ablation tuning timing theory bound bound2
    uv run python scripts/figs234.py && uv run python src/eval.py --tag full

Outputs go to `runs/` (git-ignored); tables go to `runs/tables/`. Each cell writes per-replication costs, PCLI1 flags,
the lower bound and timings. Every entry point first checks the JAX model against frozen numpy trajectories to 1e-12.
The original runs used 3 parallel single-threaded workers; `full_all.sh` runs the same jobs one after another.

## Tests
`uv run python -m pytest` (65 tests, about 2 min on CPU, no external data):
- `test_toys.py`: hand-derived toy values, exact enumeration, the numpy oracle (1.5e-14), the two lower-bound paths, key tree and tie-break.
- `test_official_parity.py`: Tables I-IV against the TeX source, and g, f, MP index and G(P, z) against the authors'
  vector figures (`tests/fixtures/paper_*`). It also runs a whole-episode differential test against the numpy oracle.
- `test_smoke.py`: config vs `sources.yaml`, phi shapes and positive definiteness, one closed-loop run of every arm, one 4-D step. `test_grads.py`: finite-difference checks of the differentiable pieces.
- `test_extras.py`: our own rules (PCLI1 handling, revisit horizon, blind arms, base-length knob).
- `test_full_scripts.py`: helper code of the theory and bound scripts on small instances.

## Checked against
No authors' code exists. The oracle is the authors' published output in the arXiv TeX source of 2312.07858v1 (tarball
sha256 `b453b186...`): Tables I-IV as typeset and the polyline vertices of the vector figures. Each was extracted into
`tests/fixtures/` (about 1.2 MB). `sh scripts/official/setup.sh` refetches the source, and `scripts/make_official_fixtures.py`
(plus `--gz`) rebuilds the fixtures. A second check is an independent numpy implementation written from the equations
(`src/oracle.py`), plus exact enumeration on tiny cases.
