# Bridging the Imitation Gap by Adaptive Insubordination
Luca Weihs, Unnat Jain, Iou-Jen Liu, Jordi Salvador, Svetlana Lazebnik, Aniruddha Kembhavi, Alexander Schwing,
NeurIPS 2021. Paper: https://arxiv.org/abs/2007.12173 (v3). Official code:
[allenai/advisor](https://github.com/allenai/advisor) (commit `6849755`, on AllenAct `f00445e4`).

## What this is
A teacher that acts on information the student cannot observe is "privileged". The paper shows that imitating such a
teacher converges to the teacher's policy averaged over what the student cannot see (Prop. 1), and that this average
can be arbitrarily bad: the "imitation gap". ADVISOR adds an auxiliary actor head trained only by imitation and
weights the student's per-state loss as `w * imitation + (1 - w) * RL`, with `w = exp(-alpha * KL(teacher || aux))`
(Eqs. 2-3). Where the teacher cannot be imitated, `w` is small and the student falls back to RL.
This folder reimplements ADVISOR and its baselines in JAX as one vmapped on-policy trainer. It also ports three of the
paper's environments to pure JAX: Poisoned Doors, 2D-Lighthouse and MiniGrid Lava Crossing Once Switch.

Labels: E1 Poisoned Doors (PD), E2 2D-Lighthouse (2D-LH), E3 MiniGrid LC Once Switch (MG). Arms: PPO, BC, DAgger,
BC^tf=1 (BC with full teacher forcing), BC+PPO (static sum), BC->PPO, DAgger->ADV and ADV (ADVISOR). The rule was
fixed before the runs: MATCH means within 5 % or inside the comparator's reported spread, CLOSE means within 20 %, and
DIVERGENT means anything else. "Authors' TSV" means the per-run result files shipped in the official repository
(`hp_runs/`), which hold the paper's data.

## Results
| Quantity | Paper | This reproduction (mean ± SE, n) | Verdict |
|---|---|---|---|
| E1 PD, ADV final return (alpha 8) | +0.992 ± 0.002 (TSV, lr band, 15 runs); Table A.5 RR@10 1.00 | +0.999 ± 0.000 (5) | MATCH |
| E1 PD, every imitation arm (BC, DAgger, BC^tf=1, BC+PPO) | -0.638 to -0.646 (IL value -2/3) | -0.660 ± 0.03 (5 each) | MATCH |
| E1 PD, BC->PPO / PPO | -0.631 / 0.000 | -0.633 ± 0.031 / 0.000 (5) | MATCH |
| E1 PD, RR@10 over the authors' hp draws 0-15 (ADV, PPO, BC, DAgger) | 1.00, 0, -0.60, -0.59 (50 draws) | 1.000, 0.003, -0.588, -0.586 (16 draws) | MATCH, 4 of 4 |
| E1 PD, RR@10 BC->PPO / DAgger->ADV | -0.17 / -0.10 (50 draws) | -0.122 / -0.126 (16 draws) | MATCH by spread (see caveats) |
| E1 claims "ADV beats each of 6 baselines" (paired, Holm) | yes | +0.999 to +1.66, Holm p <= 1.3e-6, 5/5 seeds each | HOLDS 6/6 |
| E3 MG, ADV final reward at 1e6 steps (lr 1e-3, alpha 8) | 0.809 (authors' 4 runs at alpha 8 in our lr band); Table A.5 RR@10 0.80 | +0.841 ± 0.019 (5) | MATCH |
| E3 MG, ADV at the authors' draws 3 / 7 / 48 | +0.802 ± 0.025 (TSV); authors' code on our hardware +0.751 ± 0.085 | +0.808 ± 0.058 (3) | MATCH vs TSV, CLOSE vs authors' code |
| E3 MG, PPO, BC, BC^tf=1, BC+PPO, BC->PPO | 0.000-0.003 | 0.000-0.0004 (5 each) | MATCH, 5 of 5 |
| E3 claims "ADV beats each of 5 baselines" | yes | +0.841 ± 0.019, Holm p 3.9e-6, 5/5 each | HOLDS 5/5 |
| E2 2D-LH, 33 (cell, arm) medians vs the TSV, 25 seeds per cell | TSV medians | 29 MATCH, 3 CLOSE, 1 DIVERGENT | see caveats |
| E2 claims "ADV shorter than BC / BC->PPO" in 4 gap cells | yes | 7 of 8 hold; ADV vs BC->PPO at (1,15) unresolved (-3.4 ± 3.9, Holm p 0.20) | HOLDS 7/8 |
| E2 BC gap grows with j - i, non-monotonically (Prop. 1, Fig. 6), row i = 1 | TSV medians | Spearman 0.976 vs the TSV, same maximum at j = 11 | HOLDS |
| Ablation: alpha = 0 (no adaptive weight) | implied: falls to the imitation value | PD -0.661 ± 0.032; 2D-LH (1,15) 438 ± 30, (5,15) 319 ± 19 steps, back to BC's level | HOLDS |
| PD return ordered in alpha (0, 1, 2, 4, 8) | implied by the exact-flow toy | -0.661, +0.431, +0.969, +0.996, +0.999 | HOLDS |

Caveats:
- **2D-LH divergences.** The CLOSE cells are (3,3) BC->PPO at +5.8 %, (5,5) DAgger at -12.8 % and (1,7) BC at -9.9 %.
  The (5,5) DAgger cell is MATCH (-0.4 %) against the authors' code run on our hardware. The DIVERGENT cell is PPO
  (5,5) on seeds 0-24 (median 136 vs 349), and we attribute it to seed-batch noise. 25 fresh seeds give 334 (MW p 0.98
  vs the TSV), the same seeds trained in a separate program give 276, and the authors' code gives 355
  (`results/lh_ppo_rep.json`). 2D-LH PPO is chaotic: a 1e-5 difference changes a run's outcome, not the distribution.
- **Every TSV 2D-LH test value is one shared draw.** AllenAct tests with seed 12345 for every run. Our Table A.1
  analogue at i = 1 (fraction near-optimal: ADV 0.84, BC 0.72, DAgger 0.52 vs the TSV's 1, 1, 0.88) is therefore
  compared with its expectation over independent test keys (ADV 0.95, BC 0.79). Those match the authors' code under
  fresh test seeds (ADV 1.00, BC 0.78; `results/table_a1_expected.json`).
- **PD RR@10 uses 16 of the paper's 50 hp draws** (CPU budget). Against the authors' runs at the same 16 draws,
  BC->PPO and DAgger->ADV are CLOSE (-0.122 vs -0.131 and -0.126 vs -0.118). Against Table A.5 they lie inside the 5-95 %
  range of the authors' own RR@10 over random 16-of-50 subsets (BC->PPO [-0.570, -0.020], DAgger->ADV [-0.595, +0.429];
  `results/rr10_subset_spread.json`). The verdict is therefore MATCH by spread only.
- **MiniGrid Table A.5 RR@10 over 50 draws was not reproduced.** MG ran one setting (lr 1e-3, alpha 8, 5 seeds) plus
  three of the authors' draws. Every non-ADV method is <= 0.007 on all 50 TSV draws, so no lr rescues the baselines.
- **Alpha slope.** -ln(1 - R) over alpha in {1, 2, 4} has slope 1.53, against 1.22 for the exact-flow toy. The level
  offset is explained by PPO's advantage normalization (the measured advantage std is < 1).

## What differs from the paper / known gaps
- **Not run:** PointNav, ObjectNav and CoopNav (simulators and budget). Also not run: the GAIL and demonstration
  baselines, the other MiniGrid tasks (WC, LC base / Corrupt / Faulty), and BC^tf=1->ADV.
- **Hyperparameters.** PD seed runs tune lr over {3e-4, 1e-3, 3e-3} on seeds 100-101 with the same grid for every arm.
  alpha is fixed at 8 and the stage split at 0.5. MiniGrid uses lr 1e-3 for every arm without tuning. 2D-LH uses the
  official lr 0.0242 and alpha 20. Every value and its source is in `src/configs/sources.yaml`.
- **Fixes taken from the official code**, each followed by a re-run of the affected rows:
  - two LSTM biases, as in torch;
  - torch's `clip_grad_norm_` epsilon;
  - a literal port of `StopIfNearOptimal`, applied only to stages without teacher forcing;
  - the official 2D-LH test-goal list (`src/configs/lh_official_test_goals.json`, not balanced);
  - the X->ADV warm-up trains both heads;
  - the official PD evaluation doors (an int32 overflow had changed 999 of 1000);
  - the networkx tie-break of the MiniGrid teacher (`crossing.OFFICIAL_TIEBREAK`).
- **Results files predate some verdict revisions.** `results/lh.json` / `lh.txt` grade (5,5) DAgger CLOSE against the
  TSV; the MATCH against the authors' code is from `table_a1_expected.json` and the reference runs. `pd_draws_rr.json`
  compares to the same draws only; the Table A.5 MATCH-by-spread comes from `rr10_subset_spread.json`.
  `toys.json`, `prelim_pd_report.json` and `official_reference_summary.json` come from the preliminary stage.
- **Determinism.** 2D-LH training is not bitwise reproducible on GPU (scatter-add in the gather backward): in a re-run,
  812 of 875 runs stopped at the same step. PD full-scale runs were on CPU; the other grids ran on one RTX 3090.
- **Extension, not in the paper.** `--teacher public --n_wrong k` in `scripts/run_grid.py` trains against a
  non-privileged PD teacher that gets the last k code digits wrong. Its results are not part of this reproduction.

## Layout
- `src/envs/pd.py`, `src/envs/lighthouse.py`, `src/envs/crossing.py`: pure-JAX ports of the official environments and
  teachers. 2D-LH uses the official 6400-d factorial-design observation.
- `src/model.py`: actor-critic with the auxiliary imitation head (PD embed-LSTM, 2D-LH linear, MiniGrid embed-LSTM).
- `src/algorithm.py`: ADVISOR, PPO, imitation and BC+PPO losses, GAE, advantage normalization.
- `src/train.py`: one vmapped trainer; every arm is a stage table. Holds the PRNG key tree, early stopping and resume.
- `src/eval.py`: RobustReward@K, the authors'-TSV reader and the judgment rule. `src/data.py`: env registry.
- `src/common.py`: logging, GPU bootstrap, git provenance, and `OFFICIAL_ROOT`.
- `src/configs/`: `sources.yaml`, the official 2D-LH test goals, patsy subterms and `plans/` (run groups).
- `scripts/train_all.sh`: every training run and report, sequentially.
- `scripts/toys.py`: exact toys (exact-flow fixed points, Prop. 1 gaps, RobustReward@K).
- `scripts/run_grid.py`: vmapped grids of arms x seeds x lrs x cells, or the authors' hp draws (`--draws`, `--plan`).
- `scripts/report_grid.py`, `scripts/lh_compare.py`, `scripts/report.py`: summaries, TSV comparisons and the paired
  Holm tests. `report.py` takes `mg`, `lh`, `pd` or `pd_seeds`.
- `scripts/rr_report.py`, `scripts/pd_door_noise.py`, `scripts/reeval_pd.py`: PD hp-draw RobustReward, its
  evaluation-noise check, and re-evaluation of saved checkpoints.
- `scripts/table_a1.py`, `scripts/tsv_reference.py`, `scripts/mg_tiebreak_effect.py`, `scripts/flops.py`,
  `scripts/paired.py`, `scripts/aggregate_seeds.py`: Table A.1 analogue, TSV-only rows, teacher tie-break effect,
  per-arm FLOPs, a paired-test helper, and the per-seed index.
- `scripts/official/`: `setup.sh` and `reference_all.sh` (the authors' code and its reference runs);
  `run_reference.py` and `make_fixtures.py` (run in the authors'-code venv); `summarize_reference.py`,
  `lh_eval_crosscheck.py` and `mg_eval_crosscheck.py` (ours, read the reference outputs).

## Run it
From this folder, with the root environment (`uv sync` at the repo root):

    sh scripts/official/setup.sh            # authors' code + TSVs into external/official (needed by the reports)
    JAX_PLATFORMS=cpu uv run python scripts/toys.py                   # exact toys, CPU
    sh scripts/train_all.sh                                           # everything; see below
    uv run python scripts/run_grid.py --env pd --tag pd_adv --arms advisor --seeds 0 1 2 3 4 --lrs 3e-3   # one arm

Outputs go to `runs/` (git-ignored). Original costs: the preliminary PD grid took 227 s for 147 runs, and full-scale
MiniGrid took 577 s for 33 runs at 1e6 steps. 2D-LH took 181 s for 875 runs, all on one RTX 3090. Full-scale PD
(121 runs) took 71 min on 6 CPU cores. In total about 1.4 GPU-hours, including re-runs. Float32 in training and
evaluation; float64 in the tests and toys.

## Tests
`uv run pytest` (29 tests, about 1.5 min on CPU, no external data):
- `test_smoke.py`: one update of every arm on PD and 2D-LH.
- `test_grads.py`: finite-difference gradients of every loss mode and the PD LSTM unroll, GAE against AllenAct's loop,
  and the X->ADV warm-up loss.
- `test_envs.py`: vectorized vs sequential envs, MiniGrid teacher, official 2D-LH test goals and PD evaluation doors,
  the public teacher, and the PD step cap as a time limit.
- `test_official_parity.py`: replays `tests/fixtures/official_*.npz`, recorded from the authors' code. It covers the PD,
  2D-LH and MiniGrid traces, the 6400x6400 design matrix, ADVISOR / PPO / imitation losses and gradients (rtol
  1e-10), RobustReward@K, `StopIfNearOptimal`, the MiniGrid expert at every state, and one complete AllenAct
  update per model (rtol 1e-9). `scripts/official/make_fixtures.py` regenerates the fixtures after `setup.sh`.

## Checked against
allenai/advisor @ `6849755042c6dab1488f64cf21bde2322add3cc1` on AllenAct @ `f00445e4ae8724ccc001b3300af5c56a7f882614`,
run unmodified on CPU. The venv used Python 3.8, torch 1.13.1+cpu, gym 0.17.3, gym-minigrid 1.0.3 and numpy 1.23.5.
`scripts/official/setup.sh` rebuilds it under `$OFFICIAL_ROOT` (default `external/official`); that script has not
been run end to end in this repository. `scripts/official/reference_all.sh` repeats the reference runs.
- After the fixes above, ours equals theirs everywhere it can be compared exactly. That includes one complete update
  per model at 1e-9 (the committed parity tests).
- Their code on our hardware reproduces their paper: PD ADV +0.997 ± 0.002 and MG ADV at 1e6 +0.751 ± 0.085 (TSV
  +0.802). It is as dispersed as ours on 2D-LH PPO (`results/official_reference_summary.json`).
- Their trained 2D-LH and MiniGrid weights score the same in our evaluation harness as in theirs, with no harness
  offset (`results/official_lh_crosscheck.json`, `results/official_mg_crosscheck.json`).
- Their own test protocol, replayed on 30 of their trained 2D-LH policies, gives bit-exact values under test seed
  12345. That shows every TSV 2D-LH test value is one shared draw (DAgger (1,1) 71.8 vs 79.1 under fresh seeds).
