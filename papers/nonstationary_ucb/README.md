# On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems
Aurélien Garivier and Eric Moulines, arXiv:0805.3415v1 (22 May 2008). Paper: https://arxiv.org/abs/0805.3415.
Official code: none released for this paper. The authors' stationary-bandit package pymaBandits (Cappé & Garivier
2012) and the third-party SMPyBandits were used as oracles (see "Checked against").

## What this is
The paper studies a K-armed bandit whose reward distributions are piecewise constant and change at unknown
breakpoints. It analyzes Discounted UCB (D-UCB) and the new Sliding-Window UCB (SW-UCB), proves regret bounds of
order sqrt(T Υ) log T and sqrt(T Υ log T), and a matching lower bound up to logs. Two Monte-Carlo examples (Figure 1)
show both policies beating UCB-1 and EXP3.S. This folder reimplements in JAX (float64, `lax.scan` over rounds,
`vmap` over runs) UCB-1, D-UCB, SW-UCB, EXP3.S and a labeled oracle-restart UCB-1, the two Figure 1 experiments, and
checks of the theory as labeled extensions.

Scope: every claim refers to arXiv v1, Section 5 and Figure 1. The ALT 2011 version ("... for Switching Bandit
Problems") was not read. Its talk slides are a separate, labeled secondary target (E1b/E2b, T = 2·10⁴).

Labels used below:
- **E1**: K = 3, T = 10⁴, two breakpoints (t = 3000, 5000). The jumping arm goes 0.3 → 0.9 → 0.3, as plotted (the
  text's version is a labeled ablation, `e1_text`). **E2**: K = 2, p_t(1) = 0.5 + 0.4 cos(6πt/T), p_t(2) = 0.5.
- **C1**: D-UCB and SW-UCB beat EXP3.S and UCB-1 on both. **C2/C5**: D-UCB about as good as SW-UCB on E1/E2.
  **C3**: UCB-1 best before t = 3000 on E1. **C4**: the forgetting policies track E2's cycles; UCB-1 does not.
- **X1-X6** (extensions, never run by the paper): X1 regret-rate exponents, X2 Theorem 1/7 bound validity, X3 tuning
  sweeps, X4 lower bound (Theorem 13), X5 Theorem 18 by Monte Carlo, X6 oracle-restart UCB-1.
- Claim families tested one-sided with Holm inside each family (unit = seed mean, n = 5): **P** (C1, 8 tests),
  **S1** (C1 with D-UCB padding B), **S2** (removing forgetting hurts), **S3** (C1 on the slide variants), **T**
  (C1 with every arm tuned on a one-factor grid), and post hoc **Tj** (joint grids), **S2B**, **Td** (against an
  EXP3.S with a decoupled learning rate).
- **Recheck**: a second pass over our own implementation after the main runs. RC1 replaced eye-read Figure 1 values
  by values extracted from the PDF's vector paths; RC2 added a single-realization reading; RC3 built the strongest
  EXP3.S baseline we could (decoupled η); RC4 swept the gap of the X1 instance.

## Results
Final regret R_T at T = 10⁴. Paper = vector-extracted from Figure 1 (±1). Ours = 5 seeds × 2000 runs, mean ± SE over
the 5 seed means [run-level SE]. MATCH ≤ 5 %, CLOSE ≤ 20 %, DIVERGENT beyond. "Per run" says whether the paper value
lies in our 5-95 % per-run band; the figure's curves are probably single realizations (no run count is stated).

| quantity | paper | this reproduction | verdict (mean) | per run (pct) |
|---|---|---|---|---|
| E1 UCB-1 | 909.5 | 837.2 ± 0.4 [0.5] | CLOSE (−7.9 %) | CONSISTENT (95) |
| E1 EXP3.S | 512.7 | 535.5 ± 0.3 [0.6] | MATCH (+4.5 %) | CONSISTENT (38) |
| E1 D-UCB as printed (padding 2B) | 376.8 | 610.9 ± 0.1 [0.2] | DIVERGENT (+62.1 %) | OUTSIDE (0) |
| E1 SW-UCB | 267.5 | 338.7 ± 0.8 [0.7] | DIVERGENT (+26.6 %) | CONSISTENT (8) |
| E2 UCB-1 | 835.4 | 944.3 [1.1] | CLOSE (+13.0 %) | CONSISTENT (12) |
| E2 EXP3.S | 455.4 | 665.1 [0.8] | DIVERGENT (+46.0 %) | OUTSIDE (0.1) |
| E2 D-UCB as printed (2B) | 223.8 | 350.0 [0.2] | DIVERGENT (+56.3 %) | OUTSIDE (0) |
| E2 SW-UCB | 238.4 | 259.8 [0.4] | CLOSE (+9.0 %) | CONSISTENT (29) |
| E1 / E2 D-UCB with padding B (secondary arm) | 376.8 / 223.8 | 381.4 [0.3] / 218.2 [0.2] | MATCH (+1.2 % / −2.5 %) | CONSISTENT (45 / 65) |
| C1 ordering, family P | holds | 7 of 8 tests hold (fails: printed D-UCB vs EXP3.S on E1) | mostly reproduced | |
| C1 with padding B / one-factor tuning / joint tuning / vs decoupled EXP3.S | holds | S1 4/4, T 8/8, Tj 8/8, Td 4/4 (margins −41 % to −54 %) | reproduced | |
| Removing forgetting raises regret to UCB-1 level | (mechanism) | S2 4/4, S2B 2/2 | reproduced | |
| Regret-rate exponents (Remarks 3, 9) on the paper-like instance up to T = 10⁶ | 0.5 (+log) | D-UCB 0.83, SW-UCB 0.70 (band upper edge) | not observed | |

Overall: **partial**. 1 MATCH, 3 CLOSE and 4 DIVERGENT of the 8 headline cells; 5 of 8 paper values are consistent
with one of our runs.

Caveats and final reading of each divergence:
- **D-UCB as printed does not reproduce Figure 1; D-UCB with padding B reproduces both panels.** The paper
  contradicts itself: the index formula prints padding 2B, while its own γ = 1 remark and Figure 1 imply B.
  SMPyBandits' `DiscountedUCB` also defaults to padding B. This is a text contradiction, not a tuning divergence.
- **EXP3.S on E2** rests on a parameter the paper never states for E2 (we used Υ = 2 as on E1). With the number of
  switches defined consistently on both experiments (E2 S = 6 or 7, post hoc), EXP3.S is within 7 % of the paper.
  The pre-declared verdict stays DIVERGENT.
- **SW-UCB on E1** is DIVERGENT on the mean but consistent with one realization (8th percentile; median +19.7 %).
  The mechanism is a heavy-tailed wait before SW-UCB re-explores after the change at t = 3000. Our implementation is
  bit-identical to SMPyBandits, and unpatched SMPyBandits gives 338.0 against our 338.7.
- "Roughly optimal" tuning holds only for SW-UCB on E1 on the one-factor grid; on the joint grids it holds for no
  arm. Every tuned arm is tuned on the evaluation instance with fresh noise only, as in the paper.
- Theory: the bounds hold wherever they can fail (9-50× loose); X4 never violates the lower bound. The paper's
  printed lower-bound constant is 2× larger than its proof gives. The rate exponents are not reached on the Δ = 0.1
  instance up to T = 10⁶. The RC4 gap sweep confirmed 3 of 4 pre-declared predictions: for SW-UCB a pre-asymptotic
  reading is supported; for D-UCB it is consistent but not established (its Δ = 0.4 prediction failed, 0.711).
- SEs are small (10⁴ runs per cell), so MATCH/CLOSE boundaries are set by the read-off, not by Monte-Carlo noise. The
  bootstrap over 5 seed means has only 126 distinct resamples, so its CIs are coarse; every claim that holds has
  p_Holm ≤ 4·10⁻⁸.
- `results/full_report*.{json,txt}` are the main report after the RC1 read-off fix; `results/recheck_report*` holds
  the before/after tables, the three-way paper / SMPyBandits / ours table, RC2-RC4 and family Td. Some prose
  verdicts were reworded after these files were made (D-UCB and SW-UCB wording, the Tj margin against EXP3.S is
  −52 % to −69 %, the gap-sweep reading); the numbers in the files did not change, and the bullets above are final.

## What differs from the paper / known gaps
- Assumptions (all in `src/configs/sources.yaml`): SW-UCB τ = 4 sqrt(T log T) = 1214; E2 uses E1's parameters
  (unstated in the paper); the "R" in E2's "cos(6πRt/T)" is read as a typesetting artifact; UCB indices use log of
  the current round; random tie-break; EXP3.S in log space.
- Figure 1's run count and seeds are unknown; the paper values above are probably single runs.
- No author code exists for D-UCB, SW-UCB or EXP3.S, and the authors' Figure 1 script is not available.
- `src/configs/padb_e*.yaml` still carry the old eye-read values; the report never reads them.
- The paper's mechanism was also checked at a short horizon outside the paper; that extension is not included here.
- SMPyBandits' `Exp3S` never renormalizes its weights and would overflow at X1 horizons (code reading, not run); it
  was not used as an oracle there.

## Layout
- `src/model.py`: policy state structs (discounted / windowed / count statistics, EXP3.S log-weights).
- `src/data.py`: environments as closed-form mean schedules (abrupt, periodic, constant, X1 alternating) with hashes.
- `src/algorithm.py`: UCB-1, D-UCB, SW-UCB, EXP3.S, oracle-restart UCB-1, the jitted vmapped scan runner, PRNG key tree.
- `src/theory.py`: the theorems' constants and bounds (Theorems 1, 7, 13, 18; Remarks 2, 8, 20).
- `src/train.py`: experiment driver; writes `runs/<exp>/seed_<n>/metrics.json`; chunked runs with orbax resume.
- `src/eval.py`: toy checks, a pure-Python reference implementation, run summaries, the small-scale check.
- `src/sweep.py`: X1 rate grids and X4 lower-bound drivers.
- `src/verdicts.py`: MATCH/CLOSE/DIVERGENT with precision and Monte-Carlo flags, per-run reading, log-log slope.
- `src/common.py`: float64 switch, logging re-exports, git info. `src/configs/*.yaml`: one config per experiment;
  `sources.yaml` gives the source of every value; `full_plan.yaml` / `posthoc_plan.yaml` the test families.
- `scripts/small.sh`, `scripts/small_verdict.py`: E1/E2 at 3 seeds × 500 runs plus the D-UCB padding diagnostic.
- `scripts/full.sh`: all full-scale runs (core, sweeps, tuned, X1, X4, post-hoc joint sweeps, ablations, report).
- `scripts/make_sweep_configs.py`, `scripts/tuned.py`: X3 grid configs and the argmin configs for families T / Tj.
- `scripts/report.py`: verdict tables, claim families, ablations, X1-X4 (`--readoff eye` gives the pre-RC1 tables).
- `scripts/divergent.py`: tests the top-ranked cause of each DIVERGENT cell.
- `scripts/perrun_curves.py`: per-run regret at the read-off times. `scripts/collect.py`: per-seed records and hashes.
- `scripts/flops.py`: FLOP count of one step. `scripts/x1_widegap.py`, `scripts/x1_gapsweep.py`: X1 on wider gaps.
- `scripts/x1_resume_test.sh`: kills an X1 run mid-way and checks the resumed result is bit-identical.
- `scripts/recheck.sh`, `scripts/make_recheck_configs.py`, `scripts/recheck_report.py`: RC3/RC4 runs and the recheck report.
- `scripts/extract_fig1.py`: extracts Figure 1 curves from the PDF's vector paths (`PAPER_PDF`, needs pymupdf).
- `scripts/official/`: oracle harness. `gen_ours.py` / `gen_ours_x1.py` dump our trajectories and reward tables
  (repo env); `oracle_replay.py`, `oracle_replay_x1.py`, `oracle_convention_paired.py`, `authors_ucb_replay.py`,
  `oracle_reference.py`, `aggregate_reference.py`, `env_schedule_check.py` run in the oracle venv.

## Run it
From this folder, with the root environment:

    uv run python -m src.eval --toys --out runs/toys        # toy checks, ~15 s
    sh scripts/small.sh                                     # small-scale check, ~1 min on GPU
    sh scripts/full.sh core                                 # headline + secondary arms + ablations, ~7 min on GPU
    sh scripts/full.sh sweep && sh scripts/full.sh tuned    # X3 and family T
    sh scripts/full.sh x1 a; sh scripts/full.sh x1 b; sh scripts/full.sh x1 c025; sh scripts/full.sh x1 c05
    sh scripts/full.sh x4 1000,3000,10000,30000,100000
    sh scripts/full.sh sweepj; sh scripts/full.sh sweepjx; sh scripts/full.sh tunedj; sh scripts/full.sh abl2; sh scripts/full.sh perrun
    uv run python scripts/divergent.py; uv run python scripts/x1_widegap.py
    sh scripts/full.sh report                               # -> runs/full/report.json, report_tables.txt (~45 s CPU)
    sh scripts/recheck.sh all && uv run python scripts/recheck_report.py
    uv run python -m src.train --config src/configs/e1.yaml --seeds 0,1,2,3,4 --reps 2000 --out runs/e1   # one experiment

Original cost on one RTX 3090: about 1.5 GPU-hours of process time for everything (about 0.9 h elapsed), plus
0.45 h for the recheck. The scan is latency-bound (65-110 µs per step regardless of batch). The largest single run
is X1 at T = 10⁶ (450 s for 4 policies). Runs are bit-identical on rerun with `--xla_gpu_deterministic_ops`.
`recheck_report.py` also reads the SMPyBandits reference runs from `runs/official/ref/`.

## Tests
`uv run pytest` (26 tests, about 40 s on CPU, no external data):
- `test_smoke.py`: shapes, one step per policy, chunk invariance, common random numbers across policies, batched
  seeds equal sequential, orbax kill-and-resume is bit-identical, EXP3.S with η multiplier 1 is bit-identical.
- `test_grads.py`: there are no gradients, so it tests the estimators: closed forms, direct sums, hand-computed
  indices, Monte-Carlo variance, exact replays against the pure-Python reference, Theorem 18.

## Checked against
No official D-UCB, SW-UCB or EXP3.S code exists. Oracles live under `OFFICIAL_ROOT` (default `external/official`,
git-ignored): `smpybandits/` (SMPyBandits, https://github.com/SMPyBandits/SMPyBandits, commit
`012fc13a9d82e361930fdf701dfff9a91439d60c`, third party), `pymabandits_dl/py3port/` (pymaBandits,
https://www.di.ens.fr/~cappe/Code/pymaBandits.zip, sha256 `5f4c8a1c…cd08d9`, print statements ported to Python 3
only) and `.venv/` (Python 3.12, numpy 1.26, scipy 1.13, matplotlib; CPU).
- Authors' pymaBandits UCB vs our UCB-1: 40/40 action sequences bit-identical.
- SMPyBandits `Exp3S` (unchanged) vs ours: 20/20 + 20/20 bit-identical on E1 / E2.
- SMPyBandits UCB-1, D-UCB (2B and B) and SW-UCB, with two conventions aligned (log argument and +∞ only at N = 0):
  bit-identical in all 8 cells, and over all 10⁵ / 10⁶ rounds at X1 horizons. SMPyBandits as shipped differs by
  at most about 1 % of regret.
- SMPyBandits' piecewise-stationary environment matches our E1 schedule row for row.
- Unpatched SMPyBandits reference runs (4 seeds × 60) agree with our means within |z| ≤ 1.3 on all 10 cells
  (`results/reference_summary_table.txt`, `results/recheck_report_tables.txt`), so the divergences above are the
  same under the third-party code.
