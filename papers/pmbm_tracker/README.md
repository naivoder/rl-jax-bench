# Poisson multi-Bernoulli mixture filter: direct derivation and implementation
Ángel F. García-Fernández, Jason L. Williams, Karl Granström, Lennart Svensson, IEEE Transactions on Aerospace and
Electronic Systems 54(4):1883-1901, 2018. Paper: https://arxiv.org/abs/1703.04264 (v4). Official code:
[Agarciafernandez/MTT](https://github.com/Agarciafernandez/MTT), folder `PMBM filter` (commit `cc5c6df`).

## What this is
The paper derives the Poisson multi-Bernoulli mixture (PMBM) filter for multi-target tracking with standard point
measurements. It gives a linear/Gaussian implementation (k-best data association, hypothesis pruning) and three
estimators for reading target states from the PMBM density. In a 4-target scenario (Section VII, Fig. 3) with
varying detection probability p_d and clutter rate lambda_c, it reports that PMBM has a lower RMS OSPA error than
TOMB, MOMB, GM-CPHD and GM-PHD (Table II, Fig. 4). This folder reimplements the PMBM filter and the GM-PHD / GM-CPHD
baselines in JAX (float64), the scenario, and the OSPA / GOSPA metrics. It reruns the whole Table II sweep. This is a
Bayesian filter: nothing is trained, and "runs" are Monte-Carlo runs over measurement noise.

Paper claims, as tested here: C1 the best PMBM estimator beats TOMB, MOMB, CPHD and PHD; C2 Estimator 1 beats
Estimators 2 and 3 for p_d >= 0.9; C3 Estimator 2 is lowest for p_d < 0.9; C4 the baseline order MOMB < TOMB <
CPHD < PHD. Est 1/2/3 are the paper's Estimators 1-3 (Section VI).

## Results
Table II, all 15 (p_d, lambda_c) cells x 100 Monte-Carlo runs, on the paper's own Fig. 3 truth read from the PDF's
vector graphics. Measurements come from root `PRNGKey(3003)`, and all arms see byte-identical measurement sets.
Grades (fixed before the sweep) are MATCH <= 5 %, CLOSE <= 20 %, DIVERGENT > 20 % relative to the printed value.
SE is the delta-method SE over runs. Every cell is in `results/table2.md`.

| Quantity | Paper | This reproduction (100 runs per cell) | Verdict |
|---|---|---|---|
| PMBM Est 1, 15 cells | 2.10-3.71 | within -2.4 .. +3.7 % (SE ~0.02-0.05) | MATCH 15/15 |
| PMBM Est 2, Est 3, 15 cells | 2.10-3.69 / 2.10-3.71 | largest deviation +4.0 % (Est 3 at (0.7, 15)) | MATCH 15/15 each |
| PMBM Est 1 at the Fig. 4 cell (0.9, 10) | 2.23 | 2.269 ± 0.017 | MATCH (+1.7 %) |
| TOMB (Williams' code, r > 0.7) | 2.32-4.09 | -4.3 .. +5.8 % | 14 MATCH / 1 CLOSE |
| MOMB (Williams' code, r > 0.7) | 2.10-3.82 | +8 .. +27 % | 9 CLOSE / 6 DIVERGENT |
| GM-CPHD (merge 4, cap 100) | 2.57-5.61 | mostly -2.5 .. -7.1 %; (0.9, 20) +33 % | 8 MATCH / 6 CLOSE / 1 DIVERGENT |
| GM-PHD (Vo-Ma extraction, merge 4, cap 100) | 6.34-9.18 | -23.5 .. -27.3 % | DIVERGENT 15/15 |
| C1 (Holm, 135-test family) | yes | 60/60 significant | REPRODUCED |
| C2 | yes | 6/6 required tests significant | REPRODUCED |
| C3 | yes, by < 0.05 | Est 2 < Est 1 by 0.6-1.2 % (p < 0.02 in 9/9 cells) | REPRODUCED as far as the paper separates |
| C4 | yes | TOMB < CPHD < PHD 30/30; MOMB < TOMB fails (11/15 cells significant the other way) | NOT REPRODUCED |

Caveats:
- **The PMBM numbers are the filter of the paper, not of the authors' current code.** The paper's Algorithm 1 keeps
  the top N_h = 200 global hypotheses with no weight floor, as does the authors' first release (2019-03-20, `4950e45`).
  The current code adds a 1e-4 global-weight floor (introduced 2019-05-08, `7f2d1b8`). With that floor
  (`--prune-w 1e-4`) PMBM grades 6 MATCH / 9 CLOSE, +6.4 .. +10.3 % at p_d >= 0.9. An earlier stage of this
  reproduction used the floor plus a synthetic truth draw; its +1.4 % MATCH was two errors cancelling.
- **PHD.** The graded constants (merge 4, cap 100) come from the works the paper cites. They were taken from memory of
  those works, not checked against their text. With the authors' repository constants (merge 0.1, cap 30) and the
  original Vo-Ma extraction, the same code matches the paper's PHD column in 15/15 cells within 0.7 %. This choice
  was made after seeing the numbers, so it is identification evidence, not a graded result. The repository's default
  estimator (`GMPHD_estimation2`) is -34 .. -51 % in every cell. Our GM-PHD equals the authors' Octave GM-PHD to 2.4e-11.
- **CPHD (0.9, 20).** The printed 2.57 breaks the monotone trend of its neighbors. Against a suspected typo value of
  3.57 we would be at -4.2 %. This cannot be confirmed.
- **MOMB.** With the paper's r > 0.7 extraction rule, MOMB's existence probabilities stay below 0.7 for real targets
  at low p_d. At (0.6, 10) it reports 2.82 targets on average, against a true 3.49. With the MAP-cardinality rule of
  Williams' `main.m` it grades 14 MATCH / 1 CLOSE (labelled sensitivity). C4 fails only through this rule. PMBM Est 1
  still beats MOMB-MAP in 14/15 cells, and the (0.95, 10) cell is a tie.
- **One truth draw.** The paper's protocol shares one truth over all runs, so a Table II value is conditional on that
  draw. Across 10 synthetic truth draws, Est 1 moves by a standard deviation of about 3 %.
- **Fig. 4** (per-step curve at (0.9, 10)) was graded by windows against curves extracted from the PDF's vector
  paths. Windows k = 8-39, 40-43 and 44-81 MATCH. The early transient k = 3-7 is CLOSE (+7.5 %). A residual at
  k = 5-8 (+21 %) is unexplained. These window numbers come from the 100-run confirmatory set (root 4242), whose
  outputs are not shipped.
- The GM-PHD/CPHD row with merge 0.1 / cap 100 was not completed and is not in the tables.

## What differs from the paper / known gaps
- Fixed-shape arrays: T = 256 track slots, H = 200 hypotheses per track (= N_h, so no per-track cap can bind), P = 8
  Poisson components, M = 48 measurements per scan, E = 16 estimates. Every overflow counter (`slot_full`,
  `cap_rows_dropped`, `kbest_ovf`, `child_trunc`, `pois_evict`, `meas_ignored`, `est_over`, dropped measurements)
  must read 0, or `src.train` stops the run. All 3000 graded PMBM runs had every counter at 0.
- The k-best rule k_g = ceil(N_h W_g) is the paper's own approximation. An exact mode (`k_per_parent`) is used only in
  the Bayes-exactness tests.
- Estimator 3 uses r >= 0.5, the authors' code r > 0.5. No recorded estimate sits exactly at 0.5.
- TOMB / MOMB are not reimplemented: they are Williams' ancillary code (arXiv 1203.2995), run under GNU Octave with
  its own untuned settings.
- The authors' MATLAB code runs under GNU Octave 10.3, not MATLAB. Their `kBest2DAssign` (needs an external library)
  is swapped for their own `murty.m`. Rerunning two cases with the original solver gave identical states.
- `PMBMConfig` carries a few options (`pred_batch`, `trk_q`/`pm_qform`, `pm_argmax`, float32 dtypes) that the paper's
  experiments do not use. They are off on the paper path (`model.paper_pmbm_config`, float64).
- Open: the PHD/CPHD constants of the cited works are unverified. The Fig. 4 k = 5-8 residual is unexplained.

## Layout
- `src/algorithm.py`: PMBM core (predict, update with k-best association, Estimators 1-3, pruning), generic over a
  `PMBMModel`; `run_filter` scans a whole run.
- `src/model.py`: the Section VII linear/Gaussian model and `paper_pmbm_config` (paper-path capacities).
- `src/kbest.py`: exact k-best assignment, a host heapq A* backend and a pure-JAX A* backend.
- `src/data.py`: scenario, Fig. 3 truth loader, synthetic truth, measurements, PRNG key tree, sha256 helpers.
- `src/phd.py`: GM-PHD and GM-CPHD baselines.
- `src/eval.py`: OSPA, GOSPA with decomposition, Table II aggregation, SEs, paired tests, Holm.
- `src/train.py`: Monte-Carlo runner. Writes `runs/<tag>/seed_<r>/metrics.json` per run, resumes, enforces the
  zero-counter rule.
- `src/configs/sources.yaml`: every setting with its source (paper section, official code, or ASSUMED).
- `data/fig3_truth.npz`, `data/fig4_paper_curves.json`: the Fig. 3 trajectories and Fig. 4 curves, read from the
  arXiv v4 PDF's vector paths by `scripts/extract_paper_figures.py`.
- `scripts/table2_all.sh`: the full Table II sweep (all arms, all cells) and its report.
- `scripts/export_table2_meas.py`: exports the 15 measurement sets for Octave, with sha256.
- `scripts/eval_williams.py`: OSPA of the Octave TOMB / MOMB outputs, with the r > 0.7 and MAP rules.
- `scripts/table2_report.py`: Table II grades, paired orderings, Holm families, sensitivity rows.
- `scripts/summarize.py`, `scripts/paired.py`: one tag's Table II summary; a paired comparison of two arms.
- `scripts/run_phd_smoke.py`: quick GM-PHD / GM-CPHD run at (0.9, 10).
- `scripts/export_meas.py`: exports the confirmatory (0.9, 10) sets (root 4242) for the Octave oracles.
- `scripts/extract_paper_figures.py`: regenerates `data/` from the PDF (needs `uv sync --extra pdf`; PDF path in
  `$PMBM_PAPER_PDF`).
- `scripts/official/`: our harness for the authors' code. It holds `setup.sh`, `run_octave.sh`, the Octave drivers
  `drive_pmbm.m`, `drive_williams.m`, `gospa_ref.m` and `test_murty.m`, the `shims/` (`mvnpdf`, `poissrnd`) and the
  `patches/`. The Python side is `npz_to_mat.py`, `make_parity_inputs.py`, `load_dump.py`, `parity_lib.py`,
  `diff_run.py`, `check_dump.py`, `build_fixtures.py`, `reference_run.py`, `truth_spread.py`, `make_gospa_input.py`
  and `gen_smoke_input.py`.

## Run it
From this folder, with the root environment (`uv sync` at the repo root):

    uv run python -m src.train --tag demo --truth fig3 --meas-seed 3003 --pd 0.9 --lam 10 --runs 10 --workers 1 --algos pmbm,phd,phd_vm,cphd
    uv run python scripts/summarize.py demo
    sh scripts/official/setup.sh          # authors' code + Octave, needed only for TOMB / MOMB and the oracle scripts
    sh scripts/table2_all.sh              # whole Table II sweep, sequential

A PMBM run (81 steps) takes about 35-40 s on one CPU core, and the first run includes the JIT compile. The original
sweep was about 42 CPU-hours: 2 x 1500 PMBM runs, the PHD/CPHD variants, and Williams' code under Octave at about
95 s per run. It was run on CPU only, with no GPU.

## Tests
`uv run pytest` (CPU, float64; about 6 minutes, 75 tests):
- `test_grads.py`: exact Bayes conjugacy on a hand-derived 1-D toy. The filter's posterior divided by prior x
  likelihood is constant to 4e-17 on 7 sets. A three-step version runs in exact mode. Also checks moment matching
  and the estimate-set overflow counter.
- `test_smoke.py`, `test_backend_parity.py`: a short run, and the host and JAX k-best backends agree.
- `test_kbest.py`: k-best against brute force, including ties and overflow.
- `test_data.py`: key tree, clutter region, zero-counter rule, paper-path capacities, Fig. 3 truth structure and
  sha256. The Fig. 4 curves give Table II (0.9, 10).
- `test_eval.py`: OSPA / GOSPA on hand examples and brute force, and against Stone Soup 1.9.1 (cached reference
  values; a live re-run needs `$STONESOUP_PYTHON`, see `setup.sh`).
- `test_paper_path_prune.py`: the paper path has no global-weight floor.
- `test_official_parity.py`: replays fixtures recorded from the authors' code (`tests/fixtures/official/`, built by
  `scripts/official/build_fixtures.py`). Our filter must match their state after every step at 1e-9. Also covers
  Murty k-best problems, GOSPA, the JAX backend, and six negative controls that must be flagged.
- `test_phd.py`: GM-PHD / GM-CPHD against recorded outputs of the authors' Octave code (`tests/oracle_phd_ref.npz`).
No test needs external data.

## Checked against
The authors' code, github.com/Agarciafernandez/MTT at commit `cc5c6df0e9343dc1345d91ed6ab95f01db4d1821`. It was run
under GNU Octave 10.3.0 (conda-forge) with two patches: `murty.m` instead of `kBest2DAssign`, plus a logging hook, and a
one-token Octave syntax fix in `murty.m`. The comments were re-encoded to UTF-8, and `mvnpdf` / `poissrnd` are shims.
For TOMB / MOMB, Williams' arXiv 1203.2995 ancillary files were used. `scripts/official/setup.sh` rebuilds all of
this under `$OFFICIAL_ROOT` (default `external/official`). The script itself was not run end to end in this folder.
- Step-by-step differential test on six cases (paper settings, the authors' defaults, the hardest Table II corner,
  small N_h, a non-default gate, no global-weight floor), 81 steps each. Global-hypothesis weights agree to 4e-12,
  means to 3e-11, relative covariances to 1e-15 and estimates to 1e-11, with equal counts at every step. These are
  the fixtures behind `test_official_parity.py`.
- Table II (0.9, 10), first 20 confirmatory sets. The authors' code with no floor gives 2.279 / 2.347 / 2.375 for
  Est 1/2/3 (+2.2 / +0.3 / +0.6 % vs the paper). With its current 1e-4 floor it gives 2.412 / 2.462 / 2.497. Ours
  is identical in both settings (per-step difference <= 3.4e-12). The 2019 release gives the same numbers to 4.6e-12.
- GOSPA against the authors' `GOSPA.m` on 400 random set pairs agrees to 2.8e-14.
- GM-PHD / GM-CPHD against the authors' Octave code: 2.4e-11.
- OSPA against Stone Soup 1.9.1: GOSPA agrees to 1e-9. Stone Soup's OSPA picks its assignment on the sum of
  distances rather than d^p, so it differs from true OSPA on 2/240 pairs. Ours is the true minimum, checked by brute
  force.

`results/table2.md` / `table2.json` are the sweep report. `results/table2_meas_sha256.json` holds the sha256 of each
measurement set. The verdicts above are final; the report files were not revised after the sweep.
