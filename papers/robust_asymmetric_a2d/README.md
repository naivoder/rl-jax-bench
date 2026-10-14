# Robust Asymmetric Learning in POMDPs
Andrew Warrington, J. Wilder Lavington, Adam Ścibior, Mark Schmidt, Frank Wood, ICML 2021. Paper:
https://arxiv.org/abs/2012.15566 (v3). Official code: [plai-group/a2d](https://github.com/plai-group/a2d) (commit
`368aaf5`).

## What this is
The paper shows that asymmetric imitation learning (AIL: a trainee that sees only partial observations imitates an
expert that sees the full state) converges to the "implicit policy", which can be arbitrarily bad (Thm 1). It
proposes A2D, which updates the expert by a policy gradient through the trainee's own rollouts so that the imitated
expert stays imitable, and shows A2D reaching the POMDP optimum on the Frozen Lake and Tiger Door gridworlds where
AIL fails. This folder reimplements the gridworlds, A2D, AIL and the TRPO baselines in JAX, plus an exact tabular
analysis of every environment, and reruns the gridworld experiments.

Labels used in code and tables:
- Environments: FL = Frozen Lake (LavaGap), TD = Tiger Door, TD1/TD2/TD3 = the three Tiger Door variants of App. B.1.
- Arms: `rl_mdp` (TRPO on the full state, the expert), `rl_compact`, `rl_image` (symmetric TRPO), `rl_asym`
  (asymmetric critic), `ail_compact`, `ail_image`, `a2d_compact`, `a2d_image`, `a2d_q` / `a2d_noq` (App. B.1 with and
  without the Q critic). "Compact" trainees see a flat partial vector, "Image" trainees a 42x42 render.
- Experiments: E1 = Figs. 4/5/B.3 (all arms, FL and TD), E2 = Fig. B.1 (Q vs no-Q on TD1-3), E3 = Fig. B.2 (lambda
  sweep on TD2). C1-C6 are the paper's claims (rows of the table below). Pre-declared test families: P (A2D(Image)
  beats AIL(Image) / RL(Image) / RL(Asym), n = 3, Holm over 6), C (A2D(Compact) beats AIL(Compact), n = 20, Holm over
  2), A (ablations, Holm over 8).
- Run stages, also used as run-folder names: `toys` (exact checks), `small` (every arm at small scale), `full`
  (full scale), `full_extra` (follow-up runs after reviewing the full-scale results), `recheck` (re-runs after the
  log-likelihood clamp fix described below).

## Results
Raw undiscounted stochastic return. "Final" is the per-seed mean of the last 5 evaluations (one every 5 iterations of
2000 interactions). The paper reports a per-x median and interquartile band, decoded here from its vector figures
(`data/paper_figures/`); "paper statistic" is the same median [IQR] computed from our seeds at the final x. Judgment
rule, fixed before the runs: MATCH within 5 % or inside the paper's IQR, CLOSE within 20 %, DIVERGENT otherwise.
"Authors' code" is plai-group/a2d run unmodified on CPU (with `--verbose 1`, see "Checked against"). Numbers are the
final ones, after the clamp fix (`results/recheck_report.json`, `results/recheck_tables.md`).

| Quantity | Paper | Authors' code | This reproduction (n seeds) | Verdict |
|---|---|---|---|---|
| C1 AIL converges to the implicit policy (Thm 1), FL / TD | -29.35 / -55.03 (plot); -26.6 / -54 (text, greedy) | -29.68 ± 0.78 / -54.09 ± 1.10 (3) | -27.03 ± 0.79 / -54.53 ± 0.42 (20); stochastic plot value -29.12 / -54.80; exact fixed point of each seed's expert within 3 % of \|R\| | MATCH |
| C2 E1 FL A2D(Compact) | 4.00 [3.98, 4.00] | 3.92 ± 0.05 (3) | 3.99 [3.98, 4.00]; mean 3.85 ± 0.08 (20) | MATCH |
| C2 E1 TD A2D(Compact) | 2.00 [2.00, 2.00] | no failing seed in 7 runs | 1.99 [1.97, 2.00]; mean -11.51 ± 10.44, 2/20 seeds fail (20) | MATCH on the paper statistic; mean DIVERGENT (open) |
| C2 E1 A2D(Image) pilot, FL / TD | ~4 / ~2 | FL 3.89 (1 run, 30 it) | FL 3.83, 1.06, 3.72; TD 1.81, 1.73, -63.87 (3, 80 it) | unstable at n = 3 (TD MATCH on the paper statistic, FL CLOSE on the median curve) |
| C3 RL(Image), FL / TD (n = 3) | -28.69 / -12.71 | - | -29.20 / -12.78 (paper statistic) | MATCH |
| C3 RL(Asym), FL / TD (n = 3) | -28.91 / [-14.69, -7.79] | - | -32.39 / -7.93 | CLOSE / MATCH |
| C4 A2D divergence F much smaller than AIL's | yes | yes | F inside the paper's IQR on FL and TD | holds |
| C5 E2 TD1 no-Q / Q | -42.06 / 15.87 | -43.06, -40.95 / 15.98, 15.95 | -42.14 ± 0.14 / 15.90 ± 0.01 (20) | MATCH / MATCH |
| C5 E2 TD2 no-Q = Q | both ~14 | - | 13.92 / 13.92 (20) | MATCH |
| C5 E2 TD3 no-Q | 11.98 [11.97, 11.99] | 7/10 converge, 3 stuck at -2.8 ... -5.9 | 11.97 [-4.68, 12.00]; mean 4.66 ± 2.07; 8/20 stuck at -4 ... -10 (20) | MATCH vs authors' code (p 0.70); mean DIVERGENT vs paper |
| C5 Q - no-Q (Holm over 3) | TD1 > 0, TD2/TD3 = 0 | - | TD1 +58.04 ± 0.14 (p_Holm 2e-38), TD2 0.00 ± 0.02, TD3 +7.25 ± 2.08 (p_Holm 0.005) | TD1, TD2 hold; TD3 differs (the stuck seeds) |
| C6 E3 B.1 lambda 0.25 / 0.5 / 0.75 | 13.99 / 13.99 / 13.81 | 13.99 / - / 4 of 4 converge | 13.99 / 13.99 / 13.82, 20/20 converge each (20) | MATCH |
| C6 E3 B.1 lambda 1 | -8.65 [-8.92, -6.37] | median -6.20, 0/4 converge | -4.62 [-6.65, -3.27]; 2/20 converge (20) | fails as claimed; MATCH vs authors' code, DIVERGENT raw (+47 %) / CLOSE normalized vs paper |
| C6 E3 B.1 lambda 0: late degradation | median curve drops 8.5 | drops 3.2 (3) | drops 0.5 (20) | DIVERGENT (open, authors' code n = 3) |
| C6 E3 ordering of F by lambda; lambda 0.5 - lambda 1 | increasing; > 0 | - | Spearman rho +0.92 (p < 1e-4); +17.46 ± 1.67 (p 1.3e-9) | holds |
| E3 under the main args (official lambda launcher), lambda .25 / .5 / 1 converged | no figure | 1/2, 2/2, 1/2 | 8/10, 10/10, 2/10 (pooled 20/30 vs 4/6, p 1.0) | MATCH vs authors' code |
| Family P: A2D(Image) beats AIL / RL(Image) / RL(Asym) | yes | - | 3 of 6 reject (FL 3/3, TD 0/3: one failing TD seed) | PARTIAL (n = 3 pilot) |
| Family C: A2D(Compact) beats AIL(Compact) | yes | - | FL +30.88 ± 0.78 (20/20), TD +43.02 ± 10.31 (18/20), both p_Holm < 3e-4 | HOLDS |
| Ablations: beta schedule, critic form (float64, n = 10) | (A2D needs both mechanisms) | - | no detectable effect, every p_Holm 1.0; expert-update-off (= AIL) is the whole effect | the beta / critic predictions are not confirmed |

Caveats:
- The TD A2D(Compact) mean is dragged down by 2 of 20 seeds stuck far from the optimum. The authors' code showed 0 of
  7 at the same settings (Fisher p 1.0), so the failure rate is not pinned down. At the shipped launcher's
  lambda_a2d = 0.9 (instead of the 0.95 we use for TD), ours fails on 10/20 seeds and the authors' code on 1/5
  (p 0.34). Whole-iteration replays from the authors' trained states match to 5e-9 and locate no defect. Open.
- TD3 no-Q: the authors' own code shows the same plateau tail (3/10 stuck), so the gap to the paper's tight band is
  not attributable to either side. A 10-seed figure at the authors' 3/10 rate shows at most 2 stuck seeds with
  probability 0.38. Fig. B.1 may also have been made with the main-args Q launcher. Our run of that configuration
  (TD3, main args + Q, lambda_a2d 0.9) was cut at iteration 210 with a mean of -29, and no authors'-code run of it
  exists. Open.
- The image arms are a pilot: 3 seeds, float32 on GPU (non-deterministic reductions), A2D(Image) for 80 iterations
  (1.6e5 interactions) instead of the paper's 4e5, AIL(Image) for 60 iterations. They are unstable at n = 3.
- `results/` holds report files from each stage. The verdicts above are the final ones. `small_report.json`,
  `full_report_f32.json`, `full_report.json` and `full_extra_report.json` predate the clamp fix; several of their cells
  (E3 lambda 0.75, the main-args lambda sweep, family P, the TD F statistic) were revised by `recheck_report.json`.
  The original verdict on the main-args lambda sweep was DIVERGENT (0/50 converged vs the authors' 4/6); the clamp
  fix turned it into a match.

## What differs from the paper / known gaps
- **Official-code configuration, not the paper text.** Every run uses the settings of the authors' code, which
  produced the figures: horizon T = 100 treated as terminal (paper: 200), no L2 regularization (paper: 1e-3, never
  applied in the code), a 4-conv / 64-d image encoder (paper: 2 conv / 50-d), lambda-return value targets, entropy
  added to the advantage, advantages centered. Each value and its source is in `src/configs/sources.yaml`. The
  paper-text setup was run only with the authors' code (`scripts/official/ref_run2.py`, variant `paper`).
- **Labeled non-official options**, run as ablations: `--set estimator=paper` (Eq. 27: IS-weighted advantage
  scaled, not centered, no entropy term) and `--set critic_targets=own` (each value net fit to its own lambda-return,
  as App. D.1 states). Neither changes a result detectably after the clamp fix.
- **The log-likelihood clamp.** The authors' code computes action log-likelihoods with torch
  `Categorical.log_prob`, which clamps probabilities to [2.2e-16, 1 - 2.2e-16] (zero gradient outside). Our first
  implementation used the exact `log_softmax`. That mattered only once the expert became near-deterministic, so the
  fresh-network differential tests passed. A replay from a trained official state found it (1e-2 without the clamp,
  4.7e-8 with). `loglik_clamp=True` is now the default. The clamp fix is what turned the main-args lambda sweep and
  E3 lambda 0.75 into matches. All "after" numbers above are with the clamp.
- **TRPO freeze.** Without the clamp, once a flat policy is near-deterministic the Fisher-vector product of the
  official KL form is non-finite. The official code then doubles the damping and never steps again. float32 reaches
  this early, which is why the compact arms train in float64 (as the official code does). `kl_fallback=True`
  recomputes such a step from a log-prob KL of the same value, and the entropy is recomputed with 0 log 0 = 0. Both are
  bitwise inert until they fire, and with the clamp on they never fired in any float64 run.
- **Initial buffer.** The official A2D code pushes one batch of the untrained expert's samples into the AIL buffer
  and never clears it (a swallowed exception). We reproduce it as the opt-in `official_buffer_seed=True`. Its effect
  is within noise (Fisher p 1.0), so it is off by default.
- **AIL axis.** The paper charges AIL the 8e4 interactions of training its expert; our expert is the best evaluation
  checkpoint of a full 4e5-interaction RL(MDP) run. `metrics.json` writes both axes.
- **Not reproduced:** the Pre-Enc baseline (Fig. 4) and the CARLA experiment (Sec. 6.2): no simulator and no budget.
- **Bit reproducibility.** A float64 CPU run reproduces bit for bit only when the whole original seed batch is re-run
  (same `--seeds` list, same core count), because seeds are vmapped. GPU runs are seeded and paired by seed index
  but not bit-reproducible.
- **Scripts that reconstruct the original launches.** The original runs went through machine-specific queues
  (pinned cores, waits on other jobs, re-balancing). The shell scripts here run the same commands sequentially. The
  `kl_fallback` / `loglik_clamp` values on each script reproduce the code state of the time. The official
  reference-run list in `scripts/official/reference_runs.sh` was rebuilt from the original job files and logs.
- **Paper figure decoding.** `data/paper_figures/*.json` was decoded from the paper's vector figures; the decoder
  (PDF drawing parser) is not shipped.

## Layout
- `src/data.py`: the environments (there is no dataset). FL / TD / TD1-3 as pure JAX step functions; observations
  are computed from a latent (position, hidden variable, illumination, noise key), so buffers store latents. Images
  come from a re-implementation of MiniGrid's tile painter with N(0, 0.05) noise, stored as float16.
- `src/model.py`: flax networks. 2x64 tanh MLP for flat inputs, the official 4-conv PixelEncoder for images, official
  initializations.
- `src/algorithm.py`: GAE, value / Q losses, the TRPO step (CG + line search, fallbacks), the A2D importance weights,
  the AIL forward-KL projection with the official epoch sampler, and the image augmentation.
- `src/train.py`: one jitted iteration per method (flat arms vmapped over seeds, image arms `lax.map`), checkpoints for
  image arms, and per-seed `metrics.json`.
- `src/eval.py`: evaluation on complete episodes, clean KL, aggregation.
- `src/exact.py`: exact tabular toy analysis: MDP / belief-MDP optima, exact AIL fixed point, exact tabular A2D.
- `src/toys.py`: the five toy checks (env and pieces vs official code, exact optima, sampled AIL vs exact fixed point,
  TD1 closed-form gradient, exact A2D with Q vs Monte-Carlo targets).
- `src/configs/sources.yaml`: every setting with its source (paper section or official-code flag).
- `data/paper_figures/`: numbers decoded from the paper's Figs. 4, 5, B.1, B.2, B.3.
- `scripts/small.sh`, `small_ail_rerun.sh`, `small_report.py`: every arm at small scale, the AIL re-run after the
  pre-training fix, and the report with its pre-declared pass criteria.
- `scripts/full_gpu.sh`, `full_f32.sh`, `full_f64.sh`: full-scale image arms (GPU), compact arms in float32
  (superseded), compact arms in float64 (headline). `x64_probe.py`, `x64_probes.sh`: float64 probes of the TD
  failures.
- `scripts/full_report.py`: full-scale tables and families P, C, E2, E3, A. `theory_thm1.py`: the per-seed exact AIL
  fixed point vs sampled AIL (Thm 1). `flops.py`: analytic FLOP estimate per iteration.
- `scripts/full_extra.sh`, `full_extra_report.py`, `e3_counts.py`, `exact_occupancy.py`: follow-up runs (E3 in the
  App. B.1 configuration, main-args lambda sweep, float64 ablations, E2 at 20 seeds), their report, E3 seed counts at
  the pre-declared threshold, and exact A2D under two projection occupancies.
- `scripts/recheck.sh`, `recheck_gpu.sh`, `recheck_report.py`, `recheck_tables.py`: re-runs after the clamp fix and
  the before / after, paper-statistic and authors'-code tables.
- `scripts/parse_official_ref.py`, `parse_official_ref2.py`, `parse_official_ref3.py`: collect the authors'-code
  reference runs into `runs/official_ref/`.
- `scripts/official/`: harness run inside the authors' venv. `ref_run.py`, `ref_run2.py`, `ref_run3.py` (reference
  drivers), `run_ref.sh`, `reference_runs.sh` (every reference run), `dump.py`, `dump_iter.py`, `dump_iter_td2.py`,
  `compact_fixture.py`, `record_trained_state.py` (record the test fixtures).

## Run it
From this folder, with the root environment (`uv sync` at the repo root):

    uv run python -m src.toys                                       # toy checks, ~3 min CPU -> runs/toys/toys.json
    uv run python -m src.train --exp NAME --env FL --arm a2d_compact --seeds 0-19 --iters 200 --set dtype=float64
    sh scripts/small.sh && uv run python scripts/small_report.py    # small scale; image arms want a GPU
    sh scripts/full_f64.sh && sh scripts/full_gpu.sh                # full scale (rl_mdp first: AIL loads its expert)
    FULL_COMPACT_EXP=full_f64 FULL_E2_EXP=full_f64_e2 FULL_REPORT=runs/full/report_f64.json uv run python scripts/full_report.py
    sh scripts/full_extra.sh && uv run python scripts/full_extra_report.py
    sh scripts/recheck.sh && sh scripts/recheck_gpu.sh && uv run python scripts/recheck_report.py

The AIL arms load `runs/<exp>/<env>/expert_best.npz`, written by the `rl_mdp` run of the same experiment. Outputs go
to `runs/` (git-ignored); per-seed metrics in `runs/<exp>/<env>/<arm>/seed_<n>/metrics.json`, indexed in
`runs/seed_<n>/metrics.json`. Compact arms run float64 on CPU (about 10-55 min per arm on 2-3 cores); image arms
float32 on GPU (about 4-7 min per 3-seed arm on an RTX 3090). Recorded cost: 1.43 GPU-hours for the reproduction, about
34 CPU-hours for the follow-up runs and 4.5 CPU-hours of authors'-code reference runs (the CPU time of the full-scale
compact runs was not recorded), plus 0.5 GPU-hours and about 38 CPU-core-hours for the recheck runs.

## Tests
`uv run pytest` (CPU, float64 where the official code is float64):
- `test_smoke.py`: env and network shapes, one iteration of three arms.
- `test_grads.py`: surrogate gradients against finite differences.
- `test_streams.py`: evaluation does not perturb the training PRNG stream.
- `test_estimators.py`: the official (centered) vs paper (Eq. 27) estimator in expectation; the TRPO freeze and both
  fallbacks; the clamp on a degenerate policy.
- `test_sources.py`: every `Config` field has a source in `src/configs/sources.yaml`.
- `test_official_parity.py`: differential tests against recorded official-code outputs (`tests/data/oracle.npz`,
  `official_iter.npz`): golden env rollouts (bit-exact), GAE, value loss, IS weights, TRPO surrogate / step, AIL
  loss and epoch sampler, and whole official training iterations of A2D, A2D+Q, AIL, RL(MDP), RL(Compact) and
  RL(Asym) replayed through `Trainer.iteration`, with the evaluation metric on identical rollouts.
- `test_official_parity_td2.py`: the same replay for the main-args TD2 lambda-sweep setting (`official_iter_g3.npz`).
- `test_official_parity_trained.py`: replay from a trained official state with clamped samples
  (`official_trained_td2_lam05.npz`): matches with the clamp, diverges without.

The fixtures in `tests/data/` (17 MB) are shipped, so no external data is needed.

## Checked against
plai-group/a2d at commit `368aaf5e0e425c2d48d25925c772a0d9a4f3823b`, run unmodified on CPU in its own venv (python
3.9, torch 1.13.1+cpu, gym 0.17.3, numpy 1.21.6; gym 0.21 breaks it). Point `OFFICIAL_ROOT` at a directory holding
the clone as `repo/` and the venv as `venv/` (default `external/official` in this folder); reference logs are written
under it. It must be run with `--verbose 1`: with the shipped default the trainee optimizer never steps on Linux.
- Environments: golden rollouts on all five layouts are bit-exact; noiseless images agree to the float16 storage
  quantization.
- Whole training iterations from fresh networks: parameters agree to <= 3e-9. From a trained state: <= 4.7e-8 with
  the clamp (1e-2 without), and <= 5e-9 / 5.5e-7 for two further trained states (TD lambda_a2d 0.9, TD3 no-Q).
- Sampler statistics and initialization distributions agree (every test p > 0.01; KS p 0.50 over 300 inits).
- Authors'-code reference runs on our hardware are the "Authors' code" column above (`results/ref*_summary.json`,
  `results/recheck_report.json`).
- Bugs found in the authors' code, reproduced on our side where they affect results: the trainee never trains on
  Linux without `--verbose 1`; AIL pre-training runs 2 projection epochs, not 1; the epoch sampler drops part of each
  wrap-around; the Q launcher's `--initialization fixed` is never read; the initial buffer seed above.
