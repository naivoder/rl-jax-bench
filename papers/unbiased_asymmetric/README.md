# Unbiased Asymmetric Reinforcement Learning under Partial Observability
Andrea Baisero and Christopher Amato, AAMAS 2022. Paper: https://arxiv.org/abs/2105.11674 (v2). Official code:
[abaisero/asym-rlpo](https://github.com/abaisero/asym-rlpo) (paper-era commit `1a065b0`), used here as the oracle.

## What this is
Asymmetric actor-critic for POMDPs: the actor sees the history `h`, and the critic, used only in training, may also
see the state `s`. The paper shows that the common state critic `V(s)` (A2C-asym-s) is biased for history-based
policies. It proposes the history-state critic `V(h,s)` (A2C-asym-hs), which is unbiased. This folder reimplements
all five compared arms in JAX: A2C-asym-hs, A2C-asym-s, A2C with `V(h)`, and A2C-react-2/4 (reactive, history
window 2 or 4). The arms differ only in the critic input and, for react-k, the history window. It also ports the
paper's environments (Heaven-Hell 3/4, Shopping 5/6, Car-Flag, Cleaner) as pure-JAX environments.

Labels used in code and tables:
- Stages: **toy check** (exact enumeration on toy POMDPs), **small-scale** (Shopping-5 at 2M, Heaven-Hell-3 at 1M),
  **full-scale** (paper settings, budget-truncated), **recheck** (a second pass that re-ran contested cells at
  n = 20 with full-f32 matmuls and fixed the curve extraction).
- H1-H4 are the paper's headline claims (below). **A1** is our extra ablation (asym-s with Monte-Carlo returns).
- "Primary" and "secondary" tests are the two pre-registered families of between-arm tests (Holm within each).

## Results
Our runs use 5 runs per cell (the paper uses 20) unless n is given. Paper values come from the paper's Figure 3/4
vector graphics (`scripts/paper_curves.py`, axis maps calibrated on the gridlines) averaged over the same step
window as ours. Truncated cells are compared at a **common step** shared by every run of both arms. The judgment rule
was fixed in advance: MATCH means within 5 % or within max(paper s.e.m., read precision); CLOSE means within 20 %
(near-zero paper values: absolute rule only, CLOSE within 4x the tolerance); DIVERGENT means anything else. "Authors' code" is unmodified `main_a2c.py` at the paper's Table 2
settings, CPU float32.

| Quantity | Paper (n = 20) | Authors' code (n) | This reproduction (mean ± SE, n) | Verdict |
|---|---|---|---|---|
| Toy check: V(h), V(h,s) policy gradients unbiased; V(s) TD(0) biased (closed form) | Thm 5.2 | - | bias ≤ 8e-16; asym-s bias matches closed form to 6e-16 | HOLDS (exact) |
| Shopping-5 asym-hs, final (1.9, 2.0]M | +3.501 ± 0.100 | +3.610 ± 0.019 (3) | +3.616 ± 0.050 (5) | MATCH |
| Shopping-5 asym-s, final | -5.403 ± 0.226 | -5.548 ± 0.137 (3) | -5.460 ± 0.129 (5) | MATCH |
| Shopping-5 A2C, final | +3.586 ± 0.060 | +3.642 ± 0.038 (3) | +3.600 ± 0.036 (5) | MATCH |
| Shopping-5 react-4, final | +2.079 ± 0.235 | +2.410 ± 0.065 (3) | +2.463 ± 0.015 (5) | CLOSE (+18 %; authors' code +16 %) |
| HH-3 asym-hs, (1.9, 2.0]M | +0.966 ± 0.033 | +1.000 ± 0.000 (3) | +0.908 ± 0.091 (5) | CLOSE (-6 %) |
| HH-3 asym-s, (1.9, 2.0]M | +0.151 ± 0.020 | +0.093 ± 0.020 (20) | +0.086 ± 0.015 (30) | DIVERGENT (-43 %; authors' code -38 %) |
| HH-3 asym-s, (3.66, 4.16]M | +0.138 ± 0.019 | +0.117 ± 0.027 (5) | +0.080 ± 0.018 (20, f32) | DIVERGENT (-41.5 %), unresolved |
| HH-4 asym-hs, (2.81, 3.31]M | +0.277 ± 0.097 | +0.328 ± 0.206 (5) | +0.207 ± 0.091 (20, f32) | MATCH |
| Per-cell count, 16 full-scale cells (HH-3/4, Shopping-5/6) | - | - | 13 MATCH / 2 CLOSE / 1 DIVERGENT | |
| Speed (time to threshold), 7 cells | - | - | 2 MATCH / 5 CLOSE | |
| H1: only asym-hs solves Heaven-Hell | yes | - | HH-3: hs - A2C +0.86, hs - asym-s +0.77 at (1.5, 2.0]M, Holm p 0.016 / 0.022, 5/5 runs reach 0.9 by 1.95M | HOLDS on HH-3; HH-4 not separated (n = 5, Holm p 0.63), 10M untested |
| H2: asym-hs at least as good and faster on Shopping | yes | - | hs - asym-s +9.1 (Sh-5), +11.7 (Sh-6); hs - A2C AUC +10.9, +44.8; Holm p ≤ 0.005, 5/5 runs | HOLDS |
| H3: asym-s collapses on Cleaner | yes | - | not run | UNTESTED (budget) |
| H4: asym-hs about as good as the best other arm everywhere | yes | - | holds on every cell run | HOLDS (cells run) |
| Fig. 4 critic-probe structure (HH-4) | yes | - | V(s) ties (a)/(c) and (b)/(d), V(h) ties (a)/(b), only V(h,s) separates all four | PARTIAL (runs stop at 3.3-6.1M of 10M) |
| A1: asym-s with MC returns beats asym-s TD(0) on Shopping-5 | (our prediction) | - | AUC -64.1 (0/5 runs, Holm p 0.005); final -21.7 (p 0.43) | NOT SUPPORTED |

No difference between our implementation and the authors' code was detected in any cell where both ran (Welch p
0.29-0.92, n = 3-20 per side; this is not an equivalence test).

Caveats:
- The single DIVERGENT cell (HH-3 asym-s) is shared by the authors' code: at 20 distinct seeds it is also 38 % below
  the paper at 2M and not separated from ours (Welch p 0.79). The residual is the stuck-at-random fraction (ours 47 %
  vs authors' 30 %, Fisher p 0.38). Tested and ruled out: implementation, TF32 matmuls, TD(lambda), the final entropy
  weight, sample size. Not tested: full-length training from the authors' initial weights.
- The HH-4 asym-hs cell was DIVERGENT (-57 %) at n = 5 and became MATCH at n = 20; it is bimodal (4 of 20 runs solve).
- Shopping-5 react-4 moved DIVERGENT to CLOSE when the paper curve was re-extracted on gridlines (+0.067).
- Small-scale HH-3 at 1M (a 10x-short proxy): asym-hs +0.56 ± 0.22 vs the paper's +0.418 (DIVERGENT, ours ahead),
  asym-s CLOSE, the other three arms MATCH (uninformative: a non-learning policy also matches).
- At the Shopping-6 common step our hs - asym-s drop is less than half of the paper's (-11.7 vs -26.2), because the
  paper's asym-s mean is still descending there while ours has reached its plateau.
- A1 is handicapped: it inherits the TD(0)-tuned Table 2 settings, and the asym-hs-MC control was not run. Its MC
  estimator agrees with the official `mc` estimator to 3.5e-15 over 150 updates in float64.
- **Result files predate the final verdicts.** `results/full_summary.json` and `results/full_tables.md` are the
  full-scale reduction before the recheck (12 MATCH / 1 CLOSE / 3 DIVERGENT, n = 5); `results/recheck_summary.json`
  and `results/recheck_tables.md` hold the n = 20 re-runs. The table above is the final state.

## What differs from the paper / known gaps
- **Scope cut by budget.** The runs cover about 3e8 of the paper's 4.4e9 env steps. HH-3/HH-4 stopped at 2.0-6.1M of
  10M and were finalized from their last checkpoint; Shopping-6 stopped at 1.06-2M of 2M. Car-Flag and Cleaner
  (ported and checked against the official environments, never trained) and Memory-Four-Rooms (not ported) are not
  reproduced. Fig. 3b (HH-4) and Fig. 4 are partial.
- **n = 5 per cell** against the paper's 20, except the recheck cells (n = 20).
- **Matmul precision.** The small-scale and full-scale Heaven-Hell cells trained on an RTX 3090 with JAX's default
  float32 matmul precision, which uses TF32. The default is now full float32 (`matmul_precision: highest`);
  `matmul_precision: null` reproduces the old runs. The recheck found no TF32 effect (p 0.84).
- **Official code over paper text.** Where they disagree we follow the code and record it in
  `src/configs/sources.yaml`: e.g. observation features 168 (paper 144), GRU size 64 (paper 128) for gridverse
  domains, and Cleaner does not terminate when the floor is clean (paper App. C.2.2 says it does;
  `tests/fixtures/official/cleaner_fullclean.npz`).
- **Figure 4 probes** are our construction of the four (history, state) pairs (marked ASSUMED).
- **Not run:** the asym-hs-MC control for A1; A1 at matched hyper-parameters.

## Layout
- `src/data.py`: `.pomdp` parser with SHA256 pins of the four gym-pomdps files in `tests/fixtures/`.
- `src/envs.py`: pure-JAX environments (Heaven-Hell, Shopping, Car-Flag, Cleaner) and the Fig. 4 probes.
- `src/model.py`: flax embeddings, a torch-equivalent GRU with a custom-VJP sequence pass, MLP heads, the five arms.
- `src/algorithm.py`: rollout, TD(0) / TD(lambda) targets, losses and the jitted update.
- `src/train.py`: trains one (env, method) cell with all runs vmapped; resumable orbax checkpoints; writes
  `runs/<tag>/<env>/<method>/seed_<n>/metrics.json`.
- `src/eval.py`: reductions for every stage (`--tag small|ablation|full|recheck`), paper and authors'-code comparison.
- `src/toys.py`: the exact toy check (Mini Heaven-Hell with and without revisits, Good/Bad POMDP).
- `src/curves.py`: interpolated threshold crossing, common-step windows, stuck-rate Fisher test.
- `src/configs/`: `sources.yaml` (every default with its source: paper section, official code or ASSUMED), one yaml
  per experiment, and `paper_fig3_curves*.json` / `paper_fig4_curves.json` (curves extracted from the paper).
- `scripts/small_scale.sh`, `scripts/full_scale.sh`, `scripts/recheck.sh`: sequential drivers of each stage.
- `scripts/finalize_from_ckpt.py`: writes `metrics.json` for a run stopped mid-way, from its last checkpoint.
- `scripts/paper_curves.py`: extracts Fig. 3 / Fig. 4 from the PDF vectors (needs `pymupdf` and `PAPER_PDF`).
- `scripts/official_reference.py`: reduces the authors'-code reference logs -> `runs/official_reference/summary.json`.
- `scripts/flops.py`: FLOPs per env step; `scripts/tf32_probe.py`: measures TF32 use at the default precision.
- `scripts/official/`: our harness for the authors' code, run in its own venv: `setup.sh` (clones and venv),
  `run_official.sh` / `run_official_defaults.sh` / `official_main_shim.py` (reference curves), `reference_runs.sh`,
  `export_*.py` (env streams, losses, schedules, rollouts), `make_parity_fixtures.py`, `parity_common.py`.

## Run it
From this folder, with the root environment (`uv sync` at the repo root):

    JAX_PLATFORMS=cpu uv run python src/toys.py                    # toy check, ~10 s -> runs/toys/toys.json
    uv run python src/train.py --config src/configs/small_shopping5.yaml --method a2c_asym_hs   # one cell
    sh scripts/small_scale.sh                                       # 5 arms x {Shopping-5 2M, HH-3 1M} x 5 runs
    sh scripts/official/reference_runs.sh                           # authors'-code reference curves (CPU)
    sh scripts/full_scale.sh                                        # needs small_scale first (reuses Shopping-5)
    sh scripts/recheck.sh                                           # n = 20 re-runs + authors'-code comparison

Outputs go to `runs/` (git-ignored). Original hardware: one RTX 3090 shared with other jobs, about 2.1 GPU-hours of
process wall clock for the GPU cells (about 0.8 h exclusive-equivalent), plus a CPU leg (Shopping-6, A1, recheck) of
single-threaded XLA processes, roughly 75 core-hours including the authors'-code runs. At paper length every cell is far larger than that budget: a
full rerun of `full_scale.sh` is several GPU-days.

## Tests
`uv run pytest` (about 4 min on CPU):
- `test_smoke.py`, `test_grads.py`, `test_init.py`, `test_equivalence.py`: one update per arm, custom-VJP GRU against
  autodiff, initialization moments, TD(lambda) diagnostic against its matrix form.
- `test_eval.py`, `test_shared_curves.py`, `test_train_config.py`: the judgment rule, crossing estimator,
  gridline-calibrated curves, `src/curves.py` against `src/eval.py` and the stored recheck numbers, and config-hash
  stability (`tests/fixtures/stored_config_hashes.json`).
- `test_official_parity.py`: replays small fixtures recorded from the authors' code (`tests/fixtures/official/`):
  parsed arrays, env streams for every environment, schedules, losses, gradients and one update for 5 methods x 4
  environments (float32, and a float64 replica at 1e-9), behavior-rollout distributions. No external data needed.
- `test_official_{env,carflag,cleaner,losses}.py`, `test_schedules.py`: full-size versions that read
  `$OFFICIAL_ROOT/out/`. They skip when it is absent; `UA_REQUIRE_ORACLE=1` turns the skips into failures.
  To make them run: `sh scripts/official/setup.sh`, then, with `$OFFICIAL_ROOT/.venv/bin/python`, run
  `scripts/official/export_{env_traces,carflag,cleaner,losses,schedules}.py` and `export_losses.py --f64`
  (the commands are also listed at the top of `setup.sh`).

## Checked against
The authors' code: asym-rlpo `1a065b0cb48b829d14c22fd11bb8ae48b70010a7` with gym-pomdps
`b52e5203d3e7ac079c032e11c3b721963c44cca9`, rl-parsers `b2197e3f7af2190641599e6b38aa4151160e0a64`, gym-gridverse
`9375a10e3dc97ef91e709785e08368dba48bc2b4` and one-to-one `0ae94b5c8ce928388cb312eb6d0adfae0629c29b`, in a Python 3.9
venv with torch 1.13.1+cpu, gym 0.21.0 and numpy 1.23.5. `OFFICIAL_ROOT` (default `external/official` in this folder)
holds the clones (asym-rlpo as `asym-rlpo-aamas2022/`), the venv as `.venv/`, reference-run logs in `runs/` and
exports in `out/`. `sh scripts/official/setup.sh` makes the clones at those commits (unpatched) and the venv with those
versions pinned; it has not been run end to end in this form (the commits were checked to exist).
`sh scripts/official/reference_runs.sh` writes the reference-run logs: 12 sequential 2M-step CPU runs of roughly
0.5-1.1 h each on one core (about 0.5 h for asym-s, 0.8-0.9 h for A2C and asym-hs on Shopping-5), so on the order of
10 hours in all.
- Environments: identical streams on every environment (Cleaner bit-identical; Car-Flag to 1e-9 in float64).
- Losses, gradients and a full update agree to 1e-15 in float64 for all five methods; bookkeeping schedules match.
- Learning curves: no significant difference from ours wherever both were run (table above).
