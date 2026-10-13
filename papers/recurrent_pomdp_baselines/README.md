# Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs
Tianwei Ni, Benjamin Eysenbach, Ruslan Salakhutdinov, ICML 2022. Paper: https://arxiv.org/abs/2110.05038 (v3).
Official code: [twni2016/pomdp-baselines](https://github.com/twni2016/pomdp-baselines) (PyTorch; commit `e7c19c3` used
as the reference oracle, semantics identical to the ICML commit `a04c8bc` for the parts used here).

## What this is
The paper argues that an off-the-shelf recurrent actor-critic (TD3, SAC or SAC-Discrete whose actor and critic read the
history through an RNN) is a strong POMDP baseline once five design factors are chosen well. Its main ablation lesson
is that **separate** RNN encoders for actor and critic beat a **shared** encoder, which the paper attributes to gradient
explosion in the shared RNN (Fig. 6 Semi-Circle, Fig. 21 Pendulum-V). This folder reimplements recurrent TD3 / SAC /
SAC-D in JAX with separate and shared encoders, the Markovian reference, and three shared-encoder variants, on
Pendulum-V/P (gym 0.18 dynamics), Semi-Circle, Wind and a cue-memory toy. It reproduces the separate-vs-shared
ablation on Pendulum-V and (at a reduced horizon) Semi-Circle. The MuJoCo / PyBullet benchmarks were not run.

Labels used in code and tables:
- **P1-P5**, the pre-declared primary family: separate > shared-official for P1 Pendulum-V sac-lstm-200-oar (51k
  steps), P2/P4 the GRU versions (not run), P3 Pendulum-V td3-lstm-200-oar (20k), P5 Semi-Circle td3-lstm-64-or (150k).
  One-sided paired t over seeds, Holm-corrected.
- **C1-C3**, the fp32 replication family on fresh seeds 8-11 (full-fp32 stage): C1 separate > shared-official,
  C2 shared-detach > shared-official, C3 separate > shared-noleak.
- **Shared variants**: *official* = the authors' shared implementation (joint loss 0.5 (LQ1 + LQ2) + Lpi, one Adam, the
  actor loss also reaches the Q heads); *noleak* = official without the actor-loss gradient into the Q heads (one
  reading of the paper's "share the encoder weights"; it still has no target encoder, the 0.5 critic weight and one
  Adam, none of which the paper describes); *detach* = the actor loss does not train the shared encoder (ours).
- **E6**: the decomposition of the shared-encoder failure with these variants. **L5**: context length 5 instead of 200.
- **Stages**: *toys* (closed-form cue-memory checks), *small* (all arms at a reduced horizon), *full* (the paper's 51k
  horizon for P1, in TF32), *full_fp32* (P1 again at true fp32 on fresh seeds), *recheck* (fp32 re-runs of P3/P5).

## Results
Reference numbers are the authors' own run logs (their "numeric results" archive, 4 seeds), which are the data behind
Fig. 6, Fig. 21 and Table 7: the final metric recomputed from them reproduces Table 7 Semi-Circle exactly (94.15 ±
0.57 sd). Judgement rule: MATCH within 5 % or inside the logs' seed spread, CLOSE within 20 %, DIVERGENT otherwise.
Values are the mean eval return over the last 20 % of the run, mean ± SE over seeds.

| Quantity | Paper (authors' logs) | This reproduction (mean ± SE, n) | Verdict |
|---|---|---|---|
| P1 separate, last 20 % of 51k, fp32 | −168.7 ± 2.9 | −177.8 ± 2.6 (4, seeds 8-11) | CLOSE (−5.4 %) |
| P1 shared-official, same | −443.5 ± 83.2 | −547.0 ± 157.5 (4) | MATCH by spread only (−23 %) |
| P1 gap separate − shared-official | +274.8 (unpaired) | +369.3 ± 157.3 paired, 4/4 seeds | sign robust, size not established |
| P1 in TF32, seeds 0-3: separate / shared / gap | as above | −174.3 ± 6.2 / −511.9 ± 83.3 / +337.6 ± 79.3 (4) | MATCH / spread only / spread only |
| P1 test separate > shared-official (TF32 seeds) | yes | p_Holm = 0.036 (m = 5) | HOLDS |
| C1 same test, fp32 fresh seeds | yes | p_Holm = 0.10 (n = 4; one oscillating shared seed at −1019) | FAILS (underpowered) |
| P3 Pendulum-V TD3, 16k-20k: separate / shared | −186.9 ± 10.4 / −1537.1 ± 91.7 | −205.9 ± 18.7 TF32 / −1552.5 ± 59.0 fp32 (4) | MATCH by spread only (metric-sensitive) / MATCH |
| P3 test | yes | p_Holm ≈ 1e-4 | HOLDS |
| P5 Semi-Circle, 120k-150k: separate / shared | 69.5 ± 5.1 / 0.30 ± 0.04 | 77.7 ± 5.6 / 0.42 ± 0.07 fp32 (4) | MATCH by spread only / at the floor, DIVERGENT on the raw rule |
| P5 test | yes | p_Holm ≈ 1e-4 (TF32); p = 4.2e-4 fp32 | HOLDS |
| C3 separate > shared-noleak, 51k fp32 | not in paper | +1097.3 ± 18.0, p_Holm = 1.5e-5 | holds |
| E6: detach ≈ separate (TOST ±25) | not in paper | +5.1 ± 2.7, equivalent; C2 detach > official p_Holm = 0.10 | mechanism NOT confirmed (exploratory) |
| L5 context length 5 vs 200 (separate) | fails | −995.8 ± 105.5 vs −174.3, paired +821.4 ± 106.4 | HOLDS |
| Gradient explosion in the shared RNN, Semi-Circle | shared ≈ 1e3-1e7 vs separate critic ≈ 1e-1 (Fig. 6) | ×6e4 | HOLDS |
| Gradient explosion, Pendulum-V SAC (shared ÷ separate critic RNN norm) | logs ×0.93 | ×1.05 (fp32 fresh seeds), ×1.37 (TF32) | no explosion here, as in the logs |
| Toy stage: closed-form Q / V / returns on the cue-memory POMDP | n/a | A, B, D, E within the per-value tolerance; C/C' (truncated context) within Monte-Carlo error but not the 0.05 per-value tolerance | PASS |

Overall: the sign of the headline (separate beats the authors' shared implementation) reproduces in every cell that was
run, 8/8 seeds on P1 across two precisions. Its fp32 confirmation is underpowered at n = 4 and the size of the gap is
not established. The paper's explanation reproduces only partly: gradient explosion is real on Semi-Circle but absent on
Pendulum-V SAC, where shared still fails. "The critic's gradient dominates the actor's" is not supported at the
implemented 0.5 weighting (equal clean norms at the shared RNN, about 2× unweighted, not orders of magnitude).

Caveats:
- The small and full stages trained both arms in TF32 on an RTX 3090; the paper's RTX 2080 Ti has no TF32. The
  full-fp32 stage and the recheck re-ran the key cells at true fp32 and found no precision effect (all p ≥ 0.57), so the
  fp32 rows above are the headline rows.
- P3 and P5 ran at reduced horizons (20k of 51k; 150k of 1.548M), so P5 is not comparable with Table 7. Both tests were
  declared after the small-stage data were seen. P3 separate and both P5 levels change verdict between the raw
  last-20 % metric and the paper's smoothed metric (`src.eval.paper_metric`); no test outcome changes.
- The P5 shared arm's eval returns are bit-identical in fp32 and TF32, consistent with a saturated policy at the
  action bounds (not verified directly).
- `results/small_report.json`, `results/full_report.json` and `results/full_fp32_report.json` predate the recheck:
  the verdicts in this README are the final ones. The fp32 rows replaced the TF32 level rows, P5 separate moved from
  CLOSE to MATCH by spread only, and noleak is no longer read as "the paper's described architecture".

## What differs from the paper / known gaps
- **Scope (budget).** Not run: P2/P4 (GRU), the inputs factor (o/oa/or/oar), L = 64, Semi-Circle at the full
  horizon, Wind, the Markovian / oracle references at scale, and all MuJoCo / PyBullet benchmarks. Total compute was
  ≈ 3.4 GPU-h.
- **Hyperparameters follow the authors' run configs, not the printed Table 5**, where they differ: γ 0.9 (Table 5
  prints 0.99), embeddings 8 (Table 5: 16), Semi-Circle RNN 64 and batch 64, replay capacity 4e6, TD3 without policy
  delay and with min(Q1, Q2) in the actor loss. Every value and its source is in `src/configs/sources.yaml`.
- **Unresolved.**
  - Shared SAC at 16k-20k sits below the 2021 logs (fp32 −774.7 ± 130.0 vs −425.0, perm p = 0.057). The authors'
    public code on our hardware shows the same deficit (−852.2 ± 171.1, indistinguishable from ours, p = 0.74), and
    it closes by 51k. Cause not localized.
  - P1 separate fp32 is −5.4 % (about 3 logs SE) below the logs at 51k. The public code was not run past 20k on this
    cell, so it is not attributed to either side.
  - Semi-Circle separate at the single 120k eval: ours 77.7 vs the public code 54.3 (n = 3, perm p = 0.057). This does
    not replicate on fresh seeds 4-7 (+4.2, p = 0.83); the eval point was chosen after seeing the data.
  - Pendulum-V TD3 shared training diagnostics (Q loss, Q gradient norm) are 2-3× the logs at both precisions
    (not significant); returns are unaffected.
- **Reproducibility.** All randomness comes from a counter-indexed key tree, and a resumed run is bit-identical to an
  uninterrupted one (`tests/test_resume.py`). A rerun is bit-identical only for the same seed set, variant set (vmapped
  batch size), precision and device. Separate (R = 4) and shared (R = 8/12) replications of one seed share keys, not
  arithmetic: the seed pairing is a common-random-numbers design.
- **The shared arm was never tuned**, here or in the paper.

## Layout
- `src/envs.py`: pure-JAX Pendulum-V/P, Semi-Circle, Wind and cue-memory envs, with the official code's quirks.
- `src/model.py`: flax leaf modules in torch weight layout, torch LSTM/GRU equations, hand-written BPTT.
- `src/algorithm.py`: losses and update steps (separate, shared official/noleak/detach, Markovian).
- `src/data.py`: device replay buffer, the official sequence sampler, the PRNG key tree.
- `src/train.py`: vmapped multi-seed runner with checkpoint / resume (`python -m src.train`).
- `src/eval.py`: final metrics, the paper's smoothed metric, the authors'-log reader, judgement, statistics.
- `src/common.py`: device bootstrap, config loading, source-tree hash.
- `src/configs/*.yaml`: experiment configs; `sources.yaml` tags every key with its source.
- `scripts/toys.py`, `scripts/toys_summary.py`: toy stage and its Monte-Carlo-error judgement.
- `scripts/small.py`, `scripts/small_report.py`: small stage (P: Pendulum-V 20k, S: Semi-Circle 150k) and report.
- `scripts/full.py`, `scripts/full_report.py`: full stage (P1 continued to 51k, L5) and the primary-family report.
- `scripts/full_fp32.py`, `scripts/full_fp32_report.py`: fp32 fresh-seed replication (family C) and report.
- `scripts/recheck_rerun.py`, `scripts/recheck_report.py`: fp32 re-runs of P3/P5 and their report (the report also
  reads the official-code GPU logs, which are not shipped).
- `scripts/review_recompute.py`: independent recompute of every headline number with scipy only.
- `scripts/official_reference_report.py`: authors' logs vs the official code on our hardware vs ours.
- `scripts/flops.py`, `scripts/random_policy.py`: FLOPs per update; random-policy floor.
- `scripts/relabel_truncated.py`: one-off migration of old run outputs to the horizon-aware metric key.
- `scripts/official/`: `setup.sh` fetches the authors' code and builds its venv (see Checked against).
  Our harness for the official code (labelled shims to run its unmodified `main.py`, fixture
  dumpers for the differential tests, the reference-run configs, `OFFICIAL_COMMIT.txt`).
- `results/`: the stage reports, FLOPs and random-policy floor as JSON.

## Run it
From this folder, CPU or one GPU:
```bash
uv run python -m pytest -q                                   # ~3 min on CPU
uv run python -m src.train --config pendulum_v.yaml --set algo=sac --set arch=separate --seeds 0-3 --out runs/x/sep
uv run python -m src.train --config pendulum_v.yaml --set arch=shared --variants official,noleak,detach --seeds 0-3 --out runs/x/shared
uv run python scripts/toys.py && uv run python scripts/toys_summary.py
uv run python scripts/small.py P && uv run python scripts/small.py S && uv run python scripts/small_report.py
uv run python scripts/full.py P1 && uv run python scripts/full.py L5 && uv run python scripts/full_report.py
uv run python scripts/full_fp32.py sep && uv run python scripts/full_fp32.py sh && uv run python scripts/full_fp32_report.py
uv run python scripts/recheck_rerun.py S5sep   # also S5sh, P3sh; then scripts/recheck_report.py
```
Outputs go to `runs/`. On an RTX 3090 a 51k-step Pendulum-V run costs about 0.05 GPU-h per separate replication
(0.03 shared) in TF32, about 1.75× more in fp32; the full-fp32 stage took 1.24 h. Reports that compare with the
authors' logs or the official code need `OFFICIAL_ROOT` (below).

## Tests
Gradients (custom VJP vs autodiff, finite differences, invariance to padded steps), the sequence sampler (χ²), envs
vs gym 0.18/0.21 and the official envs, networks / losses / three full updates vs the official PyTorch code
(`tests/fixtures/*.npz`, shipped), official eval rollouts and replay sampling, initialization parity, resume
exactness, config source tags, and a smoke run. Eight cases in `tests/test_eval.py` (regression of the final metric against the
authors' logs and Table 7) need the authors' published numeric-results archive and skip without it; `sh
scripts/official/setup.sh` downloads it (5.78 GB from Google Drive, about 1.1 GB kept) and then they run.

## Checked against
- Official code `twni2016/pomdp-baselines` @ `e7c19c3`, unmodified, run in its own venv (py3.8, torch 1.13.1, gym
  0.18) through the labelled shims in `scripts/official/`. Network / loss / optimizer tests match to rel ≤ 1e-3
  (mostly ≤ 1e-5), parameters after 3 official updates to ≤ 3e-6; eval rollouts and the metric to ≤ 1e-5; replay
  sampling exactly. End-to-end, Pendulum-V SAC (both arms, 16k-20k) and TD3 (to 20k) are statistically
  indistinguishable from ours.
- The authors' raw run logs ("numeric results" archive linked from the official README), used as the paper's numbers.
- `OFFICIAL_ROOT` (default `external/official`, git-ignored) holds `repo/` (the clone, with its `logs/` from our
  reference runs), `numeric_results/extracted/logs_txt/` (the authors' logs), `gym021/gym-0.21.0/`, the venv
  `.venv/` and `oracle/ref_logs/`. `sh scripts/official/setup.sh` creates all of them except the reference-run logs
  (clone at the pinned commit, unpatched; logs archive checked against its sha256; Python 3.8 venv with torch
  1.13.1+cpu, gym 0.18.0 built with pip 23.3.2 / setuptools 65.5.0, numpy 1.20.1). It has not been run end to end
  in this form; the pinned commit and download URLs were checked to exist. The fixture dumpers write to
  `tests/fixtures/`.
