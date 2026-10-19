# Multi-Target Radar Search and Track Using Sequence-Capable Deep Reinforcement Learning
Jan-Hendrik Ewers, David Cormack, Joe Gibbs, David Anderson, arXiv preprint 2025. Paper:
https://arxiv.org/abs/2502.13584 (v1). Official code: none released for this paper. The first author's public forks
of stable-baselines3 and imitation (the toolchain the paper names), and Stone Soup 1.9.1, were used as reference
oracles.

## What this is
The paper treats beam pointing of an electronically steered radar over a fixed field of regard as a discrete-action
RL problem. The agent reads a Stone Soup multi-target tracker's track list plus a scan-history raster. It is trained
with PPO using three track-list encoders: Flat, BiGRU, and BiGRU+MHSA. Two optional pre-training steps are used:
behavior cloning (BC) of a random teacher, and an auto-encoder (AE) trained on noise. The paper's headline claim is
that the attention encoder (MHSA) tracks best, while every learner searches about as well as Random. This folder
reimplements the environment, the tracker, the metrics, the encoders and PPO/BC/AE in JAX. It also implements the
fixed baselines (Static, Random, Coverage, a public hand rule `raster+revisit`, and a truth-cued ceiling), and runs
the paper's comparison.

Labels used below and in code:
- M_cov is the sum of track-covariance norms (lower is better).
- SR is the search (scan-value) reward.
- Σ GOSPA d and Σ missed are the episode sums of the GOSPA distance and of its missed-target term.
- H1-H6 are the paper's encoder-ordering claims (H1 MHSA < Flat on M_cov, H2 MHSA < BiGRU, H3 MHSA < Coverage, H4
  MHSA < Random on Σ GOSPA d, H5/H6 MHSA < Flat/BiGRU on Σ missed). They were tested as one-sided paired t over 5
  seeds, Holm over 6.
- S1-S3 are secondary claims (S3: each PPO arm's SR within ±20 % of Random's).
- P1-P11 are the parity checks against the oracles (see "Checked against").
- Stages: "toys" are exact checks, "small" is the small-scale run (1e5 steps, 2 seeds), "full" is the first
  full-scale evaluation (114,688 steps per lane), "paper budget" is the run at the paper's protocol, "review" and
  "recheck" are two later checking passes.

## Results
Final evaluation uses 100 held-out scenarios (`fold_in(PRNGKey(15), 0xE7A1)`), stochastic policies (primary), with
5 seeds per arm. Ratios are taken to Random per seed. The judgment rule was fixed before the runs. MATCH means inside
the band derived from the paper's spread or within 5 %. CLOSE means within 20 %. DIVERGENT means anything else.
"Random itself" names the grade a copy of Random would get; where it equals the measured grade, the row does not
discriminate. The learner rows are from the paper-budget run: early stop after 2.5e6 steps without improvement,
per-lane best model chosen on the deterministic validation return, cap at 2.95e6 steps per lane (29.5 % of the
paper's 1e7, a compute limit).

| Quantity | Paper | This reproduction (mean ± SE, n = 5) | Verdict |
|---|---|---|---|
| M_cov(Flat)/Random | 0.767 / 0.834 (band 0.652-0.955) | 0.796 ± 0.019 | MATCH (scene caveat) |
| M_cov(BiGRU)/Random | 0.807 / 0.771 (band 0.662-0.920) | 0.728 ± 0.088 | MATCH (4/5 lanes still improving at the cap) |
| M_cov(MHSA)/Random | 0.412 / 0.487 (band 0.351-0.564) | 0.809 ± 0.064 | DIVERGENT |
| M_cov(MHSA)/Random at lr 1e-3 (same random streams) | same | 0.765 ± 0.018 | DIVERGENT |
| SR(Flat / BiGRU / MHSA)/SR(Random) | 0.927-1.079 | 6.98 ± 1.46 / 10.05 ± 4.17 / 6.90 ± 2.48 | DIVERGENT (S3 fails) |
| Σ GOSPA d(MHSA)/Random, median/median | 0.952 (0.949-0.955) | 0.991 ± 0.012 | MATCH, barely discriminating |
| Σ GOSPA missed(MHSA)/Random, median/median | 0.841 | 0.983 ± 0.022 | CLOSE, non-discriminating |
| M_cov(Coverage)/Random (no learning) | 0.605 (0.532-0.680) | 1.099 ± 0.002 | DIVERGENT in all 12 scenes tried |
| M_cov(Static)/Random | 2.774 | undefined: 0-4 of 100 episodes hold a track | UNDEFINED |
| SR(Coverage)/SR(Random) | 0.380 | 0.0578 ± 0.0001 | DIVERGENT (depends on the unstated scan decay γ) |
| H1 / H2: MHSA < Flat / BiGRU on M_cov | yes | Δ +28 ± 125 / +168 ± 248, Holm p 1.0 | not supported (opposite sign, n.s.) |
| H3: MHSA < Coverage on M_cov | yes | Δ -599 ± 135, Holm p 0.034 | supported (scene caveat) |
| H4: MHSA < Random on Σ GOSPA d | yes | 5/5 seeds, Holm p 0.075 | CLOSE, not significant |
| H5: MHSA < Flat on Σ missed | yes | Flat misses less, two-sided p 0.018 | reversed |
| H6: MHSA < BiGRU on Σ missed | yes | n.s., opposite sign | not supported |

Summary:
- The tracker, metrics, PPO, BC and the full Flat and MHSA network updates match the oracles to ≤ 1e-6 of max|g|.
- All three learners learn at the paper budget. The Flat and BiGRU tracking ratios reproduce.
- The paper's headline, that MHSA is the best encoder, does not reproduce in either evaluation mode. MHSA's ratio
  stays 36 % above the top of the paper's band even at the one learning rate that helped. The planned checks for
  MHSA were all run: the env and tracker tests, the literal reward sign, both evaluation modes, the unmasked encoder,
  and lr 1e-4 / 3e-4 / 1e-3 (0.995 / 0.976 / 0.902 at 0.79e6 steps). None closed the gap.
- The paper's learners search like Random. Ours give up search for tracking: their looks concentrate, and SR falls
  7-10x below Random's.
- The public hand rule still beats every learner: on Σ GOSPA d (p ≤ 3e-5), and on M_cov for Flat and MHSA.

Caveats:
- The fixed-policy divergence is unresolved and makes every ratio to Random scene-dependent. Coverage is worse than
  Random in our scene, while in the paper it is better. Our episodes' M_cov varies across episodes by about 9 %,
  against about 100 % in the paper. This points to track survival (revisit before deletion), which depends on
  unstated scene constants.
- Deterministic evaluation (co-primary) makes the selected models collapse toward a fixed beam: 2-40 track-bearing
  episodes per seed. Its M_cov ratios (0.50-0.57) are therefore undefined. MHSA's 0.496 lands in the paper's band
  only through this Static-like collapse.
- The final parameters (instead of the selected best model) give Flat 0.756, BiGRU 0.717, MHSA 0.683. No H-test
  changes its verdict.
- The SEs are over seeds and conditional on one fixed set of 100 scenarios. A two-level bootstrap over seeds and
  scenarios widens the intervals but flips no grade (`scripts/review/scenboot.py`).
- Not run: budgets above 2.95e6 steps, and the MHSA ablation arms (no BC, no AE, literal sign) at the paper budget.
- Absolute values differ from the paper: our Random M_cov is 2067 against the paper's 948.
- `results/full_report.json` and `results/review_*.json` come from the 114,688-step full-scale run. Its learner rows
  were later relabeled NOT TESTED (policies still near uniform at 1.1 % of the paper's budget) and are superseded by
  the paper-budget rows above (`results/paper_budget_report.*`). Their fixed-baseline rows still stand.

## What differs from the paper / known gaps
- **Unstated settings are assumptions.** Many scene constants are not given in the paper: detection probability,
  scan decay γ, scan history length, deleter threshold at a 50 ms step, target kinematics, and the normalizer of the
  scan value. Each is listed with its source tag in `src/configs/sources.yaml`; values tagged "companion" come from
  the authors' earlier paper.
- **Two deliberate deviations from the printed text.** The track-reward sign is flipped so that reducing covariance is
  rewarded; the literal sign is run as an arm. BC uses cross-entropy; the printed MSE BC collapses to the beam center
  in 5/5 seeds and is run descriptively (`scripts/full_bcmse.py`).
- **Budget.** The learners got 2.95e6 steps per lane, not up to 1e7.
- **Paper numbers.** The second MHSA value (0.487) is inferred from the paper's figure, not printed.
- **DRQN arm.** `src/drqn_arm.py` and `scripts/run_drqn.py` are an extra arm that is not in the paper: the same
  encoders inside a double-DQN learner (`rl_jax_bench.drqn`). It ran only at small scale (2 seeds, 3.2e4 steps) and
  says nothing about ranking.
- dtype: environment, tracker and metrics run in float64; networks, PPO, BC and AE in float32.

## Layout
- `src/data.py`: the environment: truth motion, beam detection, a Stone-Soup-equivalent tracker (KF predict, UKF
  update, Mahalanobis gate, greedy NN or GNN, initiator, trace or Frobenius deleter, optional clutter), scan value,
  reward and per-step metrics (covariance norms, GOSPA live and library readings).
- `src/model.py`: Flat / masked BiGRU / MHSA-then-BiGRU track encoders, NatureCNN, separate policy and value MLPs, the
  factorized (az, el) head, and the AE decoder.
- `src/algorithm.py`: PPO with SB3 semantics, GAE with timeout bootstrap, BC (cross-entropy, and the printed MSE) and the AE loss.
- `src/baselines.py`: Static, Random, Coverage, hand rule and truth-cued ceiling.
- `src/train.py`: pre-training and PPO for one encoder family, vmapped over (arm, seed) lanes; best-model selection;
  orbax resume. The PRNG key tree is in its docstring.
- `src/eval.py`: per-episode metrics and their reductions for any policy.
- `src/toys.py`: AE floor check at full size.
- `src/drqn_arm.py`: the extra DRQN arm (not in the paper).
- `src/common.py`: GPU bootstrap, float64, logging.
- `src/configs/config.py`, `sources.yaml`: every constant with its source tag.
- `scripts/small_baselines.py`, `small_learners.py`, `small_report.py`: small-scale run (hand-rule tuning, all arms).
- `scripts/run_drqn.py`: the DRQN arm at small scale.
- `scripts/full_learners.py`, `full_bcmse.py`, `full_baselines.py`, `full_report.py`: full-scale run and report
  (also writes `runs/seed_<n>/metrics.json`).
- `scripts/full_diag_dt.py`, `full_compute.py`: time-step / deleter diagnostics of the baseline divergence, and
  parameter / FLOP counts.
- `scripts/review_scenes.py`, `review_scene_report.py`, `review_readings.py`, `review_ssv_sweep.py`,
  `review_learner_lib.py`: 12-scene grid for the fixed policies, alternative metric readings, scan-value convention
  sweep, library-GOSPA readings of the learners.
- `scripts/review/`: one-off checks (baseline reruns, GPU vs CPU, γ fill-in, objective drift, recomputation, two-level
  scenario bootstrap).
- `scripts/recheck/`: second checking pass: primary-scene golden check, stored-output readings, scene signatures,
  before/after tests, and probes of the coast value and of the scan-value normalizer.
- `scripts/paper_budget.py`, `paper_budget.sh`, `paper_budget_final_eval.py`, `paper_budget_report.py`: the
  paper-budget run (final-parameter evaluation included) and its report.
- `scripts/verify_block_report.py`, `lr1e3_report.py`: reports of the MHSA verification runs (lr sweep, unmasked,
  literal sign, single normalizer) and of MHSA at lr 1e-3 at the paper budget.
- `scripts/official_metric_parity.py`, `official_reference_report.py`: our metrics against Stone Soup's on the same
  rollouts; paper / authors' toolchain / ours table.
- `scripts/official/`: harnesses that run in the oracle venvs (`ss_ref.py`, `ss_metric_ref.py`, `sb3_ref.py`,
  `sb3_reference_run.py`, `mhsa_sb3.py`, `mhsa_map.py`), `setup.sh`, the pinned lock files, and the
  `sb3_reference.sh` / `sb3_mhsa.sh` drivers.

## Run it
From this folder, with the root environment (`uv sync` at the repo root). Outputs go to `runs/`.

    uv run python src/toys.py                                    # AE floor, minutes on GPU
    uv run python scripts/small_baselines.py && uv run python scripts/small_learners.py --steps 100000 --n-eval 32
    uv run python scripts/run_drqn.py --steps 30000 --seeds 0,1 && uv run python scripts/small_report.py
    uv run python scripts/full_learners.py --family mhsa --steps 114688    # then --family flat, --family bigru
    uv run python scripts/full_bcmse.py
    JAX_PLATFORMS=cpu uv run python scripts/full_baselines.py && uv run python scripts/full_report.py
    JAX_PLATFORMS=cpu uv run python scripts/review_scenes.py --scenes primary,dt5,frob,dt5_frob,gnn,clutter,clutter5,n12,na20
    sh scripts/paper_budget.sh && uv run python scripts/paper_budget_report.py        # ~1.6 GPU-h
    OUT=runs/verify_block CAP=48 FAMILIES="v_mhsa v_flat v_unmasked v_single" SEEDS=0,1,2 EXTRA=--eval-final sh scripts/paper_budget.sh
    OUT=runs/lr1e3 FAMILIES=v_mhsa_lr1e-3 EXTRA=--eval-final sh scripts/paper_budget.sh

Hardware: one RTX 3090. GPU costs were about 0.38 GPU-h for the full-scale run, 1.63 for the paper-budget run,
0.93 for the verification runs, and 0.73 for MHSA at lr 1e-3. Fixed policies and all review scripts run on CPU.

## Tests
`uv run pytest` (a few minutes on CPU):
- `test_grads.py`: the exact toys: scan value closed form, UKF = KF on linear models, association vs brute force and
  scipy, the reward table, GOSPA worked example, encoder padding / equivariance and flax / numpy references, PPO clip
  gradient, GAE identities, and `check_grads` of the full loss through every encoder.
- `test_smoke.py`: one end-to-end training and evaluation pass.
- `test_official_parity.py`: replays `tests/fixtures/official/*` (recorded from the SB3 / imitation forks, torch and
  Stone Soup) through our code: P1-P11. No external software is needed.
- `test_scene_variants.py`, `test_recheck.py`: the scene options (deleter statistic, library GOSPA, clutter, GNN,
  scan-value normalizer, truth speed) default to the primary scene and switch correctly.
- `test_select_best.py`, `test_verify_block.py`: best-model selection with early stop; the per-lane learning-rate
  switch and shared random streams.
- `test_drqn_arm.py`: the extra DRQN arm runs end to end.
- `test_oracle.py`: live differential tests against Stone Soup 1.9.1 (UKF, initiator, full tracker trajectories,
  GOSPA). They are skipped unless `sh scripts/official/setup.sh` has built `$OFFICIAL_ROOT/.venv`.

## Checked against
No author code exists for this paper. Oracles, built by `scripts/official/setup.sh` under `OFFICIAL_ROOT` (default
`external/official`):
- Stone Soup 1.9.1 + filterpy 1.4.5 (Python 3.11). Our tracker reproduces Stone Soup `MultiTargetTracker`
  trajectories to 2.8e-12, UKF steps to 1e-9 and GOSPA to 1e-9. Our M_cov and Σ GOSPA on identical rollouts agree
  with Stone Soup's metric pipeline to ≤ 3.7e-14 (`results/official_metric_parity.json`).
- The first author's forks
  [iwishiwasaneagle/stable-baselines3](https://github.com/iwishiwasaneagle/stable-baselines3) at `e46a6d5` and
  [iwishiwasaneagle/imitation](https://github.com/iwishiwasaneagle/imitation) at `f5cb8a4`, with torch 2.3.1.
  Rollout bootstrap, PPO gradients, the optimizer, BC loss, MHA, GRU, NatureCNN and the init conventions match to
  ≤ 2e-5. Full Flat and BiGRU+MHSA PPO updates on our env match to 1e-4 of max|g|.
- The forks trained on our environment (Flat, 3 seeds, `scripts/official/sb3_reference.sh`) sit at Random's M_cov at
  the full-scale budget, as ours did (1.029 vs 1.012; `results/official_reference_report.json`).
`setup.sh` was written for this folder and has not been run end to end; the fork commits were confirmed to exist.
