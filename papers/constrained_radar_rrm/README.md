# Constrained Deep Reinforcement Learning for Cognitive Radar Resource Management
Ziyang Lu, M. Cenk Gursoy, Chilukuri K. Mohan, Pramod K. Varshney, arXiv preprint 2026. Paper:
https://arxiv.org/abs/2606.05526 (v1). Official code: none released for this paper. The reference oracles were
filterpy 1.4.5, scipy 1.18.1 and CleanRL `ddpg_continuous_action_jax.py` at commit `fe8d8a0` (see "Checked against").

## What this is
The paper studies a track-while-scan radar. In each 2.5 s slot it splits its time between tracking dwells, for up to
5 targets tracked by polar EKFs, and a 360-degree scan that uses Shnidman P_d and 3-of-4 track initiation. The
time budget is enforced through a dual variable (eq. 33). The paper claims that constrained DRL (CDRL-DDPG and
CDRL-DQL) keeps the budget and beats the heuristics: equal allocation (EA) and the distance-based heuristic (DA). It
also claims the learners come close to an optimization-based allocation (OA). This folder reimplements the
environment, the three baselines (plus an "OA-myopic" addition) and both learners as pure jitted JAX functions,
vmapped over scenarios and training seeds. It then reruns the paper's Tables VI and VII and Fig. 9 on all four cases.

Labels used in code and tables:
- Calibrations. **A04** is the declared calibration: the scan constant S_ref = 5.62 is fitted to Fig. 9c.
  **A04a** is the sensitivity calibration: S_ref = 32.28 is fitted to Table VI. The paper gives neither constant.
  The tracking constant SNR0 = 1.8585 is fitted to Fig. 8's baseline curve.
- Learner variants. `dual50` uses dual step alpha = 50 instead of 5000. `fixlam` and `nolam` are the dual ablations.
  `raw` drops the reward and observation scaling.
- **H1-H9** are the pre-declared paired tests (Holm over 9):
  - H1/H2: DDPG/DQL > EA, Case I. H3/H4: > DA, Case I. H5/H6: > DA, Case II.
  - H7: DDPG T_init < OA's. H8/H9: DDPG vs OA in Cases I and II.
- **P1-P6** are predictions derived from the model, before any run.
- **A-nn** tags in the code and in `src/configs/sources.yaml` label the modeling assumptions where the paper is
  silent. Each tag sits next to the value it justifies.

## Results
5 training seeds per learner arm, each 3e5 slots. The test set is the Case I/II replica schedule read from
Figs. 3-4: 100 noise draws, paired in 5 groups of 20, on held-out keys. Baseline SEs are over the 5 pairing groups.
Verdict rule, fixed before the runs: MATCH is within 5 %, CLOSE within 20 %, DIVERGENT otherwise. Numbers are the
final re-evaluation: re-digitised replica and collision-free slot keys. Calibration is A04 unless marked.

| Quantity | Paper | This reproduction (mean ± SE, n) | Verdict |
|---|---|---|---|
| U Case I, EA / DA / OA | -24,200 / -19,400 / -18,200 | -21,040 ± 25 / -20,507 ± 8 / -20,301 ± 16 (5) | CLOSE (levels are calibrated) |
| U Case II, EA / DA / OA | -41,000 / -39,000 / -35,900 | -36,937 / -34,081 / -32,956 (SE ≤ 44) | CLOSE |
| U Cases III and IV, baselines | e.g. OA -6,720 / -50,500 | -2,718 to -3,291 / -24,452 | DIVERGENT (1.5-3x better than paper) |
| Ordering EA < DA < OA | yes (Case IV: EA > DA) | holds in every case | REPRODUCED |
| U Case I, CDRL-DDPG | -19,000 | median -32,530; 1/5 seeds catastrophic | DIVERGENT |
| U Case I, CDRL-DQL | -19,600 | -24,665 ± 1,282 (5) | DIVERGENT |
| U Case II, DDPG / DQL | -35,400 / -35,900 | -41,410 ± 241 / -36,654 ± 612 (5) | CLOSE / MATCH, both below DA |
| Table VI T_init, Case I (baselines / DDPG / DQL) | 10.5-20.3 / 6.0 / 7.8 slots | 841 / 781 / 446 slots | DIVERGENT (A04a: baselines 17.3, calibrated) |
| Budget kept (C7), per-slot satisfaction rate | "below threshold" | DDPG 0.37, DQL 0.57 (baselines 1.00) | NOT REPRODUCED |
| H1-H6: learner beats EA / DA | yes | 0 of 6 supported (A04); 0 of 6 (A04a) | NOT SUPPORTED |
| H8/H9: DDPG vs OA | OA better (I), DDPG better (II) | no detectable difference (I), OA better (II, p_Holm 3.5e-5) | partly CONTRADICTED |
| Learners vs OA at matched budget | close to OA | no learner significantly beats OA; DDPG loses (0/5 wins); DQL under A04a undetermined (p 0.30-0.44) | NOT SUPPORTED |
| Dual-variable ablation (`nolam`, `fixlam`) | the dual is the mechanism | removing it changes neither compliance nor U measurably | DOES NOT SUPPORT |
| Fig. 9a/b baseline sweep | shape and levels | shape reproduces; levels 1.21-1.39x the paper's under A04a | shape REPRODUCED, levels DIVERGENT |
| OA / DDPG decision latency (E5) | 2.84 | 2.56 (CPU, median of 1000) | CLOSE |

Caveats:
- **Catastrophic seeds.** "Catastrophic" means a seed mean U < -1e6: one starved track dominates the mean. Such
  means are not reproducible magnitudes. Evaluating the same parameters in a different vmapped program shape
  changes float32 outputs in the last bits. That flips DQL argmaxes and DDPG saturations, and the closed loop
  amplifies the change. Re-training the declared DQL configuration in a different program gave 2/5 catastrophic
  seeds instead of 0/5. Read the count and the median, not the mean.
- **Baseline levels are calibrated quantities, not tests.** SNR0 and S_ref are two independent fits, and they are
  inconsistent as one radar. Scaling S_ref over one decade moves the scan term's share of U from 97 % to 0.2 %
  (`results/sens_sref_snr0.json`). Only orderings, Fig. 9 shapes and learner-vs-baseline gaps are independent tests.
- **Paper-literal readings.** We trained the literal readings of the eq. 30 penalty, the eq. 34 state, the Table IV
  exploration schedule, zero initial targets and one Case III agent per omega (5 seeds each). None produces a
  learner that is both compliant and competitive. The eq. 30-literal penalty makes DQL keep the budget in 94 % of
  slots by starving tracks: 5/5 seeds are catastrophic.
- **Best learner result.** A per-omega DQL agent in Case III (omega 0.005) scores -3,876 ± 202, still below OA
  (-1,158, p 0.004).
- **Predictions.** P1, P3 and P6 hold, and P2, P4 and P5 (DDPG) fail.
- **Under-training not ruled out.** The 6e5-slot extension and a 10-seed estimate of the DQL starvation rate were
  declared but not run. A post-hoc check of the learner implementation also ran only partly: it listed suspects (DDPG
  critic numerics, under-training, the calibration population) that were never fixed and rerun. Every learner
  verdict above is therefore DIVERGENT with that check incomplete.

## What differs from the paper / known gaps
- **No dataset and no scenario files.** Scenarios are the spawn model (paper §VII-A2) plus the Case I/II replica
  schedule that we read off Figs. 3-4 in pixels (`src/configs/scenario_case1.json`). The first, coarser reading is
  `scenario_case1_v1.json` (`--replica v1`). It produced the first full-run numbers, and those are superseded.
  The literal reading of Fig. 4's tail has Target 1b overflying the radar, where our polar EKF diverges and OA
  collapses to -192k. That reading is the labeled sensitivity `--replica v2o`. The primary replica ends Target 1b at
  the radar. This was a scenario choice made after seeing a baseline failure.
- **Case III/IV are under-specified** (arc geometry, interference region noise). This explains the baseline
  divergences there. Which omega Table VII uses is unstated, so both are reported.
- **Single runs in the paper.** The paper reports one training run and one test episode per learner. Our 5-seed
  spread is larger than every learner-vs-heuristic gap the paper reports.
- **OA solver.** OA uses a fixed-iteration Newton solver. The paper's subgradient procedure is kept as
  `oa_subgradient` and is checked to agree with it.
- **Dtypes.** Environment, EKF, OA and metrics run in float64 (`jax_enable_x64` in `src/common.py`). Networks,
  replay and observations are float32.
- **results/** holds the re-evaluation outputs (`tables.md`, `report.json`, `recheck.json` and the side files).
  Learner training curves and per-draw evaluation files are not shipped.

## Layout
- `src/physics.py`: motion model, polar EKF, eq. 5-6 noise model, Kalata steady state, Shnidman P_d table, 3-of-4 chain.
- `src/env.py`: one slot as pre (prior, weights), policy, post (dwells, updates, scan, initiation, score, eq. 33 dual).
- `src/data.py`: spawn scenarios, Case I-IV replica schedules, interference regions.
- `src/baselines.py`: EA, DA, OA (Newton; bisection and subgradient references) and OA-myopic.
- `src/model.py`, `src/algorithm.py`: DDPG actor/critic and update, replay; DQL through `rl_jax_bench.drqn`.
- `src/train.py`: vmapped training over variants x seeds, resumable (orbax). `src/eval.py`: rollouts and metrics.
- `src/common.py`: GPU bootstrap, x64, config dataclasses. `src/configs/`: `default.yaml`, `sources.yaml` (source of
  every value), `calibration.json` (frozen SNR0 / S_ref) and the scenario schedules.
- `scripts/calibrate.py`, `calibrate_table6.py`: fit SNR0 and S_ref (A04), and S_ref from Table VI (A04a).
- `scripts/pilot_eval.py`, `write_seed_metrics.py`: pilot-scale evaluation (3 seeds, validation keys), per-seed files.
- `scripts/full_eval.py`: final evaluation (`--part baselines|learners|traces|sweep`).
- `scripts/full_report.py`, `full_tables.py`, `full_aux.py`: statistics, markdown tables, projection / Kendall side files.
- `scripts/recheck_report.py`: paper-literal arms, per-omega agents and the other re-evaluation checks.
- `scripts/oa_matched.py`: OA at each learner seed's realized budget share.
- `scripts/check_legacy_repro.py`: `--replica v1 --key_scheme legacy` reproduces the first full-run numbers bitwise.
- `scripts/cost.py`: E5 latency and FLOPs. `snr0_sens.py`, `sens_sref_snr0.py`, `calib_t6_sens.py`: calibration sensitivities.
- `scripts/oracle_dump_env.py`, `build_parity_fixture.py`: record our measurement stream, and build the parity fixture.
- `scripts/oracle_pendulum_ddpg.py`, `oracle_pendulum_compare.py`: our DDPG loop vs CleanRL on Pendulum-v1.
- `scripts/official/`: `setup.sh` plus the filterpy / scipy harnesses that run in the reference venv.

## Run it
From this folder, with the root environment (`uv sync` at the repo root):

    uv run python scripts/calibrate.py && uv run python scripts/calibrate_table6.py   # ~2 min GPU; rewrites calibration.json
    uv run python -m src.train --algo ddpg --case case1 --seeds 0,1,2,3,4 --slots 300000 --chunk 10000 --val_every 10000 \
        --variants "A04:;A04a:s_ref=s_ref_table6;dual50:alpha=50;A04a_dual50:s_ref=s_ref_table6,alpha=50;fixlam:alpha=0;nolam:alpha=0,lam0=0;raw:rs=1,cs=1,ls=1" \
        --out runs/full/ddpg_case1        # ~45 min on a shared RTX 3090; dql: same variants minus raw (~27 min)
    uv run python -m src.train --algo dql --case case2 --seeds 0,1,2,3,4 --slots 300000 --chunk 10000 --val_every 10000 \
        --variants "A04:;A04a:s_ref=s_ref_table6" --out runs/full/dql_case2      # likewise ddpg, and case3
    uv run python -m src.train --algo dql --case case1 --seeds 0,1,2,3,4 --slots 300000 --chunk 10000 --val_every 10000 \
        --variants "ctl:;req:g_req=1;lit:g_req=1,obs_lit=1;n0zero:n0lo=0,n0hi=1;eps01:eps0=0.1,eps1=0.0" --out runs/recheck/dql_case1
    uv run python -m src.train --algo dql --case case3 --seeds 0,1,2,3,4 --slots 300000 --chunk 10000 --val_every 10000 \
        --variants "w_fast:om_fast=1;w_slow:om_fast=0" --out runs/recheck/dql_case3   # ddpg likewise (no eps01)
    uv run python scripts/full_eval.py --part baselines --cases case1,case2,case3f,case3s,case4,case4n,case4p
    uv run python scripts/full_eval.py --part learners --cases all --runs runs/full/ddpg_case1,runs/full/dql_case1,...
    uv run python scripts/full_eval.py --part learners --prefix recheck_ --cases case1,case3f,case3s --runs runs/recheck/ddpg_case1,...
    uv run python scripts/full_eval.py --part traces --runs ... ; uv run python scripts/full_eval.py --part sweep
    uv run python scripts/oa_matched.py; uv run python scripts/calib_t6_sens.py; uv run python scripts/sens_sref_snr0.py
    uv run python scripts/full_report.py && uv run python scripts/full_aux.py && uv run python scripts/full_tables.py > runs/full_v2/tables.md
    uv run python scripts/recheck_report.py
    JAX_PLATFORMS=cpu uv run python scripts/cost.py

Outputs go to `runs/` (git-ignored). The original runs used about 2.8 GPU-hours on one shared RTX 3090. Training
ran at about 2250 slots/s per lane when the GPU was not shared. An OA evaluation of 150 episodes x 1e4 slots takes
2-3 min. Per-seed learner values reproduce bitwise only through the same run directory, lane set and evaluation
call.

## Tests
`uv run pytest` (about 3 min on CPU):
- `test_grads.py`: closed-form toys. Covers Kalata steady state vs the EKF, Shnidman P_d, the 3-of-4 chain against
  its absorbing-chain expectation, and the OA worked example. Newton, bisection and subgradient OA agree. Also the
  dual sequence, common random numbers across baselines, and DDPG/DQN losses and gradients vs analytic or numpy
  values.
- `test_smoke.py`: one environment slot, and a few training steps for both learners.
- `test_recheck.py`: the re-evaluation fixes. Covers the L_bm expectation, the requested vs executed budget term,
  the eq. 34-literal state, T_init censoring, the re-digitised windows and slot-key collisions.
- `test_official_parity.py`: replays `tests/fixtures/official_oracle.npz` (reference outputs only) against our
  environment, re-run live. Needs no external data.
- `test_oracle.py`: compares the EKF with filterpy and `ddpg_update` with CleanRL's update. Both tests are skipped
  until `sh scripts/official/setup.sh` has been run.

## Checked against
No authors' code exists. A search on 2026-10-19 covered the paper, the arXiv page and GitHub. We used second-best
references instead. `scripts/official/setup.sh` fetches them into `$OFFICIAL_ROOT` (default `external/official`):
CleanRL at `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`, a Python 3.12 venv with filterpy 1.4.5, scipy 1.18.1 and
numpy 2.5.4, and CleanRL's own Python 3.10 stack (jax 0.4.8, flax 0.6.8). Results:
- filterpy re-filtering of our recorded measurement stream reproduces our per-track costs and U to 1e-13.
- scipy DARE matches our Kalata closed form to 4e-12.
- scipy SLSQP matches our OA allocation to 1e-7 of the budget on 200 decision states.
- Exact ncx2 P_d agrees with our Shnidman table within 0.017.
- filterpy EKF on synthetic sequences agrees to 1.9e-12.
- CleanRL's DDPG update agrees with ours to 4.6e-6 over 30 updates.
- CleanRL's DDPG and our DDPG loop learn Pendulum-v1 equally well within seed noise, at CleanRL's settings and at
  the paper's Table V settings (`results/pendulum_compare.json`, 3 seeds each).

So the learner divergence is not explained by the EKF, cost, OA solver, P_d model, DDPG update, DDPG loop or the DQN
core. The scan model, spawn model, replica geometry and the two calibration constants have no reference.
