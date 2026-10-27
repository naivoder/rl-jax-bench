# Deep Q-learning from Demonstrations
Todd Hester, Matej Vecerik, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan, John Quan,
Andrew Sendonaris, Ian Osband, Gabriel Dulac-Arnold, John Agapiou, Joel Z. Leibo, Audrunas Gruslys, AAAI 2018. Paper: https://arxiv.org/abs/1704.03732 (v4). Official code: none released for this paper. The
loss was checked against opendilab/DI-engine (`dqfd_nstep_td_error`, commit `d0b21d06`), a third-party implementation
used as a second-best oracle.

## What this is
DQfD pre-trains a dueling double DQN on a small set of demonstrations, using 1-step and 10-step TD losses, a
large-margin supervised loss (margin 0.8) and L2. It then keeps training on one prioritized replay buffer that holds
the demonstrations permanently, with a priority bonus for them. The paper claims that this beats Prioritized Dueling
Double DQN (PDD DQN) early in training on 41 of 42 Atari games and often ends above its demonstrator. This folder
reimplements DQfD, PDD DQN, supervised imitation, the paper's two ablations and the ADET / RBS / HER comparators in JAX.
It runs them at reduced scale on gymnax MinAtar (Breakout, SpaceInvaders, Asterix, Seaquest). The demonstrators are
PPO agents of controlled quality, because the paper's human demonstrations were never released.

Labels used in code and tables: H1 = DQfD beats PDD DQN early; H2b' = DQfD's final score beats its demonstrator's
mean. C1-C12 are the paper's claims: C1 early performance, C2 catch-up, C3/C4 DQfD vs imitation at start / overall,
C5 exceeding the demonstrator, C6 imitation below the demonstrator, C8 DQfD vs PDD best window, C9 ablations,
C10 related-work comparators, C11 demo up-sample ratio.

## Results
Breakout: 500 k steps, 10 seeds. SpaceInvaders / Asterix / Seaquest: 300 k steps, 5 seeds (the paper used 4 seeds
and 200 M Atari steps). Pre-training uses 75 k updates (the paper used 750 k; 750 k was also run on Breakout and
Asterix, see below). Demonstrator: an early PPO snapshot scoring <= 0.5x the PPO expert. EARLY is the first 5 % of
the budget; FINAL is the last 3 %. Tests are one-sided paired t-tests with bootstrap CIs, Holm-corrected within the
family declared before the runs. The verdict rule is MATCH <= 5 %, CLOSE <= 20 %, DIVERGENT > 20 %. The paper's Atari
scores are not targets, so rows compare proportions and directions.

| Quantity | Paper | This reproduction (mean ± SE, n seeds) | Verdict |
|---|---|---|---|
| C1 / H1: DQfD above PDD DQN early, fraction of games | 41/42 | 4/4: Breakout +6.11 ± 0.02 (10), SpaceInvaders +78.2 ± 4.3, Asterix +4.57 ± 0.34, Seaquest +3.70 ± 0.04 (5 each), Holm p <= 4.4e-4 | MATCH |
| H1 against the paper's literal constant-eps PDD | - | 4/4 again (Breakout +5.59 ± 0.09), also in a 100 k-step window | holds |
| C2: PDD catches DQfD, fraction of budget | 83 M / 200 M = 42 % | Breakout 18 %; never within 300 k in the other three | DIVERGENT |
| C3: DQfD above imitation at the start | 31/42 | 0/4 under Holm; 2/4 by sign of the mean | DIVERGENT |
| C4: DQfD above imitation, final | 39/42 | 1/4 under Holm, 3/4 raw p < 0.05; 4/4 on the paper's own statistic (best window, sign count) | DIVERGENT (declared), CLOSE (paper's statistic) |
| C5 / H2b': DQfD final above its demonstrator | 14/42 above the best demo episode | 0/4: Breakout -0.008 ± 0.024 (10), SpaceInvaders -15.5 ± 97.9, Asterix -18.5 ± 0.3, Seaquest +0.10 ± 0.18 | NOT MET (no numeric comparison) |
| C6: imitation below the demonstrator | 42/42 | 4/4 below the best demo episode | MATCH |
| C8: DQfD best window above PDD | 20/42 | 3/4; Breakout PDD far above (final 22.95 ± 1.06 vs 6.72 ± 0.02) | DIVERGENT |
| C9a: removing the margin loss hurts | yes, 2 games | EARLY 4/4; FINAL 3/4, Breakout reversed (NoMargin +9.9 ± 1.1, 10/10) | NOT REPRODUCED (Breakout rule) |
| C9b: removing n-step hurts | yes | Breakout n.s. at EARLY and FINAL | NOT REPRODUCED |
| C10: DQfD above ADET | Montezuma yes, Q-Bert equal | all n.s. at FINAL | NOT REPRODUCED |
| C11: up-sample ratio | 1.15-5.4 | 1.80-3.31 at the end of the run, rising from about 1 | MATCH (level); rising trend NOT REPRODUCED |
| Margin mechanism: demo Q-gap after training | >= 0.8 by design | 0.85 / 0.82 / 0.61 / 0.67 (controls about 0) | MATCH, MATCH, DIVERGENT, CLOSE |
| C7: state of the art on 11 Atari games | Table 1 | - | not reproducible here |

Main reading: the early advantage reproduces in every game, but it is carried by the supervised clone. Pre-trained
DQfD starts at the imitation agent's level and, at the paper's eps = 0.01, stays at its demonstrator. In Breakout
its end-of-run greedy policy scores exactly the demonstrator's 7.0.

Checks of the "DQfD stays at its demonstrator" result (all on Breakout unless stated):
- Huber loss, one update per 4 steps (also with the target period matched in env steps), loss-based priorities,
  the max-form n-step target and lr 3e-5 / 3e-4: none unfreezes DQfD (`results/attribution_flags*.json`).
- The paper's k = 750 k pre-training with 2 M interaction steps: DQfD final 6.735 ± 0.017 vs the demonstrator's 6.730
  (5 seeds). NoMargin climbs to 27.0 ± 4.0 over the same steps (`results/verification/long.json`). On Asterix, 750 k
  pre-training does not change the verdict either (`results/attribution_pre750k.json`).
- Sticky actions (p = 0.1) with a regenerated demonstrator: DQfD still ends at its demonstrator (-0.18 ± 0.28), but
  now ends above PDD (`results/verification/sticky.json`). So the C2/C8 Breakout reversal belongs to the
  deterministic port.
- Replay capacity 1 M instead of 100 k: no change, +0.010 ± 0.021 (`results/recheck/breakout_capacity_1m.json`).
- Running-max vs current-max priority for new samples: no effect (`results/verification/maxp.json`).
- With eps = 0.1, DQfD beats its demonstrator in Breakout only (greedy 21.5 ± 1.6 vs 7.0); it does not in the other
  three games.

Caveats:
- Asterix in the main grid ran on a gymnax env with a spawn defect that shows up late in long episodes. A repaired
  env sits behind `asterix_spawn_repair` (default off, so shipped numbers are unchanged). The Asterix cell was
  re-run with the repair and a regenerated demonstrator (42.8 ± 0.6). H1 still holds (+5.32 ± 0.50, 5/5). H2b' is
  still not met (-28.3 ± 1.8). Every sign is unchanged (`results/recheck/asterix_repaired.json`).
- `results/analysis.json` and the `attribution_*` files predate a later re-grade. The re-grade read C4 on the paper's
  statistic and re-counted C3 by sign. The table above gives the final verdicts.
- The budgets are 400-1000x below the paper's, so C2 and C8 are budget-confounded.

## What differs from the paper / known gaps
- **Environment and scale.** We use MinAtar instead of Atari: no frame stack, the minimal action set, and no 30
  no-op starts (in MinAtar a Breakout ball is lost after about 7 no-ops). Breakout runs 500 k steps and the other
  games 300 k, against the paper's 200 M. Pre-training uses k = 75 k updates in the main grid, against the paper's
  750 k.
- **Two SpaceInvaders repairs.** Our gymnax 1.0.0 port differed from the original MinAtar in the reset alien columns
  and the bullet hit test. Both are repaired in `src/envs.py`. Breakout, SpaceInvaders and Seaquest match the
  original exactly; Asterix matches only with the repair flag.
- **Demonstrators** are PPO agents, not humans. The weak demonstrator is the strongest early PPO snapshot with a
  mean at most 0.5x the expert's. Demo sets hold about 17.5 k transitions (the median of the paper's Table 2).
- **Unspecified details** were assumed and tagged in `src/configs/sources.yaml`. These include the network size,
  Adam settings, PDD's eps annealing (1 -> 0.01 over 100 k steps), the priority signal, ADET with pre-training, and
  RBS / HER details.
- **Not run.** Atari, the paper's state-of-the-art table (C7), and NoNstep outside Breakout.
- **Unresolved.** In Breakout, nothing tested so far unfreezes DQfD at eps = 0.01, apart from removing the margin
  loss. The cause is attributed to the margin pinning the demonstrator's action together with low exploration, but
  this is not proven.

## Layout
- `src/envs.py`: gymnax MinAtar wrapper. Holds the SpaceInvaders repairs, the optional Asterix spawn repair, sticky
  actions, the time-cap guard and bit-packed observations.
- `src/model.py`: dueling Q network and the PPO actor-critic used for the demonstrators.
- `src/replay.py`: exact proportional PER with a permanent demo region, a demo bonus and sample-time n-step windows.
- `src/algorithm.py`: losses, the arm table (every arm differs only in traced hyper-parameters), and the
  pre-training and interaction scans.
- `src/data.py`: demo collection, saving and sha256 checks. `src/demonstrator.py`: PPO training.
- `src/train.py`: entry point for one (game, demonstrator) cell, with all arms x seeds vmapped. Resumable.
- `src/eval.py`: window scores and the declared tests. `src/configs/`: `default.yaml` and `sources.yaml`.
- `data/demos/`: shipped demo sets (`*_subopt.npz`, the primary weak demonstrator), the `*_demos.json` metadata with
  hashes, and the four PPO experts (`ppo_*.msgpack`, used by `tests/test_recheck.py`). `data/demos_sticky/` and
  `data/demos_asterix_fix/` hold the sticky-env Breakout set and the repaired-env Asterix set.
- `scripts/make_demos.py`, `scripts/run_all_demos.sh`: train the demonstrators and regenerate the demo sets.
- `scripts/run_main.sh`: the full-scale grid and the attribution runs. `scripts/run_verification.sh`: the
  verification runs. `scripts/run_eval_checks.sh`: checkpoint and demonstrator greedy evaluations.
- `scripts/analyze.py`, `scripts/tables.py`: tests and tables, written to `runs/full/analysis.json`.
  `scripts/attribution_report.py`: the attribution checks. `scripts/verification_analyze.py`: the re-grades.
  `scripts/write_seed_summaries.py`: per-seed summaries.
- `scripts/eval_ckpt.py`, `scripts/demonstrator_greedy.py`: evaluation of checkpoints and demonstrators.
  `scripts/run_maxp_variant.py`: the priority-convention variant.
- `scripts/record_official_env.py`, `scripts/policy_parity.py`: our env against the original numpy MinAtar.
- `scripts/official/`: `setup.sh` (fetches the reference code), `oracle_di_engine.py` (records the DI-engine loss
  fixtures), `ref_torch_dqfd.py` + `run_ref_matrix.sh` (independent torch reference run), and `analyze_ref.py`.

## Run it
From this folder, with the root environment (`uv sync` at the repo root):

    uv run python -m src.train --game Breakout --demonstrator subopt --seeds 5 --steps 5e5 --pretrain 75000 \
        --bin-width 5000 --out runs/breakout_first5          # Breakout seeds 0-4, all arms (the default)
    sh scripts/run_main.sh breakout; sh scripts/run_main.sh games 5; sh scripts/run_main.sh flags; sh scripts/run_main.sh pre750k
    sh scripts/run_verification.sh pddconst   # likewise target_period, sticky, maxp, long
    sh scripts/run_eval_checks.sh
    uv run python scripts/analyze.py --out runs/full/analysis.json && uv run python scripts/tables.py
    uv run python scripts/attribution_report.py flags      # likewise flags_tau, pre750k
    uv run python scripts/verification_analyze.py all

The repaired Asterix re-run used the same settings as the Asterix row of `run_main.sh games`, plus the PDDconst arm,
with `--demo-dir data/demos_asterix_fix`. The exact command line was not recorded. Regenerate the repaired demos with
`uv run python scripts/make_demos.py Asterix --max-steps 10000 --asterix-repair --demo-dir runs/demos_asterix_fix`.

Original costs on one RTX 3090 (vmapped arms x seeds in one process):
- Breakout, 35 interaction + 30 pre-training runs at 500 k: about 18 min.
- Demonstrators: about 3 min per game.
- Whole study: about 7.2 GPU-hours of process wall time, on a shared GPU.

Everything runs on CPU too, much more slowly. Float32 throughout. Observations are stored bit-packed.

## Tests
`uv run pytest` (about 2 min on CPU):
- `test_toys.py`: closed forms of every loss term, tabular fixed points, PER sampling against Monte Carlo, and the
  n-step targets.
- `test_grads.py`: finite-difference gradient check of the total loss in float64.
- `test_smoke.py`: every config key has a source, the arm table, one update plus one interaction chunk, and the
  im2col convolution against flax.
- `test_full_scale_helpers.py`: the traced learning rate equals `optax.adam`; the Huber, update-cadence and
  priority flags.
- `test_sticky.py`: sticky actions at p = 0 leave the env unchanged bit for bit.
- `test_official_parity.py`: replays fixtures recorded from DI-engine (loss and gradients, network forward, 25 Adam
  steps) and from the original MinAtar (3,000-8,000 steps per game). Also covers the n-step windows of the torch
  reference replay. No external data is needed.
- `test_recheck.py`: the Asterix spawn repair, the ring n-step brute force with truncations, and the time-cap guard.
  Six policy-driven parity tests also need the original MinAtar source. They skip until you run
  `sh scripts/official/setup.sh`.

## Checked against
There is no official code. `scripts/official/setup.sh` clones the references into `OFFICIAL_ROOT` (default
`external/official`):
- kenjyoung/MinAtar at `4241f24685622cfb52f843cab9bbcf385a53484a`, the original numpy MinAtar.
- opendilab/DI-engine at `d0b21d065b2d80317cc16b28a95301be8871d950`.
- A Python 3.12 venv with torch 2.5.1+cpu for the two torch scripts.

What the checks showed:
- The DQfD loss and its gradients match DI-engine's `dqfd_nstep_td_error` to float32 tolerance, and 25 Adam steps
  match to 1e-4 (`results/di_engine_parity.json`).
- All four envs match the original MinAtar exactly under random actions and under the PPO experts (Asterix with the
  repair flag).
- An independent torch reference run of DQfD (DI-engine loss, original MinAtar, our replay design; Breakout, 3 seeds,
  200 k steps) agrees with our results within MATCH/CLOSE. That includes NoMargin ending above DQfD and PDD
  overtaking DQfD. Its outputs are not shipped; `scripts/official/run_ref_matrix.sh` and `analyze_ref.py` repeat it.
- felix-kerkhoff/DQfD (commit `2dc59602`) was read for the J_E value; TF was not installed, so it was not run.
