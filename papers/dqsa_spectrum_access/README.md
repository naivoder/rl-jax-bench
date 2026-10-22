# Deep Multi-User Reinforcement Learning for Distributed Dynamic Spectrum Access
Oshri Naparstek and Kobi Cohen, IEEE Transactions on Wireless Communications 18(1), 2019. Paper:
https://arxiv.org/abs/1704.02613 (v3). Official code: none released for this paper. A third-party TensorFlow
implementation, [shkrwnd/Deep-Reinforcement-Learning-for-Dynamic-Spectrum-Access](https://github.com/shkrwnd/Deep-Reinforcement-Learning-for-Dynamic-Spectrum-Access)
(commit `2219c28`), was used as a reference oracle. It is a different algorithm, not the paper's.

## What this is
The paper proposes DQSA: N users share K channels, and each user picks a channel (or stays silent) every slot from
its own action history and ACKs. All users run one shared dueling LSTM Q-network, trained centrally with double-Q
and then deployed without any message exchange. The paper reports near-Pareto channel sharing on small cases and
higher network rates than slotted Aloha and two CSI-based baselines (OCA, DP) at N = 100, K = 50. This folder
reimplements the multi-user channel world (cliques, channel subsets, Rayleigh fading, primary users), the DQSA
network and training box, the three objectives, and all baselines in JAX. It reruns the small cases on CPU.

Experiment labels used in code and tables:
- E1: cliques, Fig. 3, competitive reward.
- E6: N = 4, K = 2, sum rate (arXiv v1/v2).
- E7: N = 3, K = 2, competitive (v1/v2).
- E8: N = 4, K = 2, proportional fairness (v2).
- E2 / E3: N = 100, K = 50, Figs. 4-6. E3 adds primary users. `E2proxy` / `E3proxy` are N = 20, K = 10 CPU proxies.
- E0: N = 2, K = 2 sanity toy, not a paper case.
- T1-T3, T-ALOHA, T-RATE, T-SHK, T-PATTERN, T-GRAD: toy checks with exact or closed-form answers.

## Results
All learner numbers come from selection seeds 100-109. Training is R = 10 000 iterations of the paper's box
(16 fresh episodes x 50 slots, one Adam step per iteration). Evaluation uses the frozen network with Eq. 11 at
alpha = 0, beta = 20, on 1000 shared test worlds, plateau slots 60-199. The 10 report seeds (0-9) were never
trained, and no GPU run was made.

| Quantity | Paper | This reproduction (mean ± SE, n seeds) | Verdict |
|---|---|---|---|
| Toy checks (closed forms, exact enumerations, gradients) | - | 67 / 67 pass | PASS |
| Slotted Aloha, N = 100, K = 50 (closed form 39.92 Mbps) | - | 39.92 ± 0.02 Mbps | MATCH |
| E1 plateau utilization, competitive | 0.7695 (vector read of Fig. 3) | 0.058 (1 seed); Aloha 0.403 | DIVERGENT |
| E6 throughput, sum rate | > 0.9, pattern in ~80 % of runs | 0.415 ± 0.005 (5), strict Pareto 0 | DIVERGENT |
| E7 utilization, competitive | pattern "one alone + two TDMA" in ~80 % of runs | 0.498 ± 0.087 (10, one group), pattern 0.005; Aloha 0.444 | NOT SEPARATED from Aloha |
| E8 throughput, proportional fairness | 0.6, near-equal sharing in > 95 % | 0.412 ± 0.003 (3) | DIVERGENT |
| E2 DQSA rate, N = 100, K = 50 | ~82 Mbps | not run (GPU-bound) | UNTESTED |
| E2 OCA / DP baselines (reconstructed) | ~65 / ~56 Mbps | OCA 54.6, DP tuned 62.9 at T_s = 1 ms; up to 72.2 / 75.8 over T_s 0.1-10 ms | level depends on the unstated T_s |
| E3 strongest CSI baseline | OCA ~36-37 Mbps | 38.0 Mbps (best over T_s) | comparable |

Overall: DIVERGENT. The checks of the implementation found no bug: equations, environment, metric, baselines,
gradients and the comparison with the third-party code all hold up. The learner does not reach the paper's
coordination in the paper's budget on any small case. The extra checks below did not complete.

Caveats:
- E7 was first reported as 0.603 ± 0.089 (5 seeds). Two of those seeds came from an earlier code version and
  cannot be reproduced. At the final code the same 5 seeds give 0.451 ± 0.137, and 10 seeds in one vmapped group
  give 0.498. The learning rate is not decided: 1e-4, 3e-4 and 1e-3 tie on one 10-seed E7 group
  (0.488 / 0.498 / 0.493).
- A seed's outcome depends on the size of its vmapped group. Batched reductions differ in the last bit, and the
  training dynamics amplify that (E7 seed 104: 0.851 in groups of 3 and 10, 0.421 alone). Same group plus same code
  is bit-identical (`scripts/diff_runs.py`).
- E1 collapses to "everyone transmits" (collision 0.94). With an own-reward TD target, transmitting is weakly
  dominant, so this is a property of the literal algorithm box, not a transcription error. The third-party code
  collapses too.
- E0 sits at random play (0.50) at 10k iterations on 3/3 seeds. At 40k iterations 2 of 3 seeds reach 0.9997.
  A multi-epoch reading of "train DQN1 with inputs xs and outputs Qs" leaves the plateau more often (5 epochs:
  0.612 ± 0.073, n = 9, and 0.778 ± 0.126, n = 6, vs 0.535 ± 0.036, n = 9). The difference is not significant
  at the declared threshold (p 0.09-0.24), and the escapes are unstable. Whether the gap is budget or the reading of
  that step stays undetermined.
- E6 does not move with rewards / N (0.413 ± 0.001) or with the v1/v2 episodic reward (0.049 ± 0.024 literal,
  0.388 ± 0.006 at 1/T). So the shared-reward explanation for E6 is unsupported.
- The order of the log-rate curves (Fig. 5) depends on the unit and log base, which the paper does not state.
  The order of the four baseline arms changes in 5 of 32 cells, so the objectives' ranking is reported as not
  gradable.
- Files in `results/` are the final outputs, with three exceptions. `results/baselines_v2/first_grid_proxy/` is the
  first, superseded DP tuning grid. The small-stage DQSA per-seed runs are not shipped, so the table rows above for
  E1, E6-E8 are quoted from the run logs. `results/recheck/report.json` uses the arm names A1/A2 (5 epochs, carried
  or fresh Adam), B1 (rewards / N), C1/C2 (episodic reward, literal and 1/T scale) and BASE (E6 control).

## What differs from the paper / known gaps
- **Not run:** the paper-scale DQSA runs (E2 / E3, N = 100), the report seeds 0-9, E6 lr selection, multi-epoch
  E6 / E8 and E7, multi-epoch groups of 10 seeds, the Matlab-default arm (30 epochs, fresh Adam, L2 1e-4, stopped
  at ~500 iterations at 0.24 it/s), the policy-invariant reward rescale, beta at 20 from the start, and multi-clique
  E1 batches. All were too expensive for the CPU budget.
- **Assumed** where the paper is silent (`src/configs/sources.yaml` tags every value): lr, Adam settings, the shape
  of the alpha / beta ramps, glorot init, ReLU on the 10-unit dueling layers, slot duration T_s = 1 ms, primary-user
  stay probability 0.9, one channel per clique in E1, the test horizon of E6-E8.
- **Baselines are reconstructed.** OCA and DP are described only in words. DP's literal reading (every user
  re-samples each slot) herds; `dp_tuned` adds an update fraction tuned on separate worlds. The TDMA genie is a
  reference, not an upper bound.
- **E2 / E3 input:** the primary arm uses a static capacity input (constant 1, equal bandwidths). The instantaneous
  rate (CSI) input is the labelled ablation `*csi`. Pareto on E2 / E3 is measured against the feasible matching.
- **Training-regime gap.** The third-party learner leaves the E0 / E7 plateaus within ~30k updates, faster than
  ours per iteration. Their update is one replay minibatch of 6 x 5 slots; ours is 16 episodes x N users x 50
  slots. At matched update counts (10-20k) it is not ahead on E7 (0.339 / 0.640 vs 0.498).

## Layout
- `src/data.py`: the channel-access world: cliques and channel subsets, collision rule, ACK, Rayleigh AR(1) fading, primary users, observation.
- `src/model.py`: LSTM(100) with dueling heads (Eq. 10), Eq. 11 sampling, alpha / beta schedules.
- `src/algorithm.py`: reward variants, the on-policy common-training driver (double-Q, target copy every 5 iterations), all baselines.
- `src/eval.py`: frozen-network test rollouts on shared worlds, utilization / rate / log-rate curves, Pareto and pattern metrics, Aloha closed forms.
- `src/train.py`: CLI training driver: vmapped seeds, orbax checkpoint / resume, NaN guard, `launch.json` provenance, `runs/<tag>/seed_<n>/metrics.json`.
- `src/toys.py`: toy checks (closed forms, exact enumerations, T1 learner fixed point, third-party env differential test T-SHK).
- `src/configs/experiments.yaml`, `sources.yaml`: experiment specs, and every value with its source tag.
- `data/paper_figures/`: exact curve vertices read from the paper's vector figures (v1 Fig. 2, v3 Figs. 3-6) and `exact_reads.txt`.
- `scripts/small_scale.sh`: all small-stage training runs, diagnostics, the 10-seed lr groups and the 40k / acting-rule checks.
- `scripts/baselines_all.sh`: every non-learning arm and the DP tuning.
- `scripts/baselines_eval.py`, `scripts/tune_dp.py`: one baseline evaluation; the DP (p_update x tau) tuning on tuning worlds.
- `scripts/recheck_rerun.sh`, `scripts/recheck_all.sh`: the extra-check arms (multi-epoch, reward scale, episodic reward) and their reports.
- `scripts/ts_sweep.py`, `scripts/ts_report.py`: T_s sweep of the CSI baselines with DP re-tuned per T_s; log-rate order check.
- `scripts/small_report.py`, `scripts/review_tables.py`, `scripts/recheck_report.py`, `scripts/toys_table.py`, `scripts/diag_report.py`: markdown tables.
- `scripts/e0_pooled.py`, `scripts/seed_distribution.py`, `scripts/budget_rule.py`: pooled E0 Fisher test, per-seed distribution, declared budget rule.
- `scripts/diff_runs.py`, `scripts/probe_group_size.py`, `scripts/probe_q.py`: bit-for-bit run diff, group-size probe, Q-values of a trained net.
- `scripts/official/`: third-party oracle harness: `setup.sh`, `run_shk.py` (runs their TF1 script under a tf.compat.v1 shim), `shk_all.sh`, `analyze_shk.py`, `aggregate_shk.py`, `make_fixtures.py`, `tf_requirements.txt`.

## Run it
From this folder, with the root environment (`uv sync` at the repo root). CPU only; set `JAX_PLATFORMS=cpu`.

    uv run python src/toys.py --skip-t1                     # toy checks, ~10 min; T-SHK needs scripts/official/setup.sh
    uv run python src/toys.py --only t1 --t1-lr 3e-4 --t1-iters 10000
    uv run python src/train.py --exp E7 --seeds 100 101 102 103 104 105 106 107 108 109 --lr 3e-4 --out runs/small_g10
    sh scripts/small_scale.sh                               # every small-stage run, many CPU-hours
    sh scripts/baselines_all.sh                             # baselines; paper-scale DP arms take hours
    sh scripts/recheck_all.sh                               # extra-check arms, ~19 process-hours originally

Outputs go to `runs/` (git-ignored). A 10k-iteration small-case run of 3-10 vmapped seeds took roughly one to a few
hours on one loaded CPU core. The N = 20 proxy ran at 0.68 s per iteration per seed, so the E2 / E3 grid is a GPU
job. Some original run arguments were not recorded and are reconstructed in `small_scale.sh` (noted there), and the
E1K2 replicate count is assumed. Float32 throughout; float64 only in the finite-difference gradient test.

## Tests
`uv run pytest` (about 2 min on CPU, 40 tests, no external data):
- `test_smoke.py`: env / network / rollout shapes, one training step, one eval, baselines.
- `test_grads.py`: float64 `check_grads` of the dueling-LSTM double-Q TD loss, a by-hand numpy loss, the dueling
  Jacobian identity, and full-vector regression gradient == scalar TD gradient.
- `test_resume.py`: an interrupted and resumed run ends with the same parameters as an uninterrupted one.
- `test_review_fixes.py`: feasible-matching metric, rate-weighted rewards, previous-slot OCA / DP, fixed subsets,
  monitor-world offset, launch provenance, same-group determinism.
- `test_recheck.py`: fresh-optimizer / L2 multi-epoch update, episodic reward, log-rate unit variants, tolerant
  pattern metrics, beta rescale.
- `test_official_parity.py`: replays `tests/fixtures/official/*.npz`, recorded from the third-party code by
  `scripts/official/make_fixtures.py`. It covers success vectors, their capacity input, input layout, Eq. 11
  probabilities, TD target, the forward pass, loss and all 8 gradient tensors of their QNetwork rebuilt on our LSTM
  cell, and our metrics on their rollouts.

## Checked against
No author code exists; the authors used Matlab and did not release it. The oracle is shkrwnd's third-party
TensorFlow implementation at commit `2219c28b6fd12c416dfd32763641b0786c32bafc`, run unmodified under TensorFlow
2.15.1 `tf.compat.v1` (Python 3.11, CPU). `run_shk.py` re-creates the removed `tf.contrib` symbols, and its layer
norm is our re-implementation. `sh scripts/official/setup.sh` clones it into `$OFFICIAL_ROOT/shkrwnd` (default
`external/official`) and builds the venv; it is untested in this layout.
- The environments agree exactly: 0 mismatches over 1e4 random joint actions (T-SHK) and in the fixture tests.
- Network forward pass, loss and gradients agree to float32 tolerance. TF Adam and optax Adam followed the same
  200-step trajectory within 2.3e-4 of the distance travelled (a one-off check, script not shipped).
- Reference runs (`scripts/official/shk_all.sh`, 100k online slots, utilization on their training stream, last 5k
  slots):
  - N = 3, K = 2 with our objective and Eq. 11 sampling: 0.966 ± 0.001 (2 seeds).
  - Their default: 0.882 ± 0.037 (5).
  - N = 4, K = 2: 0.266 ± 0.063 (3).
  - Their "one alone + two TDMA" pattern frequency: 0.08, against the paper's ~0.8.
  - On cliques their learner reaches 0.17, against the paper's 0.77.
  - The third-party code does not reproduce the paper either.
- The matched-update comparison runs (E0 / E6 / E7 at 10k-20k updates with the paper's settings) used flags that
  are not recorded, so they are not in `shk_all.sh`.
