# Maintaining Plasticity in Deep Continual Learning
Shibhansh Dohare, J. Fernando Hernandez-Garcia, Parash Rahman, A. Rupam Mahmood, Richard S. Sutton, arXiv preprint
2023 (v3; a revised version appeared in Nature 2024 as "Loss of plasticity in deep continual learning"). Paper:
https://arxiv.org/abs/2306.13812 (v3). Official code: [shibhansh/loss-of-plasticity](https://github.com/shibhansh/loss-of-plasticity)
(commit `a6b7958`, which tracks the Nature version).

## What this is
The paper shows that standard deep networks trained by backpropagation (BP) on a long sequence of tasks gradually lose
the ability to learn ("loss of plasticity"). It proposes continual backpropagation (CBP): a generate-and-test step that
re-initializes a small fraction of low-utility hidden units after every update. This folder reimplements CBP, its
baselines (BP, L2, Shrink-and-Perturb (S&P), Adam), the six utility measures of the Appendix D ablation and the
plasticity diagnostics (dead units, effective rank, mean weight magnitude) as vmappable JAX functions. It reruns the
Slowly-Changing Regression (SCR) experiments (Fig. B.10, Fig. D.12, the method comparison) and a width-100 Online
Permuted MNIST run. Continual ImageNet, Slippery Ant and the width-2000 MNIST figures were not run.

Labels used in code and tables: P1-P10 are the pre-declared primary tests (Holm-corrected): P1-P6 the Fig. D.12
utility orderings, P7-P10 the MNIST orderings CBP > BP, S&P > BP, L2 > BP, CBP > S&P. S-ACT is the "CBP fixes loss of
plasticity for every activation" claim, S-DIAG the MNIST diagnostic orderings, S-MECH the mechanistic claims.
"CBP" always means the official-code CBP (no bias transfer, replacement count rho times the number of eligible units).
Judgments: MATCH within 5 % of the paper (or inside its read precision plus our SE), CLOSE within 20 %, DIVERGENT
otherwise. Paper values are read from the figures of arXiv v3.

## Results
| Quantity | Paper | This reproduction (mean ± SE, n) | Verdict |
|---|---|---|---|
| Fig. B.10: BP loses plasticity (6 activations x 3 step sizes, SGD, 3 M steps) | yes | 18/18 arms, Holm p ≤ 6e-37 (100 streams) | HOLDS |
| Fig. B.10 last 40k bins | figure | 18/18 MATCH; ReLU 1.22, tanh 0.86 (authors' code 1.17, 0.78, n = 10) | MATCH |
| Fig. B.10 first 40k bins | figure | 12 MATCH / 6 CLOSE / 0 DIVERGENT on streams 0-99 (about 7 % above); on streams 100-1999, strict 5 % rule 14 / 4 / 0, +2.9 % | MATCH / CLOSE |
| Fig. B.10 linear baseline | 0.80-0.82 | 0.823 | MATCH |
| Fig. D.12, re-sample stream 1 M steps (25 cells) | figure | paper-era optimizer: 19 MATCH / 2 CLOSE / 4 DIVERGENT; current optimizer 13 / 8 / 4 (100 streams); overall utility at rho 1e-4: 0.317 ± 0.005 vs 0.32 | mostly MATCH (reading chosen after seeing results) |
| Fig. D.12, flip-one stream 3 M steps (25 cells) | figure | paper-era 3 / 15 / 7, current 2 / 15 / 8; small-rho CBP cells 14-22 % below the paper (overall 1e-4: 0.265 ± 0.004 vs 0.32) | CLOSE / DIVERGENT |
| Fig. D.12 adaptation-utility row | 0.64 / 0.63 / 0.67 / 0.80 | e.g. rho 1e-4 re-sample 0.453 ± 0.007 (authors' code 0.443 ± 0.028, n = 3) | DIVERGENT, unresolved (authors' code agrees with us) |
| Fig. D.12 orderings P1-P6 | yes | 6/6 in all four readings (largest Holm p 0.0015) | HOLDS |
| SCR method ordering CBP < S&P < L2 < BP | yes (§6 text) | all Holm p ≤ 2e-17 on held-out streams; authors' code same order (0.574 < 0.623 < 0.780 < 1.027) | HOLDS (gains vs BP about halve when every arm's step size is tuned) |
| §6: CBP removes loss of plasticity for all activations (S-ACT) | yes | no detectable late rise for 6/6 activations on 200 never-used streams (95 % UB ≤ +0.013 vs BP +0.55 / +0.62) | HOLDS |
| MNIST width 100 BP curve (Fig. 2b middle) | drop 4.98 pp (vector curve) | drop 4.93 ± 0.09 pp, 10 seeds, 110 of 150 tasks | MATCH |
| MNIST P7-P9 (CBP, S&P, L2 better than BP) | yes | 3/3, Holm p ≤ 7e-11 | HOLDS |
| MNIST P10 (CBP > S&P) | yes (width 2000) | width 100, App. A S&P (1e-3, sigma 1e-6) minus CBP = +0.77 ± 0.04 pp at tasks 26-30 (n = 3) | REVERSED at width 100 (30 tasks only) |
| MNIST CBP vs L2 | CBP better (width 2000) | L2 1e-3 leads CBP by +0.76 ± 0.04 pp (tasks 11-15, 5/5 held-out seeds) | not supported at width 100 |
| MNIST bias transfer (§6) | helps | −0.002 ± 0.062 pp | no effect |
| Ablation: random utility, rho → 0, eta = 0, m, bias transfer | drops as implied | random utility +0.087 / +0.400 vs paper +0.10 / +0.42 | MATCH |

Caveats:
- The Fig. D.12 protocol is ambiguous: Appendix D does not state the stream or horizon. The official Adam configs all
  use the re-sample stream for 1 M steps, the appendix text suggests flip-one for 3 M steps. Both are reported. The
  re-sample reading was chosen after its agreement was seen, so treat its MATCH count as post hoc.
- The official optimizer (`AdamGnT`) changed after the paper: the version at the paper's date (`dcc3619`) has a
  different bias-correction step. Both are implemented (`adam_rule: standard | paper_era`) and both are reported.
- The +2.9 % offset of the Fig. B.10 first bins on the stream population is unattributed (the paper's own stream
  sample or a ~3 % generator difference). The extra +3.9 % of streams 0-99 is stream sampling.
- The MNIST run stopped at task 110 of 150 (budget). Width 100 with 10 seeds, not width 2000 with 30.
- The S-MECH correlates are descriptive only: their pooled p-values were pseudo-replicated and are withdrawn.

## What differs from the paper / known gaps
- **SCR problem.** We use the official code's 20-bit problem, not the 22-bit problem of Table B.3. On the paper-text
  22-bit problem the Fig. B.10 first bins do not reproduce (0 MATCH / 4 CLOSE / 14 DIVERGENT, 30 streams); the official
  problem does.
- **CBP is the official-code CBP** (no bias transfer, official bias-correction indices). Config keys `convention`
  (C official | P paper Eqs. 4/8), `weight_reduction` (mean | sum), `count_base` (eligible | layer = Alg. 1 n_l * rho)
  and `bias_transfer` give the paper-text variants. None moves Fig. D.12 toward the paper.
- **Not run (budget):** Continual ImageNet (Figs. 1, 7, the headline result), Slippery Ant, width-2000 / 800-task
  Permuted MNIST, the Dropout / Adam / Online-Norm MNIST arms, S&P with the width-100 shrink beyond 2 tasks, L2 1e-3 at
  tasks 101-110, the 110-task horizon for P10.
- **Unresolved:** the Fig. D.12 adaptation row and the small-rho flip-one cells. The authors' code from both optimizer
  versions reproduces our numbers on identical streams, not the figure's.
- **Results files predate later revisions.** `results/full_tables.json` holds the first full-scale verdicts;
  `results/full_extra_tables.json` (review follow-up) and `results/recheck_tables.json` (second checking pass) revise
  several rows. The table above gives the final verdicts.
- MNIST width-100 L2 / S&P carry the width-2000 hyperparameters in the main run; the width-100 sweep is the follow-up.

## Layout
- `src/model.py`: flax MLP (activation chosen by a vmappable id), Kaiming-uniform init with the official gains.
- `src/algorithm.py`: one pure step for every method: SGD or per-element Adam (AdamGnT, both versions), L2, S&P
  noise, then CBP generate-and-test (utility conventions C/P/E, six utility types, accumulator or Bernoulli counts,
  optional bias transfer).
- `src/diagnostics.py`: effective rank, dead units, saturated units, mean |w| (jittable, vmappable).
- `src/data.py`: SCR stream and target LTU net (official and paper-text variants); MNIST loader with per-task permutations.
- `src/train.py`: vmapped runs (seeds x arms), `lax.scan` chunks, orbax checkpoints with resume, NaN guard.
- `src/eval.py`: reports (mean ± SE, paired tests), oracle-reference comparison, per-seed collection.
- `src/toys.py`: exactly computable toy checks (hand-computed utility tables, newborn units, counts, diagnostics).
- `src/configs/*.yaml`: one config per experiment; `sources.yaml` tags every setting with its source;
  `paper_reads/` holds the Fig. 2b curve read from the paper.
- `scripts/small.sh`, `scripts/small_tables.py`: small-scale version of every run (10 SCR streams, 15 MNIST tasks).
- `scripts/full.sh`, `scripts/full_probes.sh`: full-scale runs and the Fig. D.12 adaptation-row probes.
- `scripts/full_extra.sh`, `scripts/full_extra_tables.py`: review follow-up runs (paper protocol D.12, B.10 paper-text,
  D.12 sensitivity, MNIST held-out) and their tables.
- `scripts/recheck.sh`, `scripts/recheck_tables.py`: second checking pass (paper-era optimizer, never-used streams).
- `scripts/full_tables.py`, `scripts/full_markdown.py`: full-scale tables and pre-declared tests, Markdown rendering.
- `scripts/merge_parts.py`: merge runs split into stream blocks. `scripts/partial_mnist.py`: metrics of a run stopped
  at a checkpoint. `scripts/method_cost.py`: FLOPs and CPU time per method.
- `scripts/oracle/`: differential tests and reference runs against the official code. `run_oracle.py` runs inside the
  official-code venv; `record_fixtures*.py` re-record `tests/fixtures/`; `official_pipeline.py` and
  `pipeline_summary.py` run the authors' end-to-end pipeline; the rest compare their learner with ours on shared streams.
- `results/`: `toys.json`, `small_tables.json`, `full_tables.json`, `full_extra_tables.json`, `recheck_tables.json`,
  `oracle_diff.json`, `official_pipeline_summary.json`, `official_learner_on_our_streams.json`, `method_cost.json`.

## Run it
From this folder, with the root environment (`uv sync` at the repo root). MNIST (training set only, md5-checked):

    mkdir -p data/mnist && cd data/mnist && for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz; do
      curl -sSfLO https://ossci-datasets.s3.amazonaws.com/mnist/$f; done; cd ../..

    JAX_PLATFORMS=cpu uv run python -m src.toys                       # toy checks, ~1 min CPU
    uv run python -m src.train --config src/configs/scr_b10.yaml      # Fig. B.10, 100 runs x 3 M steps
    sh scripts/small.sh                                               # everything at small scale
    sh scripts/full.sh && sh scripts/full_probes.sh                   # full scale; ~1.4 GPU-h on an RTX 3090
    sh scripts/full_extra.sh d12_0   # ... d12_1 d12_2 b10pt mnist snp beta2 (CPU, ~11 CPU-h in total)
    uv run python -m scripts.merge_parts full_scr_d12 full_scr_d12_p0 full_scr_d12_p1 full_scr_d12_p2
    sh scripts/recheck.sh rs         # ... fo0-fo3 act0 act1 lin (CPU except rs, ~7 CPU-h)
    uv run python -m scripts.full_tables && uv run python -m scripts.full_extra_tables && uv run python -m scripts.recheck_tables

`--set key=value` overrides any config key. Runs resume from `runs/<tag>/ckpt`; a changed config refuses to resume
unless `fresh=true`. The original runs used about 2.5 GPU-hours and about 35 CPU-process-hours in total.

## Tests
`uv run pytest` (about 1.5 min on CPU, one thread). No external data needed.
- `test_toys.py`: the toy checks as tests (utility tables, newborn units, bias correction, counts, Adam reset).
- `test_grads.py`: finite-difference gradients (float64).
- `test_smoke.py`: SCR and MNIST-shaped steps for every method.
- `test_options.py`: exact noise skip, paper-literal replacement count, Eq. 4/8 convention, both AdamGnT rules.
- `test_official_parity.py`: replays `tests/fixtures/official*/*.npz`, recorded from the unmodified official code
  (inputs and its outputs only; `MANIFEST.json` pins the commit): LTU target, generate-and-test, S&P, diagnostics,
  multi-step BP/L2/Adam trajectories, the MNIST width-100 trajectory, and the paper-era optimizer.

## Checked against
The official code at commit `a6b79580d85f3025bdb601566d3627c5f489f13b`, plus the tree `git archive 63c35f3`
(identical except `lop/utils/AdamGnT.py`, which equals the paper-era `dcc3619`), run unmodified on CPU in a separate
venv (Python 3.11, torch 2.4.1+cpu, numpy 2.4.6, scipy 1.17.1). Point `OFFICIAL_ROOT` at a directory holding the
clone as `repo/`, the archived tree as `repo_pre_f00ca40/`, the venv as `.venv/` and a scratch `work/`; the default is
`external/official` in this folder. `LOP_ORACLE_REPO` selects the tree for `run_oracle.py`.
- Teacher-forced CBP + AdamGnT steps agree to ≤ 1.2e-6 with the matching optimizer version; the fixtures tell the
  two versions apart.
- With identical initial weights on the official generator's streams, ReLU SGD runs are identical to the official
  ones over 3 M steps.
- The authors' complete pipeline reproduces Fig. B.10, the SCR ordering and the MNIST width-100 shape and, like ours,
  not the Fig. D.12 adaptation row (`results/official_pipeline_summary.json`).
