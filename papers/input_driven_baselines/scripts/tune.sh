#!/bin/sh
# Compute and tuning parity: the SAME 8-point grid and the SAME 2 tuning seeds (100, 101) for each baseline arm
# (state, multi, maml); selection metric = last-5-checkpoint mean on the VALIDATION inputs (VAL_SEED, disjoint from
# test); full length (2,500 iterations). Then the winning config (src/configs/tuned.yaml) on seeds 0-9.
#   grid: gamma {0.995 (paper), 1.0 (official code)} x critic lr {1e-3 (paper), 1e-2}
#         x entropy decay {10000 iterations (paper App. K), 1000 (official code, -1e-3 per iteration)}
#   (for maml the critic lr is the meta (outer) lr; its inner loop stays at the small-scale choice)
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
set -e
for arm in ${ARMS:-state multi maml}; do
  if [ "$arm" = maml ]; then ckey=meta_outer_lr; else ckey=lr_critic; fi
  for g in 0.995 1.0; do for lc in 1.0e-3 1.0e-2; do for ed in 10000 1000; do
    $PY -m src.train --arm $arm --seeds 100,101 --out runs/tune/g${g}_c${lc}_e${ed} --set eval_split=val \
        --set gamma=$g --set $ckey=$lc --set entropy_decay_iters=$ed
  done; done; done
done
for arm in state multi maml; do
  $PY -m src.train --config src/configs/tuned.yaml --arm $arm --seeds 0,1,2,3,4,5,6,7,8,9 --out runs/full_tuned
done
$PY -m src.train --config src/configs/tuned.yaml --arm multi_shuffled --seeds 0,1,2,3,4 --out runs/full_tuned_ablation
$PY -m src.train --config src/configs/tuned.yaml --arm maml --seeds 0,1,2,3,4 --out runs/full_tuned_ablation_noadapt \
    --set meta_inner_steps=0
$PY -m src.eval --probe runs/full_tuned --arms state --out runs/full_tuned/probe_state.json
$PY -m src.eval --probe runs/full_tuned --arms multi,maml --out runs/full_tuned/probe_mm.json
echo TUNE_DONE
