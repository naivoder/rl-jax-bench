#!/bin/sh
# Every reference run of the authors' code behind the "Checked against" section, sequentially, on CPU.
# Run `sh scripts/official/setup.sh` first. Outputs go to $OFFICIAL_ROOT/ref_runs/{out,diag,logs}; finished runs are
# skipped. Then, in our environment: scripts/official/summarize_reference.py, lh_eval_crosscheck.py, mg_eval_crosscheck.py.
# Original cost (one process each): PD 3-4 min, 2D-LH 1-6 min, MiniGrid 10-15 min at 3e5 steps and 33-45 min at 1e6.
set -eu
cd "$(dirname "$0")/../.."
ROOT=${OFFICIAL_ROOT:-external/official}
. "$ROOT/env.sh"
mkdir -p "$ROOT/ref_runs/out" "$ROOT/ref_runs/diag" "$ROOT/ref_runs/logs"

job() {  # job <dir> <name> <total_steps or 0> <run_reference args...>
    dir=$1; name=$2; steps=$3; shift 3
    [ -f "$ROOT/ref_runs/$dir/$name.json" ] && return 0
    if [ "$steps" != "0" ]; then set -- "$@" --total_steps "$steps"; fi
    REF_TOTAL_STEPS=$([ "$steps" != "0" ] && echo "$steps" || true) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        "$ROOT/.venv/bin/python" scripts/official/run_reference.py "$@" --save_weights \
        --out "$ROOT/ref_runs/$dir/$name.json" > "$ROOT/ref_runs/logs/$name.log" 2>&1
}

# Poisoned Doors at the authors' hp draws 3, 7, 48
for d in 3 7 48; do for e in advisor ppo bc bc_then_ppo; do
    job out pd_${e}_d$d 0 --task PoisonedDoors --exp $e --draw $d
done; done
# 2D-Lighthouse, i = 1, seeds 0-2
for s in 0 1 2; do
    for e in bc advisor_ppo; do for j in 1 15; do
        job out lh_${e}_i1_j${j}_s$s 0 --task LightHouse --exp $e --view 1 --expert_view $j --seed $s
    done; done
    job out lh_ppo_i1_s$s 0 --task LightHouse --exp ppo --view 1 --expert_view 0 --seed $s
    job out full_lh_dagger_i1_s$s 0 --task LightHouse --exp dagger --view 1 --expert_view 1 --seed $s
done
for s in 0 1 2 3 4 5; do
    job out full_lh_ppo_i5_s$s 0 --task LightHouse --exp ppo --view 5 --expert_view 5 --seed $s
done
# MiniGrid LC Once Switch: 3e5-step proxy and the paper's 1e6 steps at draws 3, 7, 48
for s in 0 1 2; do job out mg_advisor_s$s 300000 --task AskForHelpLavaCrossingOnce --exp advisor --lr 1e-3 --alpha 8 --seed $s; done
job out mg_bc_s0 300000 --task AskForHelpLavaCrossingOnce --exp bc --lr 1e-3 --seed 0
job out mg_ppo_s0 300000 --task AskForHelpLavaCrossingOnce --exp ppo --lr 1e-3 --seed 0
for d in 3 7 48; do job out mg1e6_advisor_d$d 0 --task AskForHelpLavaCrossingOnce --exp advisor --draw $d; done
# trained 2D-LH weights for scripts/official/lh_eval_crosscheck.py
for e in bc advisor_ppo; do for s in 10 11 12 13 14; do
    job diag lh_${e}_i1_j1_s$s 0 --task LightHouse --exp $e --view 1 --expert_view 1 --seed $s
done; done
