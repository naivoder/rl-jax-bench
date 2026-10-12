#!/bin/sh
# Every instrument / diagnostic command behind results/, in the order it was run. Run AFTER scripts/train_all.sh.
# Sequential here; the original run spread these over <= 6 CPU workers plus one GPU process. Commands marked
# 'gpu' were run on the GPU (landscape true-reward batches); everything else on CPU (JAX_PLATFORMS=cpu). Same keys,
# same numbers on either device. Official-code lines need $OFFICIAL_ROOT (see README, Checked against).
set -e
cd "$(dirname "$0")/.."
PAPER="$(pwd)"
PY=${PY:-"uv run python"}
OFFICIAL_ROOT=${OFFICIAL_ROOT:-"$PAPER/external/official"}

# ---- E3 baselines + V* + value floor (CPU). Originally run per arm by a helper for the arms not covered by the
# e3a/e3b jobs below; report.py reads whichever of full_e3 / full_e3a / full_e3b exists. actor_batch is a memory knob. ----
for A in hopper_ppo hopper_trpo walker2d_ppo walker2d_trpo; do
  $PY -m src.eval --run runs/full/$A --its 150 300 450 --exps baseline value --seeds 0 1 2 3 4 --chunk 10000 \
    --vstar_pairs 1000000 --base_sizes 100 1000 2000 4000 6000 8000 10000 --m 10 --actor_batch 32 --agent_batch 1 --tag full_e3
done


# ---- jobs ----
# land_hopper [gpu]
for a in ppo trpo; do $PY -m src.eval --run runs/full/hopper_$a --its 0 150 300 450 --exps landscape --seeds 0 1 2 3 4 --seed_idx 0 --chunk 10000 --land_few 2048 --land_many 100000 --land_true_pairs 100000 --land_true_chunk 2000 --land_grid 7 --land_step_chunk 2000 --tag full_land || exit 1; done
# land_walker2d [gpu]
for a in ppo trpo; do $PY -m src.eval --run runs/full/walker2d_$a --its 0 150 300 450 --exps landscape --seeds 0 1 2 3 4 --seed_idx 0 --chunk 10000 --land_few 2048 --land_many 100000 --land_true_pairs 100000 --land_true_chunk 2000 --land_grid 7 --land_step_chunk 2000 --tag full_land || exit 1; done
# e3a_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 150 300 --exps baseline value --seeds 0 1 2 3 4 --chunk 10000 --vstar_pairs 1000000 --base_sizes 100 1000 2000 4000 6000 8000 10000 --m 10 --actor_batch 32 --agent_batch 1 --tag full_e3a
# e3b_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 450 --exps baseline value --seeds 0 1 2 3 4 --chunk 10000 --vstar_pairs 1000000 --base_sizes 100 1000 2000 4000 6000 8000 10000 --m 10 --actor_batch 32 --agent_batch 1 --tag full_e3b
# e3b_walker2d_trpo [cpu]
$PY -m src.eval --run runs/full/walker2d_trpo --its 450 --exps baseline value --seeds 0 1 2 3 4 --chunk 10000 --vstar_pairs 1000000 --base_sizes 100 1000 2000 4000 6000 8000 10000 --m 10 --actor_batch 32 --agent_batch 1 --tag full_e3b
# e1a_hopper_ppo [cpu]
$PY -m src.eval --run runs/full/hopper_ppo --its 0 150 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 64 --agent_batch 1 --tag full_e1a
# e1b_hopper_ppo [cpu]
$PY -m src.eval --run runs/full/hopper_ppo --its 300 450 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 64 --agent_batch 1 --tag full_e1b
# e1a_hopper_trpo [cpu]
$PY -m src.eval --run runs/full/hopper_trpo --its 0 150 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 64 --agent_batch 1 --tag full_e1a
# e1b_hopper_trpo [cpu]
$PY -m src.eval --run runs/full/hopper_trpo --its 300 450 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 64 --agent_batch 1 --tag full_e1b
# e1a_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 0 150 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 32 --agent_batch 1 --tag full_e1a
# e1b_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 300 450 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 32 --agent_batch 1 --tag full_e1b
# e1a_walker2d_trpo [cpu]
$PY -m src.eval --run runs/full/walker2d_trpo --its 0 150 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 32 --agent_batch 1 --tag full_e1a
# e1b_walker2d_trpo [cpu]
$PY -m src.eval --run runs/full/walker2d_trpo --its 300 450 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 32 --agent_batch 1 --tag full_e1b
# g1e5a_hopper_ppo [cpu]
$PY -m src.eval --run runs/full/hopper_ppo --its 0 150 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1a --chunk 10000 --actor_batch 64 --agent_batch 1 --tag full_g1e5a
# g1e5b_hopper_ppo [cpu]
$PY -m src.eval --run runs/full/hopper_ppo --its 300 450 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1b --chunk 10000 --actor_batch 64 --agent_batch 1 --tag full_g1e5b
# g1e5a_hopper_trpo [cpu]
$PY -m src.eval --run runs/full/hopper_trpo --its 0 150 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1a --chunk 10000 --actor_batch 64 --agent_batch 1 --tag full_g1e5a
# g1e5b_hopper_trpo [cpu]
$PY -m src.eval --run runs/full/hopper_trpo --its 300 450 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1b --chunk 10000 --actor_batch 64 --agent_batch 1 --tag full_g1e5b
# g1e5a_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 0 150 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1a --chunk 10000 --actor_batch 32 --agent_batch 1 --tag full_g1e5a
# g1e5b_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 300 450 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1b --chunk 10000 --actor_batch 32 --agent_batch 1 --tag full_g1e5b
# g1e5a_walker2d_trpo [cpu]
$PY -m src.eval --run runs/full/walker2d_trpo --its 0 150 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1a --chunk 10000 --actor_batch 32 --agent_batch 1 --tag full_g1e5a
# g1e5b_walker2d_trpo [cpu]
$PY -m src.eval --run runs/full/walker2d_trpo --its 300 450 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1b --chunk 10000 --actor_batch 32 --agent_batch 1 --tag full_g1e5b

# ---- jobs2 ----
# landsteps_hopper_ppo [cpu]
$PY -m src.eval --run runs/full/hopper_ppo --its 0 150 300 450 --exps landscape --seeds 0 1 2 3 4 --seed_idx 0 --land_seed_idx 0 --chunk 10000 --land_few 2048 --land_many 100000 --land_step_chunk 2000 --land_grid 7 --land_phase steps --tag full_landsteps
# landsteps_hopper_trpo [cpu]
$PY -m src.eval --run runs/full/hopper_trpo --its 0 150 300 450 --exps landscape --seeds 0 1 2 3 4 --seed_idx 0 --land_seed_idx 0 --chunk 10000 --land_few 2048 --land_many 100000 --land_step_chunk 2000 --land_grid 7 --land_phase steps --tag full_landsteps
# landsteps_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 0 150 300 450 --exps landscape --seeds 0 1 2 3 4 --seed_idx 0 --land_seed_idx 0 --chunk 10000 --land_few 2048 --land_many 100000 --land_step_chunk 2000 --land_grid 7 --land_phase steps --tag full_landsteps
# landsteps_walker2d_trpo [cpu]
$PY -m src.eval --run runs/full/walker2d_trpo --its 0 150 300 450 --exps landscape --seeds 0 1 2 3 4 --seed_idx 0 --land_seed_idx 0 --chunk 10000 --land_few 2048 --land_many 100000 --land_step_chunk 2000 --land_grid 7 --land_phase steps --tag full_landsteps
# landtrue_hopper [gpu]
for a in ppo trpo; do $PY -m src.eval --run runs/full/hopper_$a --its 0 150 300 450 --exps landscape --seeds 0 1 2 3 4 --seed_idx 0 --land_seed_idx 0 --land_true_pairs 100000 --land_true_chunk 2000 --land_grid 7 --land_phase true --land_steps_from full_landsteps --land_point_batch 33 --tag full_land || exit 1; done
# landtrue_walker2d [gpu]
for a in ppo trpo; do $PY -m src.eval --run runs/full/walker2d_$a --its 0 450 --exps landscape --seeds 0 1 2 3 4 --seed_idx 0 --land_seed_idx 0 --land_true_pairs 100000 --land_true_chunk 2000 --land_grid 7 --land_phase true --land_steps_from full_landsteps --land_point_batch 49 --tag full_land || exit 1; done
# e3a_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 150 300 --exps baseline value --seeds 0 1 2 3 4 --chunk 10000 --vstar_pairs 1000000 --base_sizes 100 1000 2000 4000 6000 8000 10000 --m 10 --actor_batch 32 --agent_batch 1 --tag full_e3a
# e3b_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 450 --exps baseline value --seeds 0 1 2 3 4 --chunk 10000 --vstar_pairs 1000000 --base_sizes 100 1000 2000 4000 6000 8000 10000 --m 10 --actor_batch 32 --agent_batch 1 --tag full_e3b
# e1a_hopper_ppo [cpu]
$PY -m src.eval --run runs/full/hopper_ppo --its 0 150 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 64 --agent_batch 1 --tag full_e1a
# e1b_hopper_ppo [cpu]
$PY -m src.eval --run runs/full/hopper_ppo --its 300 450 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 64 --agent_batch 1 --tag full_e1b
# e1a_hopper_trpo [cpu]
$PY -m src.eval --run runs/full/hopper_trpo --its 0 150 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 64 --agent_batch 1 --tag full_e1a
# e1b_hopper_trpo [cpu]
$PY -m src.eval --run runs/full/hopper_trpo --its 300 450 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 64 --agent_batch 1 --tag full_e1b
# e1a_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 0 150 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 32 --agent_batch 1 --tag full_e1a
# e1b_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 300 450 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 32 --agent_batch 1 --tag full_e1b
# e1a_walker2d_trpo [cpu]
$PY -m src.eval --run runs/full/walker2d_trpo --its 0 150 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 32 --agent_batch 1 --tag full_e1a
# e1b_walker2d_trpo [cpu]
$PY -m src.eval --run runs/full/walker2d_trpo --its 300 450 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 32 --agent_batch 1 --tag full_e1b
# g1e5a_hopper_ppo [cpu]
$PY -m src.eval --run runs/full/hopper_ppo --its 0 150 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1a --chunk 10000 --actor_batch 64 --agent_batch 1 --tag full_g1e5a
# g1e5b_hopper_ppo [cpu]
$PY -m src.eval --run runs/full/hopper_ppo --its 300 450 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1b --chunk 10000 --actor_batch 64 --agent_batch 1 --tag full_g1e5b
# g1e5a_hopper_trpo [cpu]
$PY -m src.eval --run runs/full/hopper_trpo --its 0 150 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1a --chunk 10000 --actor_batch 64 --agent_batch 1 --tag full_g1e5a
# g1e5b_hopper_trpo [cpu]
$PY -m src.eval --run runs/full/hopper_trpo --its 300 450 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1b --chunk 10000 --actor_batch 64 --agent_batch 1 --tag full_g1e5b
# g1e5a_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 0 150 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1a --chunk 10000 --actor_batch 32 --agent_batch 1 --tag full_g1e5a
# g1e5b_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 300 450 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1b --chunk 10000 --actor_batch 32 --agent_batch 1 --tag full_g1e5b
# g1e5a_walker2d_trpo [cpu]
$PY -m src.eval --run runs/full/walker2d_trpo --its 0 150 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1a --chunk 10000 --actor_batch 32 --agent_batch 1 --tag full_g1e5a
# g1e5b_walker2d_trpo [cpu]
$PY -m src.eval --run runs/full/walker2d_trpo --its 300 450 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1b --chunk 10000 --actor_batch 32 --agent_batch 1 --tag full_g1e5b

# ---- jobs3 ----
# landtrue_walker2d [gpu]
for a in ppo trpo; do $PY -m src.eval --run runs/full/walker2d_$a --its 0 450 --exps landscape --seeds 0 1 2 3 4 --seed_idx 0 --land_seed_idx 0 --land_true_pairs 100000 --land_true_chunk 2000 --land_grid 7 --land_phase true --land_steps_from full_landsteps --land_point_batch 49 --tag full_land || exit 1; done
# e5nc_hopper [cpu]
$PY -m src.train --env hopper --alg ppo --adv_baseline zero --adv_center off --seeds 0 1 2 --iters 488 --out runs/full_diag --ckpt_iters --resume_every 100
# e5zg_hopper [cpu]
$PY -m src.train --env hopper --alg ppo --adv_baseline zero_gae --seeds 0 1 2 --iters 488 --out runs/full_diag --ckpt_iters --resume_every 100
# e3a_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 150 300 --exps baseline value --seeds 0 1 2 3 4 --chunk 10000 --vstar_pairs 1000000 --base_sizes 100 1000 2000 4000 6000 8000 10000 --m 10 --actor_batch 32 --agent_batch 1 --tag full_e3a
# e3b_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 450 --exps baseline value --seeds 0 1 2 3 4 --chunk 10000 --vstar_pairs 1000000 --base_sizes 100 1000 2000 4000 6000 8000 10000 --m 10 --actor_batch 32 --agent_batch 1 --tag full_e3b
# e1a_hopper_ppo [cpu]
$PY -m src.eval --run runs/full/hopper_ppo --its 0 150 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 64 --agent_batch 1 --tag full_e1a
# e1b_hopper_ppo [cpu]
$PY -m src.eval --run runs/full/hopper_ppo --its 300 450 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 64 --agent_batch 1 --tag full_e1b
# e1a_hopper_trpo [cpu]
$PY -m src.eval --run runs/full/hopper_trpo --its 0 150 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 64 --agent_batch 1 --tag full_e1a
# e1b_hopper_trpo [cpu]
$PY -m src.eval --run runs/full/hopper_trpo --its 300 450 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 64 --agent_batch 1 --tag full_e1b
# e1a_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 0 150 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 32 --agent_batch 1 --tag full_e1a
# e1b_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 300 450 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 32 --agent_batch 1 --tag full_e1b
# e1a_walker2d_trpo [cpu]
$PY -m src.eval --run runs/full/walker2d_trpo --its 0 150 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 32 --agent_batch 1 --tag full_e1a
# e1b_walker2d_trpo [cpu]
$PY -m src.eval --run runs/full/walker2d_trpo --its 300 450 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 32 --agent_batch 1 --tag full_e1b
# g1e5a_hopper_ppo [cpu]
$PY -m src.eval --run runs/full/hopper_ppo --its 0 150 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1a --chunk 10000 --actor_batch 64 --agent_batch 1 --tag full_g1e5a
# g1e5b_hopper_ppo [cpu]
$PY -m src.eval --run runs/full/hopper_ppo --its 300 450 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1b --chunk 10000 --actor_batch 64 --agent_batch 1 --tag full_g1e5b
# g1e5a_hopper_trpo [cpu]
$PY -m src.eval --run runs/full/hopper_trpo --its 0 150 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1a --chunk 10000 --actor_batch 64 --agent_batch 1 --tag full_g1e5a
# g1e5b_hopper_trpo [cpu]
$PY -m src.eval --run runs/full/hopper_trpo --its 300 450 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1b --chunk 10000 --actor_batch 64 --agent_batch 1 --tag full_g1e5b
# g1e5a_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 0 150 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1a --chunk 10000 --actor_batch 32 --agent_batch 1 --tag full_g1e5a
# g1e5b_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 300 450 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1b --chunk 10000 --actor_batch 32 --agent_batch 1 --tag full_g1e5b
# g1e5a_walker2d_trpo [cpu]
$PY -m src.eval --run runs/full/walker2d_trpo --its 0 150 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1a --chunk 10000 --actor_batch 32 --agent_batch 1 --tag full_g1e5a
# g1e5b_walker2d_trpo [cpu]
$PY -m src.eval --run runs/full/walker2d_trpo --its 300 450 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1b --chunk 10000 --actor_batch 32 --agent_batch 1 --tag full_g1e5b

# ---- jobs4 ----
# vvar_hopper_ppo [cpu]
$PY scripts/value_mre_variants.py runs/full/hopper_ppo 150 450
# vvar_hopper_trpo [cpu]
$PY scripts/value_mre_variants.py runs/full/hopper_trpo 150 450
# vvar_walker2d_ppo [cpu]
$PY scripts/value_mre_variants.py runs/full/walker2d_ppo 150 450
# vvar_walker2d_trpo [cpu]
$PY scripts/value_mre_variants.py runs/full/walker2d_trpo 150 450
# e1b_hopper_ppo [cpu]
$PY -m src.eval --run runs/full/hopper_ppo --its 300 450 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 64 --agent_batch 1 --tag full_e1b
# e1a_hopper_trpo [cpu]
$PY -m src.eval --run runs/full/hopper_trpo --its 0 150 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 64 --agent_batch 1 --tag full_e1a
# e1b_hopper_trpo [cpu]
$PY -m src.eval --run runs/full/hopper_trpo --its 300 450 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 64 --agent_batch 1 --tag full_e1b
# e1a_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 0 150 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 32 --agent_batch 1 --tag full_e1a
# e1b_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 300 450 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 32 --agent_batch 1 --tag full_e1b
# e1a_walker2d_trpo [cpu]
$PY -m src.eval --run runs/full/walker2d_trpo --its 0 150 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 32 --agent_batch 1 --tag full_e1a
# e1b_walker2d_trpo [cpu]
$PY -m src.eval --run runs/full/walker2d_trpo --its 300 450 --exps grad --seeds 0 1 2 3 4 --grad_sizes 100 1000 2000 10000 --ref_pairs 1000000 --chunk 10000 --pool_ablation --actor_batch 32 --agent_batch 1 --tag full_e1b
# g1e5a_hopper_ppo [cpu]
$PY -m src.eval --run runs/full/hopper_ppo --its 0 150 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1a --chunk 10000 --actor_batch 64 --agent_batch 1 --tag full_g1e5a
# g1e5b_hopper_ppo [cpu]
$PY -m src.eval --run runs/full/hopper_ppo --its 300 450 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1b --chunk 10000 --actor_batch 64 --agent_batch 1 --tag full_g1e5b
# g1e5a_hopper_trpo [cpu]
$PY -m src.eval --run runs/full/hopper_trpo --its 0 150 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1a --chunk 10000 --actor_batch 64 --agent_batch 1 --tag full_g1e5a
# g1e5b_hopper_trpo [cpu]
$PY -m src.eval --run runs/full/hopper_trpo --its 300 450 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1b --chunk 10000 --actor_batch 64 --agent_batch 1 --tag full_g1e5b
# g1e5a_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 0 150 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1a --chunk 10000 --actor_batch 32 --agent_batch 1 --tag full_g1e5a
# g1e5b_walker2d_ppo [cpu]
$PY -m src.eval --run runs/full/walker2d_ppo --its 300 450 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1b --chunk 10000 --actor_batch 32 --agent_batch 1 --tag full_g1e5b
# g1e5a_walker2d_trpo [cpu]
$PY -m src.eval --run runs/full/walker2d_trpo --its 0 150 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1a --chunk 10000 --actor_batch 32 --agent_batch 1 --tag full_g1e5a
# g1e5b_walker2d_trpo [cpu]
$PY -m src.eval --run runs/full/walker2d_trpo --its 300 450 --exps grad --seeds 0 1 2 3 4 --seed_idx 0 1 2 --grad_sizes 100000 --ref_pairs 0 --ref_from full_e1b --chunk 10000 --actor_batch 32 --agent_batch 1 --tag full_g1e5b

# ---- jobs5 ----
# vstar16k_hopper_ppo [cpu]
$PY -m src.eval --run runs/full/hopper_ppo --its 450 --exps baseline value --seeds 0 1 2 3 4 --seed_idx 0 1 2 --chunk 10000 --vstar_pairs 1000000 --vstar_steps 16000 --base_sizes 2000 10000 --m 10 --actor_batch 64 --agent_batch 1 --tag full_vstar16k
# offlate_hopper_trpo_s0 [cpu]
(cd "$OFFICIAL_ROOT" && OMP_NUM_THREADS=1 .venv/bin/python -I "$PAPER/scripts/official/official_grad_ref.py" hopper trpo 0 grad_late/hopper_trpo_s0.npz its=150,300,450 sizes=100,1000,2000,10000 m=10 ref=100000)
# offlate_hopper_trpo_s1 [cpu]
(cd "$OFFICIAL_ROOT" && OMP_NUM_THREADS=1 .venv/bin/python -I "$PAPER/scripts/official/official_grad_ref.py" hopper trpo 1 grad_late/hopper_trpo_s1.npz its=150,300,450 sizes=100,1000,2000,10000 m=10 ref=100000)
# offlate_hopper_trpo_s2 [cpu]
(cd "$OFFICIAL_ROOT" && OMP_NUM_THREADS=1 .venv/bin/python -I "$PAPER/scripts/official/official_grad_ref.py" hopper trpo 2 grad_late/hopper_trpo_s2.npz its=150,300,450 sizes=100,1000,2000,10000 m=10 ref=100000)
# offlate_hopper_ppo_s0 [cpu]
(cd "$OFFICIAL_ROOT" && OMP_NUM_THREADS=1 .venv/bin/python -I "$PAPER/scripts/official/official_grad_ref.py" hopper ppo 0 grad_late/hopper_ppo_s0.npz its=150,300,450 sizes=100,1000,2000,10000 m=10 ref=100000)
# offlate_hopper_ppo_s1 [cpu]
(cd "$OFFICIAL_ROOT" && OMP_NUM_THREADS=1 .venv/bin/python -I "$PAPER/scripts/official/official_grad_ref.py" hopper ppo 1 grad_late/hopper_ppo_s1.npz its=150,300,450 sizes=100,1000,2000,10000 m=10 ref=100000)
# offlate_hopper_ppo_s2 [cpu]
(cd "$OFFICIAL_ROOT" && OMP_NUM_THREADS=1 .venv/bin/python -I "$PAPER/scripts/official/official_grad_ref.py" hopper ppo 2 grad_late/hopper_ppo_s2.npz its=150,300,450 sizes=100,1000,2000,10000 m=10 ref=100000)

# ---- extra/extra ----
# e5nc_hopper [cpu]
$PY -m src.train --env hopper --alg ppo --adv_baseline zero --adv_center off --seeds 0 1 2 3 4 --iters 488 --out runs/full_extra --ckpt_iters --resume_every 100
# e5nc_walker2d [cpu]
$PY -m src.train --env walker2d --alg ppo --adv_baseline zero --adv_center off --seeds 0 1 2 3 4 --iters 488 --out runs/full_extra --ckpt_iters --resume_every 100
# landr_hopper_ppo [cpu]
sh scripts/landscape_rerun.sh hopper_ppo 0 150 300 450
# landr_hopper_trpo [cpu]
sh scripts/landscape_rerun.sh hopper_trpo 0 150 300 450
# landr_walker2d_ppo [cpu]
sh scripts/landscape_rerun.sh walker2d_ppo 0 450
# landr_walker2d_trpo [cpu]
sh scripts/landscape_rerun.sh walker2d_trpo 0 450
# e3raw [cpu]
sh scripts/e3_uncentered.sh hopper_ppo walker2d_ppo hopper_trpo walker2d_trpo
# vstar_time_hopper_trpo [cpu]
$PY scripts/vstar_time.py runs/full/hopper_trpo 450 0 1 2 3 4
# vstar_time_hopper_ppo [cpu]
$PY scripts/vstar_time.py runs/full/hopper_ppo 450 0 1 2 3 4
# vstar_time_walker2d_ppo [cpu]
$PY scripts/vstar_time.py runs/full/walker2d_ppo 450 0 1 2 3 4
# vstar_time_walker2d_trpo [cpu]
$PY scripts/vstar_time.py runs/full/walker2d_trpo 450 0 1 2 3 4
