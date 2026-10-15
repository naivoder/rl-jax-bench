#!/bin/sh
# Full-scale runs, one after another (the original ran these groups concurrently). Run from the paper folder.
# Roughly 7 CPU-hours plus a few GPU-minutes for E3 on the original machine.
set -e
PY=${PY:-"uv run python"}
TR="$PY -m src.train"
S7="total_timesteps=3000000"
# E3 Queue of Cars
$TR --config src/configs/e3_queue_of_cars.yaml --tag full --no-ckpt
$TR --config src/configs/e3_queue_of_cars.yaml --tag full --no-ckpt --set qoc_semantics=crash_first --set name=e3_queue_of_cars_crash_first
$TR --config src/configs/e3_queue_of_cars_horizon.yaml --tag full --no-ckpt
# E7 Hopper proxy, 3e6 steps, 10 seeds in three blocks
$TR --config src/configs/e7_hopper.yaml --tag full --set $S7 --set "seeds=[0, 1000, 2000, 3000]" --set name=e7_hopper_p1
$TR --config src/configs/e7_hopper.yaml --tag full --set $S7 --set "seeds=[4000, 5000, 6000]" --set name=e7_hopper_p2
$TR --config src/configs/e7_hopper.yaml --tag full --set $S7 --set "seeds=[7000, 8000, 9000]" --set name=e7_hopper_p3
# E9 Difficult Gridworld (Buffer-Q)
$PY scripts/small/run_e9.py --tag full
$PY scripts/small/run_e9.py --tag full --set update_mode=batch_mean --set name=e9_difficult_grid_batch_mean
# E4 InvertedPendulum: gym-v4 port (30 seeds), Brax proxy, and the PPO-detail ablations
SEEDS30="seeds=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000]"
$TR --config src/configs/e4_inverted_pendulum_gymv4.yaml --tag full --no-ckpt --set "$SEEDS30"
$TR --config src/configs/e4_inverted_pendulum.yaml --tag full --no-ckpt
A="src/configs/e4_ip_gymv4_ablation.yaml"
TWO='arms={ppo_g1: {time_mode: none, T: 1000, gamma: 1.0}, ta_ppo_g1: {time_mode: elapsed_pm1, T: 1000, gamma: 1.0}}'
G099='arms={ppo_g099: {time_mode: none, T: 1000, gamma: 0.99}, ta_ppo_g099: {time_mode: elapsed_pm1, T: 1000, gamma: 0.99}}'
$TR --config $A --tag full --no-ckpt
$TR --config $A --tag full --no-ckpt --set "$TWO" --set tau_obs_norm=true --set name=e4_ip_gymv4_ablation_taunorm
$TR --config $A --tag full --no-ckpt --set "$TWO" --set adv_norm_scope=batch --set name=e4_ip_gymv4_ablation_advbatch
$TR --config $A --tag full --no-ckpt --set "$TWO" --set anneal_lr=true --set name=e4_ip_gymv4_ablation_annealLR
$TR --config $A --tag full --no-ckpt --set "$TWO" --set anneal_lr=true --set clip_vloss=false --set reward_scaling=false --set obs_clip=5.0 --set name=e4_ip_gymv4_ablation_ppo1like
# follow-up at gamma = 0.99 (the original command line of the ppo1-like gamma-0.99 variant was not recorded;
# this assumes the same flags as the gamma-1 ppo1-like run above)
$TR --config $A --tag full --no-ckpt --set "$G099" --set anneal_lr=true --set name=e4_ip_gymv4_ablation_annealLR_g099
$TR --config $A --tag full --no-ckpt --set "$G099" --set anneal_lr=true --set clip_vloss=false --set reward_scaling=false --set obs_clip=5.0 --set name=e4_ip_gymv4_ablation_ppo1like_g099
JAX_PLATFORMS=cpu $PY scripts/full/report.py
