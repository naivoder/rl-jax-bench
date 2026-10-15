#!/bin/sh
# Review-stage runs (CPU only), one after another. Run from the paper folder.
set -e
PY=${PY:-"uv run python"}
TR="$PY -m src.train"
G1='arms={ppo_g1: {time_mode: none, T: 1000, gamma: 1.0}, ta_ppo_g1: {time_mode: elapsed_pm1, T: 1000, gamma: 1.0}}'
G0='arms={ppo_g099: {time_mode: none, T: 1000, gamma: 0.99}, ta_ppo_g099: {time_mode: elapsed_pm1, T: 1000, gamma: 0.99}}'
# E5 critic dynamics to 1e6 steps, one process per gamma (orbax cannot checkpoint the MJX port's env state: --no-ckpt)
$TR --config src/configs/e5_ip_dynamics_gymv4.yaml --tag review --no-ckpt --set "$G1" --set name=e5_ip_dynamics_gymv4_g1
$TR --config src/configs/e5_ip_dynamics_gymv4.yaml --tag review --no-ckpt --set "$G0" --set name=e5_ip_dynamics_gymv4_g099
# ppo1-style readings of E4: 4 arms x 30 seeds x 1e5 steps
S30="seeds=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000]"
E4=src/configs/e4_inverted_pendulum_gymv4.yaml
P1="--set anneal_lr=true --set clip_vloss=false --set reward_scaling=false --set obs_clip=5.0 --set tau_obs_norm=true"
$TR --config $E4 --tag review --no-ckpt --set "$S30" $P1 --set name=e4_ip_gymv4_ppo1bundle
# (1.0e+10, not 1.0e10: YAML 1.1 reads the latter as a string)
$TR --config $E4 --tag review --no-ckpt --set "$S30" $P1 --set adv_norm_scope=batch --set vf_coef=1.0 --set max_grad_norm=1.0e+10 --set anneal_clip=true --set name=e4_ip_gymv4_ppo1_2018
# Hopper TA-PPO vs PPO, gamma = 1, T = 1000, 2e6 steps, 10 seeds, Brax PROXY (the gym-v4 Hopper port ran at
# 1.0e3 env-steps/s on CPU, below the 1.2e4 threshold set for using it)
C=src/configs/e4_hopper_ta.yaml
X="--set env=brax-hopper --set device_reason=port_1.0e3_steps_per_s_below_threshold"
$TR --config $C --tag review $X --set "seeds=[0, 1000, 2000, 3000, 4000]" --set name=e4_hopper_ta_proxy_p1
$TR --config $C --tag review $X --set "seeds=[5000, 6000, 7000, 8000, 9000]" --set name=e4_hopper_ta_proxy_p2
JAX_PLATFORMS=cpu $PY scripts/review/analyze_review.py
