#!/bin/sh
# Recheck re-runs under the paper-era ("2018") Baselines PPO semantics, CPU only, one after another.
# Each run writes runs/recheck/rerun/<name>/. Run from the paper folder. ~30 CPU-hours in total.
set -e
PY=${PY:-"uv run python"}
export JAX_PLATFORMS=cpu
TR="$PY -m src.train"
T="--tag recheck/rerun --no-ckpt"
S10="seeds=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]"
SFRESH="seeds=[$(seq -s ', ' 30000 1000 59000)]"
PPO2018="--set action_interface=raw --set reward_scale_reset_on_done=false --set tau_obs_norm=true"
PPO1="--set anneal_lr=true --set anneal_clip=true --set clip_vloss=false --set vf_coef=1.0 --set max_grad_norm=1.0e+10 --set adv_norm_scope=batch --set reward_scaling=false --set obs_clip=5.0 --set stop_rule=ppo1_completed_steps"
G1='arms={ppo_g1: {time_mode: none, T: 1000, gamma: 1.0}, ta_ppo_g1: {time_mode: remaining_pm1, T: 1000, gamma: 1.0}}'
G0='arms={ppo_g099: {time_mode: none, T: 1000, gamma: 0.99}, ta_ppo_g099: {time_mode: remaining_pm1, T: 1000, gamma: 0.99}}'
E5=src/configs/e5_ip_dynamics_gymv4.yaml
RO="readout_updates=[7, 14, 21, 28, 35, 42, 49, 98, 147, 196, 245, 294, 343, 392, 441]"
# E4 InvertedPendulum gym-v4 port: ppo2-2018 (paper seeds, then 30 fresh seeds), ppo1-2018, Brax proxy under ppo2-2018
$TR --config src/configs/e4_ip_gymv4_ppo2_2018.yaml $T
$TR --config src/configs/e4_ip_gymv4_ppo2_2018.yaml $T --set "$SFRESH" --set name=e4_ip_gymv4_ppo2_2018_fresh
$TR --config src/configs/e4_ip_gymv4_ppo1_2018.yaml $T
$TR --config src/configs/e4_inverted_pendulum_ppo2_2018.yaml $T
# E5 critic dynamics under ppo2-2018 and ppo1-2018
$TR --config $E5 $T $PPO2018 --set "$S10" --set "$G1" --set name=e5_ppo2_2018_g1
$TR --config $E5 $T $PPO2018 --set "$S10" --set "$G0" --set name=e5_ppo2_2018_g099
$TR --config $E5 $T $PPO2018 $PPO1 --set "$S10" --set "$G1" --set "$RO" --set name=e5_ppo1_2018_g1
$TR --config $E5 $T $PPO2018 $PPO1 --set "$S10" --set "$G0" --set "$RO" --set name=e5_ppo1_2018_g099
# Hopper proxies under ppo2-2018
$TR --config src/configs/e7_hopper_ppo2_2018.yaml $T --set "seeds=[0, 1000, 2000, 3000]" --set name=e7_hopper_ppo2_2018_p1
$TR --config src/configs/e7_hopper_ppo2_2018.yaml $T --set "seeds=[4000, 5000, 6000]" --set name=e7_hopper_ppo2_2018_p2
$TR --config src/configs/e7_hopper_ppo2_2018.yaml $T --set "seeds=[7000, 8000, 9000]" --set name=e7_hopper_ppo2_2018_p3
$TR --config src/configs/e4_hopper_ta_ppo2_2018.yaml $T --set "seeds=[0, 1000, 2000, 3000, 4000]" --set name=e4_hopper_ta_ppo2_2018_p1
$TR --config src/configs/e4_hopper_ta_ppo2_2018.yaml $T --set "seeds=[5000, 6000, 7000, 8000, 9000]" --set name=e4_hopper_ta_ppo2_2018_p2
# ppo1 with its TF1 numerics (float32 running moments, ppo1's Adam): E4 paper seeds, fresh seeds, E5
$TR --config src/configs/e4_ip_gymv4_ppo1_2018.yaml $T --set ppo1_numerics=true --set name=e4_ip_gymv4_ppo1_2018_numerics
$TR --config src/configs/e4_ip_gymv4_ppo1_2018.yaml $T --set ppo1_numerics=true --set "$SFRESH" --set name=e4_ip_gymv4_ppo1_2018_numerics_fresh
$TR --config $E5 $T $PPO2018 $PPO1 --set ppo1_numerics=true --set "$S10" --set "$G1" --set "$RO" --set name=e5_ppo1n_2018_g1
$TR --config $E5 $T $PPO2018 $PPO1 --set ppo1_numerics=true --set "$S10" --set "$G0" --set "$RO" --set name=e5_ppo1n_2018_g099
$PY scripts/recheck/analyze_recheck.py
