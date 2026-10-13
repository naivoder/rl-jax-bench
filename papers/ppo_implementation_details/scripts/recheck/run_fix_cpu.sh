#!/bin/sh
# Recheck re-runs, JAX on CPU, classic control, 500k steps. Usage: run_fix_cpu.sh <worktree of the pre-fix commit>
#   before/after: pre-fix commit (worktree at $1) vs current code, separate/shared x CartPole/Acrobot, seeds 1-10
#   vf_coef 0.1 / 0.01 for separate AND shared, CartPole/Acrobot, seeds 1-10
#   fresh seeds 101-120, separate/shared x CartPole/Acrobot
# (Originally split into two parallel lanes; here it runs sequentially.)
set -e
cd "$(dirname "$0")/../.."
R=$(pwd)
PY=${PY:-"uv run python"}
S10="seeds=[1,2,3,4,5,6,7,8,9,10]"
S20="seeds=[101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120]"
export JAX_PLATFORMS=cpu
run() { (cd "$1" && shift && $PY -m src.train --config src/configs/classic_control.yaml --no-ckpt "$@" 2>&1 | grep -E "timing|ERROR|Traceback|WARN.*nan" ); }
for env in CartPole-v1 Acrobot-v1; do
  for arm in separate shared; do
    echo "prefix $env $arm"; run "$1" --tag recheck_prefix_cpu --envs $env --arms $arm --set "$S10"
    echo "postfix $env $arm"; run $R --tag recheck/fix/postfix_cpu --envs $env --arms $arm --set "$S10"
    echo "fresh $env $arm"; run $R --tag recheck/fix/fresh_cpu --envs $env --arms $arm --set "$S20"
    for vf in 0.1 0.01; do
      echo "vf $vf $env $arm"; run $R --tag recheck/fix/vf$vf --envs $env --arms $arm --set "$S10" --set vf_coef=$vf
    done
  done
done
