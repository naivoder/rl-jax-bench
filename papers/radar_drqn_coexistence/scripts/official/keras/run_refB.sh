#!/bin/sh
# Reference run B: Keras (TF 2.2) learner, Keras's own init, full-scale streams, E1 eq3 seeds 0-2.
# Needs refB_stream_s{0,1,2}.npz in runs/keras_oracle, written by dump_streams.py; the exact dump arguments of the
# original run were not recorded (assumed: --seed $s --exp mc_p04 --reading eq3 with the full-scale defaults).
# Keras venv: $OFFICIAL_ROOT/.venv-tf22 (scripts/official/setup.sh). Sequential, CPU.
set -e
cd "$(dirname "$0")/../../.."
OFFICIAL_ROOT=${OFFICIAL_ROOT:-external/official}
KPY=${KPY:-$OFFICIAL_ROOT/.venv-tf22/bin/python}
K=scripts/official/keras
mkdir -p runs/keras_oracle && cd runs/keras_oracle
for arm in ddrqn dqn; do
  for s in 0 1 2; do
    for adam in keras optax; do
      $KPY ../../$K/keras_ref.py --stream refB_stream_s$s.npz --arm $arm --init keras --tf-seed $s --adam $adam \
          --out refB_${arm}_${adam}adam_s$s.npz > refB_${arm}_${adam}adam_s$s.log 2>&1
    done
  done
done
