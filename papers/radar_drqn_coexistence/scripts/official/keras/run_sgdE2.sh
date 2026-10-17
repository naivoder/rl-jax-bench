#!/bin/sh
# Plain-SGD oracle check (recheck rule 6): Keras (TF 2.2) learners with plain SGD (Eq. 9) on the recheck pool
# (seed_root 20261018), E2 mc_p02 eq3, seeds 0-2, 100 + 70 CPIs, OUR init (--init ours) and OUR replay indices,
# paired with runs/recheck arm_{dqn_sgd,ddrqn_sgd}. CPU only, sequential. Then: python scripts/official/keras/compare_sgdE2.py
set -e
cd "$(dirname "$0")/../../.."
OFFICIAL_ROOT=${OFFICIAL_ROOT:-external/official}
KPY=${KPY:-$OFFICIAL_ROOT/.venv-tf22/bin/python}
PY=${PY:-"uv run python"}
K=scripts/official/keras
mkdir -p runs/keras_oracle
for s in 0 1 2; do
  JAX_PLATFORMS=cpu $PY $K/dump_streams.py --seed $s --exp mc_p02 --reading eq3 --seed-root 20261018 --train-cpis 100 \
      --eval-cpis 70 --out runs/keras_oracle/sgdE2_stream_s$s.npz > runs/keras_oracle/sgdE2_dump_s$s.log 2>&1
done
cd runs/keras_oracle
for s in 0 1 2; do
  for arm in ddrqn dqn; do
    $KPY ../../$K/keras_ref.py --stream sgdE2_stream_s$s.npz --arm $arm --init ours --sgd --out sgdE2_${arm}_s$s.npz > sgdE2_${arm}_s$s.log 2>&1
  done
done
