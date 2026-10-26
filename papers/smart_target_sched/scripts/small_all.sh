#!/bin/sh
# Small-scale run (Nmc = 20 for the tables, 10 for the figures), all arms and variants, tag `small`. CPU only.
#   usage: sh scripts/small_all.sh scalar|matrix|figs
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false JAX_PLATFORMS=cpu
PY=${PY:-"uv run python"}
R="$PY src/train.py"
case "$1" in
scalar)
  $R --tables t1 t2 t3 --K 1 2 3 --nmc 20 --tag small --resume
  $R --tables t1 t2 t3 --K 1 2 3 --nmc 20 --sampled --policies whittle myopic tev passive_cont_revisit --tag small --resume
  $R --tables t1 t2 t3 --K 1 2 3 --nmc 20 --skip-neg --policies whittle myopic tev --tag small --resume ;;
matrix)
  $R --tables t4 --K 1 2 3 --nmc 20 --tag small --resume
  $R --tables t4 --K 1 2 3 --nmc 20 --ft standard --policies whittle myopic tev --tag small --resume
  $R --tables t4 --K 1 2 3 --nmc 20 --pcli1 literal --policies whittle --tag small --resume
  $R --tables t4 --K 1 2 3 --nmc 20 --sampled --policies whittle myopic tev --tag small --resume ;;
figs)
  for f in fig6 fig5a fig5b fig5c; do
    $R --fig $f --Ks 2 6 --nmc 10 --policies whittle myopic tev passive_cont passive_cont_h10 passive_cont_revisit random --lb --tag small --resume
  done
  $R --fig fig6 --Ks 2 --nmc 10 --policies rollout1 --tag small --resume
  $R --fig fig6 --Ks 10 22 --nmc 10 --policies whittle myopic tev --lb --tag small --resume ;;
*) echo "usage: sh scripts/small_all.sh scalar|matrix|figs" >&2; exit 2 ;;
esac
