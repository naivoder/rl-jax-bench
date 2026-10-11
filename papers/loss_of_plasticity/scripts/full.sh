#!/bin/sh
# Full-scale runs (pre-declared before launch). Each line is one vmapped GPU process; the wall time of
# every process is appended to runs/full_gpu_time.tsv (GPU-hour accounting).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
run() {
  tag=$1; shift
  t0=$(date +%s)
  $PY -m src.train "$@" tag=$tag
  echo "$tag	$(( $(date +%s) - t0 ))" >> runs/full_gpu_time.tsv
}
run full_scr_b10 --config src/configs/scr_b10.yaml --set
run full_scr_linear --config src/configs/scr_linear.yaml --set
run full_scr_d12_rs --config src/configs/scr_d12.yaml --set scr_mode=resample n_steps=1000000
run full_scr_methods --config src/configs/scr_methods_full.yaml --set
run full_scr_flip_methods --config src/configs/scr_flip_methods.yaml --set
run full_mnist --config src/configs/mnist_reduced.yaml --set track_wmag=false scan_unroll=4
run full_scr_d12 --config src/configs/scr_d12.yaml --set
echo FULL_RUNS_DONE
