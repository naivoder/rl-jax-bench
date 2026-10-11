#!/bin/sh
# Follow-up (review resolution) runs, CPU only. Usage: scripts/full_extra.sh <chain>
# chains: d12_0 d12_1 d12_2 (full_scr_d12 in three stream blocks), b10pt (B.10 paper-text, then D.12 sensitivity
# default + convention P), mnist (final-seed L2 1e-3 / CBP+bias transfer, 15 tasks), snp (S&P shrink 1e-3, then D.12 beta2).
set -e
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=cpu
PY=${PY:-"uv run python"}
run() {
  tag=$1; shift
  t0=$(date +%s)
  $PY -m src.train "$@" tag=$tag
  echo "$tag	$(( $(date +%s) - t0 ))	cpu" >> runs/full_extra_cpu_time.tsv
}
seq_list() { $PY -c "print('[' + ','.join(str(i) for i in range($1, $2)) + ']')"; }
case "$1" in
  d12_0) run full_scr_d12_p0 --config src/configs/scr_d12.yaml --set "seeds=$(seq_list 0 34)" ;;
  d12_1) run full_scr_d12_p1 --config src/configs/scr_d12.yaml --set "seeds=$(seq_list 34 67)" ;;
  d12_2) run full_scr_d12_p2 --config src/configs/scr_d12.yaml --set "seeds=$(seq_list 67 100)" ;;
  b10pt)
    run full_extra_scr_b10_papertext --config src/configs/scr_b10.yaml --set paper_text=true n_seeds=30
    run full_extra_d12_sens_default --config src/configs/d12_sens_default.yaml --set
    run full_extra_d12_sens_convP --config src/configs/d12_sens_convP.yaml --set ;;
  mnist)   # amended before launch (CPU budget): held-out final seeds 1000-1004, 15 tasks
    run full_extra_mnist_final15 --config src/configs/mnist_w100_final15.yaml --set ;;
  snp)     # S&P with the width-100 L2 shrink, one sigma, sweep seeds 0-2 (stopped after 2 tasks: CPU budget)
    run full_extra_mnist_snp_1e-3 --config src/configs/mnist_w100_snp_1e-3.yaml --set ;;
  beta2)
    run full_extra_d12_sens_beta2 --config src/configs/d12_sens_beta2.yaml --set ;;
esac
echo "CHAIN_DONE $1"
