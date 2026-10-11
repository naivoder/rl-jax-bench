#!/bin/sh
# Recheck re-runs (pre-declared, 2026-10-11). Usage: scripts/recheck.sh <chain>
#   fo0..fo3 : recheck_d12_fo_pe, Fig. D.12 flip-one 3 M with the paper-era AdamGnT, streams 0-99 in 4 blocks (CPU)
#   rs       : recheck_d12_rs_pe, the same on re-sample 1 M, streams 0-99 (GPU; wall time -> runs/recheck_gpu_time.tsv)
#   act0/act1: recheck_act_conf, S-ACT CBP arms on never-used streams 2000-2199 in 2 blocks (CPU)
#   lin      : recheck_act_conf_lin, linear control on streams 2000-2199 (CPU)
set -e
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
seq_list() { $PY -c "print('[' + ','.join(str(i) for i in range($1, $2)) + ']')"; }
cpu() {
  tag=$1; cores=$2; shift 2
  t0=$(date +%s)
  JAX_PLATFORMS=cpu $PY -m src.train "$@" tag=$tag
  echo "$tag	$(( $(date +%s) - t0 ))	cpu" >> runs/recheck_cpu_time.tsv
}
case "$1" in
  fo0) cpu recheck_d12_fo_pe_p0 4,5 --config src/configs/scr_d12_paper_era.yaml --set "seeds=$(seq_list 0 25)" ;;
  fo1) cpu recheck_d12_fo_pe_p1 6,7 --config src/configs/scr_d12_paper_era.yaml --set "seeds=$(seq_list 25 50)" ;;
  fo2) cpu recheck_d12_fo_pe_p2 8,9 --config src/configs/scr_d12_paper_era.yaml --set "seeds=$(seq_list 50 75)" ;;
  fo3) cpu recheck_d12_fo_pe_p3 10,11 --config src/configs/scr_d12_paper_era.yaml --set "seeds=$(seq_list 75 100)" ;;
  act0) cpu recheck_act_conf_p0 12,13 --config src/configs/recheck_act_conf.yaml --set "seeds=$(seq_list 2000 2100)" ;;
  act1) cpu recheck_act_conf_p1 14,15 --config src/configs/recheck_act_conf.yaml --set "seeds=$(seq_list 2100 2200)" ;;
  lin) cpu recheck_act_conf_lin 12,13 --config src/configs/recheck_act_conf_linear.yaml --set ;;
  rs)
    t0=$(date +%s)
    $PY -m src.train --config src/configs/scr_d12_paper_era.yaml --set scr_mode=resample n_steps=1000000 tag=recheck_d12_rs_pe
    echo "recheck_d12_rs_pe	$(( $(date +%s) - t0 ))	gpu" >> runs/recheck_gpu_time.tsv ;;
esac
echo "CHAIN_DONE $1"
