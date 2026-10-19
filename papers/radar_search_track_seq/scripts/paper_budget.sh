#!/bin/sh
# Paper-budget learner runs, one encoder family per process, run one after another (GPU recommended).
# The common iteration cap is a timing-only budget decision. It is written to <out>/train_<family>/n_it_override_<enc>
# before launch. Recorded values: 180 PPO iterations (2,949,120 steps per lane) for the paper-budget run and the
# lr-1e-3 run, 48 iterations (786,432 steps) for the verification block.
#   sh scripts/paper_budget.sh                                   # paper-budget run: mhsa, bigru, flat x seeds 0-4
#   OUT=runs/verify_block CAP=48 FAMILIES="v_mhsa v_flat v_unmasked v_single" SEEDS=0,1,2 EXTRA=--eval-final sh scripts/paper_budget.sh
#   OUT=runs/lr1e3 FAMILIES=v_mhsa_lr1e-3 EXTRA=--eval-final sh scripts/paper_budget.sh
# then: uv run python scripts/paper_budget_report.py | verify_block_report.py | lr1e3_report.py
set -eu
PY=${PY:-"uv run python"}
OUT=${OUT:-runs/paper_budget}
CAP=${CAP:-180}
for fam in ${FAMILIES:-mhsa bigru flat}; do
  # encoder of the family (the override file is named after it)
  case $fam in
    mhsa|v_mhsa|v_mhsa_lr1e-3|v_unmasked) enc=mhsa ;;
    flat|v_flat|v_single) enc=flat ;;
    bigru) enc=bigru ;;
    *) echo "unknown family $fam"; exit 1 ;;
  esac
  mkdir -p "$OUT/train_$fam"
  echo "$CAP" > "$OUT/train_$fam/n_it_override_$enc"
  $PY scripts/paper_budget.py --family "$fam" --steps 10000000 --select-best-patience 2500000 \
      --seeds "${SEEDS:-0,1,2,3,4}" ${EXTRA:-} --out "$OUT"
done
