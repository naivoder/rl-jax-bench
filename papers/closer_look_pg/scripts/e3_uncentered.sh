#!/bin/sh
# E3 'zero' arm without centring (zero_raw) on the full-scale E3 rollouts (same estimate keys); 'zero' is recomputed
# alongside as a bit-identity check against full_e3*.json. CPU. Usage: sh scripts/e3_uncentered.sh <arm...>
set -e
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
for A in "$@"; do
  $PY -m src.eval --run runs/full/$A --its 150 300 450 --exps baseline --seeds 0 1 2 3 4 --chunk 10000 \
    --base_sizes 100 1000 2000 4000 6000 8000 10000 --m 10 --base_kinds zero zero_raw --actor_batch 32 --agent_batch 1 --tag fullr_e3raw
done
