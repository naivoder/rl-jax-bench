#!/bin/sh
# full-scale probes for the Fig. D.12 adaptation row (GPU; the CPU variant was too slow on the shared host).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
for tc in full_probe_d12_adapt:d12_adapt_probe full_probe_d12_papertext:d12_papertext_probe; do
  tag=${tc%%:*}; cfg=${tc##*:}
  t0=$(date +%s)
  $PY -m src.train --config src/configs/$cfg.yaml --set tag=$tag
  echo "$tag	$(( $(date +%s) - t0 ))" >> runs/full_gpu_time.tsv
done
echo PROBES_DONE
