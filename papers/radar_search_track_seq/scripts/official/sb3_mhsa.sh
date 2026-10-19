#!/bin/sh
# The authors' SB3 / imitation forks with a torch BiGRU+MHSA extractor on our environment, seeds 0-2, CPU.
# Needs sh scripts/official/setup.sh first. The recorded run passed no extra flags (TAG bcfrozen); other variants
# were run with EXTRA set, and those flag values are not recorded.
#   sh scripts/official/sb3_mhsa.sh
#   sh scripts/official/sb3_mhsa.sh --record-update tests/fixtures/official/sb3_mhsa_update.npz   # re-record the P11 fixture
set -eu
OFFICIAL_ROOT=${OFFICIAL_ROOT:-external/official}
PY="$OFFICIAL_ROOT/.venv-sb3/bin/python"
mkdir -p runs/official/sb3_mhsa
if [ $# -gt 0 ]; then "$PY" -I scripts/official/mhsa_sb3.py --seed 0 --threads 2 "$@"; exit; fi
for s in 0 1 2; do
  "$PY" -I scripts/official/mhsa_sb3.py --seed $s --threads 2 ${EXTRA:-} > runs/official/sb3_mhsa/${TAG:-bcfrozen}_seed$s.log 2>&1
done
