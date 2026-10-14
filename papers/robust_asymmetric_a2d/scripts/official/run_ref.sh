#!/bin/sh
# One official reference run: run_ref.sh DRIVER LOGNAME ARGS...  (DRIVER = ref_run.py | ref_run2.py | ref_run3.py).
# Official source untouched; CPU, 1 thread. Layout under $OFFICIAL_ROOT (default external/official in the paper folder):
# repo/ = plai-group/a2d clone, venv/ = its python 3.9 venv. Logs: $REF_LOGS/<LOGNAME>.log (default
# $OFFICIAL_ROOT/oracle/runs3). Results land in the clone's own tests/results/AdaptAsymDagger/ tree.
HERE=$(cd "$(dirname "$0")" && pwd)
PAPER=$(cd "$HERE/../.." && pwd)
OFFICIAL_ROOT=${OFFICIAL_ROOT:-$PAPER/external/official}
REF_LOGS=${REF_LOGS:-$OFFICIAL_ROOT/oracle/runs3}
mkdir -p $REF_LOGS
DRIVER=$1 NAME=$2; shift 2
cd $OFFICIAL_ROOT/repo/a2d_gridworld
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD $OFFICIAL_ROOT/venv/bin/python $HERE/$DRIVER "$@" \
  > $REF_LOGS/$NAME.log 2>&1
echo "$(date +%T) exit=$? $DRIVER $*" >> $REF_LOGS/done.txt
