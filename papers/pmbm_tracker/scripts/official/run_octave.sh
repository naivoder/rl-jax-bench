#!/bin/sh
# GNU Octave 10.3 with the work copy of the authors' code built by setup.sh. OCTAVE_HOME must be set because the conda
# env is used without activation. PMBM_ORACLE_ROOT = the work copy (official code + patches + shims).
HERE=$(cd "$(dirname "$0")" && pwd)
OFFICIAL_ROOT=${OFFICIAL_ROOT:-$HERE/../../external/official}
export OCTAVE_HOME=${OCTAVE_HOME:-$OFFICIAL_ROOT/octave_env}
export PMBM_ORACLE_ROOT=${PMBM_ORACLE_ROOT:-$OFFICIAL_ROOT/work}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # single-threaded BLAS: no oversubscription when several oracles run
exec "$OCTAVE_HOME/bin/octave-cli" --no-gui --norc "$@"
