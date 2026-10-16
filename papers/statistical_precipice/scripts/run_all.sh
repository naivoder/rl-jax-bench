#!/bin/sh
# Every command behind results/, in order, sequentially. Run from anywhere; works in the paper folder.
# Steps marked [official] need the authors' rliable checkout and its own venv under $OFFICIAL_ROOT (see README).
# Steps marked [pdf] need the paper PDF at $PAPER_PDF (default external/data/<name>.pdf).
set -e
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
OFFICIAL_ROOT=${OFFICIAL_ROOT:-external/official}
OPY="$OFFICIAL_ROOT/.venv/bin/python"
SEEDS=$(seq -s, 1 100)

# E0: synthetic inputs, rliable reference outputs, differential test
$PY scripts/make_inputs.py                                   # -> $DATA_ROOT/e0/inputs.npz
$PY -c "from src import data; data.fetch_dopamine()"         # Dopamine ALE JSON at the pinned commit (~25 MB)
$OPY scripts/official/rliable_reference.py                   # [official] -> $DATA_ROOT/e0/oracle.npz (~15 min CPU)
$PY scripts/e0_difftest.py                                   # -> runs/e0/difftest.json
$OPY -I scripts/official/reference_ale.py                    # [official] -> $DATA_ROOT/official/reference_ale.npz
$PY scripts/official_compare.py                              # -> runs/official/compare.json

# Toy checks and the 20-seed small-scale run
$PY scripts/toys.py                                          # -> runs/toys/metrics.json
$PY -m src.train --tag small --skip-done                     # E6, seeds 1-20 (e6.yaml default)
$PY scripts/small_scale.py                                   # -> runs/small/metrics.json

# Paper read-offs from the PDF's vector drawings
$PY scripts/paper/vector_readoff.py                          # [pdf] -> data/paper_readoff/readoff.json
$PY scripts/paper/a17_readoff.py                             # [pdf] -> runs/full/paper_a17.json
$PY scripts/paper/fig10_readoff.py                           # [pdf] -> runs/full/paper_fig10.json

# Full run: ALE / DMC numbers and the 100-seed E6 study (training ~1.2 h on 3 CPU cores)
$PY scripts/full_ale.py                                      # -> runs/full/ale.json
$PY -m src.train --tag full --set "seeds=[$SEEDS]"
$PY scripts/full_e6.py                                       # -> runs/full/e6.json
$PY scripts/full_e6.py --declared --only coverage,fewrun --out runs/full/e6_declared.json
$PY scripts/data_provenance.py                               # -> runs/full/data_provenance.json
$PY scripts/full_e6_sens.py                                  # -> runs/full/e6_sens.json

# Recheck pass: theory, semi-synthetic pool, reward-scaling sensitivity
$PY scripts/recheck_theory.py                                # -> runs/full/recheck_theory.json
$PY scripts/recheck_semisynth.py                             # -> runs/full/recheck_semisynth.json
$PY -m src.train --tag recheck_rs --set "seeds=[$SEEDS]" --set reward_scaling=true
$PY -m src.train --tag recheck_rs_fresh --set "seeds=[$(seq -s, 101 200)]" --set reward_scaling=true \
    --set "e6.arms=[separate,shared]"                        # originally run on the GPU
$PY scripts/recheck_rs.py                                    # -> runs/full/recheck_rs.json

# Tables
$PY scripts/report.py all
$PY scripts/recheck_report.py
