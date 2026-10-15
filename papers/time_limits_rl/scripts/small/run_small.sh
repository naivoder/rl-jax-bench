#!/bin/sh
# Small-scale pass over every arm (a cheap preview of the full protocol). Run from the paper folder.
set -e
PY=${PY:-"uv run python"}
# E3 Queue of Cars: 1e5 steps (1/3 of the full budget), all 4 arms, 10 seeds
$PY -m src.train --config src/configs/e3_queue_of_cars.yaml --tag small --set total_timesteps=100352 --no-ckpt
$PY -m src.train --config src/configs/e3_queue_of_cars.yaml --tag small --set total_timesteps=100352 \
    --set qoc_semantics=crash_first --set name=e3_queue_of_cars_crash_first --no-ckpt
# E9 Buffer-Q: 2e5 steps (10x smaller than Figure 7), 30 seeds, 4 buffers x PEB on/off (CPU)
$PY scripts/small/run_e9.py --tag small --set total_steps=200000 --set bin_steps=20000
# E4/E5 IP proxy: the paper's 1e5 steps, 5 seeds, 4 arms (CPU)
$PY -m src.train --config src/configs/e4_inverted_pendulum.yaml --tag small --set "seeds=[0, 1000, 2000, 3000, 4000]"
# E7 hopper proxy: 3e5 steps (10x smaller than the paper's 3e6), 3 seeds, 3 arms (CPU)
$PY -m src.train --config src/configs/e7_hopper.yaml --tag small --set total_timesteps=301056 \
    --set "seeds=[0, 1000, 2000]"
$PY scripts/small/report.py
