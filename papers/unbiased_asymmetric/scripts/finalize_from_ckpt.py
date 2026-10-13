"""Write per-run metrics.json from a cell's latest orbax checkpoint (runs stopped before T_max by the GPU budget).

    JAX_PLATFORMS=cpu python scripts/finalize_from_ckpt.py --config src/configs/full_heavenhell3.yaml --method a2c
The checkpoint holds the full run state incl. the 200-slot log buffers, so the curve up to the checkpointed step is
exactly what the full-length run would have logged (T_max only sets the log cadence T_max/200; the entropy schedule is
keyed to 2M steps and the target period to 10k, independent of T_max). `final` = mean of the last 10 logged points
before the stop; the record carries truncated=True and t_stop."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import train  # noqa: E402  (runs gpu_bootstrap)

import numpy as np  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--note", default="gpu (stopped at the GPU budget; finalised from checkpoint)")
    ap.add_argument("--run-offset", type=int, default=0)   # slices launched with train.py --run-offset
    a = ap.parse_args()
    cfg = train.resolve_config(a.config, a.method, dict(run_offset=a.run_offset or None))
    off = int(cfg.get("run_offset", 0))
    sfx = "" if off == 0 else f"_off{off}"
    od = train.out_dir(cfg)
    chash = train.config_hash(cfg)
    old = open(os.path.join(od, f"ckpt{sfx}.config.sha256")).read().strip()
    if old != chash:
        raise RuntimeError(f"config sha mismatch {old} != {chash}")
    rs, _ = train.build(cfg)
    mgr = train._ckpt(os.path.join(od, "ckpt" + sfx))
    step = mgr.latest_step()
    rs = mgr.restore(step, args=ocp.args.StandardRestore(rs))
    t = np.asarray(rs.t_env)
    timing = dict(truncated=True, ckpt_chunk=int(step), t_stop_min=int(t.min()), t_stop_max=int(t.max()),
                  platform=a.note)
    out = train.write_metrics(cfg, rs, chash, timing)
    for r, rec in out.items():
        rec["truncated"] = True
        rec["t_stop"] = int(t[r])
        train.write_json(os.path.join(od, f"seed_{off + r}", "metrics.json"), rec)
    print(f"[INFO] finalised {od}: chunk {step}, t_env {t.tolist()}")


if __name__ == "__main__":
    main()
