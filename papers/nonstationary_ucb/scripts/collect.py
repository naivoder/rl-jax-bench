"""Collect the Full-scale headline records into runs/seed_<n>/metrics.json (n = 0..4) and pin the seed-0..4 hashes.

    uv run python scripts/collect.py

runs/seed_<n>/metrics.json = {experiment: the full per-seed record of runs/full/<experiment>/seed_<n>/metrics.json}
for the headline (e1, e2), secondary-arm (padb_*) and ALT-slides (e1b, e2b) experiments: per-replication final
regret and arm-1 counts, curve summaries, env hashes, git SHA, config, compile time and throughput.
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EXPS = ("e1", "e2", "padb_e1", "padb_e2", "e1b", "e2b", "padb_e1b", "padb_e2b")

hashes = json.load(open(os.path.join(ROOT, "runs", "env_hashes.json")))
for n in range(5):
    out = {}
    for e in EXPS:
        f = os.path.join(ROOT, "runs", "full", e, f"seed_{n}", "metrics.json")
        out[e] = json.load(open(f))
    for e in ("e1", "e2"):
        assert out[e]["env"]["means_sha256"] == hashes[e]["means_sha256"], f"{e}: mean schedule changed since Small-scale"
        hashes[e].setdefault("uniforms_fp_sha256_by_seed", {})[str(n)] = out[e]["env"]["uniforms_fp_sha256"]
        assert out[f"padb_{e}"]["env"]["uniforms_fp_sha256"] == out[e]["env"]["uniforms_fp_sha256"]
    for e in ("e1b", "e2b"):
        hashes.setdefault(e, {"means_sha256": out[e]["env"]["means_sha256"], "uniforms_fp_sha256_by_seed": {}})
        hashes[e]["uniforms_fp_sha256_by_seed"][str(n)] = out[e]["env"]["uniforms_fp_sha256"]
    os.makedirs(os.path.join(ROOT, "runs", f"seed_{n}"), exist_ok=True)
    with open(os.path.join(ROOT, "runs", f"seed_{n}", "metrics.json"), "w") as f:
        json.dump(out, f)
with open(os.path.join(ROOT, "runs", "env_hashes.json"), "w") as f:
    json.dump(hashes, f, indent=1)
print("wrote runs/seed_0..4/metrics.json; E1/E2 mean-schedule hashes equal the Small-scale pins")
