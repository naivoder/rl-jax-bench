"""Reconstruct the exact `python -m src.train` command of every recorded run from its stored config.

Usage: uv run python -m scripts.repro_commands > runs/repro_commands.yaml
Walks runs/<tag>/**/seed_<n>/metrics.json (skips runs/seed_* gathered copies, runs/dev and runs/audit), groups by
(tag, env, arm, git), picks the src/configs/*.yaml whose base config + ARMS[arm] needs the fewest --set overrides to
equal the stored config, and prints the command with those overrides, the seeds and the summed process timing.
No jax import: pure file reading.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
CONFIGS = ["classic_control.yaml", "pendulum.yaml", "lstm_masked_cartpole.yaml", "brax_hopper.yaml"]


def load_cfg(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        base = load_cfg(os.path.join(os.path.dirname(path), cfg.pop("base")))
        base.update(cfg)
        cfg = base
    return cfg


def arms():
    import ast
    src = open(os.path.join(REPO, "src", "train.py")).read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "ARMS" for t in node.targets):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "ARMS":
            return ast.literal_eval(node.value)
    raise RuntimeError("ARMS not found")


def main():
    A = arms()
    bases = {c: load_cfg(os.path.join(REPO, "src", "configs", c)) for c in CONFIGS}
    groups = defaultdict(lambda: {"seeds": set(), "wall_s": 0.0, "cfg": None})
    for f in sorted(glob.glob(os.path.join(REPO, "runs", "**", "metrics.json"), recursive=True)):
        rel = os.path.relpath(f, os.path.join(REPO, "runs"))
        parts = rel.split(os.sep)
        if parts[0].startswith("seed_") or parts[0] in ("dev", "audit"):
            continue
        d = json.load(open(f))
        tag = os.sep.join(parts[:-4]) if len(parts) >= 4 else parts[0]
        key = (tag, d["env_id"], d["arm"], str(d.get("git")))
        g = groups[key]
        g["seeds"].add(int(d["seed"]))
        g["cfg"] = d["config"]
        t = d.get("timing") or {}
        g["wall_s"] = max(g["wall_s"], float(t.get("compile_seconds", 0.0) or 0.0) + float(t.get("steady_seconds", 0.0) or 0.0))
        g["device"] = t.get("device", "?")
    out = []
    for (tag, env, arm, git), g in sorted(groups.items()):
        cfg = {k: v for k, v in g["cfg"].items()}
        best = None
        for c, base in bases.items():
            ref = dict(base, env_id=env, **A.get(arm, {}))  # unknown (probe-script) arm labels run as "default" + --set
            diff = {k: v for k, v in cfg.items() if k not in ("seeds", "env_id") and ref.get(k, object()) != v}
            missing = [k for k in ref if k not in cfg and k not in ("seeds",)]
            score = len(diff) + len(missing)
            if best is None or score < best[0]:
                best = (score, c, diff)
        _, c, diff = best
        seeds = sorted(g["seeds"])
        sets = " ".join(f"--set '{k}={json.dumps(v)}'" for k, v in sorted(diff.items()))
        run_arm = arm if arm in A else "default"
        cmd = (f"uv run python -m src.train --config src/configs/{c} --tag {tag} --envs {env} --arms {run_arm} "
               f"--set 'seeds={json.dumps(seeds)}' {sets}").strip()
        out.append({"tag": tag, "env": env, "arm": arm, "git": git, "n_seeds": len(seeds), "device": g.get("device", "?"),
                    "process_s": round(g["wall_s"], 1), "cmd": cmd,
                    "arm_known_now": arm in A})
    yaml.safe_dump({"runs": out}, sys.stdout, sort_keys=False, width=1000)


if __name__ == "__main__":
    main()
