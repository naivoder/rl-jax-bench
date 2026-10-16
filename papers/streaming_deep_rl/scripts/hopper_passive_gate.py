"""Proxy-acceptance gate for a hopper simulator .

Before ANY learning run on a stand-in for MuJoCo Hopper-v4, measure the passive dynamics: mean episode length (healthy
termination, 1000-step cap) under zero torque and under uniform-random actions in [-1, 1], 32 episodes each. The
proxy passes iff both lengths lie within +-30 % of real Hopper-v4's (gymnasium 0.29.1 + mujoco 2.3.7, the authors'
versions: zero 155, random 23.4 steps). The Brax `positional` backend fails (zero torque stands for all 1000 steps,
return 1013: a free ~1000 return the paper's env never gives); Brax `generalized` passes (125 / 19.1).

Usage: python hopper_passive_gate.py <reference.json> <candidate.json>  -> prints the verdict, exit 1 on FAIL.
Measuring a candidate: scripts/hopper_passive.py <backend> (Brax),
hopper_passive_mujoco.py (reference, authors' venv)."""
import json
import sys

POLICIES = ("zero", "random")
TOL = 0.30


def _len(d, pol):
    if pol in d:
        return d[pol]["len"]
    hit = [v for k, v in d.items() if k.endswith("/" + pol)]
    if len(hit) != 1:
        raise KeyError(f"policy {pol!r} not (uniquely) measured in candidate")
    return hit[0]["len"]


def passive_gate(ref: dict, cand: dict, tol: float = TOL) -> tuple[bool, dict]:
    rows = {}
    for pol in POLICIES:
        r, c = _len(ref, pol), _len(cand, pol)
        rows[pol] = dict(ref_len=r, cand_len=c, rel=c / r - 1.0, ok=abs(c / r - 1.0) <= tol)
    return all(v["ok"] for v in rows.values()), rows


if __name__ == "__main__":
    ok, rows = passive_gate(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])))
    for pol, v in rows.items():
        print(f"{pol:7s} ref {v['ref_len']:7.1f}  candidate {v['cand_len']:7.1f}  {v['rel']:+.0%}  {'ok' if v['ok'] else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
