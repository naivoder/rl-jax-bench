"""Toy checks: run every src/toys.py check (incl. the 5e6-sample float32 Welford precision test) on CPU and the oracle
differential tests (pytest), write runs/toys/toys.json.  Usage: python scripts/toy_checks.py"""
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
sys.path.insert(0, REPO)
from rl_jax_bench.jaxlog import info, metric, write_json  # noqa: E402
from src import toys  # noqa: E402

out = {}
for name, fn in toys.ALL.items():
    t = time.time()
    r = fn()
    out[name] = r
    metric("toys", check=name, passed=r["pass"], seconds=round(time.time() - t, 1))
py = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_official_parity.py", "tests/test_grads.py"],
                    cwd=REPO, capture_output=True, text=True)
out["oracle_and_grad_tests"] = dict(stdout_tail=py.stdout.strip().splitlines()[-1], **{"pass": py.returncode == 0})
out["all_pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
write_json(os.path.join(REPO, "runs", "toys", "toys.json"), out)
info("toys done", all_pass=out["all_pass"], tests=out["oracle_and_grad_tests"]["stdout_tail"])
