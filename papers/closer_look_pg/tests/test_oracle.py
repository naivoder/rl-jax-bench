"""Differential test against the official PyTorch primitives (scripts/oracle_diff.py); skipped when the isolated
oracle venv is absent. ~35 s, so it is excluded from the <30 s smoke budget (run: pytest tests/test_oracle.py)."""
import os
import subprocess
import sys

import pytest

ORACLE_PY = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official")), ".venv", "bin", "python")


@pytest.mark.skipif(not os.path.exists(ORACLE_PY), reason="needs the authors' code and its venv under $OFFICIAL_ROOT (default external/official): run sh scripts/official/setup.sh")
def test_oracle_diff():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, JAX_PLATFORMS="cpu", ORACLE_WRITE_FIXTURES="0")
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "oracle_diff.py")], capture_output=True, text=True,
                       env=env, cwd=root)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
