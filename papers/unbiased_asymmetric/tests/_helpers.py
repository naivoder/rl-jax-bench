"""Shared test helpers: import path, CPU/x64-free float32 setup, torch->flax weight porting for the oracle tests."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import jax  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
import numpy as np  # noqa: E402

OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(REPO, "external", "official"))
ORACLE = os.path.join(OFFICIAL_ROOT, "out")
# Full-size oracle tests (tests/test_official_{env,carflag,cleaner,losses}.py, tests/test_schedules.py) read the
# $OFFICIAL_ROOT/out/ exports (scripts/official/export_*.py) and SKIP on a fresh clone; tests/test_official_parity.py covers the same checks on
# small committed fixtures and always runs. UA_REQUIRE_ORACLE=1 turns those skips into failures.
REQUIRE_ORACLE = os.environ.get("UA_REQUIRE_ORACLE") == "1"


def oracle_mark(path, script):
    """skipif-marker for a full-size oracle export; a hard failure when UA_REQUIRE_ORACLE=1 and the file is missing."""
    import pytest
    if REQUIRE_ORACLE and not os.path.exists(path):
        raise FileNotFoundError(f"UA_REQUIRE_ORACLE=1 but {path} is missing: run {script} in the official venv")
    return pytest.mark.skipif(not os.path.exists(path), reason=f"needs $OFFICIAL_ROOT/out/{os.path.basename(path)}: run "
                              f"sh scripts/official/setup.sh, then $OFFICIAL_ROOT/.venv/bin/python {script}")


def _mlp(d, pfx):
    return {"l0": {"kernel": d[pfx + ".0.weight"].T, "bias": d[pfx + ".0.bias"]},
            "l1": {"kernel": d[pfx + ".2.weight"].T, "bias": d[pfx + ".2.bias"]},
            "out": {"kernel": d[pfx + ".4.weight"].T, "bias": d[pfx + ".4.bias"]}}


def _enc(d, pfx):
    cell = {"w_ih": d[pfx + ".history_model.rnn.weight_ih_l0"].T, "w_hh": d[pfx + ".history_model.rnn.weight_hh_l0"].T,
            "b_ih": d[pfx + ".history_model.rnn.bias_ih_l0"], "b_hh": d[pfx + ".history_model.rnn.bias_hh_l0"]}
    if pfx + ".observation_model.embeddings.weight" not in d:      # vector env: one-hot actions, identity obs
        return {"cell": cell}
    o = d[pfx + ".observation_model.embeddings.weight"]
    assert np.all(o[-1] == 0.0), "reset-observation embedding row must be zero (padding_idx=-1)"
    return {"inp": {"a_emb": d[pfx + ".action_model.embeddings.weight"], "o_emb": o[:-1]},
            "cell": {"w_ih": d[pfx + ".history_model.rnn.weight_ih_l0"].T,
                     "w_hh": d[pfx + ".history_model.rnn.weight_hh_l0"].T,
                     "b_ih": d[pfx + ".history_model.rnn.bias_ih_l0"], "b_hh": d[pfx + ".history_model.rnn.bias_hh_l0"]}}


VHEAD = {"hs": "vhs_model", "s": "vs_model", "h": "vh_model"}


def port(d, mode):
    """d: {torch state_dict name: array}. Returns (actor params, critic params) in our flax layout."""
    pa = {"enc": _enc(d, "agent"), "pi": _mlp(d, "agent.policy_model")}
    pc = {"v": _mlp(d, "critic." + VHEAD[mode])}
    if "h" in mode:
        pc["enc"] = _enc(d, "critic")
    if "s" in mode and "critic.state_model.embeddings.weight" in d:
        pc["sf"] = {"s_emb": d["critic.state_model.embeddings.weight"]}
    return pa, pc


def unport(tree, mode, kind):
    """Inverse naming map for gradients: our flax tree -> {torch name: array} (only for params we have)."""
    out = {}

    def mlp(t, pfx):
        for i, k in (("0", "l0"), ("2", "l1"), ("4", "out")):
            out[f"{pfx}.{i}.weight"] = np.asarray(t[k]["kernel"]).T
            out[f"{pfx}.{i}.bias"] = np.asarray(t[k]["bias"])

    def enc(t, pfx):
        if "inp" in t:
            out[pfx + ".action_model.embeddings.weight"] = np.asarray(t["inp"]["a_emb"])
            out[pfx + ".observation_model.embeddings.weight"] = np.asarray(t["inp"]["o_emb"])
        c = t["cell"]
        out[pfx + ".history_model.rnn.weight_ih_l0"] = np.asarray(c["w_ih"]).T
        out[pfx + ".history_model.rnn.weight_hh_l0"] = np.asarray(c["w_hh"]).T
        out[pfx + ".history_model.rnn.bias_ih_l0"] = np.asarray(c["b_ih"])
        out[pfx + ".history_model.rnn.bias_hh_l0"] = np.asarray(c["b_hh"])

    if kind == "actor":
        enc(tree["enc"], "agent")
        mlp(tree["pi"], "agent.policy_model")
    else:
        mlp(tree["v"], "critic." + VHEAD[mode])
        if "enc" in tree:
            enc(tree["enc"], "critic")
        if "sf" in tree:
            out["critic.state_model.embeddings.weight"] = np.asarray(tree["sf"]["s_emb"])
    return out
