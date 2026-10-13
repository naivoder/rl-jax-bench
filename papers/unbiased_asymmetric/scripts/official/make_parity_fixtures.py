"""Distil the (large, git-ignored) official exports in $OFFICIAL_ROOT/out/ into SMALL committed fixtures for
tests/test_official_parity.py. Every array here was recorded from the official code (export_env_traces.py,
export_carflag.py, export_cleaner.py, export_schedules.py); this script only subsets / bit-packs / hashes them.

    uv run python -I scripts/official/make_parity_fixtures.py
"""
import hashlib
import os

import numpy as np

PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(PAPER, "external", "official"))
HERE = os.path.dirname(os.path.abspath(__file__))
SRC, DST = os.path.join(OFFICIAL_ROOT, "out"), os.path.join(PAPER, "tests", "fixtures", "official")
ENVS = ["heavenhell_3", "heavenhell_4", "shopping_5", "shopping_6"]
N_CAT, N_CAR, N_CLEAN = 200, 200, 20


def h(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x, np.float64)).tobytes()).hexdigest()


def main():
    os.makedirs(DST, exist_ok=True)
    d = np.load(os.path.join(SRC, "env_traces.npz"))
    out = {}
    for e in ENVS:
        for k in ("s0", "act", "st", "ob", "rw", "dn", "len"):
            x = d[f"{e}/{k}"][:N_CAT]
            out[f"{e}/{k}"] = x.astype(np.int16) if x.dtype.kind == "i" else x
        for k in ("T", "O", "R", "D", "start"):     # parsed rl_parsers arrays: SHA256 of the float64 bytes + shape
            x = d[f"{e}/{k}"]
            out[f"{e}/{k}_sha256"], out[f"{e}/{k}_shape"] = np.array(h(x)), np.array(x.shape)
    np.savez_compressed(os.path.join(DST, "env_traces_small.npz"), **out)
    c = np.load(os.path.join(SRC, "carflag_traces.npz"))
    np.savez_compressed(os.path.join(DST, "carflag_traces_small.npz"),
                        **{k: c[k][:N_CAR] for k in c.files})
    c = np.load(os.path.join(SRC, "cleaner_traces.npz"))
    st, ob = c["st"][:N_CLEAN], c["ob"][:N_CLEAN]
    assert np.all((st == 0) | (st == 1)) and np.all((ob == 0) | (ob == 1)), "cleaner state/obs must be binary"
    np.savez_compressed(os.path.join(DST, "cleaner_traces_small.npz"), s0=c["s0"][:1], o0=c["o0"][:1],
                        act=c["act"][:N_CLEAN], rw=c["rw"][:N_CLEAN], st_bits=np.packbits(st.astype(np.uint8), axis=-1),
                        ob_bits=np.packbits(ob.astype(np.uint8), axis=-1), st_dim=st.shape[-1], ob_dim=ob.shape[-1])
    s = np.load(os.path.join(SRC, "schedules.npz"))
    np.savez_compressed(os.path.join(DST, "schedules.npz"), **{k: s[k] for k in s.files})
    for f in sorted(os.listdir(DST)):
        print(f, os.path.getsize(os.path.join(DST, f)))


if __name__ == "__main__":
    main()
