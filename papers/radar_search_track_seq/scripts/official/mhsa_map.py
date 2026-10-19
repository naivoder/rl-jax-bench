"""Parameter / gradient name mapping between the SB3-fork BiGRU+MHSA extractor (scripts/official/
mhsa_sb3.py, recorded with --record-update) and OUR ActorCritic(encoder="mhsa"). Used by test_P11 (recheck)."""
import jax.numpy as jnp
import numpy as np

FE = "features_extractor."


def params(d, prefix):
    g = lambda k: d[prefix + FE + k]
    conv = lambda w: jnp.asarray(w.transpose(2, 3, 1, 0))
    Wl = g("cnn.linear.0.weight")
    cnn = {"Conv_0": {"kernel": conv(g("cnn.cnn.0.weight")), "bias": jnp.asarray(g("cnn.cnn.0.bias"))},
           "Conv_1": {"kernel": conv(g("cnn.cnn.2.weight")), "bias": jnp.asarray(g("cnn.cnn.2.bias"))},
           "Conv_2": {"kernel": conv(g("cnn.cnn.4.weight")), "bias": jnp.asarray(g("cnn.cnn.4.bias"))},
           "Dense_0": {"kernel": jnp.asarray(Wl.reshape(128, 64, 2, 2).transpose(2, 3, 1, 0).reshape(256, 128)),
                       "bias": jnp.asarray(g("cnn.linear.0.bias"))}}
    W, b = g("mha.in_proj_weight"), g("mha.in_proj_bias")
    dm, h = 64, 4
    hd = dm // h
    mha = {nm: {"kernel": jnp.asarray(W[j * dm:(j + 1) * dm].T.reshape(dm, h, hd)),
                "bias": jnp.asarray(b[j * dm:(j + 1) * dm].reshape(h, hd))} for j, nm in enumerate(("query", "key", "value"))}
    mha["out"] = {"kernel": jnp.asarray(g("mha.out_proj.weight").T.reshape(h, hd, dm)), "bias": jnp.asarray(g("mha.out_proj.bias"))}
    gru = {}
    for nm, suf in (("fwd", ""), ("bwd", "_reverse")):
        wi, wh, bi, bh = (g(f"gru.{k}_l0{suf}") for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"))
        H = wh.shape[1]
        gru.update({f"{nm}_wi": jnp.asarray(wi.T), f"{nm}_wh": jnp.asarray(wh.T),
                    f"{nm}_bi": jnp.asarray(bi + np.concatenate([bh[:2 * H], np.zeros(H, np.float32)])),
                    f"{nm}_bhn": jnp.asarray(bh[2 * H:])})
    te = {"embed": {"kernel": jnp.asarray(g("embed.weight").T), "bias": jnp.asarray(g("embed.bias"))},
          "mhsa": {"mha": mha, "ln": {"scale": jnp.asarray(g("ln.weight")), "bias": jnp.asarray(g("ln.bias"))}},
          "bigru": gru}
    P = {"track_enc": te, "cnn": cnn}
    for ours, t in (("pi", "policy_net"), ("v", "value_net")):
        P[f"{ours}_0"] = {"kernel": jnp.asarray(d[f"{prefix}mlp_extractor.{t}.0.weight"].T), "bias": jnp.asarray(d[f"{prefix}mlp_extractor.{t}.0.bias"])}
        P[f"{ours}_1"] = {"kernel": jnp.asarray(d[f"{prefix}mlp_extractor.{t}.2.weight"].T), "bias": jnp.asarray(d[f"{prefix}mlp_extractor.{t}.2.bias"])}
    for k in ("action_net", "value_net"):
        P[k] = {"kernel": jnp.asarray(d[f"{prefix}{k}.weight"].T), "bias": jnp.asarray(d[f"{prefix}{k}.bias"])}
    return P


def grads_to_torch(gr):
    """Our gradient tree -> SB3 names, for the parameters with a 1:1 mapping (+ GRU summed-bias handled below)."""
    o = {}
    te = gr["track_enc"]
    o[FE + "embed.weight"] = np.asarray(te["embed"]["kernel"]).T
    o[FE + "embed.bias"] = np.asarray(te["embed"]["bias"])
    m = te["mhsa"]["mha"]
    o[FE + "mha.in_proj_weight"] = np.concatenate([np.asarray(m[nm]["kernel"]).reshape(64, 64).T for nm in ("query", "key", "value")], 0)
    o[FE + "mha.in_proj_bias"] = np.concatenate([np.asarray(m[nm]["bias"]).reshape(64) for nm in ("query", "key", "value")])
    o[FE + "mha.out_proj.weight"] = np.asarray(m["out"]["kernel"]).reshape(64, 64).T
    o[FE + "mha.out_proj.bias"] = np.asarray(m["out"]["bias"])
    o[FE + "ln.weight"] = np.asarray(te["mhsa"]["ln"]["scale"])
    o[FE + "ln.bias"] = np.asarray(te["mhsa"]["ln"]["bias"])
    for nm, suf in (("fwd", ""), ("bwd", "_reverse")):
        g = te["bigru"]
        H = 32
        o[FE + f"gru.weight_ih_l0{suf}"] = np.asarray(g[f"{nm}_wi"]).T
        o[FE + f"gru.weight_hh_l0{suf}"] = np.asarray(g[f"{nm}_wh"]).T
        o[FE + f"gru.bias_ih_l0{suf}"] = np.asarray(g[f"{nm}_bi"])                       # d/d b_ih = d/d bi
        o[FE + f"gru.bias_hh_l0{suf}"] = np.concatenate([np.asarray(g[f"{nm}_bi"])[:2 * H], np.asarray(g[f"{nm}_bhn"])])
    c = gr["cnn"]
    for i, j in ((0, 0), (1, 2), (2, 4)):
        o[FE + f"cnn.cnn.{j}.weight"] = np.asarray(c[f"Conv_{i}"]["kernel"]).transpose(3, 2, 0, 1)
        o[FE + f"cnn.cnn.{j}.bias"] = np.asarray(c[f"Conv_{i}"]["bias"])
    o[FE + "cnn.linear.0.weight"] = np.asarray(c["Dense_0"]["kernel"]).reshape(2, 2, 64, 128).transpose(3, 2, 0, 1).reshape(128, 256)
    o[FE + "cnn.linear.0.bias"] = np.asarray(c["Dense_0"]["bias"])
    for ours, t in (("pi", "policy_net"), ("v", "value_net")):
        for i, j in ((0, 0), (1, 2)):
            o[f"mlp_extractor.{t}.{j}.weight"] = np.asarray(gr[f"{ours}_{i}"]["kernel"]).T
            o[f"mlp_extractor.{t}.{j}.bias"] = np.asarray(gr[f"{ours}_{i}"]["bias"])
    for k in ("action_net", "value_net"):
        o[k + ".weight"] = np.asarray(gr[k]["kernel"]).T
        o[k + ".bias"] = np.asarray(gr[k]["bias"])
    return o


