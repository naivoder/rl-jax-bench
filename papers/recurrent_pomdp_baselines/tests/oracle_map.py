"""Map official PyTorch state_dict names <-> our parameter trees (test helper; no official code is imported)."""
import numpy as np
import jax.numpy as jnp

ORACLE_SIZES = dict(obs_embed=8, action_embed=4, reward_embed=4, rnn_hidden=16, dqn_layers=[16, 16],
                    policy_layers=[16, 16], lr=3e-4, gamma=0.9, tau=0.005, inputs="oar", alpha_lr=3e-4,
                    auto_alpha=True, entropy_alpha=0.2, exploration_noise=0.1, target_noise=0.2,
                    target_noise_clip=0.5, scan_unroll=1)


def _lin(sd, pre):
    return {"weight": jnp.asarray(sd[pre + ".weight"]), "bias": jnp.asarray(sd[pre + ".bias"])}


def _mlp(sd, pre, sac_head=False):
    out = {"fc0": _lin(sd, pre + ".fc0"), "fc1": _lin(sd, pre + ".fc1"), "last_fc": _lin(sd, pre + ".last_fc")}
    if sac_head:
        out["last_fc_log_std"] = _lin(sd, pre + ".last_fc_log_std")
    return out


def _enc(sd, pre):
    p = (pre + ".") if pre else ""
    return {"action_embedder": _lin(sd, p + "action_embedder.fc"), "reward_embedder": _lin(sd, p + "reward_embedder.fc"),
            "observ_embedder": _lin(sd, p + "observ_embedder.fc"),
            "rnn": {"w_ih": jnp.asarray(sd[p + "rnn.weight_ih_l0"]), "w_hh": jnp.asarray(sd[p + "rnn.weight_hh_l0"]),
                    "b_ih": jnp.asarray(sd[p + "rnn.bias_ih_l0"]), "b_hh": jnp.asarray(sd[p + "rnn.bias_hh_l0"])}}


def separate_params(sd, algo):
    sac = algo == "sac"
    actor = lambda pre: {"enc": _enc(sd, pre), "short": {"fc": _lin(sd, pre + ".current_observ_embedder.fc")},  # noqa
                         "pi": _mlp(sd, pre + ".policy", sac)}
    critic = lambda pre: {"enc": _enc(sd, pre), "short": {"fc": _lin(sd, pre + ".current_shortcut_embedder.fc")},  # noqa
                          "q1": _mlp(sd, pre + ".qf1"), "q2": _mlp(sd, pre + ".qf2")}
    return {"actor": actor("actor"), "critic": critic("critic"), "actor_t": actor("actor_target"),
            "critic_t": critic("critic_target")}


def shared_params(sd, algo):
    sac = algo == "sac"
    main = {"enc": _enc(sd, ""), "short_oa": {"fc": _lin(sd, "current_observ_action_embedder.fc")},
            "q1": _mlp(sd, "qf1"), "q2": _mlp(sd, "qf2"), "short_o": {"fc": _lin(sd, "current_observ_embedder.fc")},
            "pi": _mlp(sd, "policy", sac)}
    tgt = {"q1": _mlp(sd, "qf1_target"), "q2": _mlp(sd, "qf2_target"), "pi": _mlp(sd, "policy_target", sac)}
    return {"main": main, "tgt": tgt}


def state_dict(z, prefix):
    return {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}


def batch_from(z, u):
    g = lambda k: np.asarray(z[f"b{u}/{k}"])  # noqa: E731
    obs, obs2, act, rew, term, mask = g("obs"), g("obs2"), g("act"), g("rew"), g("term"), g("mask")
    B, A = act.shape[1], act.shape[2]
    b = {"obs": np.concatenate([obs[:1], obs2], 0), "act": np.concatenate([np.zeros((1, B, A)), act], 0),
         "rew": np.concatenate([np.zeros((1, B, 1)), rew], 0), "term": np.concatenate([np.zeros((1, B, 1)), term], 0),
         "mask": mask}
    b = {k: jnp.asarray(v, jnp.float32) for k, v in b.items()}
    noise = {"tgt": jnp.asarray(z[f"b{u}/noise_tgt"]), "act": jnp.asarray(z[f"b{u}/noise_act"])}
    return b, noise


def torch_name_of(path, arch):
    """our tree path (tuple of keys) -> torch state_dict name (for error messages / comparisons)."""
    return ".".join(path)


def flatten_separate(params):
    """our separate params -> {torch name: array} for the four modules."""
    out = {}
    for mod, tname in (("actor", "actor"), ("critic", "critic"), ("actor_t", "actor_target"), ("critic_t", "critic_target")):
        p = params[mod]
        _flat_enc(out, p["enc"], tname + ".")
        if "pi" in p:
            out.update(_flat_lin(p["short"]["fc"], tname + ".current_observ_embedder.fc"))
            out.update(_flat_mlp(p["pi"], tname + ".policy"))
        else:
            out.update(_flat_lin(p["short"]["fc"], tname + ".current_shortcut_embedder.fc"))
            out.update(_flat_mlp(p["q1"], tname + ".qf1"))
            out.update(_flat_mlp(p["q2"], tname + ".qf2"))
    return out


def flatten_shared(params):
    out = {}
    m, t = params["main"], params["tgt"]
    _flat_enc(out, m["enc"], "")
    out.update(_flat_lin(m["short_oa"]["fc"], "current_observ_action_embedder.fc"))
    out.update(_flat_lin(m["short_o"]["fc"], "current_observ_embedder.fc"))
    for k, n in (("q1", "qf1"), ("q2", "qf2"), ("pi", "policy")):
        out.update(_flat_mlp(m[k], n))
        out.update(_flat_mlp(t[k], n + "_target"))
    return out


def _flat_lin(p, pre):
    return {pre + ".weight": np.asarray(p["weight"]), pre + ".bias": np.asarray(p["bias"])}


def _flat_mlp(p, pre):
    out = {}
    for k, v in p.items():
        out.update(_flat_lin(v, f"{pre}.{k}"))
    return out


def _flat_enc(out, e, pre):
    for k in ("action_embedder", "reward_embedder", "observ_embedder"):
        out.update(_flat_lin(e[k], f"{pre}{k}.fc"))
    r = e["rnn"]
    out[f"{pre}rnn.weight_ih_l0"] = np.asarray(r["w_ih"])
    out[f"{pre}rnn.weight_hh_l0"] = np.asarray(r["w_hh"])
    out[f"{pre}rnn.bias_ih_l0"] = np.asarray(r["b_ih"])
    out[f"{pre}rnn.bias_hh_l0"] = np.asarray(r["b_hh"])
