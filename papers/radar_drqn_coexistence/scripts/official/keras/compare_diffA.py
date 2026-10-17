import numpy as np, json, sys
S = np.load(sys.argv[1]); ntr = int(S["ntr"])
res = {}
for arm in ("dqn", "ddrqn"):
    K = np.load(sys.argv[2].format(arm=arm))
    o = S[f"ours_{arm}_out"]; ao = S[f"ours_{arm}_act"]
    lo = np.where(o[:, 7] > 0, o[:, 5], np.nan); lk = K["losses"]
    m = ~np.isnan(lo)
    assert (m == ~np.isnan(lk)).all(), "update schedule differs"
    lo_, lk_ = lo[m], lk[m]
    rel = np.abs(lo_ - lk_) / np.maximum(np.abs(lk_), 1e-6)
    first_bad = int(np.argmax(rel > 1e-3)) if (rel > 1e-3).any() else None
    # rewards consistent with the stream
    r_ok = np.allclose(o[:, 0], S["R"][ao, S["th2"]])
    ev = slice(ntr, None)
    agree = float(np.mean(ao[ev] == K["acts"][ev]))
    same = ao == K["acts"]
    rew_eq = bool(np.all(o[same, 0] == K["rews"][same]))
    fin = {k.split("__")[1]: S[k] for k in S.files if k.startswith(f"final_{arm}__")}
    # map keras weight names
    kmap = {"params/fc0/kernel": "fc0/kernel:0", "params/fc0/bias": "fc0/bias:0", "params/fc1/kernel": "fc1/kernel:0",
            "params/fc1/bias": "fc1/bias:0", "params/fc2/kernel": "fc2/kernel:0", "params/fc2/bias": "fc2/bias:0",
            "params/q/kernel": "q/kernel:0", "params/q/bias": "q/bias:0", "params/lstm/kernel/kernel": "lstm/lstm_cell/kernel:0",
            "params/lstm/recurrent/kernel": "lstm/lstm_cell/recurrent_kernel:0", "params/lstm/kernel/bias": "lstm/lstm_cell/bias:0"}
    kf = {k.split("__")[1]: K[k] for k in K.files if k.startswith("final__")}
    prel = {}
    for k, v in fin.items():
        kk = [x for x in kf if x.endswith(kmap[k])]
        if kk:
            w = kf[kk[0]]
            prel[k] = float(np.abs(v - w).max() / (np.abs(w).max() + 1e-12))
    res[arm] = dict(n_updates=int(m.sum()), loss_rel_max_first500=float(rel[:500].max()), loss_rel_median_all=float(np.median(rel)),
                    loss_rel_max_all=float(rel.max()), first_update_rel_gt_1e3=first_bad,
                    ours_rewards_equal_R_table=bool(r_ok), eval_action_agreement=agree, rewards_equal_where_actions_equal=rew_eq,
                    final_param_rel_maxabs=prel, loss_first5_ours=lo_[:5].tolist(), loss_first5_keras=lk_[:5].tolist(),
                    J_eval_ours=float(o[ev, 0].mean()), J_eval_keras=float(K["rews"][ev].mean()))
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[3], "w"), indent=1)
