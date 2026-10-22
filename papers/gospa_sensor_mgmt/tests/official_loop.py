"""Re-run of the authors' main.py closed loop (one algorithm block) with injectable rollout choices, used only as a test oracle.

Mirrors main.py's per-step order exactly (predict -> plan -> measure -> S1 update (lbp, tomb) -> S2 update -> merge -> [cap if myopic]
-> prune -> region -> threshold -> GOSPA).  Differences from main.py: numpy>=2 needs the flat existence vector between the two
updates (values identical), the truth is the saved array, and rollout choices come from the injected uniforms."""
import numpy as np
import official


class Inj:
    mode = "joint"
    k = 0
    it = -1
    j = 0
    UJ = None     # [K, Bj, Tm]
    UI = None     # [K, 2, Bi, Tm]
    Bi = 1


def _patch_classes(off):
    saved = []
    for cls in (off["MCTS"].MonteCarloTreeSearchNode_individual, off["MCTS_joint"].MonteCarloTreeSearchNode_joint):
        orig = cls.rollout

        def make(orig):
            def rollout(self):
                Inj.it += 1
                Inj.j = 0
                return orig(self)
            return rollout

        def rollout_policy(self, avail):
            if Inj.mode == "joint":
                u = Inj.UJ[Inj.k][Inj.it][Inj.j]
            else:
                u = Inj.UI[Inj.k][Inj.it // Inj.Bi][Inj.it % Inj.Bi][Inj.j]
            Inj.j += 1
            return avail[int(np.floor(u * len(avail)))]
        cls.rollout = make(orig)
        cls.rollout_policy = rollout_policy
        saved.append((cls, orig))
    return saved


def _unpatch(saved):
    for cls, orig in saved:
        cls.rollout = orig
        del cls.rollout_policy


def run(alg_cost, T, bj, bi, truth, UJ, UI, lam=2.0, n_steps=200, seed=123546, cap=None, inject=True):
    off = official.load()
    Pm = off["Parameters"]
    Pm.lambda_false_alarm = lam / (np.pi * 40.0 ** 2)
    Pm.clutter_rate = lam
    sens_mod = off["sensor"]
    saved = _patch_classes(off) if inject else []
    try:
        np.random.seed(seed)
        o1, o2, oM = sens_mod.GOSPA_Sensor_updated_maths(), sens_mod.GOSPA_Sensor_updated_maths(), sens_mod.GOSPA_Sensor_updated_maths()
        sl = [o1, o2]
        o1.current_sensor_position = np.array([[5.0], [150.0]]); o2.current_sensor_position = np.array([[-5.0], [150.0]])
        o1.movement_model.state = o1.current_sensor_position.copy(); o2.movement_model.state = o2.current_sensor_position.copy()
        prob_existence = [0]
        means_existing = np.zeros(Pm.birth_means[:, [0]].shape)
        covariances_existing = np.zeros(Pm.birth_covariances[[0], :, :].shape)
        lambdau = np.zeros(Pm.lambdau.shape); means_new = np.zeros(Pm.birth_means.shape); covariances_new = np.zeros(Pm.birth_covariances.shape)
        rec = dict(z=[], spos=[], gospa=[], loc=[], miss=[], fal=[], joint=[], nest=[])
        myopic = (T == 1)
        cap = myopic if cap is None else cap
        Inj.Bi = bi; Inj.UJ = UJ; Inj.UI = UI
        for k in range(n_steps):
            prob_existence, means_existing, covariances_existing = oM.predictor.predict_pmb_empty_poisson(
                prob_existence.copy(), means_existing.copy(), covariances_existing.copy())
            Inj.k, Inj.it = k, -1
            Pm.simulation_search_depth = T
            if np.linalg.norm(o1.current_sensor_position - o2.current_sensor_position) < Pm.proximity_distance:
                Pm.MCTS_simulation_number = bj
                Inj.mode = "joint"
                oM.manage_multiple_sensors_mb_full_jointly_calculated_MCTS([o1, o2], prob_existence, means_existing, covariances_existing, cost_function=alg_cost)
                rec["joint"].append(True)
            else:
                Pm.MCTS_simulation_number = bi
                Inj.mode = "ind"
                oM.manage_multiple_sensors_mb_individually_MCTS([o1, o2], prob_existence, means_existing, covariances_existing, cost_function=alg_cost)
                rec["joint"].append(False)
            rec["spos"].append(np.stack([o1.current_sensor_position.reshape(2), o2.current_sensor_position.reshape(2)]))
            tk = truth[:, :, k].T
            o1.measurements.generate_measurements_multiple_targets(tk, o1.current_sensor_position, k)
            o2.measurements.generate_measurements_multiple_targets(tk, o2.current_sensor_position, k)
            rec["z"].append([o1.measurements.all_measurements[k].copy(), o2.measurements.all_measurements[k].copy()])
            lu, mn, cn, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = oM.updater.update_pmb_empty_poisson(
                lambdau, means_new, covariances_new, prob_existence, means_existing, covariances_existing, o1.measurements.all_measurements[k], [o1])
            pupd, pnew = off["lbp"].lbp(wupd, wnew)
            prob_existence, means_existing, covariances_existing = off["tomb"].tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
            lu2, mn2, cn2, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = oM.updater.update_pmb_empty_poisson(
                lu, mn, cn, prob_existence, means_existing, covariances_existing, o2.measurements.all_measurements[k], [o2])
            pupd, pnew = off["lbp"].lbp(wupd, wnew)
            prob_existence, means_existing, covariances_existing = off["tomb"].tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
            if len(prob_existence) > 0:
                prob_existence, means_existing, covariances_existing = off["utils"].merge_gaussians(prob_existence, means_existing, covariances_existing)
            if cap:
                prob_existence[prob_existence >= 1] = 1
            fi = prob_existence > Pm.pruning_threshold
            prob_existence, means_existing, covariances_existing = prob_existence[fi], means_existing[:, fi], covariances_existing[fi]
            xc = (means_existing[0, :] >= -250) & (means_existing[0, :] <= 250)
            yc = (means_existing[2, :] >= -250) & (means_existing[2, :] <= 250)
            cc = xc & yc
            prob_existence, means_existing, covariances_existing = prob_existence[cc].copy(), means_existing[:, cc].copy(), covariances_existing[cc].copy()
            ss = prob_existence > Pm.existence_threshold
            est = means_existing[:, ss]
            gt = tk[:, ~np.any(np.isnan(tk), axis=0)]
            g, _, loc, mis, fal = off["gospa"].calculate_gospa(gt.T, est.T, Pm.c, Pm.p)
            rec["gospa"].append(g ** 2); rec["loc"].append(loc); rec["miss"].append(mis); rec["fal"].append(fal)
            rec["nest"].append(int(ss.sum()))
        out = {k: np.array(v) for k, v in rec.items() if k != "z"}
        out["z"] = rec["z"]          # list over steps of [z_sensor1 (2,m1), z_sensor2 (2,m2)]
        return out
    finally:
        _unpatch(saved)
