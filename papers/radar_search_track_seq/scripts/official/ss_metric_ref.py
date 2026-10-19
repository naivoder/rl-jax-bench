"""Stone Soup 1.9.1 METRIC PIPELINE on detections recorded from our environment (ORACLE; NOT our implementation).

Run ONLY with the isolated oracle venv:
    $OFFICIAL_ROOT/.venv/bin/python -I scripts/official/ss_metric_ref.py <in.npz> <out.npz>
Input: traj_z [S,T,n,3], traj_valid [S,T,n], truth [S,T,n_tgt,6] (the truth at each step's timestamp), dt, q, R,
vel_sigma, gate, missed, del_thr, c, p, analytic_jac.
The tracker is the B6-verified MultiTargetTracker (KalmanPredictor + UnscentedKalmanUpdater + DistanceHypothesiser /
Mahalanobis + DistanceGater + NearestNeighbour + SimpleMeasurementInitiator + CovarianceBasedDeleter).
Two readings of the episode metrics:
  live     at each step, the tracker's CURRENT track set (what an online reward / env step sees)
  library  the Stone Soup tutorial pipeline: every track ever produced is accumulated (tracks.update(current)) and
           handed to a MultiManager with SumofCovarianceNormsMetric and GOSPAMetric; a deleted track keeps its last
           (predicted, trace > threshold) state at the deletion timestamp.
Episode reductions: M_cov = mean over track-bearing timestamps of the per-timestamp sum of ||P||_F; GOSPA = sums over
timestamps of the (rooted) distance and of the localisation / missed / false components.
"""
import sys
import datetime

import numpy as np
from stonesoup.types.state import GaussianState
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.updater.kalman import UnscentedKalmanUpdater
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.gater.distance import DistanceGater
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import NearestNeighbour
from stonesoup.initiator.simple import SimpleMeasurementInitiator
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.metricgenerator.ospametric import GOSPAMetric
from stonesoup.metricgenerator.uncertaintymetric import SumofCovarianceNormsMetric
from stonesoup.metricgenerator.manager import MultiManager

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ss_ref import analytic_jacobian, prior  # noqa: E402  (our own oracle harness, same directory)

T0 = datetime.datetime(2025, 1, 1)


def gospa_vals(m):
    v = m.value
    return [v["distance"], v["localisation"], v["missed"], v["false"]]


def main():
    d = dict(np.load(sys.argv[1]))
    if int(d.get("analytic_jac", 0)):
        CartesianToElevationBearingRange.jacobian = analytic_jacobian
    mm = CartesianToElevationBearingRange(ndim_state=6, mapping=(0, 2, 4), noise_covar=np.asarray(d["R"]))
    dt, q = float(d["dt"]), float(d["q"])
    S, T, n, _ = d["traj_z"].shape
    p, c = int(d["p"]), float(d["c"])
    g = GOSPAMetric(p=p, c=c)
    out = {k: [] for k in ("live_m_cov", "live_gospa", "lib_m_cov", "lib_gospa", "lib_n_ts", "live_n_ts",
                           "n_tracks_total")}
    for s in range(S):
        tm = CombinedLinearGaussianTransitionModel([ConstantVelocity(q)] * 3)
        pr = KalmanPredictor(tm)
        up = UnscentedKalmanUpdater(measurement_model=mm)
        hyp = DistanceHypothesiser(pr, up, measure=Mahalanobis(), missed_distance=float(d["missed"]))
        gat = DistanceGater(hyp, measure=Mahalanobis(), gate_threshold=float(d["gate"]))
        steps, truths = [], [GroundTruthPath() for _ in range(d["truth"].shape[2])]
        for t in range(T):
            ts = T0 + datetime.timedelta(seconds=dt * (t + 1))
            steps.append((ts, {Detection(d["traj_z"][s, t, m].reshape(3, 1), timestamp=ts, measurement_model=mm)
                               for m in range(n) if d["traj_valid"][s, t, m]}))
            for j, path in enumerate(truths):
                path.append(GroundTruthState(d["truth"][s, t, j].reshape(6, 1), timestamp=ts))
        tracker = MultiTargetTracker(
            initiator=SimpleMeasurementInitiator(prior_state=prior(float(d["vel_sigma"])), measurement_model=mm),
            deleter=CovarianceBasedDeleter(covar_trace_thresh=float(d["del_thr"])),
            detector=iter(steps), data_associator=NearestNeighbour(gat), updater=up)
        all_tracks = set()
        live_cov, live_g = [], np.zeros(4)
        for t, (ts, cur) in enumerate(tracker):
            all_tracks.update(cur)
            states = [tr.state for tr in cur]
            if states:
                live_cov.append(sum(np.linalg.norm(st.covar) for st in states))
            truth_states = [path[t] for path in truths]
            m, _ = g.compute_gospa_metric(states, truth_states)
            v = gospa_vals(m)
            live_g += np.array(v)
        out["live_m_cov"].append(np.mean(live_cov) if live_cov else np.nan)
        out["live_n_ts"].append(len(live_cov))
        out["live_gospa"].append(live_g)
        gens = [GOSPAMetric(p=p, c=c, tracks_key="tracks", truths_key="truths")]
        if all_tracks:                       # the library generator raises on an empty track set
            gens.append(SumofCovarianceNormsMetric())
        mgr = MultiManager(gens)
        mgr.add_data({"tracks": all_tracks, "truths": set(truths)})
        res = mgr.generate_metrics()
        vals = [m.value for m in res["sumCovNorms_generator"]["Sum of Covariance Norms Metric"].value] \
            if all_tracks else []
        out["lib_m_cov"].append(np.mean(vals) if vals else np.nan)
        out["lib_n_ts"].append(len(vals))
        gm = res["gospa_generator"]["GOSPA Metrics"]
        out["lib_gospa"].append(np.sum([gospa_vals(m) for m in gm.value], 0))
        out["n_tracks_total"].append(len(all_tracks))
    import stonesoup
    np.savez(sys.argv[2], **{k: np.asarray(v, float) for k, v in out.items()},
             stonesoup_version=np.array(stonesoup.__version__))


if __name__ == "__main__":
    main()
