"""Stone Soup 1.9.1 reference outputs (ORACLE, library the authors name; NOT our implementation).

Run ONLY with the isolated oracle venv:
    $OFFICIAL_ROOT/.venv/bin/python scripts/official/ss_ref.py <in.npz> <out.npz>
Input npz fields select the cases:
  ukf_x [N,6], ukf_P [N,6,6], ukf_z [N,3], R [3,3]        -> UKF measurement prediction, Mahalanobis distance, update
  init_z [M,3], R, vel_sigma                               -> SimpleMeasurementInitiator
  traj_z [S,T,n,3], traj_valid [S,T,n], dt, q, R, vel_sigma, gate, missed, del_thr -> full MultiTargetTracker
  gospa_truth [G,n,6], gospa_tracks [G,K,6], gospa_valid [G,K], c, p -> GOSPAMetric
"""
import sys
import datetime

import numpy as np
from stonesoup.types.state import GaussianState, State
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.updater.kalman import UnscentedKalmanUpdater
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.gater.distance import DistanceGater
from stonesoup.measures import Mahalanobis, Euclidean
from stonesoup.dataassociator.neighbour import NearestNeighbour
from stonesoup.initiator.simple import SimpleMeasurementInitiator
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.metricgenerator.ospametric import GOSPAMetric

T0 = datetime.datetime(2025, 1, 1)


def mm_of(R):
    return CartesianToElevationBearingRange(ndim_state=6, mapping=(0, 2, 4), noise_covar=np.asarray(R))


def ukf_cases(d, out):
    mm = mm_of(d["R"])
    up = UnscentedKalmanUpdater(measurement_model=mm)
    zh, S, C, D, xu, Pu = [], [], [], [], [], []
    for x, P, z in zip(d["ukf_x"], d["ukf_P"], d["ukf_z"]):
        pred = GaussianStatePrediction(x.reshape(6, 1), P, timestamp=T0)
        mp = up.predict_measurement(pred, measurement_model=mm)
        det = Detection(z.reshape(3, 1), timestamp=T0, measurement_model=mm)
        D.append(Mahalanobis()(mp, det))
        upd = up.update(SingleHypothesis(pred, det))
        zh.append(np.asarray(mp.state_vector, float).ravel())
        S.append(np.asarray(mp.covar, float))
        C.append(np.asarray(mp.cross_covar, float))
        xu.append(np.asarray(upd.state_vector, float).ravel())
        Pu.append(np.asarray(upd.covar, float))
    out.update(ukf_zhat=np.array(zh), ukf_S=np.array(S), ukf_C=np.array(C), ukf_D=np.array(D),
               ukf_xu=np.array(xu), ukf_Pu=np.array(Pu))


def prior(vs):
    return GaussianState(np.zeros((6, 1)), np.diag([0.0, vs ** 2, 0.0, vs ** 2, 0.0, vs ** 2]), timestamp=T0)


def init_cases(d, out):
    mm = mm_of(d["R"])
    ini = SimpleMeasurementInitiator(prior_state=prior(float(d["vel_sigma"])), measurement_model=mm)
    xs, Ps = [], []
    for z in d["init_z"]:
        tr = list(ini.initiate({Detection(z.reshape(3, 1), timestamp=T0, measurement_model=mm)}, T0))[0]
        xs.append(np.asarray(tr.state_vector, float).ravel())
        Ps.append(np.asarray(tr.covar, float))
    out.update(init_x=np.array(xs), init_P=np.array(Ps))


def traj_cases(d, out):
    mm = mm_of(d["R"])
    dt, q = float(d["dt"]), float(d["q"])
    S, T, n, _ = d["traj_z"].shape
    tm = CombinedLinearGaussianTransitionModel([ConstantVelocity(q)] * 3)
    pr = KalmanPredictor(tm)
    up = UnscentedKalmanUpdater(measurement_model=mm)
    K = int(d["k_max"])
    xs = np.full((S, T, K, 6), np.nan)
    Ps = np.full((S, T, K, 6, 6), np.nan)
    cnt = np.zeros((S, T), int)
    for s in range(S):
        hyp = DistanceHypothesiser(pr, up, measure=Mahalanobis(), missed_distance=float(d["missed"]))
        gat = DistanceGater(hyp, measure=Mahalanobis(), gate_threshold=float(d["gate"]))
        steps = []
        for t in range(T):
            ts = T0 + datetime.timedelta(seconds=dt * (t + 1))
            dets = {Detection(d["traj_z"][s, t, m].reshape(3, 1), timestamp=ts, measurement_model=mm)
                    for m in range(n) if d["traj_valid"][s, t, m]}
            steps.append((ts, dets))
        tracker = MultiTargetTracker(
            initiator=SimpleMeasurementInitiator(prior_state=prior(float(d["vel_sigma"])), measurement_model=mm),
            deleter=CovarianceBasedDeleter(covar_trace_thresh=float(d["del_thr"])),
            detector=iter(steps), data_associator=NearestNeighbour(gat), updater=up)
        for t, (ts, tracks) in enumerate(tracker):
            tl = sorted(tracks, key=lambda tr: tuple(np.asarray(tr.state_vector, float).ravel()[[0, 2, 4]]))
            cnt[s, t] = len(tl)
            for j, tr in enumerate(tl[:K]):
                xs[s, t, j] = np.asarray(tr.state_vector, float).ravel()
                Ps[s, t, j] = np.asarray(tr.covar, float)
    out.update(traj_x=xs, traj_P=Ps, traj_count=cnt)


def gospa_cases(d, out):
    g = GOSPAMetric(p=int(d["p"]), c=float(d["c"]))
    res = []
    for tru, trk, v in zip(d["gospa_truth"], d["gospa_tracks"], d["gospa_valid"]):
        ms = [State(x.reshape(6, 1), timestamp=T0) for x, vv in zip(trk, v) if vv]
        ts = [State(x.reshape(6, 1), timestamp=T0) for x in tru]
        m, _ = g.compute_gospa_metric(ms, ts)
        val = m.value
        res.append([val["distance"], val["localisation"], val["missed"], val["false"]])
    out.update(gospa=np.array(res, float))


def analytic_jacobian(self, state, **kwargs):
    """Exact Jacobian of (asin(z/rho), atan2(y, x), rho) w.r.t. the 6-D state (positions at 0, 2, 4)."""
    x, y, z = (float(state.state_vector[i, 0]) for i in (0, 2, 4))
    rho2 = x * x + y * y + z * z
    rho = np.sqrt(rho2)
    rh2 = x * x + y * y
    rh = np.sqrt(rh2)
    J = np.zeros((3, 6))
    J[0, [0, 2, 4]] = [-x * z / (rho2 * rh), -y * z / (rho2 * rh), rh / rho2]
    J[1, [0, 2]] = [-y / rh2, x / rh2]
    J[2, [0, 2, 4]] = [x / rho, y / rho, z / rho]
    return J


def main():
    d = dict(np.load(sys.argv[1]))
    if int(d.get("analytic_jac", 0)):
        CartesianToElevationBearingRange.jacobian = analytic_jacobian
    out = {}
    if "ukf_x" in d:
        ukf_cases(d, out)
    if "init_z" in d:
        init_cases(d, out)
    if "traj_z" in d:
        traj_cases(d, out)
    if "gospa_truth" in d:
        gospa_cases(d, out)
    import stonesoup
    out["stonesoup_version"] = np.array(stonesoup.__version__)
    np.savez(sys.argv[2], **out)


if __name__ == "__main__":
    main()
