"""The method has no gradient (nothing is optimised; the README, so instead of check_grads this file asserts
the estimators' closed-form / exactly-enumerable statistical properties (toy checks at reduced size):
D9 toy points + 64-atom bootstrap enumeration, D4.2 closed-form bootstrap moments, D2.5 P(X>Y) identities,
D3 profile unbiasedness and variance, Fig 8 area identities, D5.2 lift identity, D10 coverage prediction.
The vendored PPO's own check_grads tests live in the ppo_implementation_details reproduction (papers/ppo_implementation_details/tests)."""
import jax

from src import algorithm as A
from src import eval as E

KEY = A.branch_key(0, A.EXP_IDS["tests"], 99)


def test_d9_toy_points_enumeration_and_mc():
    r = E.toys_toy(jax.random.fold_in(KEY, 1), B=60_000)
    assert r["points_pass"], r["points"]
    assert r["pct_pass"], r["pct"]
    assert r["boot_pass"], r["boot"]


def test_d4_2_closed_form_bootstrap_moments():
    r = E.toys_closed_forms(jax.random.fold_in(KEY, 2), n_cases=9, B=20_000)
    assert r["exact_max_abs_err"] <= 1e-12
    assert r["pass_"], r


def test_d2_5_poi_identities():
    assert E.toys_poi_identities(jax.random.fold_in(KEY, 3), n_cases=40)["pass_"]


def test_d3_profile_unbiased_and_avg_profile_biased():
    r = E.toys_profile_unbiased(jax.random.fold_in(KEY, 4), reps=6000, N=3)
    assert r["max_abs_z_run"] <= 4.5 and r["max_abs_z_avg"] > 10, r
    # recheck : the average-score profile's bias against ITS OWN target shrinks with N; its gap to the run-score
    # tail (a different estimand) grows with N
    rows = r["avg_rows"]
    assert rows[-1]["max_abs_bias_vs_own_target"] < rows[0]["max_abs_bias_vs_own_target"], rows
    assert rows[-1]["avg_gap_to_run_tail"] > rows[0]["avg_gap_to_run_tail"], rows
    assert r["pass_"]


def test_fig8_area_identities():
    assert E.toys_area_identities(jax.random.fold_in(KEY, 5), n_cases=20)["pass_"]


def test_d5_2_lift_identity():
    assert E.toys_lift_identity(jax.random.fold_in(KEY, 6))["pass_"]


def test_d10_coverage_prediction_small():
    r = E.toys_coverage(jax.random.fold_in(KEY, 8), Ns=(3,), M=26, sets=400, inner_B=1000)
    assert r["pass_"], r
