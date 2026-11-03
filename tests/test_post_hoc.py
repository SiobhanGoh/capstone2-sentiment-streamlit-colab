import numpy as np

from training.post_hoc import cross_validated, ece, fit, fit_temperature, mcnemar_exact, predict, softmax


def test_temperature_recovers_a_known_scale():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(4000, 3)) * 2
    y = np.array([rng.choice(3, p=p) for p in softmax(true)])
    assert abs(fit_temperature(true * 3, y) - 3) < 0.35


def test_mcnemar_counts_disagreements_and_is_symmetric():
    y = np.array([0, 0, 0, 0, 1, 1])
    a = np.array([0, 0, 1, 1, 1, 1])
    b = np.array([0, 1, 0, 0, 1, 0])
    r = mcnemar_exact(y, a, b)
    # Reference right only on rows 1 and 5; candidate right only on rows 2 and 3.
    assert (r["only_reference_right"], r["only_candidate_right"]) == (2, 2)
    assert r["p_value"] == mcnemar_exact(y, b, a)["p_value"] == 1.0
    assert mcnemar_exact(y, y, y)["p_value"] == 1.0


def test_blend_weight_one_matches_the_model_alone():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(50, 3))
    t = softmax(rng.normal(size=(50, 3)))
    params = {"temperature": 1.0, "weight": 1.0, "offsets": [0, 0, 0]}
    assert (predict("blend", params, z, t) == z.argmax(1)).all()


def test_offsets_can_rescue_an_underpredicted_class():
    y = np.array([0] * 40 + [1] * 40)
    # Class 1 rows still score class 0 highest, so class 1 is never predicted without an offset.
    z = np.c_[np.where(y == 0, 1.0, 0.7), np.where(y == 1, 0.5, 0.0), np.full(80, -5.0)]
    t = softmax(np.zeros((80, 3)))
    assert (predict("mbert", fit("mbert", z, t, y), z, t) == 0).all()
    tuned = fit("mbert+offsets", z, t, y)
    assert (predict("mbert+offsets", tuned, z, t) == y).mean() > 0.95
    assert 0 <= cross_validated("mbert+offsets", z, t, y) <= 1


def test_ece_is_zero_for_perfectly_calibrated_certain_predictions():
    y = np.array([0, 1, 2])
    assert ece(np.eye(3), y) == 0
