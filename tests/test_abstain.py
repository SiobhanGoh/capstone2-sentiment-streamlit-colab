import numpy as np
import pytest

from training.abstain import (TARGET_ACCURACY, app_scores, at_cutoff, aurc, choose_cutoff, evaluate, reliability,
                              risk_coverage, signal)
from training.post_hoc import softmax


def test_cutoff_is_the_lowest_that_reaches_the_target():
    score = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.991, 0.999])
    correct = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    # Labelling from 0.6 up gives 8 of 8 right; from 0.5 up gives 8 of 9 (88.9%), below 90%.
    assert choose_cutoff(score, correct, 0.9) == 0.6
    r = at_cutoff(score, correct, 0.6)
    assert (r["answered"], r["not_sure"], r["answered_accuracy"], r["not_sure_accuracy"]) == (8, 2, 1.0, 0.0)


def test_risk_coverage_labels_the_most_confident_first():
    score = np.array([0.9, 0.2, 0.8, 0.1])
    correct = np.array([1, 0, 1, 1], dtype=bool)
    cov, acc = risk_coverage(score, correct)
    assert list(cov) == [0.25, 0.5, 0.75, 1.0]
    assert list(acc) == [1.0, 1.0, 2 / 3, 0.75]
    # A signal that ranks every wrong answer last has a lower area than one that does not.
    assert aurc(score, correct) < aurc(-score, correct)


def test_margin_and_confidence_signals():
    p = np.array([[0.6, 0.3, 0.1], [0.4, 0.35, 0.25]])
    assert np.allclose(signal(p, "confidence"), [0.6, 0.4])
    assert np.allclose(signal(p, "margin"), [0.3, 0.05])


def test_reliability_bins_cover_every_row():
    rng = np.random.default_rng(0)
    p = softmax(rng.normal(size=(500, 3)) * 2)
    y = rng.integers(0, 3, 500)
    rows = reliability(p, y)
    assert sum(r["count"] for r in rows) == 500
    assert rows[0]["from"] == pytest.approx(1 / 3, abs=1e-4) and rows[-1]["to"] == 1.0


def synthetic_parts(seed=1, n=600):
    rng = np.random.default_rng(seed)
    parts = {}
    for split in ["validation", "test"]:
        y = rng.integers(0, 3, n)
        z = rng.normal(size=(n, 3))
        z[np.arange(n), y] += 2.0  # informative logits
        t = softmax(rng.normal(size=(n, 3)))
        parts[split] = {"y": y, "z": z, "t": t}
    return parts


def test_evaluate_picks_on_validation_and_checks_parity():
    parts = synthetic_parts()
    scoring = {"temperature": 1.1, "blend_weight": 0.65}
    pt = app_scores(parts["test"]["z"], parts["test"]["t"], scoring)
    expected = float((pt.argmax(1) == parts["test"]["y"]).mean())
    report = evaluate(parts, scoring, expected)
    v = report["validation"]
    assert v["chosen_signal"] in ("confidence", "margin")
    assert v["answered_accuracy"] >= TARGET_ACCURACY
    assert report["test"]["parity"]["matches_browser_record"]
    assert report["test"]["curve"][0]["coverage"] == 1.0
    assert report["test"]["curve"][0]["answered_accuracy"] == pytest.approx(expected)
    # A wrong browser record must stop the run.
    with pytest.raises(ValueError, match="Parity"):
        evaluate(parts, scoring, expected - 0.01)
