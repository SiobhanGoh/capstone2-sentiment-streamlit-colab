import numpy as np

from training import label_review


def test_flag_marks_confident_disagreement_only():
    labels = np.array([0, 0, 1, 1, 2, 2])
    probs = np.array([
        [0.9, 0.05, 0.05],   # agrees
        [0.1, 0.1, 0.8],     # labelled 0, confidently 2 -> flagged
        [0.2, 0.7, 0.1],     # agrees
        [0.45, 0.4, 0.15],   # labelled 1, prefers 0 weakly -> not flagged
        [0.05, 0.05, 0.9],   # agrees
        [0.3, 0.3, 0.4],     # agrees
    ])
    flagged, margin, thresholds = label_review.flag(probs, labels)
    assert flagged.tolist() == [False, True, False, False, False, False]
    assert margin[1] == 0.8 - 0.1 and thresholds.shape == (3,)


def test_item_ids_are_stable_and_hide_the_text():
    a = label_review.item_id("servis lambat")
    assert a == label_review.item_id("servis lambat") and len(a) == 16 and "servis" not in a
