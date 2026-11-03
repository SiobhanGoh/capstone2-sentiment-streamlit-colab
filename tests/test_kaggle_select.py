import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from training import kaggle_select
from training.protocol import load_prepared, prepare_csv


@pytest.fixture
def setup(tmp_path):
    rng = np.random.default_rng(0)
    # Text carries no label signal, so the TF-IDF baseline is near chance and cannot decide the test.
    vocab = ["kopi", "teh", "nasi", "roti", "mee", "air", "hari", "esok"]
    rows = [{"Tweet": " ".join(rng.choice(vocab, 4)) + f" {i}", "Sentiment": s}
            for s in ["NEGATIVE", "NEUTRAL", "POSITIVE"] for i in range(120)]
    pd.DataFrame(rows).to_csv(tmp_path / "c.csv", index=False)
    prepare_csv(tmp_path / "c.csv", tmp_path / "data")
    frame, _ = load_prepared(tmp_path / "data")
    split = {k: frame[frame.split == k] for k in ["train", "validation", "test"]}
    base = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=500)).fit(split["train"].text, split["train"].label_id)
    (tmp_path / "baseline").mkdir()
    joblib.dump(base, tmp_path / "baseline" / "baseline.joblib")

    def noisy(y, acc):
        # Logits that are right about `acc` of the time.
        z = rng.normal(0, 1, (len(y), 3))
        right = rng.random(len(y)) < acc
        z[np.arange(len(y)), np.where(right, y, (y + 1) % 3)] += 3
        return z.astype(np.float32)

    ys = {k: v.label_id.to_numpy() for k, v in split.items()}
    (tmp_path / "app").mkdir()
    for k in ["validation", "test"]:
        np.save(tmp_path / "app" / f"fp16_{k}.npy", noisy(ys[k], 0.6))
    for fam, acc, seeds in [("weak", 0.5, 2), ("strong", 0.95, 3)]:
        for s in range(seeds):
            d = tmp_path / "results" / fam / "out" / "runs" / f"{fam}-s{s}"
            d.mkdir(parents=True)
            z = {k: noisy(ys[k], acc) for k in ["validation", "test", "train"]}
            np.savez(d / "logits.npz", **z)
            f1 = float((z["validation"].argmax(1) == ys["validation"]).mean())
            (d / "training.json").write_text(json.dumps({"status": "complete", "spec": {"family": fam},
                                                         "selected_validation": {"macro_f1": f1, "accuracy": f1}}))
    stage = tmp_path / "results" / "weak" / "out" / "runs" / "weak-stage1"
    stage.mkdir()
    (stage / "training.json").write_text(json.dumps({"status": "complete", "spec": {"family": "stage1"}}))
    return tmp_path


def test_selection_prefers_the_strong_family_and_scores_test_once(setup):
    r = kaggle_select.run(setup / "results", setup / "data", setup / "baseline", setup / "app", setup / "out.json")
    assert set(r["runs"]) == {"weak-s0", "weak-s1", "strong-s0", "strong-s1", "strong-s2"}  # first stages left out
    names = set(r["validation_cv_macro_f1"])
    for c in ["app", "weak-best", "weak-mean", "strong-best", "strong-mean", "all-mean", "app+weak", "app+strong"]:
        assert c in names and c + "+tfidf" in names
    assert r["selected"].removesuffix("+tfidf") in {"strong-best", "strong-mean", "all-mean", "app+strong"}
    assert r["test"]["selected"]["accuracy"] > r["test"]["current_app"]["accuracy"]
    assert {"p_value", "accuracy_diff_95ci"} <= set(r["test"]["comparison"])
    with pytest.raises(FileExistsError):
        kaggle_select.run(setup / "results", setup / "data", setup / "baseline", setup / "app", setup / "out.json")
