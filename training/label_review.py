"""Pick training comments whose label looks wrong, for a blind human review.

Each training comment gets the prediction of the cross-validation model that never saw it (the
'oof' Kaggle run). A comment is flagged when that model prefers another class with confidence
above the average confidence it gives that class on comments labelled with it (the "confident
learning" rule of Northcutt et al., 2021), ranked by how far the other class beats the given label.
Random comments where the model agrees are mixed in as controls: they show how often the reviewer
agrees with the original labels when nothing looks wrong. The review order is shuffled, so the
reviewer cannot tell flagged comments from controls, and the page never shows labels or guesses.

Only the train split is reviewed; validation and test labels are left exactly as they are.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.protocol import LABELS, load_prepared


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def out_of_fold(results_dir, n_train):
    """Held-out probabilities for every training row, from the fold model that did not train on it."""
    probs = np.full((n_train, 3), np.nan)
    runs = sorted(Path(results_dir).rglob("runs/oof-*/logits.npz"))
    for path in runs:
        with np.load(path) as z:
            k = int(path.parent.name.rsplit("-f", 1)[1])
            held = z["train_fold"] == k
            probs[held] = softmax(z["train"][held])
    if np.isnan(probs).any():
        raise ValueError(f"{int(np.isnan(probs).any(1).sum())} training rows have no held-out prediction.")
    return probs, len(runs)


def flag(probs, labels):
    """Confident-learning flags and a disagreement margin for each row."""
    thresholds = np.array([probs[labels == k, k].mean() for k in range(3)])
    pred = probs.argmax(1)
    confident = probs[np.arange(len(labels)), pred] >= thresholds[pred]
    flagged = (pred != labels) & confident
    margin = probs[np.arange(len(labels)), pred] - probs[np.arange(len(labels)), labels]
    return flagged, margin, thresholds


def item_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build(data_dir, results_dir, n_flagged=270, n_controls=30, seed=7):
    frame, manifest = load_prepared(data_dir)
    train = frame[frame.split == "train"].reset_index(drop=True)
    y = train.label_id.to_numpy()
    probs, n_runs = out_of_fold(results_dir, len(train))
    flagged, margin, thresholds = flag(probs, y)
    rng = np.random.default_rng(seed)
    top = np.where(flagged)[0][np.argsort(-margin[flagged], kind="stable")][:n_flagged]
    agree = np.where(probs.argmax(1) == y)[0]
    controls = rng.choice(agree, size=min(n_controls, len(agree)), replace=False)
    chosen = np.concatenate([top, controls])
    rows = pd.DataFrame({
        "row": chosen, "id": [item_id(t) for t in train.text.iloc[chosen]], "text": train.text.iloc[chosen].to_numpy(),
        "original_label": [LABELS[i] for i in y[chosen]], "model_label": [LABELS[i] for i in probs[chosen].argmax(1)],
        "model_confidence": probs[chosen].max(1).round(4), "role": ["flagged"] * len(top) + ["control"] * len(controls),
    })
    rows = rows.iloc[rng.permutation(len(rows))].reset_index(drop=True)
    rows["n"] = np.arange(len(rows))
    stats = {"split_sha256": manifest["split_sha256"], "train_rows": len(train), "fold_models": n_runs,
             "out_of_fold_accuracy": float((probs.argmax(1) == y).mean()),
             "flagged_rows": int(flagged.sum()), "flagged_share": float(flagged.mean()),
             "flagged_by_label": {LABELS[k]: int((flagged & (y == k)).sum()) for k in range(3)},
             "thresholds": thresholds.round(4).tolist(), "review_rows": len(rows),
             "reviewed_flagged": int(len(top)), "reviewed_controls": int(len(controls))}
    return rows, stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True, help="Private folder (never commit): review.csv and stats.json")
    a = p.parse_args()
    rows, stats = build(a.data, a.results)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "review.csv", index=False)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
