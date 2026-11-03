"""Cheap, CPU-only improvements on top of the published browser model, chosen on validation only.

Options compared (fixed before looking at any score):
  mbert          the published model as it is
  mbert+offsets  three class offsets added to its log-probabilities
  blend          w * mBERT probabilities + (1 - w) * TF-IDF baseline probabilities
  blend+offsets  the blend with class offsets
A temperature is always fitted on validation for calibration; it never changes a label on its own.

Each option's parameters are fitted on validation. To stop options with more parameters from looking
better simply because they were fitted and judged on the same rows, options are compared by 5-fold
cross-validation inside the validation split. The winner is then refitted on all validation rows and
the test split is scored once. A McNemar test and paired bootstrap intervals compare it with mbert.
"""
import argparse
import json
from math import comb
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from training.protocol import load_prepared

OPTIONS = ["mbert", "mbert+offsets", "blend", "blend+offsets"]
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)
OFFSETS = np.round(np.arange(-1.0, 1.0001, 0.1), 2)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y):
    """Temperature minimising negative log-likelihood, by a fine 1-D search."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(5), 400))
    nll = [-np.log(softmax(logits / t)[np.arange(len(y)), y] + 1e-12).mean() for t in grid]
    return float(grid[int(np.argmin(nll))])


def macro_f1(y, pred):
    return float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0))


def probs(option, params, z, p_tfidf):
    p = softmax(z / params["temperature"])
    if option.startswith("blend"):
        p = params["weight"] * p + (1 - params["weight"]) * p_tfidf
    return p


def predict(option, params, z, p_tfidf):
    log_p = np.log(probs(option, params, z, p_tfidf) + 1e-12)
    return (log_p + np.asarray(params.get("offsets", [0, 0, 0]))).argmax(1)


def fit(option, z, p_tfidf, y):
    """Fit an option's parameters on these rows by maximising macro-F1 (temperature by likelihood)."""
    params = {"temperature": fit_temperature(z, y)}
    weights = WEIGHTS if option.startswith("blend") else [1.0]
    best = None
    for w in weights:
        base = {**params, "weight": float(w)}
        log_p = np.log(probs(option, base, z, p_tfidf) + 1e-12)
        grid = [(0.0, a, b) for a in OFFSETS for b in OFFSETS] if option.endswith("offsets") else [(0.0, 0.0, 0.0)]
        for off in grid:
            score = macro_f1(y, (log_p + np.asarray(off)).argmax(1))
            # Prefer smaller adjustments on ties so the choice is stable.
            key = (round(score, 6), -abs(w - 1.0) if option.startswith("blend") else 0, -sum(abs(o) for o in off))
            if best is None or key > best[0]:
                best = (key, {**base, "offsets": [float(o) for o in off]})
    return best[1]


def cross_validated(option, z, p_tfidf, y, folds=5, seed=42):
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(np.where(y == k)[0]) for k in range(3)])
    fold_of = np.empty(len(y), dtype=int)
    fold_of[order] = np.arange(len(y)) % folds
    pred = np.empty(len(y), dtype=int)
    for f in range(folds):
        train, held = fold_of != f, fold_of == f
        params = fit(option, z[train], p_tfidf[train], y[train])
        pred[held] = predict(option, params, z[held], p_tfidf[held])
    return macro_f1(y, pred)


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def mcnemar_exact(y, a, b):
    """Two-sided exact McNemar test on the rows where exactly one of the two predictions is right."""
    only_a = int(((a == y) & (b != y)).sum())
    only_b = int(((a != y) & (b == y)).sum())
    n, k = only_a + only_b, min(only_a, only_b)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"only_reference_right": only_a, "only_candidate_right": only_b, "p_value": float(p)}


def bootstrap(y, a, b, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    acc, f1 = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        acc.append(accuracy_score(y[i], b[i]) - accuracy_score(y[i], a[i]))
        f1.append(macro_f1(y[i], b[i]) - macro_f1(y[i], a[i]))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"accuracy_diff_95ci": q(acc), "macro_f1_diff_95ci": q(f1)}


def run(data_dir, baseline_dir, logits_dir, output_path):
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("A post-hoc result already exists; keep it and choose a new path.")
    frame, manifest = load_prepared(data_dir)
    baseline = joblib.load(Path(baseline_dir) / "baseline.joblib")
    parts = {}
    for split in ["validation", "test"]:
        rows = frame[frame.split == split].reset_index(drop=True)
        parts[split] = {"y": rows.label_id.to_numpy(), "z": np.load(Path(logits_dir) / f"fp16_{split}.npy"),
                        "t": baseline.predict_proba(rows.text)}
        assert len(parts[split]["z"]) == len(rows)
    v, t = parts["validation"], parts["test"]
    # 1. Choose an option by cross-validation inside validation, before any test score.
    cv = {o: cross_validated(o, v["z"], v["t"], v["y"]) for o in OPTIONS}
    selected = max(OPTIONS, key=lambda o: (round(cv[o], 6), -OPTIONS.index(o)))
    fitted = {o: fit(o, v["z"], v["t"], v["y"]) for o in OPTIONS}
    # 2. Score the test split once.
    ref_pred = predict("mbert", fitted["mbert"], t["z"], t["t"])
    results = {}
    for o in OPTIONS:
        pred = predict(o, fitted[o], t["z"], t["t"])
        p = probs(o, fitted[o], t["z"], t["t"])
        results[o] = {"params": fitted[o], "validation_cv_macro_f1": cv[o],
                      "test": {"accuracy": float(accuracy_score(t["y"], pred)), "macro_f1": macro_f1(t["y"], pred)},
                      "test_ece": ece(p, t["y"]), "comparison_with_mbert": {**mcnemar_exact(t["y"], ref_pred, pred),
                                                                            **bootstrap(t["y"], ref_pred, pred)}}
    raw = softmax(t["z"])
    report = {
        "protocol": manifest["protocol"], "split_sha256": manifest["split_sha256"], "test_rows": int(len(t["y"])),
        "model": "SiobhanGoh/malay-english-sentiment-onnx fp16 (published browser model)",
        "selection_rule": "Highest 5-fold cross-validated macro-F1 inside the validation split; test scored once afterwards.",
        "selected": selected,
        "calibration": {"temperature": fitted["mbert"]["temperature"], "test_ece_before": ece(raw, t["y"]),
                        "test_ece_after": results["mbert"]["test_ece"]},
        "options": results,
        "note": "Retrospective internal evaluation on the existing corpus; not external validation.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--logits", required=True)
    parser.add_argument("--out", required=True)
    a = parser.parse_args()
    r = run(a.data, a.baseline, a.logits, a.out)
    print(json.dumps({"selected": r["selected"], "calibration": r["calibration"],
                      **{o: {"cv": round(x["validation_cv_macro_f1"], 4), **{k: round(v, 4) for k, v in x["test"].items()},
                             "p": round(x["comparison_with_mbert"]["p_value"], 4)} for o, x in r["options"].items()}}, indent=1))


if __name__ == "__main__":
    main()
