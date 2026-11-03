"""Export the TF-IDF + logistic regression baseline to compact JSON for the web app.

The browser recomputes exactly what scikit-learn does: word tokens of two or more word characters,
unigrams and bigrams, sublinear term frequency, smoothed IDF, L2 normalisation, then a softmax over
the linear scores. `reference_proba` below reimplements that in plain Python and is checked against
scikit-learn before anything is written.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import joblib
import numpy as np

TOKEN = re.compile(r"(?u)\b\w\w+\b")
DIGITS = 6  # significant digits kept for weights


def terms(text):
    tokens = TOKEN.findall(text.lower())
    return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def reference_proba(text, vocab, idf, coef, intercept):
    counts = {}
    for t in terms(text):
        if t in vocab:
            counts[vocab[t]] = counts.get(vocab[t], 0) + 1
    weights = {i: (1 + math.log(c)) * idf[i] for i, c in counts.items()}
    norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
    scores = [intercept[k] + sum(coef[k][i] * w / norm for i, w in weights.items()) for k in range(3)]
    m = max(scores)
    e = [math.exp(s - m) for s in scores]
    return [x / sum(e) for x in e]


def rounded(values):
    return [float(f"{v:.{DIGITS}g}") for v in values]


def export(baseline_dir, data_dir, out_path):
    from training.protocol import load_prepared
    model = joblib.load(Path(baseline_dir) / "baseline.joblib")
    vec, lr = model.steps[0][1], model.steps[1][1]
    assert list(lr.classes_) == [0, 1, 2]
    assert vec.get_params()["token_pattern"] == TOKEN.pattern and vec.ngram_range == (1, 2) and vec.sublinear_tf
    order = sorted(vec.vocabulary_, key=vec.vocabulary_.get)
    payload = {
        "format": "rojak-tfidf-v1",
        "terms": order,
        "idf": rounded(vec.idf_),
        "coef": [rounded(row) for row in lr.coef_],
        "intercept": rounded(lr.intercept_),
    }
    vocab = {t: i for i, t in enumerate(payload["terms"])}
    frame, manifest = load_prepared(data_dir)
    texts = frame[frame.split != "train"].text.tolist()
    exact = model.predict_proba(texts)
    mine = np.array([reference_proba(t, vocab, payload["idf"], payload["coef"], payload["intercept"]) for t in texts])
    diff = float(np.abs(mine - exact).max())
    same = float((mine.argmax(1) == exact.argmax(1)).mean())
    assert diff < 1e-4 and same == 1.0, (diff, same)
    payload["checked"] = {"rows": len(texts), "max_probability_diff": diff, "split_sha256": manifest["split_sha256"]}
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    Path(out_path).write_text(text, encoding="utf-8")
    return {"terms": len(order), "bytes": len(text.encode()), "sha256": hashlib.sha256(text.encode()).hexdigest(), **payload["checked"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(json.dumps(export(a.baseline, a.data, a.out), indent=1))
