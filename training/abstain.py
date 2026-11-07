"""When should the app say "not sure"? Reliability and accuracy-vs-coverage for the live scores.

Concept: a model that answers every comment is forced to guess on the hard ones. If it may instead
say "not sure" on its least confident inputs, the comments it does label are right more often. The
price is coverage (the share it labels). This script measures that trade-off for the exact scores
the web app shows, and picks the cut-off honestly:

  1. Rebuild the app's scores for the frozen validation and test splits: the published 16-bit
     student's ONNX logits, temperature-scaled, blended with the TF-IDF file the browser loads
     (weights from web/src/model_release.json). A parity check stops the run unless the test labels
     reproduce the recorded browser accuracy exactly.
  2. Choose the confidence signal (top score, or the gap between the top two scores) and the
     cut-off on VALIDATION ONLY: the lowest cut-off whose answered comments reach TARGET_ACCURACY.
     Both were fixed before any test number was looked at.
  3. Score the test split once at that cut-off, with bootstrap intervals, and record the full
     test reliability table and accuracy-vs-coverage curve for the About page.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from training.export_tfidf import reference_proba
from training.post_hoc import ece, macro_f1, softmax
from training.protocol import load_prepared, prepare_csv

TARGET_ACCURACY = 0.90  # fixed before scoring: labelled comments should be right 9 times in 10
SIGNALS = ["confidence", "margin"]
COVERAGE_POINTS = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
ROOT = Path(__file__).resolve().parents[1]


def app_file(name, web_path):
    """The app's own file: bundled next to this script on Kaggle, or from web/ in a checkout."""
    for path in [ROOT / "training" / name, ROOT / web_path]:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"{name} not found (looked in training/ and {web_path}).")


def tfidf_probabilities(texts, model):
    vocab = {t: i for i, t in enumerate(model["terms"])}
    return np.array([reference_proba(t, vocab, model["idf"], model["coef"], model["intercept"]) for t in texts])


def app_scores(logits, p_tfidf, scoring):
    """What the browser shows: softmax(logits / T), then w * that + (1 - w) * TF-IDF."""
    p = softmax(np.asarray(logits, dtype=np.float64) / scoring["temperature"])
    w = scoring["blend_weight"]
    return w * p + (1 - w) * p_tfidf


def signal(p, kind):
    top = np.sort(p, axis=1)
    return top[:, -1] if kind == "confidence" else top[:, -1] - top[:, -2]


def risk_coverage(score, correct):
    """Answered accuracy when only the k most confident comments are labelled, for every k."""
    order = np.argsort(-score, kind="stable")
    hits = np.cumsum(correct[order])
    k = np.arange(1, len(score) + 1)
    return k / len(score), hits / k


def aurc(score, correct):
    """Area under the risk-coverage curve: mean error over all coverage levels (lower is better)."""
    _, acc = risk_coverage(score, correct)
    return float(np.mean(1 - acc))


def choose_cutoff(score, correct, target=TARGET_ACCURACY):
    """Lowest cut-off whose answered comments are at least `target` accurate (most coverage)."""
    cutoffs = np.unique(score)
    for c in cutoffs:  # ascending: the first that meets the target labels the most comments
        keep = score >= c
        if keep.any() and correct[keep].mean() >= target:
            return float(c)
    return float(cutoffs[-1])


def at_cutoff(score, correct, cutoff):
    keep = score >= cutoff
    return {
        "coverage": float(keep.mean()),
        "answered": int(keep.sum()),
        "not_sure": int((~keep).sum()),
        "answered_accuracy": float(correct[keep].mean()) if keep.any() else None,
        "not_sure_accuracy": float(correct[~keep].mean()) if (~keep).any() else None,
    }


def bootstrap(score, correct, cutoff, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    cov, acc = [], []
    for _ in range(n):
        i = rng.integers(0, len(score), len(score))
        r = at_cutoff(score[i], correct[i], cutoff)
        cov.append(r["coverage"])
        if r["answered_accuracy"] is not None:
            acc.append(r["answered_accuracy"])
    q = lambda a: [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]
    return {"coverage_95": q(cov), "answered_accuracy_95": q(acc)}


def reliability(p, y, bins=10):
    conf = p.max(1)
    correct = p.argmax(1) == y
    edges = np.linspace(1 / 3, 1, bins + 1)  # three classes: the top score is never below 1/3
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        rows.append({"from": round(float(lo), 4), "to": round(float(hi), 4), "count": int(m.sum()),
                     "mean_confidence": float(conf[m].mean()) if m.any() else None,
                     "accuracy": float(correct[m].mean()) if m.any() else None})
    return rows


def evaluate(parts, scoring, expected_accuracy):
    v, t = parts["validation"], parts["test"]
    pv, pt = app_scores(v["z"], v["t"], scoring), app_scores(t["z"], t["t"], scoring)
    yv, yt = v["y"], t["y"]
    pred_t = pt.argmax(1)
    # Parity: these must be the labels the browser gave (recorded browser accuracy on the same split).
    hits, expected_hits = int((pred_t == yt).sum()), round(expected_accuracy * len(yt))
    if hits != expected_hits:
        raise ValueError(f"Parity check failed: {hits} correct, the browser record says {expected_hits}.")
    cv, ct = pv.argmax(1) == yv, pred_t == yt
    # 1. Signal and cut-off from validation alone.
    val_aurc = {s: aurc(signal(pv, s), cv) for s in SIGNALS}
    chosen = min(SIGNALS, key=lambda s: (round(val_aurc[s], 6), SIGNALS.index(s)))
    cutoff = choose_cutoff(signal(pv, chosen), cv)
    # 2. Test, once.
    st = signal(pt, chosen)
    cov, acc = risk_coverage(st, ct)
    curve = []
    for c in COVERAGE_POINTS:
        k = max(1, int(round(c * len(st))))
        curve.append({"coverage": c, "answered_accuracy": float(acc[k - 1])})
    return {
        "target_accuracy": TARGET_ACCURACY,
        "selection_rule": "Signal with the lower area under the validation risk-coverage curve; cut-off = lowest value "
                          "whose answered validation comments reach the target accuracy. Test scored once afterwards.",
        "validation": {"aurc": val_aurc, "chosen_signal": chosen, "cutoff": cutoff,
                       **at_cutoff(signal(pv, chosen), cv, cutoff)},
        "test": {
            "rows": int(len(yt)),
            "parity": {"accuracy": float(accuracy_score(yt, pred_t)), "macro_f1": macro_f1(yt, pred_t),
                       "matches_browser_record": True},
            "ece": ece(pt, yt),
            "aurc": {s: aurc(signal(pt, s), ct) for s in SIGNALS},
            "at_cutoff": {**at_cutoff(st, ct, cutoff), **bootstrap(st, ct, cutoff)},
            "curve": curve,
            "reliability": reliability(pt, yt),
        },
    }


def find_one(root, pattern):
    found = sorted(Path(root).rglob(pattern))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one {pattern} under {root}, found {len(found)}.")
    return found[0]


def kaggle_run(input_dir, out_dir, work_dir="/tmp/rojak"):
    """On Kaggle: the private corpus and the 'export' run's ONNX logits are attached as inputs."""
    from training.kaggle_job import SPLIT_SHA256, find_corpus
    data_dir = Path(work_dir) / "data"
    if not (data_dir / "manifest.json").exists():
        prepare_csv(find_corpus(input_dir), data_dir)
    frame, manifest = load_prepared(data_dir)
    if manifest["split_sha256"] != SPLIT_SHA256:
        raise ValueError("The prepared split does not match the frozen evaluation split.")
    release = app_file("model_release.json", "web/src/model_release.json")
    model = app_file("tfidf.json", "web/public/tfidf.json")
    with np.load(find_one(input_dir, "onnx_eval/model_fp16/logits.npz")) as z:
        logits = {k: z[k] for k in ["validation", "test"]}
    parts = {}
    for split in ["validation", "test"]:
        rows = frame[frame.split == split].reset_index(drop=True)
        assert len(rows) == len(logits[split])
        parts[split] = {"y": rows.label_id.to_numpy(), "z": logits[split], "t": tfidf_probabilities(rows.text, model)}
    report = {"protocol": manifest["protocol"], "split_sha256": manifest["split_sha256"],
              "what": "Live app scores (fp16 student + TF-IDF blend) on the frozen splits: when to say 'not sure'.",
              "scoring": release["scoring"], **evaluate(parts, release["scoring"], release["evaluation"]["browser_accuracy"])}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "abstain.json").write_text(json.dumps(report, indent=2))
    (out / "summary.json").write_text(json.dumps({"experiment": "abstain", "validation": report["validation"],
                                                  "test_at_cutoff": report["test"]["at_cutoff"]}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--out", default="/kaggle/working/out")
    a = parser.parse_args()
    r = kaggle_run(a.input, a.out)
    print(json.dumps({"validation": r["validation"], "test_at_cutoff": r["test"]["at_cutoff"]}, indent=1))


if __name__ == "__main__":
    main()
