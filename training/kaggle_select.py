"""Choose among the Kaggle runs on validation only, then score the test split once.

Written before any Kaggle result existed. Candidates, built from whatever runs finished:
  <family>-best   the family's run with the best validation macro-F1
  <family>-mean   the average of every finished run in the family (seed ensemble)
  all-mean        the average of all family means
  app             the current web app's model (published mBERT) for reference
and each of these blended with the TF-IDF baseline, the weight fitted on validation exactly as in
training/post_hoc.py. Members are averaged as temperature-scaled probabilities, each temperature
fitted on validation. The choice is the highest 5-fold cross-validated macro-F1 inside the
validation split, preferring the simpler candidate on ties; the test split is then scored once
and compared with the current app (McNemar exact test, paired bootstrap intervals).

Caveat recorded in the output: each run's epoch, and a family's best seed, were already chosen on
the same validation split, so validation scores are slightly optimistic for every candidate.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

from training import post_hoc
from training.protocol import load_prepared


def member_probs(logits_by_split, y_val):
    t = post_hoc.fit_temperature(logits_by_split["validation"], y_val)
    return {k: post_hoc.softmax(v / t) for k, v in logits_by_split.items()}, t


def load_runs(results_dir):
    runs = {}
    for meta_path in sorted(Path(results_dir).rglob("runs/*/training.json")):
        meta = json.loads(meta_path.read_text())
        family = meta.get("spec", {}).get("family")
        if meta.get("status") != "complete" or family in (None, "stage1", "smoke"):
            continue
        with np.load(meta_path.parent / "logits.npz") as z:
            runs[meta_path.parent.name] = {"family": family, "val_f1": meta["selected_validation"]["macro_f1"],
                                           "logits": {k: z[k] for k in z.files}}
    return runs


def log_mean(prob_list):
    return {s: np.log(np.mean([p[s] for p in prob_list], axis=0) + 1e-12) for s in ["validation", "test"]}


def candidates(runs, y_val, app_logits=None):
    """name -> {'members': [...], 'z': {split: scores}}, simplest first.

    A single model keeps its raw logits (post_hoc fits its temperature); an ensemble is the log of
    its members' averaged temperature-scaled probabilities.
    """
    raw = {name: {k: r["logits"][k] for k in ["validation", "test"]} for name, r in runs.items()}
    probs = {name: member_probs(raw[name], y_val)[0] for name in runs}
    out = {}
    if app_logits is not None:
        out["app"] = {"members": ["published mBERT"], "z": app_logits}
    families = sorted({r["family"] for r in runs.values()})
    means = {}
    for f in families:
        names = sorted(n for n, r in runs.items() if r["family"] == f)
        best = max(names, key=lambda n: (runs[n]["val_f1"], n))
        out[f"{f}-best"] = {"members": [best], "z": raw[best]}
        if len(names) > 1:
            means[f] = {s: np.mean([probs[n][s] for n in names], axis=0) for s in ["validation", "test"]}
            out[f"{f}-mean"] = {"members": names, "z": log_mean([probs[n] for n in names])}
    if len(means) > 1:
        out["all-mean"] = {"members": sorted(n for f in means for n, r in runs.items() if r["family"] == f),
                           "z": log_mean(list(means.values()))}
    return out


def select(cands, y_val, t_val, folds=5):
    """CV macro-F1 inside validation for each candidate with and without the TF-IDF blend."""
    table = {}
    for i, (name, c) in enumerate(cands.items()):
        z = c["z"]["validation"]
        table[name] = {"cv": post_hoc.cross_validated("mbert", z, t_val, y_val, folds), "order": 2 * i}
        table[name + "+tfidf"] = {"cv": post_hoc.cross_validated("blend", z, t_val, y_val, folds), "order": 2 * i + 1}
    chosen = max(table, key=lambda k: (round(table[k]["cv"], 6), -table[k]["order"]))
    return chosen, {k: v["cv"] for k, v in table.items()}


def predict(cands, name, y_val, t_val, t_test):
    base = name.removesuffix("+tfidf")
    option = "blend" if name.endswith("+tfidf") else "mbert"
    zv, zt = cands[base]["z"]["validation"], cands[base]["z"]["test"]
    params = post_hoc.fit(option, zv, t_val, y_val)
    return post_hoc.predict(option, params, zt, t_test), params


def run(results_dir, data_dir, baseline_dir, app_logits_dir, output_path):
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("A selection result already exists; keep it and choose a new path.")
    frame, manifest = load_prepared(data_dir)
    val, test = frame[frame.split == "validation"], frame[frame.split == "test"]
    y_val, y_test = val.label_id.to_numpy(), test.label_id.to_numpy()
    baseline = joblib.load(Path(baseline_dir) / "baseline.joblib")
    t_val, t_test = baseline.predict_proba(val.text), baseline.predict_proba(test.text)
    app = {s: np.load(Path(app_logits_dir) / f"fp16_{s}.npy") for s in ["validation", "test"]}
    runs = load_runs(results_dir)
    cands = candidates(runs, y_val, app)
    # 1. The choice, from validation alone.
    chosen, cv = select(cands, y_val, t_val)
    record = {"protocol": manifest["protocol"], "split_sha256": manifest["split_sha256"],
              "runs": {n: {"family": r["family"], "validation_macro_f1": r["val_f1"]} for n, r in runs.items()},
              "validation_cv_macro_f1": cv, "selected": chosen, "selected_members": cands[chosen.removesuffix("+tfidf")]["members"]}
    # 2. The test split, once. The reference is the current app: published mBERT + TF-IDF blend.
    ref_pred, _ = predict(cands, "app+tfidf", y_val, t_val, t_test)
    pred, params = predict(cands, chosen, y_val, t_val, t_test)
    record["test"] = {"selected": {"accuracy": float(accuracy_score(y_test, pred)), "macro_f1": post_hoc.macro_f1(y_test, pred)},
                      "current_app": {"accuracy": float(accuracy_score(y_test, ref_pred)), "macro_f1": post_hoc.macro_f1(y_test, ref_pred)},
                      "comparison": {**post_hoc.mcnemar_exact(y_test, ref_pred, pred), **post_hoc.bootstrap(y_test, ref_pred, pred)}}
    record["selected_params"] = params
    record["note"] = ("Retrospective internal evaluation on the existing corpus. Run epochs and best seeds were chosen on "
                      "the same validation split, so validation scores are slightly optimistic for every candidate.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--app-logits", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    r = run(a.results, a.data, a.baseline, a.app_logits, a.out)
    print(json.dumps({k: r[k] for k in ["selected", "selected_members", "test"]}, indent=1))
