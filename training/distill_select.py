"""Choose one distilled student for the browser on validation only, then score the test split once.

Candidates are deployable on their own: each single student, alone or blended with the TF-IDF
baseline (temperature and blend weight fitted on validation exactly as in training/post_hoc.py).
Ensembles of students are left out because the web app can only download one model. The choice is
the highest 5-fold cross-validated macro-F1 inside validation; the test split is then scored once
and compared with the live app (published mBERT + TF-IDF), the original mBERT, and the teacher.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

from training import post_hoc
from training.protocol import load_prepared


def load_students(results_dir):
    out = {}
    for meta in sorted(Path(results_dir).rglob("runs/student-*/training.json")):
        m = json.loads(meta.read_text())
        if m.get("status") == "complete":
            with np.load(meta.parent / "logits.npz") as z:
                out[meta.parent.name] = {"validation": z["validation"], "test": z["test"],
                                         "validation_macro_f1": m["selected_validation"]["macro_f1"]}
    return out


def run(results_dir, data_dir, baseline_dir, app_logits_dir, teacher_record, output_path):
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("A student selection already exists; keep it and choose a new path.")
    frame, manifest = load_prepared(data_dir)
    val, test = frame[frame.split == "validation"], frame[frame.split == "test"]
    yv, yt = val.label_id.to_numpy(), test.label_id.to_numpy()
    base = joblib.load(Path(baseline_dir) / "baseline.joblib")
    tv, tt = base.predict_proba(val.text), base.predict_proba(test.text)
    students = load_students(results_dir)
    # 1. The choice, from validation alone. Simpler (no blend) first on ties.
    cv = {}
    for name in sorted(students):
        cv[name] = post_hoc.cross_validated("mbert", students[name]["validation"], tv, yv)
        cv[name + "+tfidf"] = post_hoc.cross_validated("blend", students[name]["validation"], tv, yv)
    order = list(cv)
    chosen = max(order, key=lambda k: (round(cv[k], 6), -order.index(k)))
    name, option = chosen.removesuffix("+tfidf"), "blend" if chosen.endswith("+tfidf") else "mbert"
    params = post_hoc.fit(option, students[name]["validation"], tv, yv)
    # 2. The test split, once.
    pred = post_hoc.predict(option, params, students[name]["test"], tt)
    app = {s: np.load(Path(app_logits_dir) / f"fp16_{s}.npy") for s in ["validation", "test"]}
    app_pred = post_hoc.predict("blend", post_hoc.fit("blend", app["validation"], tv, yv), app["test"], tt)
    original = app["test"].argmax(1)

    def compare(ref):
        return {**post_hoc.mcnemar_exact(yt, ref, pred), **post_hoc.bootstrap(yt, ref, pred)}

    teacher = json.loads(Path(teacher_record).read_text())
    report = {
        "protocol": manifest["protocol"], "split_sha256": manifest["split_sha256"],
        "selection_rule": "Highest 5-fold CV macro-F1 inside validation among single students, alone or + TF-IDF.",
        "students_validation_macro_f1": {k: v["validation_macro_f1"] for k, v in students.items()},
        "validation_cv_macro_f1": cv, "selected": chosen, "selected_params": params,
        "test": {"selected": {"accuracy": float(accuracy_score(yt, pred)), "macro_f1": post_hoc.macro_f1(yt, pred),
                              "ece": post_hoc.ece(post_hoc.probs(option, params, students[name]["test"], tt), yt)},
                 "live_app": {"accuracy": float(accuracy_score(yt, app_pred)), "macro_f1": post_hoc.macro_f1(yt, app_pred)},
                 "original_mbert": {"accuracy": float(accuracy_score(yt, original)), "macro_f1": post_hoc.macro_f1(yt, original)},
                 "teacher": teacher["test"]["selected"],
                 "vs_live_app": compare(app_pred), "vs_original_mbert": compare(original)},
        "note": ("Retrospective internal evaluation. Validation has guided many choices in this project (epochs, seeds, "
                 "teacher, student), and the test split has been scored for earlier experiments too, so treat small "
                 "differences with caution."),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for a in ["results", "data", "baseline", "app-logits", "teacher-record", "out"]:
        p.add_argument(f"--{a}", required=True)
    a = p.parse_args()
    r = run(a.results, a.data, a.baseline, a.app_logits, a.teacher_record, a.out)
    print(json.dumps({k: r[k] for k in ["validation_cv_macro_f1", "selected", "test"]}, indent=1))
