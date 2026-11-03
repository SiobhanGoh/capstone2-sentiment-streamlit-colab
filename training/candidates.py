"""Compare stronger base models under the corrected protocol.

Every candidate is fine-tuned on the same frozen train split and its checkpoint is chosen on
validation macro-F1, exactly like the original mBERT run. The winner is chosen by validation
macro-F1 alone, before any test score is computed. Test scores for every candidate are then
recorded once, for transparency, and must not be used to change the choice afterwards.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from training.protocol import LABELS, clean_text, load_prepared, normalize_label
from training.run import predict_bert, scores, train_bert, write_json

# Fixed before any training. Order does not matter; selection is by validation only.
CANDIDATES = {
    "mesolitica-debertav2": {"base": "mesolitica/malaysian-debertav2-base", "revision": "14a359fdf0068cbb2a052bf084449bcb9e90872b",
                             "learning_rate": 3e-5,
                             "why": "Pretrained on 17B tokens of Malaysian text including local Malay and Manglish."},
    "mesolitica-roberta": {"base": "mesolitica/roberta-base-bahasa-cased", "revision": "7dac0262dde0ad7bdbd7acb7b34e3e2cda623b8c",
                           "learning_rate": 2e-5,
                           "why": "Pretrained on local Twitter, Instagram, news, IIUM confessions and Malay Wikipedia."},
    "xlm-t": {"base": "cardiffnlp/twitter-xlm-roberta-base", "revision": "4c365f1490cb329b52150ad72f922ea467b5f4e6",
              "learning_rate": 2e-5,
              "why": "XLM-RoBERTa further pretrained on about 198M multilingual tweets."},
}
REFERENCE = Path(__file__).resolve().parents[1] / "verification/corrected_comparison.json"


def select_by_validation(trainings):
    """Pick the candidate with the best validation macro-F1; ties go to the earlier name."""
    complete = {k: v for k, v in trainings.items() if v.get("status") == "complete"}
    if not complete:
        raise ValueError("No candidate finished training.")
    return max(sorted(complete), key=lambda k: complete[k]["selected_validation"]["macro_f1"])


def load_check_set(path):
    """An optional extra check set: Tweet/Sentiment CSV, or the web app's corrections CSV."""
    raw = pd.read_csv(path)
    if {"Text", "Your label"}.issubset(raw.columns):
        raw = raw.rename(columns={"Text": "Tweet", "Your label": "Sentiment"})
    if not {"Tweet", "Sentiment"}.issubset(raw.columns):
        raise ValueError("The check set needs Tweet and Sentiment columns, or the app's corrections export.")
    frame = pd.DataFrame({"text": raw.Tweet.map(clean_text), "label": raw.Sentiment.map(normalize_label)})
    frame = frame[(frame.text != "") & frame.label.notna()].drop_duplicates("text")
    frame["label_id"] = frame.label.map({l: i for i, l in enumerate(LABELS)})
    return frame.reset_index(drop=True)


def train_all(data_dir, runs_dir, names=None, epochs=4):
    runs = Path(runs_dir)
    for name in names or CANDIDATES:
        spec = CANDIDATES[name]
        out = runs / name
        if (out / "training.json").exists():
            if json.loads((out / "training.json").read_text()).get("status") == "complete":
                print(json.dumps({"skipping_finished": name}), flush=True)
                continue
            # An interrupted run (for example a Colab disconnect) is kept aside and restarted from scratch.
            n = 1
            while (runs / f"{name}.interrupted-{n}").exists():
                n += 1
            out.rename(runs / f"{name}.interrupted-{n}")
        print(json.dumps({"training": name, "base": spec["base"]}), flush=True)
        train_bert(data_dir, out, epochs=epochs, base=spec["base"], revision=spec["revision"],
                   learning_rate=spec["learning_rate"])


def evaluate_all(data_dir, runs_dir, output_path, check_csv=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("A candidate comparison already exists; keep it and choose a new path for a new experiment.")
    frame, manifest = load_prepared(data_dir)
    test = frame[frame.split == "test"]
    check = load_check_set(check_csv) if check_csv else None
    trainings = {}
    for name in CANDIDATES:
        meta = Path(runs_dir) / name / "training.json"
        if meta.exists():
            trainings[name] = json.loads(meta.read_text())
            if trainings[name]["split_sha256"] != manifest["split_sha256"]:
                raise ValueError(f"{name} was trained on a different split.")
    # The choice is fixed here, before any test prediction is made.
    winner = select_by_validation(trainings)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for name, meta in trainings.items():
        if meta.get("status") != "complete":
            results[name] = {"status": "incomplete"}
            continue
        model_dir = Path(runs_dir) / name / "model"
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(device)
        entry = {
            "base": meta["base_model"], "base_revision": meta["base_revision"], "learning_rate": meta["learning_rate"],
            "selected_epoch": meta["selected_epoch"], "validation": meta["selected_validation"],
            "test": scores(test.label_id, predict_bert(model, tokenizer, test, torch, device)),
            "model_sha256": hashlib.sha256((model_dir / "model.safetensors").read_bytes()).hexdigest(),
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        if check is not None:
            entry["check_set"] = scores(check.label_id, predict_bert(model, tokenizer, check, torch, device))
        results[name] = entry
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    reference = json.loads(REFERENCE.read_text())
    report = {
        "protocol": manifest["protocol"], "split_sha256": manifest["split_sha256"], "source_sha256": manifest["source_sha256"],
        "test_rows": len(test), "check_rows": None if check is None else len(check),
        "selection_rule": "Highest validation macro-F1, fixed before test scores were computed.",
        "selected": winner,
        "reference_mbert": {
            "test": {k: reference["mbert"][k] for k in ["accuracy", "macro_f1"]},
            "validation_macro_f1": max(h["validation"]["macro_f1"] for h in reference["mbert_selection"]["history"]),
        },
        "candidates": results,
        "note": "Retrospective internal comparison on the same corpus as the original work; not external validation. "
                "One seed per candidate, so small differences may be noise.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(json.dumps({"selected": winner, **{k: {m: v["test"][m] for m in ["accuracy", "macro_f1"]}
                                            for k, v in results.items() if "test" in v}}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--runs", required=True)
    t.add_argument("--only", nargs="*", choices=list(CANDIDATES))
    e = sub.add_parser("evaluate")
    e.add_argument("--data", required=True)
    e.add_argument("--runs", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--check-csv")
    args = parser.parse_args()
    if args.command == "train":
        train_all(args.data, args.runs, args.only)
    else:
        evaluate_all(args.data, args.runs, args.out, args.check_csv)


if __name__ == "__main__":
    main()
