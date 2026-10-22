"""Train on train, select on validation, then evaluate both locked models together."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

from training.protocol import LABELS, LABEL_TO_ID, clean_text, load_prepared, prepare_csv


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def scores(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
            "classification_report": classification_report(y, pred, labels=[0, 1, 2], target_names=LABELS,
                                                          output_dict=True, zero_division=0)}


def train_baseline(data_dir, output_dir):
    frame, manifest = load_prepared(data_dir)
    train, val = frame[frame.split == "train"], frame[frame.split == "validation"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    best_score, best_model, selected_c = -1, None, None
    history = []
    # The complete grid is fixed before validation; no test access during selection.
    for c in [.1, 1.0, 10.0]:
        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000,
                                             sublinear_tf=True),
                              LogisticRegression(C=c, max_iter=2000, random_state=42))
        model.fit(train.text, train.label_id)
        metrics = scores(val.label_id, model.predict(val.text))
        history.append({"C": c, "validation": metrics})
        if metrics["macro_f1"] > best_score:
            best_score, best_model, selected_c = metrics["macro_f1"], model, c
    joblib.dump(best_model, output/"baseline.joblib")
    write_json(output/"training.json", {"model": "TF-IDF logistic regression", "selected_C": selected_c,
                                        "split_sha256": manifest["split_sha256"], "history": history})
    print(json.dumps({"baseline_selected_C": selected_c, "validation_macro_f1": best_score}), flush=True)


def token_batches(frame, tokenizer, batch_size, torch, device, shuffle=False):
    indices = np.random.permutation(len(frame)) if shuffle else np.arange(len(frame))
    for start in range(0, len(indices), batch_size):
        batch = frame.iloc[indices[start:start+batch_size]]
        inputs = tokenizer(batch.text.tolist(), padding=True, truncation=True, max_length=128,
                           return_tensors="pt")
        yield {k: v.to(device) for k, v in inputs.items()}, torch.tensor(batch.label_id.to_numpy(), dtype=torch.long, device=device)


def predict_bert(model, tokenizer, frame, torch, device, batch_size=8):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for inputs, _ in token_batches(frame, tokenizer, batch_size, torch, device):
            with torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else nullcontext():
                logits = model(**inputs).logits
            predictions.extend(logits.argmax(-1).cpu().tolist())
    return predictions


def train_bert(data_dir, output_dir, *, epochs=4, batch_size=4, accumulation=4, revision=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    frame, manifest = load_prepared(data_dir)
    train, val = frame[frame.split == "train"], frame[frame.split == "validation"]
    output = Path(output_dir)
    if min(epochs, batch_size, accumulation) < 1:
        raise ValueError("Epochs, batch size and accumulation must be positive.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Training output already contains artifacts; choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = "google-bert/bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, revision=revision, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=LABEL_TO_ID,
        use_safetensors=True,
    ).to(device)
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    n_batches = math.ceil(len(train)/batch_size)
    steps = math.ceil(n_batches/accumulation) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(.1*steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best, history = -1, []
    config = {"base_model": base, "base_revision": model.config._commit_hash,
              "split_sha256": manifest["split_sha256"], "epochs": epochs, "batch_size": batch_size,
              "accumulation": accumulation, "learning_rate": 2e-5, "max_length": 128,
              "seed": 42, "device": device, "selection": "validation macro-F1 only"}
    write_json(output/"training.json", {**config, "status": "running", "history": history})
    print(json.dumps({"training_device": device, "train_rows": len(train), "epochs": epochs}), flush=True)
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        start = time.time()
        total_loss = 0.0
        for step, (inputs, labels) in enumerate(token_batches(train, tokenizer, batch_size, torch, device, shuffle=True), 1):
            # Correct normalization for a final incomplete accumulation group.
            group_start = ((step-1)//accumulation)*accumulation
            group_size = min(accumulation, n_batches-group_start)
            with torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else nullcontext():
                loss = model(**inputs, labels=labels).loss
            total_loss += float(loss.detach())
            scaler.scale(loss/group_size).backward()
            if step % accumulation == 0 or step == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before: scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0:
                print(json.dumps({"epoch": epoch, "batch": step, "batches": n_batches,
                                  "elapsed_seconds": round(time.time()-start)}), flush=True)
        metrics = scores(val.label_id, predict_bert(model, tokenizer, val, torch, device))
        record = {"epoch": epoch, "train_loss": total_loss/n_batches, "validation": metrics,
                  "seconds": round(time.time()-start, 2)}
        history.append(record)
        if metrics['macro_f1'] > best:
            best = metrics['macro_f1']
            selected_epoch = epoch
            model.save_pretrained(output/"model", safe_serialization=True)
            tokenizer.save_pretrained(output/"model")
        write_json(output/"training.json", {**config, "status": "running", "selected_epoch": selected_epoch, "history": history})
        print(json.dumps(record), flush=True)
    write_json(output/"training.json", {**config, "status": "complete", "selected_epoch": selected_epoch, "history": history})


def evaluate(data_dir, baseline_dir, bert_dir, output_path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    output = Path(output_path)
    if output.exists(): raise FileExistsError("Final evaluation already exists; do not silently overwrite a viewed test result.")
    frame, manifest = load_prepared(data_dir)
    test = frame[frame.split == "test"]
    baseline_meta = json.loads((Path(baseline_dir)/"training.json").read_text())
    bert_meta = json.loads((Path(bert_dir)/"training.json").read_text())
    if bert_meta.get("status") != "complete": raise ValueError("Transformer training is incomplete.")
    for meta in [baseline_meta, bert_meta]:
        if meta["split_sha256"] != manifest["split_sha256"]:
            raise ValueError("Models were trained using different data partitions.")
    # Load only artifacts produced by this local training pipeline.
    baseline = joblib.load(Path(baseline_dir)/"baseline.joblib")
    model_dir = Path(bert_dir)/"model"
    # BERT uses BertPreTokenizer; the Mistral-specific regex patch is inapplicable.
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, fix_mistral_regex=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True, use_safetensors=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    baseline_predictions = baseline.predict(test.text).tolist()
    bert_predictions = predict_bert(model, tokenizer, test, torch, device)
    result = {"protocol": manifest, "test_rows": len(test),
              "baseline": scores(test.label_id, baseline_predictions),
              "mbert": scores(test.label_id, bert_predictions),
              "baseline_selection": baseline_meta, "mbert_selection": bert_meta,
              "model_sha256": hashlib.sha256((model_dir/"model.safetensors").read_bytes()).hexdigest(),
              "limitation": "Corrected retrospective internal comparison on an existing corpus; not new external validation. "
                            "Earlier development used this corpus. Near-duplicates, temporal/source shift and calibration are not established."}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps({"baseline": result["baseline"], "mbert": result["mbert"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--csv", required=True)
    prepare.add_argument("--out", required=True)
    for name in ["baseline", "mbert"]:
        sub = commands.add_parser(name)
        sub.add_argument("--data", required=True)
        sub.add_argument("--out", required=True)
        if name == "mbert":
            sub.add_argument("--revision")
            sub.add_argument("--epochs", type=int, default=4)
            sub.add_argument("--batch-size", type=int, default=4)
            sub.add_argument("--accumulation", type=int, default=4)
    final = commands.add_parser("evaluate")
    final.add_argument("--data", required=True)
    final.add_argument("--baseline", required=True)
    final.add_argument("--mbert", required=True)
    final.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "prepare": print(json.dumps(prepare_csv(args.csv, args.out)), flush=True)
    elif args.command == "baseline": train_baseline(args.data, args.out)
    elif args.command == "mbert": train_bert(args.data, args.out, epochs=args.epochs, batch_size=args.batch_size,
                                              accumulation=args.accumulation, revision=args.revision)
    else: evaluate(args.data, args.baseline, args.mbert, args.out)


if __name__ == "__main__": main()
