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


def token_batches(frame, tokenizer, batch_size, torch, device, shuffle=False, with_index=False):
    indices = np.random.permutation(len(frame)) if shuffle else np.arange(len(frame))
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start+batch_size]
        batch = frame.iloc[idx]
        inputs = tokenizer(batch.text.tolist(), padding=True, truncation=True, max_length=128,
                           return_tensors="pt")
        labels = torch.tensor(batch.label_id.to_numpy(), dtype=torch.long, device=device)
        if with_index:
            yield {k: v.to(device) for k, v in inputs.items()}, labels, idx
        else:
            yield {k: v.to(device) for k, v in inputs.items()}, labels


def soften(probs, temperature):
    """Teacher probabilities at a distillation temperature: p ** (1/T), renormalised."""
    z = np.log(np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1)) / temperature
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def distillation_loss(logits, labels, teacher, alpha, temperature, torch):
    """Per batch: (1 - alpha) * T^2 * KL(teacher_T || student_T) + alpha * CE on rows with a gold label.

    labels of -1 mark rows with no gold label (teacher only). teacher is already softened to T.
    """
    F = torch.nn.functional
    logits = logits.float()
    log_student = F.log_softmax(logits / temperature, -1)
    kd = (teacher * (teacher.clamp_min(1e-12).log() - log_student)).sum(-1) * temperature ** 2
    has_label = (labels >= 0).float()
    ce = F.cross_entropy(logits, labels.clamp_min(0), reduction="none") * has_label
    return ((1 - alpha) * kd + alpha * ce).mean()


def predict_logits(model, tokenizer, frame, torch, device, batch_size=8):
    """Raw class logits as a float32 array, in the frame's row order."""
    model.eval()
    out = []
    with torch.inference_mode():
        for inputs, _ in token_batches(frame, tokenizer, batch_size, torch, device):
            with torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else nullcontext():
                logits = model(**inputs).logits
            out.append(logits.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 3), dtype=np.float32)


def predict_bert(model, tokenizer, frame, torch, device, batch_size=8):
    return predict_logits(model, tokenizer, frame, torch, device, batch_size).argmax(1).tolist()


DEFAULT_BASE = "google-bert/bert-base-multilingual-cased"
HEAD_PREFIXES = ("classifier.", "pooler.", "score.", "pre_classifier.")


def check_pretrained_weights(loading):
    """Refuse to fine-tune if the encoder itself did not load, which would silently start from random weights."""
    # Only the new classification head (and a pooler some checkpoints omit) may start untrained.
    missing = [k for k in loading.get("missing_keys", []) if not (k.startswith(HEAD_PREFIXES) or ".pooler." in k)]
    if missing:
        raise ValueError(f"{len(missing)} encoder weights did not load (for example {missing[0]}); "
                         "this checkpoint does not match the sequence-classification architecture.")


def train_bert(data_dir, output_dir, *, epochs=4, batch_size=4, accumulation=4, revision=None,
               base=DEFAULT_BASE, learning_rate=2e-5, seed=42, train_frame=None, init_from=None,
               train_name="train split", eval_batch_size=8, subfolder=None, soft_targets=None,
               kd_alpha=0.5, kd_temperature=2.0):
    """Fine-tune and keep the epoch with the best validation macro-F1.

    train_frame replaces the gold train split (for example extra tweets for a first stage); the
    validation split is always the frozen one. init_from continues from a local checkpoint.
    soft_targets (teacher probabilities aligned with train_frame) switches to distillation; rows
    with label_id -1 then learn from the teacher alone.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    frame, manifest = load_prepared(data_dir)
    train, val = frame[frame.split == "train"], frame[frame.split == "validation"]
    if train_frame is not None:
        train = train_frame[["text", "label_id"]].reset_index(drop=True)
    if init_from is not None:
        base, revision = str(init_from), None
    output = Path(output_dir)
    if min(epochs, batch_size, accumulation) < 1:
        raise ValueError("Epochs, batch size and accumulation must be positive.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Training output already contains artifacts; choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local = init_from is not None
    sub = {} if subfolder is None else {"subfolder": subfolder}
    tokenizer = AutoTokenizer.from_pretrained(base, revision=revision, local_files_only=local, **sub)
    # A new 3-class head is initialised; some community checkpoints publish only pytorch_model.bin.
    model, loading = AutoModelForSequenceClassification.from_pretrained(
        base, revision=revision, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=LABEL_TO_ID,
        output_loading_info=True, local_files_only=local, **sub,
    )
    check_pretrained_weights(loading)
    model.to(device)
    # Non-reentrant checkpointing: DeBERTa-v2 reuses one relative-position tensor across layers,
    # and the reentrant mode frees it after the first layer's backward pass ("backward through the
    # graph a second time"). The non-reentrant mode works for every architecture here.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    n_batches = math.ceil(len(train)/batch_size)
    steps = math.ceil(n_batches/accumulation) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(.1*steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    if soft_targets is not None:
        if len(soft_targets) != len(train):
            raise ValueError("Soft targets must align with the training rows.")
        teacher_all = soften(soft_targets, kd_temperature)
    best, history = -1, []
    config = {"base_model": base, "base_revision": None if local else model.config._commit_hash,
              "init_from": None if init_from is None else str(init_from), "train_data": train_name,
              "train_rows": len(train),
              "split_sha256": manifest["split_sha256"], "epochs": epochs, "batch_size": batch_size,
              "accumulation": accumulation, "learning_rate": learning_rate, "max_length": 128,
              "seed": seed, "device": device, "selection": "validation macro-F1 only", "subfolder": subfolder,
              "distillation": None if soft_targets is None else {"alpha": kd_alpha, "temperature": kd_temperature,
                                                                 "rows_without_gold_label": int((train.label_id < 0).sum())}}
    write_json(output/"training.json", {**config, "status": "running", "history": history})
    print(json.dumps({"training_device": device, "train_rows": len(train), "epochs": epochs}), flush=True)
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        start = time.time()
        total_loss = 0.0
        for step, (inputs, labels, idx) in enumerate(token_batches(train, tokenizer, batch_size, torch, device,
                                                                     shuffle=True, with_index=True), 1):
            # Correct normalization for a final incomplete accumulation group.
            group_start = ((step-1)//accumulation)*accumulation
            group_size = min(accumulation, n_batches-group_start)
            with torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else nullcontext():
                if soft_targets is None:
                    loss = model(**inputs, labels=labels).loss
                else:
                    teacher = torch.as_tensor(teacher_all[idx], device=device)
                    loss = distillation_loss(model(**inputs).logits, labels, teacher, kd_alpha, kd_temperature, torch)
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
        metrics = scores(val.label_id, predict_bert(model, tokenizer, val, torch, device, eval_batch_size))
        record = {"epoch": epoch, "train_loss": total_loss/n_batches, "validation": metrics,
                  "seconds": round(time.time()-start, 2)}
        history.append(record)
        if metrics['macro_f1'] > best:
            best = metrics['macro_f1']
            selected_epoch = epoch
            model.save_pretrained(output/"model", safe_serialization=True)
            best_val = metrics
            tokenizer.save_pretrained(output/"model")
        write_json(output/"training.json", {**config, "status": "running", "selected_epoch": selected_epoch, "history": history})
        print(json.dumps(record), flush=True)
    write_json(output/"training.json", {**config, "status": "complete", "selected_epoch": selected_epoch,
                                        "selected_validation": {"accuracy": best_val["accuracy"], "macro_f1": best_val["macro_f1"]},
                                        "history": history})


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
