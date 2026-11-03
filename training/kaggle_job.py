"""Longer training runs on Kaggle's free GPU, under the same frozen protocol.

Experiments (fixed before any run):
  large   XLM-RoBERTa-large fine-tuned on the gold train split, five seeds.
  silver  First stage on ~160k ChatGPT-labelled Malaysian tweets (training/extra_data.py), then
          fine-tuned on the gold train split: XLM-RoBERTa-large (three seeds) and mBERT (two seeds;
          the browser model's own architecture, so a winner could ship without distillation).
  smoke   A two-minute check of the whole pipeline with tiny settings; its numbers mean nothing.

Each run keeps the epoch with the best validation macro-F1. For every run the job saves logits on
validation, test, train and the extra tweets, so that ensembles and distillation can be chosen
later from validation alone, test scored once at the end (training/kaggle_select.py). Only the
best-validation model of each family is kept, saved in 16-bit to fit Kaggle's output limit.
The private corpus and the extra tweets stay in /tmp and are never written to the outputs.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from training.protocol import load_prepared, prepare_csv

SOURCE_SHA256 = "3ff66de7b91cb1b2fd06ac5dd136f495859114808d19b1688c66a21ba1d74d77"
SPLIT_SHA256 = "692d03d208495810fcd1c7fa85906b2cfc1f4fc18628b6a39998c898770b9929"
XLMR = {"base": "FacebookAI/xlm-roberta-large", "revision": "c23d21b0620b635a76227c604d44e43a9f0ee389"}
MBERT = {"base": "google-bert/bert-base-multilingual-cased", "revision": None}
BATCH = {"batch_size": 16, "accumulation": 2}  # effective batch 32
BUDGET_HOURS = 11.0  # Kaggle stops a session at 12 hours

EXPERIMENTS = {
    "large": [
        {"name": f"xlmr-large-s{s}", "family": "xlmr-large", **XLMR, "learning_rate": 1e-5, "epochs": 5, "seed": s, **BATCH}
        for s in range(1, 6)
    ],
    "silver": [
        {"name": "xlmr-large-silver-stage1", "family": "stage1", **XLMR, "learning_rate": 1e-5, "epochs": 1, "seed": 1,
         "data": "silver", **BATCH},
        *[{"name": f"xlmr-large-silver-s{s}", "family": "xlmr-large-silver", "init": "xlmr-large-silver-stage1",
           "learning_rate": 1e-5, "epochs": 4, "seed": s, **BATCH} for s in range(1, 4)],
        {"name": "mbert-silver-stage1", "family": "stage1", **MBERT, "learning_rate": 2e-5, "epochs": 1, "seed": 1,
         "data": "silver", **BATCH},
        *[{"name": f"mbert-silver-s{s}", "family": "mbert-silver", "init": "mbert-silver-stage1",
           "learning_rate": 2e-5, "epochs": 4, "seed": s, **BATCH} for s in range(1, 3)],
    ],
    "smoke": [
        {"name": "smoke-stage1", "family": "stage1", **MBERT, "learning_rate": 2e-5, "epochs": 1, "seed": 1,
         "data": "silver", "batch_size": 16, "accumulation": 1, "limit": 300},
        {"name": "smoke-s1", "family": "smoke", "init": "smoke-stage1", "learning_rate": 2e-5, "epochs": 1, "seed": 1,
         "batch_size": 16, "accumulation": 1, "limit": 300},
    ],
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_corpus(input_dir):
    """The uploaded CSV whose checksum matches the frozen corpus."""
    for path in sorted(Path(input_dir).rglob("*.csv")):
        if sha256_file(path) == SOURCE_SHA256:
            return path
    raise FileNotFoundError("No CSV in the Kaggle dataset matches the frozen corpus checksum.")


def run(experiment, input_dir, out_dir, work_dir="/tmp/rojak", train_fn=None, logits_fn=None, silver_fn=None):
    from training import extra_data
    from training.run import train_bert
    train_fn = train_fn or train_bert
    plan = EXPERIMENTS[experiment]
    out, work = Path(out_dir), Path(work_dir)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    data_dir = work / "data"
    if not (data_dir / "manifest.json").exists():
        prepare_csv(find_corpus(input_dir), data_dir)
    gold, manifest = load_prepared(data_dir)
    if manifest["split_sha256"] != SPLIT_SHA256:
        raise ValueError("The prepared split does not match the frozen evaluation split.")
    silver, audit = (silver_fn or (lambda g: extra_data.build(g, extra_data.download(str(work / "hf")))))(gold)
    (out / "extra_data_audit.json").write_text(json.dumps(audit, indent=2))
    frames = {"validation": gold[gold.split == "validation"], "test": gold[gold.split == "test"],
              "train": gold[gold.split == "train"], "transfer": silver}
    summary = {"experiment": experiment, "protocol": manifest["protocol"], "split_sha256": SPLIT_SHA256,
               "transfer_text_sha256": audit["retained_text_sha256"], "runs": {}, "skipped": []}
    best = {}
    last_hours = 0.0
    for i, spec in enumerate(plan):
        elapsed = (time.time() - started) / 3600
        if elapsed + last_hours > BUDGET_HOURS:
            summary["skipped"] = [s["name"] for s in plan[i:]]
            break
        t0 = time.time()
        run_dir = work / "runs" / spec["name"]
        train_rows = silver if spec.get("data") == "silver" else None
        if spec.get("limit") and train_rows is not None:
            train_rows = train_rows.head(spec["limit"])
        kwargs = dict(epochs=spec["epochs"], batch_size=spec["batch_size"], accumulation=spec["accumulation"],
                      learning_rate=spec["learning_rate"], seed=spec["seed"], eval_batch_size=64)
        if spec.get("init"):
            kwargs["init_from"] = work / "runs" / spec["init"] / "model"
        else:
            kwargs.update(base=spec["base"], revision=spec["revision"])
        if train_rows is not None:
            kwargs.update(train_frame=train_rows, train_name="extra tweets (silver)")
        elif spec.get("limit"):
            kwargs.update(train_frame=frames["train"].head(spec["limit"]), train_name="train split (smoke subset)")
        train_fn(data_dir, run_dir, **kwargs)
        meta = json.loads((run_dir / "training.json").read_text())
        limit = spec.get("limit")
        logits = (logits_fn or saved_logits)(run_dir / "model",
                                              {k: (v.head(limit) if limit and k == "transfer" else v) for k, v in frames.items()})
        dest = out / "runs" / spec["name"]
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "training.json").write_text(json.dumps({**meta, "spec": spec}, indent=2))
        np.savez_compressed(dest / "logits.npz", **logits)
        val = meta["selected_validation"]["macro_f1"]
        summary["runs"][spec["name"]] = {"family": spec["family"], "validation": meta["selected_validation"],
                                         "selected_epoch": meta["selected_epoch"], "hours": round((time.time() - t0) / 3600, 3)}
        family = spec["family"]
        if family != "stage1":
            if family not in best or val > best[family][1]:
                if family in best:
                    shutil.rmtree(work / "runs" / best[family][0] / "model", ignore_errors=True)
                best[family] = (spec["name"], val)
            else:
                shutil.rmtree(run_dir / "model", ignore_errors=True)
        last_hours = (time.time() - t0) / 3600
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for family, (name, _) in best.items():
        save_half(work / "runs" / name / "model", out / "models" / name)
    summary["kept_models"] = {f: n for f, (n, _) in best.items()}
    summary["hours"] = round((time.time() - started) / 3600, 3)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def saved_logits(model_dir, frames):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from training.run import predict_logits
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(device)
    result = {k: predict_logits(model, tokenizer, f, torch, device, batch_size=64) for k, f in frames.items()}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def save_half(src, dest):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(src, local_files_only=True)
    model.half().save_pretrained(dest, safe_serialization=True)
    AutoTokenizer.from_pretrained(src, local_files_only=True).save_pretrained(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=list(EXPERIMENTS), required=True)
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--out", default="/kaggle/working/out")
    a = parser.parse_args()
    print(json.dumps(run(a.experiment, a.input, a.out), indent=1), flush=True)


if __name__ == "__main__":
    main()
