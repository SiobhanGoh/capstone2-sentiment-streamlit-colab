"""Longer training runs on Kaggle's free GPU, under the same frozen protocol.

Experiments (fixed before any run):
  large   XLM-RoBERTa-large fine-tuned on the gold train split, five seeds.
  large2  The same model with a longer schedule (8 epochs, batch 16, learning rate 2e-5), three seeds.
  silver  First stage on ~160k ChatGPT-labelled Malaysian tweets (training/extra_data.py), then
          fine-tuned on the gold train split: XLM-RoBERTa-large (three seeds) and mBERT (two seeds;
          the browser model's own architecture, so a winner could ship without distillation).
  distill The browser model's architecture (mBERT) taught by the teacher chosen on validation
          (training/distill_teacher.json, written by training/kaggle_select.py): soft labels on the
          gold train split and the extra tweets, plus the gold labels. Students start from the
          published checkpoint or from plain mBERT. Reads the teacher's logits from the earlier
          Kaggle runs' outputs, attached as inputs.
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
import pandas as pd

from training.protocol import load_prepared, prepare_csv

SOURCE_SHA256 = "3ff66de7b91cb1b2fd06ac5dd136f495859114808d19b1688c66a21ba1d74d77"
SPLIT_SHA256 = "692d03d208495810fcd1c7fa85906b2cfc1f4fc18628b6a39998c898770b9929"
XLMR = {"base": "FacebookAI/xlm-roberta-large", "revision": "c23d21b0620b635a76227c604d44e43a9f0ee389"}
MBERT = {"base": "google-bert/bert-base-multilingual-cased", "revision": None}
APP = {"base": "SiobhanGoh/CP2_Emotion_Model_New", "revision": "9bc6645ce7a8e02761c59386ffd16969bed1bc55",
       "subfolder": "corrected-2026-10"}  # the model the web app runs today
BATCH = {"batch_size": 16, "accumulation": 2}  # effective batch 32
TEACHER_FILE = Path(__file__).with_name("distill_teacher.json")
FIXES_FILE = Path(__file__).with_name("train_label_fixes.json")
# Qwen2.5-7B-Instruct further trained by mesolitica on Malaysian data (Manglish, state dialects).
QWEN = {"base": "mesolitica/Malaysian-Qwen2.5-7B-Instruct", "revision": "83a0e145c726385502898ab7e016982eae1b684d"}
FOLDS = 5
# A 7B model predicts ~25 rows/s on a T4, so it labels a fixed random subset of the extra tweets.
TRANSFER_SUBSET = 50000


def transfer_subset(n_rows, n, seed=0):
    """Sorted row positions of a fixed random subset of the extra tweets (all of them if n >= n_rows)."""
    if n >= n_rows:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(seed).choice(n_rows, n, replace=False))
BUDGET_HOURS = 11.0  # Kaggle stops a session at 12 hours

EXPERIMENTS = {
    "large": [
        {"name": f"xlmr-large-s{s}", "family": "xlmr-large", **XLMR, "learning_rate": 1e-5, "epochs": 5, "seed": s, **BATCH}
        for s in range(1, 6)
    ],
    # Added after "large" stopped improving only because its schedule ended (validation macro-F1 still
    # rising at the last epoch; 715 updates against mBERT's ~1,150). Fixed before any further result.
    "large2": [
        {"name": f"xlmr-large2-s{s}", "family": "xlmr-large2", **XLMR, "learning_rate": 2e-5, "epochs": 8, "seed": s,
         "batch_size": 16, "accumulation": 1} for s in range(1, 4)
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
    "distill": [
        *[{"name": f"student-app-s{s}", "family": "student-app", **APP, "learning_rate": 2e-5, "epochs": 3, "seed": s,
           "distill": True, **BATCH} for s in range(1, 3)],
        *[{"name": f"student-mbert-s{s}", "family": "student-mbert", **MBERT, "learning_rate": 3e-5, "epochs": 3, "seed": s,
           "distill": True, **BATCH} for s in range(1, 3)],
    ],
    # A 7B chat model as a classifier (training/llm.py, QLoRA), two seeds. Fixed before any result.
    "llm": [
        {"name": f"qwen7b-s{s}", "family": "qwen7b", "kind": "llm", **QWEN, "learning_rate": 1e-4, "epochs": 3, "seed": s,
         "batch_size": 8, "accumulation": 2, "transfer_subset": TRANSFER_SUBSET} for s in (1, 2)
    ],
    "llm-smoke": [
        {"name": "qwen7b-smoke", "family": "smoke", "kind": "llm", **QWEN, "learning_rate": 1e-4, "epochs": 1, "seed": 1,
         "batch_size": 8, "accumulation": 1, "limit": 64},
    ],
    # Out-of-fold predictions on the train split, to find comments whose label looks wrong: each fold's
    # model never saw the comments it predicts. Same settings as large2.
    "oof": [
        {"name": f"oof-xlmr-f{k}", "family": "oof", **XLMR, "learning_rate": 2e-5, "epochs": 8, "seed": 1,
         "batch_size": 16, "accumulation": 1, "fold": k, "skip_transfer": True} for k in range(FOLDS)
    ],
    # Cleaner training labels (training/train_label_fixes.json), same settings as large2, three seeds each.
    # Fixed before any result: relabel only where a blind relabel and the out-of-fold model agree, or drop
    # every flagged training row. Validation and test labels are never changed.
    "clean": [
        {"name": f"xlmr-{fix}-s{s}", "family": f"xlmr-{fix}", **XLMR, "learning_rate": 2e-5, "epochs": 8, "seed": s,
         "batch_size": 16, "accumulation": 1, "label_fix": fix} for fix in ("relabel", "dropflag") for s in range(1, 4)
    ],
    # Not training: converts the chosen student to ONNX and scores the files (training/kaggle_export.py).
    "export": [],
    "abstain": [],  # CPU only: reads the export run's ONNX logits, see training/abstain.py
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


def fixed_train(train, fix, fixes):
    """The train split with cleaner labels: 'relabel' changes agreed rows, 'dropflag' removes flagged rows."""
    train = train[["text", "label_id"]].reset_index(drop=True)
    if fixes["train_rows"] != len(train):
        raise ValueError("The label fixes were made for a different train split.")
    if fix == "relabel":
        train = train.copy()
        for row, label in fixes["relabel"].items():
            train.loc[int(row), "label_id"] = int(label)
        return train
    if fix == "dropflag":
        return train.drop(index=fixes["flagged"]).reset_index(drop=True)
    raise ValueError(f"Unknown label fix {fix!r}.")


def fold_ids(labels, folds=FOLDS, seed=42):
    """Stratified fold number for each training row, reproducible anywhere from the labels alone."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(np.where(labels == k)[0]) for k in range(3)])
    out = np.empty(len(labels), dtype=int)
    out[order] = np.arange(len(labels)) % folds
    return out


def load_teacher(input_dir, frames, teacher=None):
    """Teacher probabilities on train and transfer rows: the members' temperature-scaled softmax, averaged."""
    teacher = teacher or json.loads(TEACHER_FILE.read_text())
    loaded = []
    for m in teacher["members"]:
        found = sorted(Path(input_dir).rglob(f"runs/{m['run']}/logits.npz"))
        if len(found) != 1:
            raise FileNotFoundError(f"Expected one logits file for teacher member {m['run']}, found {len(found)}.")
        with np.load(found[0]) as z:
            index = z["transfer_index"] if "transfer_index" in z.files else np.arange(len(frames["transfer"]))
            if len(z["train"]) != len(frames["train"]) or len(z["transfer"]) != len(index):
                raise ValueError(f"Teacher {m['run']} logits do not match the train or transfer rows.")
            loaded.append((m, z["train"], z["transfer"], index))
    # Extra tweets every member labelled (a 7B member labels a subset).
    common = loaded[0][3]
    for *_, index in loaded[1:]:
        common = np.intersect1d(common, index)
    out = {"train": [], "transfer": []}
    for m, train, transfer, index in loaded:
        pick = np.searchsorted(index, common)
        for k, v in [("train", train), ("transfer", transfer[pick])]:
            v = v.astype(np.float64) / m["temperature"]
            v = np.exp(v - v.max(1, keepdims=True))
            out[k].append(v / v.sum(1, keepdims=True))
    return {"train": np.mean(out["train"], axis=0), "transfer": np.mean(out["transfer"], axis=0), "transfer_index": common}


def run(experiment, input_dir, out_dir, work_dir="/tmp/rojak", train_fn=None, logits_fn=None, silver_fn=None,
        teacher=None):
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
    folds = fold_ids(frames["train"].label_id.to_numpy())
    fixes = None
    if any(spec.get("label_fix") for spec in plan):
        fixes = json.loads(FIXES_FILE.read_text())
        if fixes["split_sha256"] != SPLIT_SHA256:
            raise ValueError("The label fixes were made for a different split.")
    soft = None
    if any(spec.get("distill") for spec in plan):
        teacher = teacher or json.loads(TEACHER_FILE.read_text())
        if teacher["transfer_text_sha256"] != audit["retained_text_sha256"]:
            raise ValueError("The extra tweets differ from the ones the teacher saw.")
        probs = load_teacher(input_dir, frames, teacher)
        students = pd.concat([frames["train"][["text", "label_id"]],
                              frames["transfer"].iloc[probs["transfer_index"]][["text"]].assign(label_id=-1)], ignore_index=True)
        soft = np.concatenate([probs["train"], probs["transfer"]]).astype(np.float32)
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
            if spec.get("subfolder"):
                kwargs["subfolder"] = spec["subfolder"]
        if spec.get("distill"):
            kwargs.update(train_frame=students, soft_targets=soft, train_name="gold train + extra tweets, teacher soft labels")
        elif train_rows is not None:
            kwargs.update(train_frame=train_rows, train_name="extra tweets (silver)")
        elif spec.get("label_fix"):
            kwargs.update(train_frame=fixed_train(frames["train"], spec["label_fix"], fixes),
                          train_name=f"train split, label fix '{spec['label_fix']}'")
        elif "fold" in spec:
            kwargs.update(train_frame=frames["train"][folds != spec["fold"]],
                          train_name=f"train split without fold {spec['fold']} of {FOLDS}")
        elif spec.get("limit"):
            kwargs.update(train_frame=frames["train"].head(spec["limit"]), train_name="train split (smoke subset)")
        llm = spec.get("kind") == "llm"
        if llm and train_fn is train_bert:
            from training.llm import train_llm as fn
        else:
            fn = train_fn
        fn(data_dir, run_dir, **kwargs)
        meta = json.loads((run_dir / "training.json").read_text())
        limit = spec.get("limit")
        index = transfer_subset(len(frames["transfer"]), spec.get("transfer_subset", len(frames["transfer"])))
        if limit:
            index = index[:limit]
        wanted = {k: (v.iloc[index] if k == "transfer" else v) for k, v in frames.items()
                  if not (spec.get("skip_transfer") and k == "transfer")}
        if logits_fn is not None:
            logits = logits_fn(run_dir / "model", wanted)
        elif llm:
            from training.llm import saved_logits as llm_logits
            logits = llm_logits(run_dir / "model", wanted)
        else:
            logits = saved_logits(run_dir / "model", wanted)
        if "fold" in spec:
            logits["train_fold"] = folds
        if "transfer" in wanted and len(index) != len(frames["transfer"]):
            logits["transfer_index"] = index
        dest = out / "runs" / spec["name"]
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "training.json").write_text(json.dumps({**meta, "spec": spec}, indent=2))
        np.savez_compressed(dest / "logits.npz", **logits)
        val = meta["selected_validation"]["macro_f1"]
        summary["runs"][spec["name"]] = {"family": spec["family"], "validation": meta["selected_validation"],
                                         "selected_epoch": meta["selected_epoch"], "hours": round((time.time() - t0) / 3600, 3)}
        family = spec["family"]
        if family == "oof":
            shutil.rmtree(run_dir / "model", ignore_errors=True)  # only the predictions are needed
        elif family != "stage1":
            if family not in best or val > best[family][1]:
                if family in best:
                    shutil.rmtree(work / "runs" / best[family][0] / "model", ignore_errors=True)
                best[family] = (spec["name"], val)
            else:
                shutil.rmtree(run_dir / "model", ignore_errors=True)
        last_hours = (time.time() - t0) / 3600
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for family, (name, _) in best.items():
        src = work / "runs" / name / "model"
        if (src / "adapter_config.json").exists():  # a LoRA adapter is already small; copy it as it is
            shutil.copytree(src, out / "models" / name)
        else:
            save_half(src, out / "models" / name)
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
    if a.experiment == "abstain":
        from training.abstain import kaggle_run as abstain_run
        report = abstain_run(a.input, a.out)
        print(json.dumps({"validation": report["validation"], "test_at_cutoff": report["test"]["at_cutoff"]}, indent=1), flush=True)
        return
    if a.experiment == "export":
        from training.kaggle_export import run as export_run
        print(json.dumps(export_run(a.input, a.out)["variants"], indent=1), flush=True)
        return
    print(json.dumps(run(a.experiment, a.input, a.out), indent=1), flush=True)


if __name__ == "__main__":
    main()
