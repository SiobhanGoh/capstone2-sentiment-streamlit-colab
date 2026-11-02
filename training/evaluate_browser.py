"""Score the browser (ONNX) model files on the frozen test partition.

Uses the same prepared split as the corrected comparison (its fingerprint is checked on load),
the same tokenizer settings (truncation at 128 tokens) and reports, for each ONNX file:
accuracy, macro-F1, the confusion matrix, and label agreement with the full-precision export.
Only aggregate results are written; no raw text leaves the private data folder.

    python -m training.evaluate_browser --data runs/data --model browser-model --out verification/browser_model_evaluation.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from training.protocol import LABELS, load_prepared

VARIANTS = ["model.onnx", "model_quantized.onnx", "model_fp16.onnx"]


def logits_for(session, tokenizer, texts, batch_size=32):
    wanted = {i.name for i in session.get_inputs()}
    out = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                          max_length=128, return_tensors="np")
        out.append(session.run(["logits"], {k: v.astype("int64") for k, v in batch.items() if k in wanted})[0])
    return np.concatenate(out)


def main():
    import onnxruntime as ort
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--predictions", help="Optional private folder for per-row predictions (never commit).")
    args = parser.parse_args()
    frame, manifest = load_prepared(args.data)
    test = frame.loc[frame.split == "test"].reset_index(drop=True)
    texts = test.text.tolist()
    y = test.label_id.to_numpy()
    model_dir = Path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    export = json.loads((model_dir / "export_manifest.json").read_text())
    results, labels = {}, {}
    for name in VARIANTS:
        path = model_dir / "onnx" / name
        if not path.exists():
            continue
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        logits = logits_for(session, tokenizer, texts)
        pred = logits.argmax(1)
        labels[name] = pred
        results[name] = {
            "sha256": export["files"][f"onnx/{name}"]["sha256"],
            "bytes": export["files"][f"onnx/{name}"]["bytes"],
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        }
        if args.predictions:
            folder = Path(args.predictions)
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / f"{name}.logits.npy", logits)
    reference = labels["model.onnx"]
    for name, pred in labels.items():
        results[name]["label_agreement_with_full_precision"] = float((pred == reference).mean())
    report = {
        "protocol": manifest["protocol"],
        "split_sha256": manifest["split_sha256"],
        "source_sha256": manifest["source_sha256"],
        "test_inputs": len(test),
        "labels": LABELS,
        "source_checkpoint": export["source"],
        "variants": results,
        "note": ("Retrospective internal evaluation on the same frozen test partition as the corrected "
                 "comparison; it checks that conversion preserved behaviour, not external validity."),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {m: v[m] for m in ["accuracy", "macro_f1", "label_agreement_with_full_precision", "bytes"]}
                      for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
