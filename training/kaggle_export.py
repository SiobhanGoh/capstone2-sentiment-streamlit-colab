"""On Kaggle: convert the chosen distilled student to the web app's ONNX files and score them.

Nothing is published here. The student (kept by the 'distill' run, attached as an input) is
exported exactly like the current browser model (training/export_browser.py), then every ONNX file
is run on the frozen validation and test splits. Only numbers reach the collected outputs: an
evaluation summary and the ONNX logits. The ONNX files stay in the private Kaggle output under
out/models/ until a separate, approved step publishes them.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from training.protocol import load_prepared, prepare_csv

STUDENT = "student-app-s1"  # chosen by training/distill_select.py on validation (verification/distill_selection.json)
VARIANTS = ["model.onnx", "model_quantized.onnx", "model_fp16.onnx"]


def find_student(input_dir, name=STUDENT):
    found = [p for p in sorted(Path(input_dir).rglob(f"models/{name}")) if (p / "model.safetensors").exists()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one kept model {name}, found {len(found)}.")
    return found[0]


def student_logits(input_dir, name=STUDENT):
    found = sorted(Path(input_dir).rglob(f"runs/{name}/logits.npz"))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one logits file for {name}, found {len(found)}.")
    with np.load(found[0]) as z:
        return {k: z[k] for k in ["validation", "test"]}


def run(input_dir, out_dir, work_dir="/tmp/rojak"):
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from training import export_browser, kaggle_job
    from training.evaluate_browser import logits_for
    out, work = Path(out_dir), Path(work_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = work / "data"
    if not (data_dir / "manifest.json").exists():
        prepare_csv(kaggle_job.find_corpus(input_dir), data_dir)
    frame, manifest = load_prepared(data_dir)
    if manifest["split_sha256"] != kaggle_job.SPLIT_SHA256:
        raise ValueError("The prepared split does not match the frozen evaluation split.")
    source = find_student(input_dir)
    model_dir = out / "models" / "browser-model"
    onnx_dir = model_dir / "onnx"
    onnx_dir.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    # The kept student is stored in 16-bit; export from 32-bit like the current browser model.
    model = AutoModelForSequenceClassification.from_pretrained(source, local_files_only=True,
                                                               dtype=torch.float32).eval()
    if [str(model.config.id2label[i]).upper() for i in range(3)] != export_browser.LABELS:
        raise SystemExit("Label order does not match the evaluation protocol.")
    tokenizer.save_pretrained(model_dir)
    model.config.save_pretrained(model_dir)
    export_browser.export_fp32(model, tokenizer, onnx_dir / "model.onnx")
    notes = export_browser.make_variants(onnx_dir / "model.onnx", onnx_dir)
    files = {str(p.relative_to(model_dir)): {"sha256": export_browser.sha256(p), "bytes": p.stat().st_size}
             for p in sorted(model_dir.rglob("*")) if p.is_file()}
    reference = student_logits(input_dir)
    texts = {s: frame[frame.split == s].text.tolist() for s in ["validation", "test"]}
    ys = {s: frame[frame.split == s].label_id.to_numpy() for s in texts}
    results = {}
    for name in VARIANTS:
        path = onnx_dir / name
        if not path.exists():
            continue
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        logits = {s: logits_for(session, tokenizer, t) for s, t in texts.items()}
        dest = out / "onnx_eval" / name.removesuffix(".onnx")
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / "logits.npz", **logits)
        results[name] = {
            "sha256": files[f"onnx/{name}"]["sha256"], "bytes": files[f"onnx/{name}"]["bytes"],
            **{f"{s}_accuracy": float((logits[s].argmax(1) == ys[s]).mean()) for s in texts},
            **{f"{s}_label_agreement_with_student": float((logits[s].argmax(1) == reference[s].argmax(1)).mean())
               for s in texts},
            "test_max_abs_logit_diff_vs_student": float(np.abs(logits["test"] - reference["test"]).max()),
        }
    summary = {"student": STUDENT, "source_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                     for p in sorted(source.iterdir()) if p.is_file()},
               "split_sha256": manifest["split_sha256"], "files": files, "variants": results, "notes": notes,
               "public_sentence_check": export_browser.compare(model, tokenizer, onnx_dir)}
    (out / "onnx_eval" / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
