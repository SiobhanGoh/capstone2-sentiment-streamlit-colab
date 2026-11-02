"""Convert the pinned, evaluated checkpoint into ONNX files for in-browser inference.

The output follows the folder layout transformers.js expects:
    config.json, tokenizer files, onnx/model.onnx, onnx/model_quantized.onnx, onnx/model_fp16.onnx

Weights are checked against the evaluated SHA-256 before export. Each ONNX variant is then
compared with PyTorch on public example sentences only; test-set evaluation is a separate
step (training/evaluate_browser.py) that needs the private corpus.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.protocol import clean_text  # noqa: E402

RELEASE = Path(__file__).resolve().parents[1] / "hf_space_streamlit/src/model_release.json"
FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "vocab.txt"]
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
PUBLIC_SENTENCES = [
    "Service dekat kedai ni bagus, staff friendly dan sangat membantu.",
    "I really enjoyed this video. Thank you for the clear explanation.",
    "Saya akan hantar dokumen itu esok pagi.",
    "The store opens at 10 AM.",
    "Service lambat, order salah, memang mengecewakan.",
    "I am disappointed. I waited a long time and the problem is still unresolved.",
    "best gila video ni, bila nak upload part 2?",
    "makanan sedap tapi service slow sikit lah",
    "tak faham kenapa orang suka benda ni",
    "ok je, nothing special",
    "terbaik! terima kasih sebab share",
    "worst experience ever, never coming back",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(spec, cache):
    from huggingface_hub import snapshot_download
    prefix = spec["subfolder"] + "/"
    snapshot = snapshot_download(spec["repo_id"], revision=spec["revision"], cache_dir=cache,
                                 allow_patterns=[prefix + f for f in FILES])
    folder = Path(snapshot) / spec["subfolder"]
    if sha256(folder / "model.safetensors") != spec["model_sha256"]:
        raise SystemExit("Model checksum does not match the evaluated release.")
    return folder


def export_fp32(model, tokenizer, path):
    import torch
    sample = tokenizer(["export sample", "a slightly longer export sample"], padding=True, return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    axes = {n: {0: "batch_size", 1: "sequence_length"} for n in names}
    axes["logits"] = {0: "batch_size"}
    torch.onnx.export(model, tuple(sample[n] for n in names), str(path), input_names=names,
                      output_names=["logits"], dynamic_axes=axes, opset_version=17,
                      do_constant_folding=True, dynamo=False)


def make_variants(fp32_path, onnx_dir):
    """Write the 8-bit file, and a 16-bit file when it converts cleanly. Returns notes."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(fp32_path), str(onnx_dir / "model_quantized.onnx"), weight_type=QuantType.QInt8)
    notes = {}
    try:
        import onnxruntime as ort
        from onnxruntime.transformers.optimizer import optimize_model
        optimised = optimize_model(str(fp32_path), model_type="bert", num_heads=12, hidden_size=768,
                                   opt_level=0, use_gpu=False)
        optimised.convert_float_to_float16(keep_io_types=True)
        optimised.save_model_to_file(str(onnx_dir / "model_fp16.onnx"))
        ort.InferenceSession(str(onnx_dir / "model_fp16.onnx"), providers=["CPUExecutionProvider"])
    except Exception as error:  # 16-bit is optional; never block the 8-bit release on it.
        (onnx_dir / "model_fp16.onnx").unlink(missing_ok=True)
        notes["model_fp16.onnx"] = f"not produced: {type(error).__name__}: {str(error)[:300]}"
    return notes


def compare(model, tokenizer, onnx_dir):
    import numpy as np
    import onnxruntime as ort
    import torch
    texts = [clean_text(t) for t in PUBLIC_SENTENCES]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="np")
    with torch.inference_mode():
        reference = model(**{k: torch.from_numpy(v) for k, v in inputs.items()}).logits.numpy()
    report = {}
    for name in ["model.onnx", "model_quantized.onnx", "model_fp16.onnx"]:
        if not (onnx_dir / name).exists():
            continue
        session = ort.InferenceSession(str(onnx_dir / name), providers=["CPUExecutionProvider"])
        wanted = {i.name for i in session.get_inputs()}
        logits = session.run(["logits"], {k: v.astype("int64") for k, v in inputs.items() if k in wanted})[0]
        report[name] = {
            "max_abs_logit_diff": float(np.abs(logits - reference).max()),
            "label_agreement": float((logits.argmax(1) == reference.argmax(1)).mean()),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", default=".hf-cache")
    args = parser.parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    spec = json.loads(RELEASE.read_text())
    source = download(spec, args.cache)
    out = Path(args.out)
    if out.exists():
        raise SystemExit("Output folder exists; choose a new one so earlier exports are kept.")
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(parents=True)
    for name in FILES:
        if name != "model.safetensors":
            shutil.copy2(source / name, out / name)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(source, local_files_only=True).eval()
    if [str(model.config.id2label[i]).upper() for i in range(3)] != LABELS:
        raise SystemExit("Label order does not match the evaluation protocol.")
    export_fp32(model, tokenizer, onnx_dir / "model.onnx")
    notes = make_variants(onnx_dir / "model.onnx", onnx_dir)
    manifest = {
        "source": {k: spec[k] for k in ["repo_id", "revision", "subfolder", "model_sha256"]},
        "files": {str(p.relative_to(out)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                  for p in sorted(out.rglob("*")) if p.is_file()},
        "public_sentence_check": compare(model, tokenizer, onnx_dir),
        "notes": notes,
    }
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["public_sentence_check"], indent=2))


if __name__ == "__main__":
    main()
