"""Publish the checked distilled student to Hugging Face and record the commit to pin.

Run by the Kaggle workflow (action "publish") on the files downloaded from the private 'export'
run. Refuses to upload unless every file matches the SHA-256 recorded when the files were scored
(verification/student_onnx_evaluation.json). Only the 16-bit model and the files transformers.js
needs are uploaded; the web app keeps its old pinned revision until it is switched separately.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

CARD = Path(__file__).with_name("browser_model_card.md")
KEEP = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage(model_dir, evaluation, out):
    """Copy the verified files into `out`, ready to upload. Raises on any checksum mismatch."""
    model_dir, out = Path(model_dir), Path(out)
    chosen = f"onnx/{evaluation['published_variant']}"
    for name in KEEP + [chosen]:
        expected = evaluation["files"][name]["sha256"]
        if sha256(model_dir / name) != expected:
            raise SystemExit(f"{name} does not match the checksum recorded at evaluation.")
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_dir / name, out / name)
    (out / "README.md").write_text(CARD.read_text(encoding="utf-8"), encoding="utf-8")
    (out / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    return chosen


def main():
    from huggingface_hub import HfApi
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--evaluation", required=True)
    p.add_argument("--staging", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    evaluation = json.loads(Path(a.evaluation).read_text())
    chosen = stage(a.model, evaluation, a.staging)
    commit = HfApi().upload_folder(repo_id=a.repo, folder_path=a.staging, repo_type="model",
                                   commit_message=f"Distilled student ({evaluation['student']}), 16-bit")
    record = {"repo_id": a.repo, "revision": commit.oid, "onnx_file": chosen,
              "onnx_sha256": evaluation["files"][chosen]["sha256"], "student": evaluation["student"]}
    Path(a.out).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
