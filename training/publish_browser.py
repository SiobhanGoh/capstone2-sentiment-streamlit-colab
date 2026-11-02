"""Publish evaluated ONNX files to a Hugging Face model repo and record the commit to pin.

Refuses to upload unless every published file matches the SHA-256 in the export manifest, and
the chosen ONNX file matches the one scored in verification/browser_model_evaluation.json.
"""
import argparse
import hashlib
import json
from pathlib import Path

CARD = Path(__file__).with_name("browser_model_card.md")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    from huggingface_hub import HfApi
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    folder = Path(args.model)
    evaluation = json.loads(Path(args.evaluation).read_text())
    manifest = json.loads((folder / "export_manifest.json").read_text())
    chosen = evaluation["published_variant"]
    for name, info in manifest["files"].items():
        if (folder / name).exists() and sha256(folder / name) != info["sha256"]:
            raise SystemExit(f"{name} does not match the export manifest.")
    if manifest["files"][f"onnx/{chosen}"]["sha256"] != evaluation["variants"][chosen]["sha256"]:
        raise SystemExit("The chosen ONNX file is not the one that was evaluated.")
    keep = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "vocab.txt", "export_manifest.json", f"onnx/{chosen}"}
    (folder / "README.md").write_text(CARD.read_text(encoding="utf-8"), encoding="utf-8")
    keep.add("README.md")
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    commit = api.upload_folder(repo_id=args.repo, folder_path=str(folder), repo_type="model",
                               allow_patterns=sorted(keep),
                               commit_message=f"Publish evaluated browser model ({chosen})")
    revision = commit.oid
    record = {
        "repo_id": args.repo,
        "revision": revision,
        "onnx_file": f"onnx/{chosen}",
        "onnx_sha256": evaluation["variants"][chosen]["sha256"],
        "export_run_id": evaluation["export_run_id"],
    }
    Path(args.out).write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
