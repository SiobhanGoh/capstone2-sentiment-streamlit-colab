import hashlib

import pytest

from training import publish_student


def test_stage_copies_only_verified_files_and_refuses_changes(tmp_path):
    model = tmp_path / "model"
    (model / "onnx").mkdir(parents=True)
    names = publish_student.KEEP + ["onnx/model_fp16.onnx", "onnx/model.onnx"]
    for n in names:
        (model / n).write_text(n)
    ev = {"published_variant": "model_fp16.onnx", "student": "s",
          "files": {n: {"sha256": hashlib.sha256(n.encode()).hexdigest()} for n in names}}
    out = tmp_path / "out"
    publish_student.stage(model, ev, out)
    staged = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert "onnx/model.onnx" not in staged and "onnx/model_fp16.onnx" in staged and "README.md" in staged
    (model / "onnx/model_fp16.onnx").write_text("tampered")
    with pytest.raises(SystemExit, match="does not match"):
        publish_student.stage(model, ev, tmp_path / "out2")
