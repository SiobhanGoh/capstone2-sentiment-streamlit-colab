import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import extra_data, kaggle_job
from training.protocol import prepare_csv


def jl(*rows):
    return [json.dumps(r) for r in rows]


def test_read_silver_cleans_maps_and_drops_conflicts_and_duplicates():
    frame, audit = extra_data.read_silver(jl(
        {"text": "Bagus GILA @abu https://x.co", "sentiment": "positive"},
        {"text": "bagus gila", "sentiment": "positive"},          # duplicate after cleaning
        {"text": "entah", "sentiment": "neutral"},
        {"text": "entah", "sentiment": "negative"},               # conflicting label
        {"text": "", "sentiment": "negative"},
        {"text": "teruk", "sentiment": "angry"},                  # unknown label
        {"text": "teruk sangat", "sentiment": "Negative"},
    ) + [""])
    assert frame.to_dict("records") == [{"text": "bagus gila", "label_id": 2}, {"text": "teruk sangat", "label_id": 0}]
    assert audit == {"parsed_rows": 5, "conflicting_label_rows": 2, "duplicate_rows": 1}


def test_near_duplicates_uses_word_set_jaccard_including_unknown_words():
    protected = ["servis lambat sangat teruk hari ini"]
    candidates = [
        "servis lambat sangat teruk hari ini lah",   # 6/7 = 0.857 -> near duplicate
        "servis lambat sangat teruk",                # 4/6 = 0.667 -> kept
        "servis lambat sangat teruk hari kau tahu",  # 5/8 = 0.625 (two unseen words) -> kept
        "makanan sedap",
    ]
    assert extra_data.near_duplicates(candidates, protected, chunk=2).tolist() == [True, False, False, False]


def gold_frame():
    return pd.DataFrame({
        "text": ["train satu dua tiga", "val empat lima enam", "test tujuh lapan sembilan"],
        "label_id": [2, 0, 1], "split": ["train", "validation", "test"]})


def test_build_removes_held_out_overlap_and_audits(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(jl(
        {"text": "train satu dua tiga", "sentiment": "positive"},       # exact train overlap, agrees
        {"text": "val empat lima enam", "sentiment": "positive"},       # exact validation overlap
        {"text": "test tujuh lapan sembilan sepuluh", "sentiment": "neutral"},  # near test (4/5)
        {"text": "train satu dua tiga empat", "sentiment": "neutral"},  # near train only: kept
        {"text": "lain langsung", "sentiment": "negative"},
    )), encoding="utf-8")
    silver, audit = extra_data.build(gold_frame(), path)
    assert silver.text.tolist() == ["lain langsung", "train satu dua tiga empat"]
    assert audit["exact_overlap_train"] == 1 and audit["exact_overlap_validation"] == 1 and audit["exact_overlap_test"] == 0
    assert audit["near_duplicate_of_validation_or_test"] == 1
    assert audit["label_agreement_on_train_overlap"] == 1.0
    assert audit["retained_rows"] == 2 and audit["class_counts"] == {"NEGATIVE": 1, "NEUTRAL": 1, "POSITIVE": 0}
    assert audit["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_experiments_are_fixed_and_well_formed():
    names = [s["name"] for p in kaggle_job.EXPERIMENTS.values() for s in p]
    assert len(names) == len(set(names))
    for plan in kaggle_job.EXPERIMENTS.values():
        seen = set()
        for spec in plan:
            assert spec.get("init") is None or spec["init"] in seen  # first stage runs before it is used
            assert ("init" in spec) != ("base" in spec)
            seen.add(spec["name"])
    assert {s["seed"] for s in kaggle_job.EXPERIMENTS["large"]} == {1, 2, 3, 4, 5}


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    words = {"NEGATIVE": "teruk", "NEUTRAL": "biasa", "POSITIVE": "bagus"}
    rows = [{"Tweet": f"{w} nombor {i}", "Sentiment": s} for s, w in words.items() for i in range(20)]
    inp = tmp_path / "input" / "rojak-sentiment-private"
    inp.mkdir(parents=True)
    csv = inp / "corpus.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    (inp / "other.csv").write_text("Tweet,Sentiment\nx,POSITIVE\n")
    split = prepare_csv(csv, tmp_path / "check")["split_sha256"]
    monkeypatch.setattr(kaggle_job, "SOURCE_SHA256", hashlib.sha256(csv.read_bytes()).hexdigest())
    monkeypatch.setattr(kaggle_job, "SPLIT_SHA256", split)
    monkeypatch.setattr(kaggle_job, "save_half", lambda src, dest: shutil.copytree(src, dest))
    return tmp_path


def fake_train(record):
    scores = iter([0.70, 0.60, 0.75, 0.50, 0.80, 0.40, 0.65, 0.55, 0.62, 0.61])

    def train(data_dir, out, **kw):
        record.append((Path(out).name, kw))
        (Path(out) / "model").mkdir(parents=True)
        f1 = next(scores)
        Path(out, "training.json").write_text(json.dumps({"status": "complete", "selected_epoch": 1,
                                                          "selected_validation": {"accuracy": f1, "macro_f1": f1}}))
    return train


def test_run_keeps_only_the_best_model_per_family_and_writes_no_text(corpus):
    calls = []
    silver = pd.DataFrame({"text": ["a b", "c d", "e f"], "label_id": [0, 1, 2]})
    audit = {"retained_text_sha256": "x", "retained_rows": 3}
    logits = lambda model_dir, frames: {k: np.zeros((len(v), 3), np.float32) for k, v in frames.items()}
    out = corpus / "out"
    summary = kaggle_job.run("silver", corpus / "input", out, work_dir=corpus / "work",
                             train_fn=fake_train(calls), logits_fn=logits, silver_fn=lambda g: (silver, audit))
    assert [c[0] for c in calls] == [s["name"] for s in kaggle_job.EXPERIMENTS["silver"]]
    # Stage two continues from stage one; stage one trains on the extra tweets.
    assert calls[0][1]["train_frame"] is silver and calls[1][1]["init_from"].parent.name == "xlmr-large-silver-stage1"
    assert summary["kept_models"] == {"xlmr-large-silver": "xlmr-large-silver-s2", "mbert-silver": "mbert-silver-s2"}
    assert sorted(p.name for p in (out / "models").iterdir()) == ["mbert-silver-s2", "xlmr-large-silver-s2"]
    with np.load(out / "runs" / "mbert-silver-s2" / "logits.npz") as z:
        assert {k: z[k].shape for k in z.files} == {"validation": (9, 3), "test": (9, 3), "train": (42, 3), "transfer": (3, 3)}
    # Nothing in the outputs contains corpus text.
    texts = [f"nombor {i}" for i in range(20)]
    for path in out.rglob("*.json"):
        assert not any(t in path.read_text() for t in texts), path


def test_run_refuses_a_different_split(corpus, monkeypatch):
    monkeypatch.setattr(kaggle_job, "SPLIT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="frozen evaluation split"):
        kaggle_job.run("smoke", corpus / "input", corpus / "out", work_dir=corpus / "w",
                       train_fn=fake_train([]), silver_fn=lambda g: (g, {}))


def test_run_stops_before_the_session_budget(corpus, monkeypatch):
    monkeypatch.setattr(kaggle_job, "BUDGET_HOURS", -1)
    summary = kaggle_job.run("large", corpus / "input", corpus / "out", work_dir=corpus / "w", train_fn=fake_train([]),
                             silver_fn=lambda g: (g.head(0)[["text", "label_id"]], {"retained_text_sha256": ""}))
    assert summary["runs"] == {} and len(summary["skipped"]) == 5


def test_kaggle_entry_embeds_this_commits_training_code(tmp_path):
    import base64, io, re, tarfile
    from training import kaggle_entry
    script = kaggle_entry.render("smoke", "abc1234")
    compile(script, "kaggle_entry.py", "exec")
    assert "EXPERIMENT, COMMIT = 'smoke', 'abc1234'" in script
    blob = re.search(r"b64decode\('([A-Za-z0-9+/=]+)'\)", script).group(1)
    names = tarfile.open(fileobj=io.BytesIO(base64.b64decode(blob)), mode="r:gz").getnames()
    assert {"training/kaggle_job.py", "training/run.py", "training/protocol.py", "training/extra_data.py"} <= set(names)
    assert all(n.startswith("training/") and n.endswith(".py") for n in names)
    with pytest.raises(ValueError):
        kaggle_entry.render("nope", "x")
