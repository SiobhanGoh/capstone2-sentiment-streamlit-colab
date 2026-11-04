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
    assert all(n.startswith("training/") and (n.endswith(".py") or n in ("training/distill_teacher.json", "training/train_label_fixes.json")) for n in names)
    with pytest.raises(ValueError):
        kaggle_entry.render("nope", "x")


def test_soften_and_distillation_loss_match_the_formula():
    torch = pytest.importorskip("torch")
    from training.run import distillation_loss, soften
    p = np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
    t = soften(p, 2.0)
    expected = np.sqrt(p) / np.sqrt(p).sum(1, keepdims=True)
    assert np.allclose(t, expected, atol=1e-6)
    logits = np.array([[2.0, 0.0, -1.0], [0.5, 0.1, 0.3]])
    labels = np.array([0, -1])  # second row has no gold label
    loss = float(distillation_loss(torch.tensor(logits), torch.tensor(labels), torch.tensor(t), 0.5, 2.0, torch))
    ls = logits / 2 - np.log(np.exp(logits / 2).sum(1, keepdims=True))
    kd = (t * (np.log(t) - ls)).sum(1) * 4
    ce = np.array([-(logits[0, 0] - np.log(np.exp(logits[0]).sum())), 0.0])
    assert loss == pytest.approx(float((0.5 * kd + 0.5 * ce).mean()), rel=1e-5)


def teacher_logits(tmp_path, kernel, run, frames, fill):
    d = tmp_path / "input" / kernel / "out" / "runs" / run
    d.mkdir(parents=True)
    np.savez(d / "logits.npz", **{k: np.full((n, 3), 0.0) + fill for k, n in frames.items()})


def test_load_teacher_averages_temperature_scaled_members(tmp_path):
    frames = {"train": pd.DataFrame({"text": ["a", "b"]}), "transfer": pd.DataFrame({"text": ["c"]})}
    teacher_logits(tmp_path, "rojak-sentiment-large", "xlmr-large-s1", {"train": 2, "transfer": 1}, np.array([2.0, 0, 0]))
    teacher_logits(tmp_path, "rojak-sentiment-silver", "mbert-silver-s1", {"train": 2, "transfer": 1}, np.array([0, 0, 4.0]))
    spec = {"members": [{"run": "xlmr-large-s1", "temperature": 2.0}, {"run": "mbert-silver-s1", "temperature": 4.0}]}
    probs = kaggle_job.load_teacher(tmp_path / "input", frames, spec)
    e = np.exp([1.0, 0, 0]); a = e / e.sum()
    b = np.exp([0, 0, 1.0]) / np.exp([0, 0, 1.0]).sum()
    assert np.allclose(probs["train"], (a + b) / 2) and probs["transfer"].shape == (1, 3)
    with pytest.raises(FileNotFoundError):
        kaggle_job.load_teacher(tmp_path / "input", frames, {"members": [{"run": "missing", "temperature": 1}]})
    with pytest.raises(ValueError, match="do not match"):
        kaggle_job.load_teacher(tmp_path / "input", {**frames, "transfer": pd.DataFrame({"text": ["c", "d"]})}, spec)


def test_distill_run_teaches_students_on_gold_and_extra_rows(corpus):
    calls = []
    silver = pd.DataFrame({"text": ["a b", "c d", "e f"], "label_id": [0, 1, 2]})
    audit = {"retained_text_sha256": "x", "retained_rows": 3}
    teacher_logits(corpus, "rojak-sentiment-large", "xlmr-large-s1", {"train": 42, "transfer": 3}, np.array([1.0, 0, 0]))
    spec = {"members": [{"run": "xlmr-large-s1", "temperature": 1.0}], "transfer_text_sha256": "x"}
    logits = lambda model_dir, frames: {k: np.zeros((len(v), 3), np.float32) for k, v in frames.items()}
    summary = kaggle_job.run("distill", corpus / "input", corpus / "out", work_dir=corpus / "work", train_fn=fake_train(calls),
                             logits_fn=logits, silver_fn=lambda g: (silver, audit), teacher=spec)
    assert [c[0] for c in calls] == [s["name"] for s in kaggle_job.EXPERIMENTS["distill"]]
    kw = calls[0][1]
    assert kw["subfolder"] == "corrected-2026-10" and kw["base"] == "SiobhanGoh/CP2_Emotion_Model_New"
    rows, soft = kw["train_frame"], kw["soft_targets"]
    assert len(rows) == len(soft) == 45 and (rows.label_id == -1).sum() == 3 and (rows.label_id >= 0).sum() == 42
    assert set(summary["kept_models"]) == {"student-app", "student-mbert"}
    with pytest.raises(ValueError, match="extra tweets differ"):
        kaggle_job.run("distill", corpus / "input", corpus / "out2", work_dir=corpus / "work2", train_fn=fake_train([]),
                       logits_fn=logits, silver_fn=lambda g: (silver, audit), teacher={**spec, "transfer_text_sha256": "y"})


def test_train_bert_distils_end_to_end_with_a_tiny_model(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from training.run import train_bert
    words = ["teruk", "biasa", "bagus", "nombor"] + [str(i) for i in range(20)]
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]))
    tiny = tmp_path / "tiny"
    transformers.BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True).save_pretrained(tiny)
    config = transformers.BertConfig(vocab_size=len(words) + 5, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                     intermediate_size=32, num_labels=3)
    transformers.BertForSequenceClassification(config).save_pretrained(tiny)
    rows = [{"Tweet": f"{w} nombor {i}", "Sentiment": s} for s, w in
            {"NEGATIVE": "teruk", "NEUTRAL": "biasa", "POSITIVE": "bagus"}.items() for i in range(20)]
    pd.DataFrame(rows).to_csv(tmp_path / "c.csv", index=False)
    prepare_csv(tmp_path / "c.csv", tmp_path / "data")
    train = pd.DataFrame({"text": ["teruk nombor 1", "bagus nombor 2", "biasa 3"], "label_id": [0, 2, -1]})
    soft = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.2, 0.6, 0.2]], dtype=np.float32)
    train_bert(tmp_path / "data", tmp_path / "out", epochs=1, batch_size=2, accumulation=1, init_from=tiny,
               train_frame=train, soft_targets=soft)
    meta = json.loads((tmp_path / "out" / "training.json").read_text())
    assert meta["status"] == "complete" and meta["distillation"]["rows_without_gold_label"] == 1
    with pytest.raises(ValueError, match="align"):
        train_bert(tmp_path / "data", tmp_path / "out2", epochs=1, init_from=tiny, train_frame=train, soft_targets=soft[:2])


def test_kaggle_export_converts_and_scores_the_student(corpus, monkeypatch):
    pytest.importorskip("onnxruntime")
    transformers = pytest.importorskip("transformers")
    from training import kaggle_export
    from training.protocol import LABELS
    words = ["teruk", "biasa", "bagus", "nombor"] + [str(i) for i in range(20)]
    vocab = corpus / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]))
    student = corpus / "input" / "rojak-sentiment-distill" / "out" / "models" / "student-app-s1"
    transformers.BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True).save_pretrained(student)
    config = transformers.BertConfig(vocab_size=len(words) + 5, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                     intermediate_size=32, num_labels=3, id2label=dict(enumerate(LABELS)),
                                     label2id={l: i for i, l in enumerate(LABELS)})
    transformers.BertForSequenceClassification(config).half().save_pretrained(student)
    runs = corpus / "input" / "rojak-sentiment-distill" / "out" / "runs" / "student-app-s1"
    runs.mkdir(parents=True)
    np.savez(runs / "logits.npz", validation=np.zeros((9, 3)), test=np.zeros((9, 3)))
    summary = kaggle_export.run(corpus / "input", corpus / "out", work_dir=corpus / "w")
    assert "model.onnx" in summary["variants"] and "onnx/model.onnx" in summary["files"]
    v = summary["variants"]["model.onnx"]
    assert 0 <= v["test_accuracy"] <= 1 and v["sha256"] == summary["files"]["onnx/model.onnx"]["sha256"]
    with np.load(corpus / "out" / "onnx_eval" / "model" / "logits.npz") as z:
        assert z["test"].shape == (9, 3) and z["validation"].shape == (9, 3)
    assert (corpus / "out" / "models" / "browser-model" / "onnx" / "model.onnx").exists()


def test_fold_ids_are_stratified_and_reproducible():
    labels = np.array([0] * 50 + [1] * 30 + [2] * 20)
    f = kaggle_job.fold_ids(labels)
    assert (f == kaggle_job.fold_ids(labels)).all() and set(f) == set(range(5))
    for k in range(5):
        assert np.bincount(labels[f == k], minlength=3).tolist() == [10, 6, 4]


def test_oof_run_trains_without_each_fold_and_keeps_no_models(corpus):
    calls = []
    silver = pd.DataFrame({"text": ["a b"], "label_id": [0]})
    seen = []
    logits = lambda model_dir, frames: seen.append(sorted(frames)) or {k: np.zeros((len(v), 3), np.float32) for k, v in frames.items()}
    out = corpus / "out"
    summary = kaggle_job.run("oof", corpus / "input", out, work_dir=corpus / "work", train_fn=fake_train(calls),
                             logits_fn=logits, silver_fn=lambda g: (silver, {"retained_text_sha256": "x"}))
    assert len(calls) == 5 and all(sorted(s) == ["test", "train", "validation"] for s in seen)
    sizes = [len(c[1]["train_frame"]) for c in calls]
    assert sum(sizes) == 4 * 42 and summary["kept_models"] == {}
    with np.load(out / "runs" / "oof-xlmr-f0" / "logits.npz") as z:
        held = z["train_fold"] == 0
        assert held.sum() == 42 - sizes[0] and "transfer" not in z.files


def test_teacher_uses_only_tweets_every_member_labelled(tmp_path):
    frames = {"train": pd.DataFrame({"text": ["a"]}), "transfer": pd.DataFrame({"text": ["x", "y", "z"]})}
    full = tmp_path / "input" / "large2" / "out" / "runs" / "xlmr"
    full.mkdir(parents=True)
    np.savez(full / "logits.npz", train=np.zeros((1, 3)), transfer=np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]))
    sub = tmp_path / "input" / "llm" / "out" / "runs" / "qwen"
    sub.mkdir(parents=True)
    np.savez(sub / "logits.npz", train=np.zeros((1, 3)), transfer=np.array([[1.0, 0, 0], [0, 0, 1.0]]), transfer_index=np.array([0, 2]))
    spec = {"members": [{"run": "xlmr", "temperature": 1.0}, {"run": "qwen", "temperature": 1.0}]}
    probs = kaggle_job.load_teacher(tmp_path / "input", frames, spec)
    assert probs["transfer_index"].tolist() == [0, 2] and probs["transfer"].shape == (2, 3)
    assert probs["transfer"][1].argmax() == 2  # tweet z, labelled by both members


def test_transfer_subset_is_fixed_sorted_and_capped():
    a = kaggle_job.transfer_subset(1000, 100)
    assert len(a) == 100 and (np.diff(a) > 0).all() and (a == kaggle_job.transfer_subset(1000, 100)).all()
    assert kaggle_job.transfer_subset(10, 50).tolist() == list(range(10))


def test_length_order_puts_short_texts_first():
    from training.llm import length_order
    texts = ["panjang sekali ayat ini", "ok", "agak sederhana"]
    order = length_order(texts)
    assert [texts[i] for i in order] == ["ok", "agak sederhana", "panjang sekali ayat ini"]


def test_fixed_train_relabels_or_drops_only_listed_rows():
    train = pd.DataFrame({"text": list("abcd"), "label_id": [0, 1, 2, 0]})
    fixes = {"train_rows": 4, "relabel": {"1": 0, "3": 2}, "flagged": [1, 2]}
    assert kaggle_job.fixed_train(train, "relabel", fixes).label_id.tolist() == [0, 0, 2, 2]
    assert kaggle_job.fixed_train(train, "dropflag", fixes).text.tolist() == ["a", "d"]
    assert train.label_id.tolist() == [0, 1, 2, 0]  # the original frame is untouched
    with pytest.raises(ValueError):
        kaggle_job.fixed_train(train.head(3), "relabel", fixes)
