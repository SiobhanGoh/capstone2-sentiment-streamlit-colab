import pandas as pd
import pytest

from training.candidates import CANDIDATES, load_check_set, select_by_validation


def run(f1, status="complete"):
    return {"status": status, "selected_validation": {"accuracy": f1, "macro_f1": f1}}


def test_selection_uses_validation_only_and_skips_unfinished_runs():
    trainings = {"a": run(0.74), "b": run(0.79), "c": run(0.99, status="running")}
    assert select_by_validation(trainings) == "b"


def test_ties_are_broken_by_name_so_the_choice_is_repeatable():
    assert select_by_validation({"zeta": run(0.8), "alpha": run(0.8)}) == "alpha"


def test_selection_needs_a_finished_candidate():
    with pytest.raises(ValueError):
        select_by_validation({"a": run(0.8, status="running")})


def test_candidates_are_fixed_with_bases_and_learning_rates():
    assert len(CANDIDATES) >= 2
    for spec in CANDIDATES.values():
        assert "/" in spec["base"] and 0 < spec["learning_rate"] < 1e-3


def test_check_set_accepts_app_corrections_export(tmp_path):
    path = tmp_path / "label_corrections.csv"
    pd.DataFrame({"Text": ["best gila @ali", "best gila", "teruk"], "Model label": ["NEGATIVE"] * 3,
                  "Your label": ["POSITIVE", "POSITIVE", "neg"]}).to_csv(path, index=False)
    frame = load_check_set(path)
    assert frame.text.tolist() == ["best gila", "teruk"]
    assert frame.label_id.tolist() == [2, 0]


def test_training_refuses_a_checkpoint_whose_encoder_did_not_load():
    from training.run import check_pretrained_weights
    check_pretrained_weights({"missing_keys": ["classifier.weight", "pooler.dense.bias", "bert.pooler.dense.weight",
                                               "classifier.out_proj.bias"]})
    with pytest.raises(ValueError):
        check_pretrained_weights({"missing_keys": ["roberta.encoder.layer.3.intermediate.dense.weight"]})
    with pytest.raises(ValueError, match="encoder weights did not load"):
        check_pretrained_weights({"missing_keys": ["classifier.weight", "deberta.encoder.layer.0.attention.self.query_proj.weight"]})


def test_interrupted_runs_are_kept_aside_and_finished_ones_skipped(tmp_path, monkeypatch):
    import json
    import training.candidates as c
    calls = []
    monkeypatch.setattr(c, "train_bert", lambda data, out, **kw: calls.append((out.name, kw["base"])))
    (tmp_path / "xlm-t").mkdir()
    (tmp_path / "xlm-t" / "training.json").write_text(json.dumps({"status": "complete"}))
    (tmp_path / "mesolitica-roberta").mkdir()
    (tmp_path / "mesolitica-roberta" / "training.json").write_text(json.dumps({"status": "running"}))
    c.train_all("data", tmp_path)
    assert [name for name, _ in calls] == ["mesolitica-debertav2", "mesolitica-roberta"]
    assert (tmp_path / "mesolitica-roberta.interrupted-1").exists()
