import json
import pandas as pd
import pytest
from training.protocol import clean_text, prepare_frame, prepare_csv, load_prepared, fingerprint


def examples():
    return pd.DataFrame([{"Tweet": f"Example {label} item {i}", "Sentiment": label}
                         for label in ["NEGATIVE", "NEUTRAL", "POSITIVE"] for i in range(40)])


def test_normalization_does_not_keep_usernames_or_urls():
    assert clean_text("  @name #Hello https://example.com/test  WORLD\n") == "hello world"


def test_duplicate_and_conflicting_inputs_never_cross_partitions():
    raw = examples()
    extra = pd.DataFrame([{"Tweet": "Example NEGATIVE item 0", "Sentiment": "NEG"},
                          {"Tweet": "Example NEGATIVE item 1", "Sentiment": "POS"},
                          {"Tweet": None, "Sentiment": "POSITIVE"}])
    frame, audit = prepare_frame(pd.concat([raw, extra], ignore_index=True))
    assert audit['duplicate_rows'] == 1
    assert audit['conflicting_label_rows'] == 2
    assert audit['invalid_or_empty_rows'] == 1
    assert frame.text_id.is_unique
    groups = [set(frame[frame.split == name].text_id) for name in ['train', 'validation', 'test']]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert audit['raw_rows'] == sum(audit[k] for k in ['retained_rows','duplicate_rows','conflicting_label_rows','invalid_or_empty_rows'])


def test_partition_is_reproducible_independent_of_input_order():
    first, _ = prepare_frame(examples())
    reordered, _ = prepare_frame(examples().sample(frac=1, random_state=7))
    assert fingerprint(first) == fingerprint(reordered)


def test_frozen_split_rejects_replacement_and_tampering(tmp_path):
    csv = tmp_path/'input.csv'
    examples().to_csv(csv, index=False)
    output = tmp_path/'data'
    prepare_csv(csv, output)
    frame, _ = load_prepared(output)
    with pytest.raises(FileExistsError): prepare_csv(csv, output)
    frame.loc[0, 'split'] = 'test' if frame.loc[0, 'split'] != 'test' else 'train'
    frame.to_csv(output/'prepared.csv', index=False)
    with pytest.raises(ValueError, match='partitions changed'): load_prepared(output)


def test_baseline_vocabulary_is_fitted_on_training_only(tmp_path):
    import joblib
    from training.run import train_baseline
    csv = tmp_path/'input.csv'
    examples().to_csv(csv, index=False)
    output = tmp_path/'data'
    prepare_csv(csv, output)
    frame, _ = load_prepared(output)
    train_baseline(output, tmp_path/'baseline')
    model = joblib.load(tmp_path/'baseline/baseline.joblib')
    vocabulary = model.named_steps['tfidfvectorizer'].vocabulary_
    analyzer = model.named_steps['tfidfvectorizer'].build_analyzer()
    training_terms = {term for text in frame[frame.split == 'train'].text for term in analyzer(text)}
    assert set(vocabulary).issubset(training_terms)
    assert set(model.classes_) == {0,1,2}
    # No test scores are created by the training/selection command.
    meta=json.loads((tmp_path/'baseline/training.json').read_text())
    assert all('validation' in row and 'test' not in row for row in meta['history'])
