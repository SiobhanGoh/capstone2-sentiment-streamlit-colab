"""One deterministic, duplicate-safe protocol shared by both sentiment models."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
PROTOCOL_VERSION = "sentiment-disjoint-v1"


def clean_text(value):
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    return re.sub(r"\s+", " ", text.replace("#", "")).strip()


def normalize_label(value):
    mapping = {"NEG": "NEGATIVE", "NEU": "NEUTRAL", "POS": "POSITIVE"}
    label = str(value).strip().upper()
    label = mapping.get(label, label)
    return label if label in LABELS else None


def prepare_frame(raw, seed=42):
    if not {"Tweet", "Sentiment"}.issubset(raw.columns):
        raise ValueError("CSV must contain Tweet and Sentiment columns.")
    frame = pd.DataFrame({"text": raw.Tweet.map(clean_text),
                          "label": raw.Sentiment.map(normalize_label)})
    audit = {"raw_rows": len(frame)}
    invalid = frame.label.isna() | frame.text.eq("")
    audit["invalid_or_empty_rows"] = int(invalid.sum())
    frame = frame.loc[~invalid].copy()
    frame["text_id"] = frame.text.map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())
    # Conflicting labels for identical normalized inputs cannot be resolved by guessing.
    conflict_ids = frame.groupby("text_id").label.nunique()
    conflicts = set(conflict_ids[conflict_ids > 1].index)
    audit["conflicting_label_rows"] = int(frame.text_id.isin(conflicts).sum())
    frame = frame.loc[~frame.text_id.isin(conflicts)]
    before = len(frame)
    frame = frame.drop_duplicates("text_id").sort_values("text_id").reset_index(drop=True)
    audit["duplicate_rows"] = before - len(frame)
    if frame.label.value_counts().reindex(LABELS, fill_value=0).min() < 10:
        raise ValueError("Need at least 10 unique examples per sentiment after cleaning.")
    train, held = train_test_split(frame.index, test_size=.30, random_state=seed, stratify=frame.label)
    validation, test = train_test_split(held, test_size=.50, random_state=seed,
                                      stratify=frame.loc[held, "label"])
    frame["split"] = ""
    for name, indices in [("train", train), ("validation", validation), ("test", test)]:
        frame.loc[indices, "split"] = name
    frame["label_id"] = frame.label.map(LABEL_TO_ID)
    audit["retained_rows"] = len(frame)
    audit["split_counts"] = {name: int((frame.split == name).sum()) for name in ["train", "validation", "test"]}
    audit["class_counts"] = {name: {label: int(((frame.split == name) & (frame.label == label)).sum())
                                  for label in LABELS} for name in audit["split_counts"]}
    audit["protocol"] = PROTOCOL_VERSION
    audit["seed"] = seed
    audit["language_filter"] = "No automatic language filtering; supplied Malay-English corpus retained."
    validate_frame(frame)
    return frame, audit


def validate_frame(frame):
    if frame.text_id.duplicated().any():
        raise ValueError("Normalized text duplicates or split overlap detected.")
    if set(frame.split) != {"train", "validation", "test"}:
        raise ValueError("All three partitions are required.")
    if not frame.label.map(LABEL_TO_ID).eq(frame.label_id).all():
        raise ValueError("Label mapping does not match the protocol.")
    if not frame.text.map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()).eq(frame.text_id).all():
        raise ValueError("Text fingerprint mismatch.")


def fingerprint(frame):
    manifest = frame[["text_id", "label_id", "split"]].sort_values("text_id").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def prepare_csv(csv_path, output_dir, seed=42):
    output_dir = Path(output_dir)
    if (output_dir/"manifest.json").exists():
        raise FileExistsError("Prepared data already exists; choose a new directory to avoid changing a frozen split.")
    source = Path(csv_path)
    frame, audit = prepare_frame(pd.read_csv(source), seed)
    audit["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    audit["split_sha256"] = fingerprint(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir/"prepared.csv", index=False)
    (output_dir/"manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def load_prepared(directory):
    directory = Path(directory)
    frame = pd.read_csv(directory/"prepared.csv", keep_default_na=False)
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    validate_frame(frame)
    if fingerprint(frame) != manifest["split_sha256"]:
        raise ValueError("Prepared data or partitions changed after freezing.")
    return frame, manifest
