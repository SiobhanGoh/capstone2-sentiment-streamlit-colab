"""Extra, machine-labelled Malaysian tweets for a first training stage ("silver" data).

Source: mesolitica/chatgpt-explain-sentiment, sentiment.jsonl, pinned below. 162,902 Malaysian
tweets labelled negative / neutral / positive by ChatGPT-3.5, so the labels are noisier than the
gold corpus and may define "neutral" differently. The dataset states research use and asks that
it not be redistributed, so it is only downloaded inside the training job and never committed.

To keep the frozen evaluation honest, any tweet that matches a gold validation or test input
exactly after cleaning, or nearly (word-set Jaccard >= NEAR), is removed. Exact matches with the
gold train split are removed too, so the first stage never sees a gold row twice.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from training.protocol import LABEL_TO_ID, clean_text

REPO_ID = "mesolitica/chatgpt-explain-sentiment"
REVISION = "93e03257d1ab80cf482306daacc160e59d5e7b22"
FILENAME = "sentiment.jsonl"
NEAR = 0.8
TOKEN = r"(?u)\b\w\w+\b"


def download(cache_dir=None):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO_ID, FILENAME, repo_type="dataset", revision=REVISION, cache_dir=cache_dir)


def read_silver(lines):
    """Parse jsonl lines into cleaned (text, label_id); drop empty, unknown and conflicting rows."""
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        label = str(item.get("sentiment", "")).strip().upper()
        text = clean_text(item.get("text"))
        if text and label in LABEL_TO_ID:
            rows.append((text, LABEL_TO_ID[label]))
    frame = pd.DataFrame(rows, columns=["text", "label_id"])
    audit = {"parsed_rows": len(frame)}
    labels_per_text = frame.groupby("text").label_id.nunique()
    conflicted = set(labels_per_text[labels_per_text > 1].index)
    audit["conflicting_label_rows"] = int(frame.text.isin(conflicted).sum())
    frame = frame[~frame.text.isin(conflicted)]
    before = len(frame)
    frame = frame.drop_duplicates("text")
    audit["duplicate_rows"] = before - len(frame)
    return frame.reset_index(drop=True), audit


def near_duplicates(candidates, protected, threshold=NEAR, chunk=8000):
    """Boolean mask over candidates: word-set Jaccard >= threshold with any protected text."""
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(token_pattern=TOKEN, binary=True, lowercase=True).fit(list(protected))
    p = vec.transform(protected).astype(np.int32)
    p_sizes = np.asarray(p.sum(1)).ravel()
    # Words outside the protected vocabulary can only lower Jaccard, so count them in the size.
    analyzer = CountVectorizer(token_pattern=TOKEN, lowercase=True).build_analyzer()
    c_sizes = np.array([len(set(analyzer(t))) for t in candidates])
    c = vec.transform(candidates).astype(np.int32)
    mask = np.zeros(len(candidates), dtype=bool)
    pt = p.T.tocsc()
    for start in range(0, len(candidates), chunk):
        inter = (c[start:start + chunk] @ pt).toarray()
        union = c_sizes[start:start + chunk, None] + p_sizes[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            jac = np.where(union > 0, inter / union, 0.0)
        mask[start:start + chunk] = (jac >= threshold).any(1)
    return mask


def build(gold, path):
    """Silver training rows plus an audit. gold: the prepared frame with text, label_id, split."""
    raw = open(path, "rb").read()
    silver, audit = read_silver(raw.decode("utf-8").splitlines())
    audit = {"source": f"{REPO_ID}@{REVISION}/{FILENAME}", "source_sha256": hashlib.sha256(raw).hexdigest(), **audit}
    by_text = gold.set_index("text")
    exact = silver.text.isin(by_text.index)
    for split in ["train", "validation", "test"]:
        audit[f"exact_overlap_{split}"] = int(silver.text.isin(gold.text[gold.split == split]).sum())
    # Label agreement is reported on train overlaps only; held-out labels are not inspected.
    tr = silver[silver.text.isin(gold.text[gold.split == "train"])]
    if len(tr):
        audit["label_agreement_on_train_overlap"] = float((tr.label_id.to_numpy() == by_text.loc[tr.text, "label_id"].to_numpy()).mean())
    silver = silver[~exact]
    held = gold.text[gold.split != "train"].tolist()
    near = near_duplicates(silver.text.tolist(), held)
    audit["near_duplicate_of_validation_or_test"] = int(near.sum())
    silver = silver[~near].sort_values("text").reset_index(drop=True)
    audit["retained_rows"] = len(silver)
    audit["class_counts"] = {k: int((silver.label_id == v).sum()) for k, v in LABEL_TO_ID.items()}
    audit["retained_text_sha256"] = hashlib.sha256("\n".join(silver.text).encode("utf-8")).hexdigest()
    return silver, audit
