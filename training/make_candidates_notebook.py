"""Writes colab_notebooks/CP3_candidate_models.ipynb. Run: python training/make_candidates_notebook.py"""
import json
from pathlib import Path

BRANCH = "browser-model"
SOURCE_SHA = "3ff66de7b91cb1b2fd06ac5dd136f495859114808d19b1688c66a21ba1d74d77"
SPLIT_SHA = "692d03d208495810fcd1c7fa85906b2cfc1f4fc18628b6a39998c898770b9929"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Can a Malaysian model beat mBERT?

This notebook fine-tunes three stronger base models under exactly the same corrected protocol as the published mBERT (76.04% accuracy, macro-F1 0.7557 on 981 test inputs):

| Candidate | Why it might do better |
|---|---|
| `mesolitica/malaysian-debertav2-base` | Pretrained on 17B tokens of Malaysian text, including local Malay and Manglish |
| `mesolitica/roberta-base-bahasa-cased` | Pretrained on local Twitter, Instagram, news and IIUM confessions |
| `cardiffnlp/twitter-xlm-roberta-base` | XLM-RoBERTa further pretrained on about 198M tweets |

**The rule, fixed before any training:** each candidate picks its best epoch on validation. The winner is the candidate with the highest validation macro-F1, chosen *before* any test score is computed. Test scores for all candidates are then recorded once, for transparency, and are not used to change the choice.

**How to run:** Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all**. It takes roughly 1–1.5 hours. Finished models are saved to your Google Drive, so if Colab disconnects, run all again and it carries on from where it stopped.

**What to send back:** the `candidates_comparison.json` file the last cell downloads. It holds only scores, never any comment text.
"""),
    md("## 1. Check the GPU and get the code"),
    code(f"""
import subprocess
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
assert gpu.returncode == 0, "No GPU found. Use Runtime → Change runtime type → T4 GPU, then run again."
print("GPU:", gpu.stdout.strip())

import os
if not os.path.exists("/content/capstone2-sentiment-streamlit-colab"):
    !git clone -q --branch {BRANCH} https://github.com/SiobhanGoh/capstone2-sentiment-streamlit-colab.git /content/capstone2-sentiment-streamlit-colab
%cd /content/capstone2-sentiment-streamlit-colab
!git log --oneline -1
%pip install -q -r requirements-training.txt
"""),
    md("## 2. Load your corpus and rebuild the frozen split\nThe file must be the exact CSV used for the corrected evaluation. Its checksum and the split fingerprint are both checked, so the test set is guaranteed to be the same 981 inputs."),
    code(f"""
import hashlib, json
from pathlib import Path
from google.colab import drive, files

drive.mount("/content/drive")
RUNS = Path("/content/drive/MyDrive/rojak-sentiment-runs/candidates_v1")
RUNS.mkdir(parents=True, exist_ok=True)

EXPECTED_SOURCE = "{SOURCE_SHA}"
EXPECTED_SPLIT = "{SPLIT_SHA}"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

candidates = [Path("/content/drive/MyDrive/Notebooks/cleaned_tweets_sentiments.csv"), RUNS / "cleaned_tweets_sentiments.csv"]
DATASET = next((p for p in candidates if p.exists() and sha(p) == EXPECTED_SOURCE), None)
if DATASET is None:
    print("Couldn't find the corpus on Drive. Choose cleaned_tweets_sentiments.csv:")
    up = files.upload()
    DATASET = Path(next(iter(up)))
assert sha(DATASET) == EXPECTED_SOURCE, "This isn't the CSV used for the corrected evaluation (checksum differs)."
print("Corpus:", DATASET)

from training.protocol import load_prepared, prepare_csv
DATA = RUNS / "data"
if not (DATA / "manifest.json").exists():
    prepare_csv(DATASET, DATA)
frame, manifest = load_prepared(DATA)
assert manifest["split_sha256"] == EXPECTED_SPLIT, "The rebuilt split doesn't match the published one."
print("Split matches the published one:", manifest["split_counts"])
"""),
    md("## 3. Optional: a slang check set\nIf you have used **Wrong label?** in the web app, download *Labels you corrected* as CSV and choose it here. It is scored separately and never used to pick the winner. Press **Cancel** to skip."),
    code("""
CHECK = None
existing = RUNS / "label_corrections.csv"
if existing.exists():
    CHECK = existing
else:
    try:
        up = files.upload()
        if up:
            name = next(iter(up))
            Path(name).rename(existing)
            CHECK = existing
    except Exception:
        pass
print("Check set:", CHECK or "none")
"""),
    md("## 4. Fine-tune the candidates\nEach run prints progress every 100 batches and a validation score after each epoch. Finished candidates are skipped if you run this again."),
    code("""
from training.candidates import CANDIDATES, train_all
for name, spec in CANDIDATES.items():
    print(f"{name}: {spec['base']} @ {spec['revision'][:7]}, lr {spec['learning_rate']}. {spec['why']}")
train_all(DATA, RUNS)
"""),
    md("## 5. Pick the winner, then score every candidate once\nThe winner is fixed by validation macro-F1 first. This cell refuses to run twice into the same file, so a viewed test result can't be quietly replaced."),
    code("""
import pandas as pd
from training.candidates import evaluate_all
OUT = RUNS / "candidates_comparison.json"
report = json.loads(OUT.read_text()) if OUT.exists() else evaluate_all(DATA, RUNS, OUT, CHECK)

rows = [{"model": "mBERT (published)", "validation macro-F1": report["reference_mbert"]["validation_macro_f1"],
         "test accuracy": report["reference_mbert"]["test"]["accuracy"], "test macro-F1": report["reference_mbert"]["test"]["macro_f1"]}]
for name, r in report["candidates"].items():
    if "test" in r:
        row = {"model": name + ("  ← selected" if name == report["selected"] else ""),
               "validation macro-F1": r["validation"]["macro_f1"], "test accuracy": r["test"]["accuracy"],
               "test macro-F1": r["test"]["macro_f1"], "parameters (M)": round(r["parameters"] / 1e6)}
        if "check_set" in r:
            row["check-set accuracy"] = r["check_set"]["accuracy"]
        rows.append(row)
pd.DataFrame(rows).round(4)
"""),
    md("## 6. Download the results file\nSend `candidates_comparison.json` back in the chat. Nothing is published yet; that only happens after we've looked at the numbers together."),
    code("""
files.download(str(OUT))
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).resolve().parents[1] / "colab_notebooks/CP3_candidate_models.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(out)
