<p align="center"><img src="assets/sentiment-hero.svg" width="100%" alt="Malay–English Sentiment Analysis — Transformer NLP" /></p>

<p align="center"><strong>Multilingual text classification · Social-comment analytics · Explainability</strong></p>
<p align="center"><a href="https://huggingface.co/spaces/SiobhanGoh/emotion-detection-malaybert-skeleton">Hugging Face demo</a> · <a href="#corrected-evaluation">Evaluation</a> · <a href="#reproduce-the-comparison">Reproduce</a> · <a href="#run-the-app">Run the app</a></p>

## Overview

A **2025–26 degree capstone** combining Malay–English sentiment modelling with a Streamlit app for single-text predictions, optional YouTube comment analysis and downloadable reports. Predictions are **Negative / Neutral / Positive**. Existing filenames say “emotion detection”; the implemented task is three-class sentiment polarity.

The evaluation pipeline was repaired in October 2026 to use **one shared, disjoint train/validation/test split** for both TF-IDF logistic regression and multilingual BERT. Historical scores from overlapping splits are superseded. This is a corrected retrospective comparison on the existing corpus, not new external validation.

## Explore the system

| Capability | Implementation |
|---|---|
| Transformer inference | Hugging Face tokenizer and sequence classifier, three logits and softmax |
| Single-text analysis | Sentiment label and class-score distribution |
| Comment analytics | YouTube Data API v3 retrieval or up to 50 pasted comments |
| Reporting | Aggregate sentiment, comment tables, session views and PDF/CSV exports |
| Reproducible modelling | Deterministic preprocessing, duplicate controls, frozen split fingerprints, validation-based selection and common final evaluation |

<p align="center"><img src="assets/architecture.svg" width="100%" alt="Malay English sentiment inference pipeline" /></p>

## Corrected evaluation

Both models were evaluated once on the same **981 test inputs** after validation selection. No settings were changed using these final results.

| Model | Test accuracy | Test macro-F1 | Selection |
|---|---:|---:|---|
| TF-IDF + logistic regression | 66.46% | 0.6553 | C = 10 |
| Fine-tuned mBERT | 76.04% | 0.7557 | Epoch 4 of 4 |

Full class metrics, confusion matrices, row reconciliation, selection history and model checksum: [corrected comparison](verification/corrected_comparison.json). Tokenizer reload consistency was checked on all 4,577 training inputs: [verification](verification/tokenizer_roundtrip.json).

The original CSV contains **6,999 rows**. The corrected deterministic process excludes **148 rows with conflicting labels for identical normalised text** and **312 remaining duplicates**, retaining **6,539 unique inputs**. There are no empty-text or invalid-label exclusions in this source. The counts reconcile exactly: 6,999 − 148 − 312 = 6,539.

| Partition | Negative | Neutral | Positive | Total |
|---|---:|---:|---:|---:|
| Train | 1,864 | 1,193 | 1,520 | 4,577 |
| Validation | 399 | 256 | 326 | 981 |
| Test | 400 | 255 | 326 | 981 |

- Unicode normalisation, case-folding, URL/mention removal, hashtag-marker removal and whitespace collapse are applied before duplicate grouping. No automatic language filter is used. This is a documented replacement of the old preprocessing, not an explanation of its unreconciled 6,448-row output.
- Identical normalised inputs cannot cross partitions. Conflicting labels are excluded rather than guessed. Near-duplicate detection remains future work.
- The shared stratified **70/15/15** split uses seed 42 and a content fingerprint. The training commands use only train and validation partitions; final evaluation compares both selected models on the same 981 test inputs.
- Baseline: TF-IDF unigrams/bigrams, minimum document frequency 2, up to 30,000 features; logistic regression C is selected from 0.1, 1 and 10 using validation macro-F1. Vocabulary is fitted on training data only.
- mBERT: `google-bert/bert-base-multilingual-cased`, maximum length 128, AdamW at 2e-5, weight decay 0.01, 10% warmup, four epochs, effective batch size 16. The checkpoint with the best validation macro-F1 is selected before the final comparison.
- The run records the base-model revision, split fingerprint and resulting model checksum. GPU training can still vary across hardware/software; a seed does not guarantee bit-for-bit identity.

**Interpretation:** The test partition is disjoint within this corrected run, but earlier development used the broader corpus. These results are internal estimates, not a blind new benchmark or proof of deployment performance. Temporal/source shift, code-switched slang, near duplicates and calibration need separate assessment. Softmax scores are not calibrated probabilities of correctness.

## Reproduce the comparison

Use Python 3.12 in a separate training environment. Supply your own permitted CSV with `Tweet` and `Sentiment` columns; raw social text and model weights are deliberately excluded from this repository.

```bash
git clone https://github.com/SiobhanGoh/capstone2-sentiment-streamlit-colab.git
cd capstone2-sentiment-streamlit-colab
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-training.txt
python -m training.run prepare --csv /path/to/cleaned_tweets_sentiments.csv --out runs/data
python -m training.run baseline --data runs/data --out runs/baseline
python -m training.run mbert --data runs/data --out runs/mbert --revision 3f076fdb1ab68d5b2880cb87a0886f315b8146f8
python -m training.run evaluate --data runs/data --baseline runs/baseline --mbert runs/mbert --out runs/comparison.json
```

Prepared partitions and final results reject silent overwrites. Choose a new run directory for a genuinely new experiment, and keep the previous result. Do not tune against the final test result. On Windows, a short `HF_HOME` path can avoid long-cache-path errors. CUDA is optional but substantially faster; install the appropriate official PyTorch build for your hardware.

The [Colab notebook](colab_notebooks/CP2_malaybert_training.ipynb) runs the same pipeline. GitHub Actions runs 32 protocol and app checks without downloading model weights or the private corpus. Install the app requirements plus scikit-learn, joblib, pytest and pypdf, then run `python -m pytest -q tests`.

## App and model status

The corrected checkpoint was deployed on **23 October 2026**. The app pins model revision `9bc6645ce7a8e02761c59386ffd16969bed1bc55`, subfolder `corrected-2026-10`, in `SiobhanGoh/CP2_Emotion_Model_New`. It verifies weights SHA-256 `771a19b8c14865f8d8d8c7adc7df95d859b08a866195f5b7bfe3ac44723ec61b` before loading, checks label order and uses the same normalisation and 128-token limit as the corrected training pipeline. The running app exposes the effective revision and checksum.

Public checks confirmed a single-text prediction, retrieval and analysis of 30 real YouTube comments, and CSV/PDF downloads. This establishes feature operation, not predictive accuracy on YouTube. The app rejects empty cleaned input, excludes unusable comments, clears stale results, handles retrieval failures explicitly and supports pasted comments without an API key. Session history remains per session; reruns do not duplicate predictions.

The summaries show sentiment counts and class scores. Unvalidated emotion-intensity, engagement and business-risk scores have been removed. Softmax values are labelled as uncalibrated model outputs. PDFs paginate and escape comment text; characters unsupported by the PDF font use explicit names/codes, while CSV preserves original text and neutralises spreadsheet-formula prefixes.

Live exploratory checks also found errors: the straightforward negative Malay sentence “Servis lambat dan sangat mengecewakan. 😭” was predicted Positive (model score 0.8927), and “Kedai dibuka pada pukul sembilan.” was predicted Positive (0.8190). These hand-written probes are not an accuracy benchmark, but they demonstrate that the internal test score does not establish reliable everyday-domain behaviour. Do not treat the demonstration as a production-ready general sentiment service.

![Public single-text demo using the corrected checkpoint, 23 October 2026](assets/app-live.png)

Screenshot: a functional positive example, not accuracy evidence. [Deployment file verification](verification/deployment_source.json) records the exact deployed Space revision and file checksums. [Feature checks](verification/app_feature_checks.json) separates public browser checks, automated checks and observed prediction limitations.

## Run the app

The app uses Python 3.12 with pinned dependencies. The Docker Space installs the CPU build of PyTorch. From the repository root:

```bash
cd hf_space_streamlit
python -m venv .venv
# Activate this environment, then:
python -m pip install -r requirements.txt
python -m streamlit run src/streamlit_app.py
```

`src/model_release.json` fixes the model revision, subfolder, checksum and evaluation provenance. `SENTIMENT_MODEL_PATH` can point to a local copy of the same checkpoint for local testing; the checksum is still enforced. `YOUTUBE_API_KEY` enables comment retrieval and must be stored privately as an environment variable or Hugging Face Space secret. Never commit it. Paste comments works without a key.

## Explainability and original work

The original exploration contains LIME wrappers for the baseline and mBERT. SHAP LinearExplainer covers the **TF-IDF logistic-regression baseline only**. These explanation experiments have not been rerun for the corrected checkpoint; no transformer SHAP or explanation-fidelity claim is made.

The corrected notebook replaces the old independent splits. Original CP2 code is kept in [an explicitly disabled historical notebook](colab_notebooks/archive/CP2_original_exploration.ipynb), with all saved outputs removed. Original outputs remain in Git history and should not be used as valid accuracy evidence. CP1 is historical exploratory work; [the notebook guide](colab_notebooks/README.md) explains the distinction.

## Repository map

```text
training/                 # Shared preprocessing, selection and final evaluation
requirements-training.txt # Tested training dependencies
tests/                    # Protocol, inference helpers, exports and app checks
verification/             # Aggregate results; no raw social text
colab_notebooks/           # Corrected workflow and historical source
hf_space_streamlit/       # Verified deployed app source and pinned environment
assets/                   # Architecture and real app screenshot
```

## Remaining work

- Assess near-duplicates, source/time-based generalisation and probability calibration.
- Evaluate a new external sample before making broader deployment-performance claims.
- Rerun explainability experiments for the corrected checkpoint if needed; historical explanations do not describe this release.

Scope: sentiment analysis and social-comment analytics; no clinical inference, validated fine-grained emotion detection or live Twitter streaming.

---

**Siobhan Goh** · Master of Artificial Intelligence, Monash University Malaysia · [LinkedIn](https://www.linkedin.com/in/siobhan-goh/)
