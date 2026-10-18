<p align="center"><img src="assets/sentiment-hero.svg" width="100%" alt="Malay–English Sentiment Analysis — Transformer NLP" /></p>

<p align="center"><strong>Multilingual text classification · Social-comment analytics · Explainability</strong></p>
<p align="center">
<a href="https://huggingface.co/spaces/SiobhanGoh/emotion-detection-malaybert-skeleton">Hugging Face demo</a> · <a href="#explore-the-system">Explore the system</a> · <a href="#modelling-and-interpretability">Methods</a> · <a href="#run-locally">Run locally</a>
</p>

## Overview

An applied **Malay–English sentiment analysis** capstone joining NLP modelling with an interactive **Streamlit** experience. Users can classify text as **Negative / Neutral / Positive**, review aggregate comment sentiment from an optional YouTube integration and export summaries for further analysis.

> **Terminology:** Existing application and model filenames use “emotion detection”. The inspected implementation is a **three-class sentiment-polarity classifier**, not a validated multi-emotion or mental-health diagnostic model.

> **Evaluation status:** Historic test scores are withheld from the project summary pending a corrected, disjoint data split. [Technical limitations](#modelling-and-interpretability) are documented below.

## Explore the system

| Capability | What is implemented |
|---|---|
| **Transformer inference** | Hugging Face tokenizer + sequence-classification model, three logits and softmax |
| **Single-text UI** | Enter text, inspect predicted sentiment and class-score distribution |
| **Video-comment analysis** | YouTube Data API v3 ingestion when the authorised API key is configured |
| **Insight reporting** | Comment tables, aggregate distributions, session views and Matplotlib-based PDF exports |
| **Training notebooks** | Baseline TF-IDF/logistic regression, mBERT training, LIME and baseline SHAP explanations |

The public Hugging Face Space was observed returning a prediction for its built-in positive example on **18 October 2026**. The YouTube retrieval/export path exists in source but was not fully exercised during that live check.

<p align="center"><img src="assets/architecture.svg" width="100%" alt="Malay English sentiment inference pipeline" /></p>

## Modelling and interpretability

The project uses social text with Malay–English code-switching, informal language and three sentiment labels. The training notebook documents a **TF-IDF + Logistic Regression** baseline and **`bert-base-multilingual-cased`** fine-tuning, using PyTorch and the Hugging Face Transformers library.

<details>
<summary><strong>Inspect the modelling approach</strong></summary>

- Token and text normalisation, removal of obvious URLs/mentions/noise and label harmonisation.
- Stratified dataset splitting appears in both baseline and mBERT sections, but the splits were independently created and **do not yield a guaranteed shared disjoint test**.
- mBERT sequence classification uses three output classes, token length 128, AdamW and warmup in the saved notebook.
- The deployed app obtains a model from `SiobhanGoh/CP2_Emotion_Model_New` via `MODEL_ID` and supports CPU or CUDA inference.

The specific deployment revision was not cryptographically matched to the audited training snapshot.
</details>

<details>
<summary><strong>What the explanations actually cover</strong></summary>

- **LIME:** prediction wrappers for the baseline and the multilingual BERT classifier.
- **SHAP LinearExplainer:** global and local n-gram contributions for the **TF-IDF logistic-regression baseline only**.
- No verified SHAP explanations of the transformer itself or measured explanation fidelity.
</details>

<details>
<summary><strong>Evaluation transparency</strong></summary>

The original clean-text CSV contains **6,999 rows**, but the saved modelling output implies **6,448**, without a fully reconciled row-level transformation record. The mBERT notebook also independently resplits data from a population containing the baseline's test rows, then evaluates against that baseline test selection. This compromises an independent comparison. Observed scores are retained in archived source outputs for reproducibility, but should **not be promoted as trusted unseen-data generalisation** until a consistent, disjoint split is rebuilt and retested.

Model output probabilities are not calibrated probabilities of real-world correctness. Code-switched slang, domain shift, class imbalance and ambiguous neutral content warrant separate assessment.
</details>

## Repository map

```text
colab_notebooks/
  CP1_training.ipynb
  CP2_malaybert_training.ipynb
  README.md
hf_space_streamlit/
  src/streamlit_app.py
  requirements.txt
  Dockerfile
  README.md              # Existing Space configuration metadata
  hf.yaml
README.md
LICENSE
```

## Run locally

```bash
git clone https://github.com/SiobhanGoh/capstone2-sentiment-streamlit-colab.git
cd capstone2-sentiment-streamlit-colab/hf_space_streamlit
python -m venv .venv
# Activate your environment (Windows: .venv\Scripts\activate; macOS/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt
python -m streamlit run src/streamlit_app.py
```

The default inference checkpoint is set by the `MODEL_ID` environment variable. The source defaults to `SiobhanGoh/CP2_Emotion_Model_New`. The YouTube feature additionally requires a privately configured `YOUTUBE_API_KEY`; **never commit credentials to this repository**.

**Environment note:** The existing dependency versions are comparatively old; compatibility and deployment method should be verified before claiming one-command reproducibility on current Python versions. The repository contains a Dockerfile and Hugging Face Streamlit SDK metadata whose active relationship requires a hosting check.

## Limitations and roadmap

- [ ] Rebuild consistent data preprocessing and create one fully disjoint train/validation/test protocol.
- [ ] Re-evaluate baseline and transformer on the same untouched test partition.
- [ ] Identify the deployed model checkpoint/revision and align its README with source.
- [ ] Test YouTube API and PDF/CSV export in the deployed environment.
- [ ] Add privacy-safe, real application screenshots and a short demo recording.

**Scope:** Applied sentiment analysis and audience-insight prototyping; not medical inference, fine-grained emotion recognition or live Twitter streaming.

---

**Siobhan Goh** · Master of Artificial Intelligence, Monash University Malaysia · [LinkedIn](https://www.linkedin.com/in/siobhangoh/)