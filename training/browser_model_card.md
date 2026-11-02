---
language: [ms, en]
license: apache-2.0
library_name: transformers.js
pipeline_tag: text-classification
base_model: google-bert/bert-base-multilingual-cased
tags: [onnx, sentiment-analysis, malay, code-switching]
---

# Malay–English sentiment (ONNX, for browsers)

Three-class sentiment polarity (negative, neutral, positive) for Malay, English and code-switched social text. This is an ONNX conversion of the fine-tuned multilingual BERT checkpoint from a 2025–26 degree capstone, made so the model can run in a web browser with [transformers.js](https://github.com/huggingface/transformers.js).

- Source checkpoint: `SiobhanGoh/CP2_Emotion_Model_New`, revision `9bc6645ce7a8e02761c59386ffd16969bed1bc55`, subfolder `corrected-2026-10`, weights SHA-256 `771a19b8c14865f8d8d8c7adc7df95d859b08a866195f5b7bfe3ac44723ec61b`.
- Labels, in order: `NEGATIVE`, `NEUTRAL`, `POSITIVE`. Inputs are cleaned the same way as in training and cut to 128 tokens.
- File hashes for everything here are in `export_manifest.json`.

## Evaluation

Scored on the same frozen 981-input test partition as the capstone's corrected comparison. The split fingerprint and source file checksum were verified before scoring.

{{EVALUATION_TABLE}}

These are retrospective internal results on the existing corpus, not new external validation. Softmax scores are uncalibrated. Expect errors on sarcasm, slang, heavy code-switching and topics unlike the training data.

## Use

```js
import { AutoModelForSequenceClassification, AutoTokenizer } from '@huggingface/transformers';
const id = 'SiobhanGoh/malay-english-sentiment-onnx';
const tokenizer = await AutoTokenizer.from_pretrained(id, { revision: '<pinned commit>' });
const model = await AutoModelForSequenceClassification.from_pretrained(id, { revision: '<pinned commit>', dtype: '{{DTYPE}}' });
```

Apply the project's text cleaning first (Unicode NFKC, case-folding, link and @mention removal, `#` removal, whitespace collapse). Source, training protocol and the web app: https://github.com/SiobhanGoh/capstone2-sentiment-streamlit-colab
