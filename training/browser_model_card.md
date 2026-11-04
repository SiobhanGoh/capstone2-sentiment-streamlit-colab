---
language: [ms, en]
license: other
license_name: non-commercial-research
library_name: transformers.js
pipeline_tag: text-classification
base_model: google-bert/bert-base-multilingual-cased
tags: [onnx, sentiment-analysis, malay, code-switching, knowledge-distillation]
---

# Malay–English sentiment (ONNX, for browsers)

Three-class sentiment polarity (negative, neutral, positive) for Malay, English and code-switched social text, from a 2025–26 degree capstone. It runs in a web browser with [transformers.js](https://github.com/huggingface/transformers.js).

Since November 2026 this is a **distilled** model: the capstone's fine-tuned multilingual BERT, retrained to copy a team of three XLM-RoBERTa-large models (about 3.3 GB together) while keeping BERT-base size (356 MB in 16-bit).

- Labels, in order: `NEGATIVE`, `NEUTRAL`, `POSITIVE`. Inputs are cleaned the same way as in training and cut to 128 tokens.
- Earlier version (the original fine-tuned checkpoint, 76.04%): revision `80f942af84c6b05e7e5c7b56a4e7d0bda6d85595` of this repo.

## How it was trained

1. Teacher: XLM-RoBERTa-large fine-tuned on the capstone's 4,577 training comments, three seeds; chosen over nine alternatives by cross-validated macro-F1 inside the validation split.
2. Student: initialised from the original checkpoint (`SiobhanGoh/CP2_Emotion_Model_New`, revision `9bc6645`, subfolder `corrected-2026-10`) and trained on the gold labels plus the teacher's soft labels, over the training comments and 150,276 Malaysian tweets used as unlabelled text ([mesolitica/chatgpt-explain-sentiment](https://huggingface.co/datasets/mesolitica/chatgpt-explain-sentiment); its own ChatGPT labels were not used). Tweets matching any validation or test comment, exactly or nearly (word-set Jaccard ≥ 0.8), were removed; none did.
3. The student and its epoch were chosen on validation; the test split was scored once afterwards.

## Evaluation

The same frozen 981-input test partition as the capstone's corrected comparison.

| Model | Test accuracy | Macro-F1 |
|---|---:|---:|
| Original checkpoint | 76.04% | 0.7557 |
| **This model, 16-bit file** | 79.61% | 0.7943 |
| This model + the TF-IDF baseline (as in the web app) | **80.33%** | **0.8001** |

Against the original checkpoint the web app's combination gets 93 test inputs right that the original gets wrong, and 51 the other way round (McNemar exact p < 0.001); against the previous web app (77.37%) the figures are 78 and 49 (p = 0.013). The 16-bit file gives the same label as the trained PyTorch student on every validation and test input. The 8-bit conversion breaks this model and is not published.

These are retrospective internal results on the existing corpus, not new external validation. Validation guided many choices across the project and the test split had been scored for earlier experiments, so the true gain is probably somewhat smaller than measured. Expect errors on sarcasm, slang, heavy code-switching and topics unlike the training data.

## Licence

Training used text from mesolitica's Malaysian datasets, whose maintainers ask for research, non-commercial use. This model is therefore offered for non-commercial research and education only.

## Use

```js
import { AutoModelForSequenceClassification, AutoTokenizer } from '@huggingface/transformers';
const id = 'SiobhanGoh/malay-english-sentiment-onnx';
const tokenizer = await AutoTokenizer.from_pretrained(id, { revision: '<pinned commit>' });
const model = await AutoModelForSequenceClassification.from_pretrained(id, { revision: '<pinned commit>', dtype: 'fp16' });
```

Apply the project's text cleaning first (Unicode NFKC, case-folding, link and @mention removal, `#` removal, whitespace collapse). Source, training protocol and the web app: https://github.com/SiobhanGoh/capstone2-sentiment-streamlit-colab
