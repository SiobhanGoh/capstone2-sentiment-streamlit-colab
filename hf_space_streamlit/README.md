---
title: Malay English Sentiment Analysis
emoji: 💬
colorFrom: pink
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

Malay–English three-class sentiment demo for a 2025–2026 degree capstone.

Features: single-text predictions, YouTube or pasted comments, sentiment summaries,
CSV/PDF reports, and session history. YouTube retrieval needs a `YOUTUBE_API_KEY`
Space secret; pasted comments work without it.

The app pins the corrected checkpoint and verifies its SHA-256 before loading.
The corrected internal test accuracy is 76.04%, with macro-F1 0.7557 on 981 inputs.
These are retrospective internal results, not new external validation.
Softmax scores are uncalibrated and do not measure emotion intensity or business risk.

Source and evaluation protocol:
https://github.com/SiobhanGoh/capstone2-sentiment-streamlit-colab
