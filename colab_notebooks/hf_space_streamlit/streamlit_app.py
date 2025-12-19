# src/streamlit_app.py
# ============================================================
# Emotion Detection — Malay BERT (Sentiment, HF Spaces Ready)
# Single-text + YouTube comment analytics
# ============================================================
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import textwrap

# ------------------------------------------------------------
# Global Matplotlib Style — Pastel Soft Theme (HF Spaces Safe)
# ------------------------------------------------------------
plt.rcParams.update(
    {
        "axes.edgecolor": "#334155",
        "axes.linewidth": 1.2,
        "axes.labelcolor": "#64748b",
        "axes.titlesize": 13,
        "axes.titlecolor": "#334155",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "grid.color": "#cbd5e1",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
    }
)

# ------------------------------------------------------------
# Page config (must be early)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Emotion Detection — Malay BERT",
    page_icon="💬",
    layout="wide",
)

# ------------------------------------------------------------
# Global CSS (modern pastel dashboard, scoped, HF-safe)
# ------------------------------------------------------------
st.markdown(
    """
<style>
/* Root background */
html, body, [data-testid="stAppViewContainer"], .main {
    background: radial-gradient(circle at top left, #ffe6f7, #f3f1ff 40%, #e7fcff 100%) !important;
}

/* App title / subtitle */
.app-main-title {
    text-align: center;
    font-weight: 800;
    font-size: 2.0rem;
    letter-spacing: -0.02em;
    margin-top: 0.2rem;
    margin-bottom: 0.1rem;
    color: #0f172a;
}

.app-subtitle {
    text-align: center;
    font-size: 0.95rem;
    color: #6b7280;
    margin-bottom: 1.1rem;
}

/* Center tabs */
div[data-testid="stTabs"] > div[role="tablist"] {
    justify-content: center;
}

div[role="tab"] > div[data-testid="stMarkdownContainer"] p {
    font-weight: 600;
}

/* Pill chips for examples */
#emotion-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-bottom: 0.4rem;
}

#emotion-chips button {
    border-radius: 999px;
    padding: 0.40rem 1.0rem;
    min-width: 135px;
    height: 40px;
    font-size: 0.92rem;
    font-weight: 600;
    background: linear-gradient(135deg, #ffdde8, #e0f2fe);
    border: none;
    color: #374151;
    box-shadow: 0 0 0 rgba(255, 182, 193, 0.6);
    transition: all 0.22s ease-in-out;
}

#emotion-chips button:hover {
    transform: translateY(-2px) scale(1.03);
    box-shadow: 0 0 16px rgba(255, 182, 193, 0.65);
    filter: brightness(1.1);
}

/* Primary action: Analyze Sentiment */
#analyze-btn-wrap button,
#analyze-btn-wrap button[kind="secondary"],
#analyze-btn-wrap button[data-testid="baseButton-secondary"],
#analyze-btn-wrap * button,
#analyze-btn-wrap * button[kind="secondary"],
#analyze-btn-wrap * button[data-testid="baseButton-secondary"] {
    border-radius: 999px !important;
    padding: 0.6rem 1.8rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    background: linear-gradient(90deg, #22c55e, #4ade80, #2dd4bf) !important;
    border: none !important;
    box-shadow: 0 0 12px rgba(45,212,191,0.65) !important;
    transition: all 0.25s ease-in-out !important;
}

#analyze-btn-wrap button:hover,
#analyze-btn-wrap button[kind="secondary"]:hover,
#analyze-btn-wrap button[data-testid="baseButton-secondary"]:hover,
#analyze-btn-wrap * button:hover,
#analyze-btn-wrap * button[kind="secondary"]:hover,
#analyze-btn-wrap * button[data-testid="baseButton-secondary"]:hover {
    transform: translateY(-3px) scale(1.05) !important;
    box-shadow: 0 0 26px rgba(45,212,191,0.95) !important;
    filter: brightness(1.15) !important;
}

/* Secondary action (YouTube fetch) */
.secondary-action button[data-testid="baseButton-secondary"],
.secondary-action button[data-testid="baseButton-primary"] {
    border-radius: 999px !important;
    padding: 0.55rem 1.8rem !important;
    font-size: 1.0rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    background: linear-gradient(90deg, #fb7185, #fb923c) !important;
    border: none !important;
    box-shadow: 0 0 12px rgba(251,146,60,0.7) !important;
    transition: all 0.25s ease-in-out !important;
}

.secondary-action button[data-testid="baseButton-secondary"]:hover,
.secondary-action button[data-testid="baseButton-primary"]:hover {
    transform: translateY(-2px) scale(1.03) !important;
    box-shadow: 0 0 22px rgba(251,146,60,0.85) !important;
    filter: brightness(1.10) !important;
}

/* Session action buttons (download / clear) */
#session-actions button[data-testid="baseButton-secondary"],
#session-actions button[data-testid="baseButton-primary"] {
    border-radius: 999px !important;
    padding: 0.50rem 1.4rem !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    border: none !important;
    color: #374151 !important;
    background: linear-gradient(120deg, #fef3c7, #fde68a) !important;
    box-shadow: 0 0 12px rgba(253,230,138,0.55) !important;
    transition: all 0.25s ease-in-out !important;
}

#session-actions button[data-testid="baseButton-secondary"]:hover,
#session-actions button[data-testid="baseButton-primary"]:hover {
    transform: translateY(-2px) scale(1.03) !important;
    box-shadow: 0 0 20px rgba(253,230,138,0.85) !important;
    filter: brightness(1.08) !important;
}

/* Generic button safety */
button[kind="secondary"], button[kind="primary"] {
    transition: all 0.18s ease-in-out;
}

/* Card base */
.card {
    background: rgba(255,255,255,0.98);
    border-radius: 18px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 18px 40px rgba(148,163,184,0.25);
    border: 1px solid rgba(226,232,240,0.9);
}

/* Primary result card */
.primary-result-card {
    background: radial-gradient(circle at top left, #ffe6f7, #e7fcff 60%, #f5f3ff 100%);
}

/* Summary cards for YouTube & advanced metrics */
.summary-card {
    border-radius: 18px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 18px 40px rgba(148,163,184,0.25);
    border: 1px solid rgba(226,232,240,0.9);
    background: #ffffff;
}

.summary-card h4 {
    font-size: 0.90rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.25rem;
    color: #6b7280;
}

.summary-card .value {
    font-size: 1.5rem;
    font-weight: 700;
}

/* Color variants */
.summary-positive {
    background: linear-gradient(135deg, #ffe6f7, #dcfce7);
}

.summary-negative {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
}

.summary-neutral {
    background: linear-gradient(135deg, #e0f2fe, #e5e7eb);
}

.summary-total {
    background: linear-gradient(135deg, #e0f2fe, #f5f3ff);
}

.summary-confidence {
    background: linear-gradient(135deg, #fef3c7, #e0f2fe);
}

/* Standardized heights for cleaner grid layout */
.overview-card {
    height: 260px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.metric-card {
    height: 240px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

/* OVERVIEW CARDS internal spacing */
.overview-card h4 {
    margin-top: 0rem !important;
    margin-bottom: 0.35rem !important;
}

.overview-card .value {
    margin-top: 0.25rem !important;
    margin-bottom: 0.40rem !important;
}

.overview-card p {
    margin-top: 0.40rem !important;
}

/* METRIC CARDS internal spacing */
.metric-card h4 {
    margin-top: 0rem !important;
    margin-bottom: 0.30rem !important;
}

.metric-card .value {
    margin-top: 0.20rem !important;
    margin-bottom: 0.40rem !important;
}

.metric-card p {
    margin-top: 0.40rem !important;
}

/* Explainer cards */
.explainer {
    background: linear-gradient(135deg, #fdf2ff, #e0f2fe 40%, #fefce8);
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    margin-top: 0.8rem;
    box-shadow: 0 10px 28px rgba(148,163,184,0.22);
    border: 1px solid rgba(226,232,240,0.9);
    font-size: 0.95rem;
    color: #4b5563;
    line-height: 1.55;
}

.explainer-title {
    font-size: 1.1rem;
    font-weight: 750;
    margin-bottom: 0.4rem;
    color: #374151;
}

/* Probability table inside explainer */
.prob-card {
    margin-top: 0.9rem;
}

.prob-card table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 0.6rem;
    font-size: 0.92rem;
}

.prob-card th, .prob-card td {
    padding: 0.45rem 0.6rem;
    border-bottom: 1px solid rgba(226,232,240,0.9);
}

.prob-card th {
    text-align: left;
    font-weight: 700;
    color: #4b5563;
}

.prob-card tr:last-child td {
    border-bottom: none;
}

/* DataFrame tweaks */
.stDataFrame thead tr th {
    font-weight: 600 !important;
}

.stDataFrame td {
    line-height: 1.3;
}

/* Footer styling */
.app-footer {
    text-align: center;
    color: #6b7280;
    font-size: 0.90rem;
    margin-top: 2rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Model + label config (Sentiment, HF-safe)
# ------------------------------------------------------------
MODEL_ID = os.getenv("MODEL_ID", "SiobhanGoh/CP2_Emotion_Model_New")

# Default / fallback ordering; may be overwritten by model.config.id2label if available
EMOTION_LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

EMOTION_DISPLAY = {
    "NEGATIVE": "Negative",
    "NEUTRAL": "Neutral",
    "POSITIVE": "Positive",
}

# Pastel colors for charts
EMOTION_COLORS = {
    "NEGATIVE": "#e49aa3",  # pastel rose
    "NEUTRAL": "#9fc9e8",   # pastel blue
    "POSITIVE": "#7cd9a5",  # pastel green
}

# ------------------------------------------------------------
# Cache decorator (works on old + new Streamlit)
# ------------------------------------------------------------
if hasattr(st, "cache_resource"):
    _cache_model = st.cache_resource
else:
    _cache_model = st.cache

# ------------------------------------------------------------
# Load Model (cached, HF Spaces / Docker friendly)
# ------------------------------------------------------------
@_cache_model(show_spinner=True)
def load_model():
    """
    Loads tokenizer + model once per space instance.
    Safe on CPU/GPU and respects HF Spaces caching.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    except Exception:
        st.error(
            f"Unable to load model with id '{MODEL_ID}'. "
            "Please verify that the model exists in this Space or in your Hugging Face account."
        )
        st.stop()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Try to align EMOTION_LABELS with model.config.id2label safely
    global EMOTION_LABELS
    try:
        id2label = getattr(model.config, "id2label", None)
        if isinstance(id2label, dict) and len(id2label) == 3:
            def _to_int(k):
                try:
                    return int(k)
                except Exception:
                    return k

            sorted_keys = sorted(id2label.keys(), key=_to_int)
            labels = [str(id2label[k]).upper() for k in sorted_keys]
            if len(labels) == 3:
                EMOTION_LABELS = labels
    except Exception:
        # Fallback silently to default EMOTION_LABELS
        pass

    return tokenizer, model, device


tokenizer, model, device = load_model()

# ------------------------------------------------------------
# Single-label sentiment prediction helper (softmax)
# ------------------------------------------------------------
def predict_single(text: str):
    """
    Single-label sentiment prediction:
    - Softmax over 3 classes (e.g., NEGATIVE, NEUTRAL, POSITIVE)
    - Returns top label, confidence, and full probability vector.
    """
    if not text or not text.strip():
        return {
            "top_label": None,
            "top_conf": 0.0,
            "probs": np.zeros(len(EMOTION_LABELS), dtype=float),
        }

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # Ensure length alignment
    if len(probs) != len(EMOTION_LABELS):
        probs_fixed = np.zeros(len(EMOTION_LABELS), dtype=float)
        for i in range(min(len(probs), len(EMOTION_LABELS))):
            probs_fixed[i] = probs[i]
        probs = probs_fixed

    top_idx = int(np.argmax(probs))
    top_label = EMOTION_LABELS[top_idx]
    top_conf = float(probs[top_idx])

    return {
        "top_label": top_label,
        "top_conf": top_conf,
        "probs": probs,
    }

# ------------------------------------------------------------
# NEW build_full_report_pdf
# Full multi-page report for Tab 2 + Tab 3 (with Glass Cards)
# ------------------------------------------------------------
def build_full_report_pdf(summary, df_comments):
    """
    Creates a multi-page pastel report that combines:
    - Tab 2 overview + explanation
    - Tab 3 advanced metrics + creator-focused insights
    - Standalone pages for the full comment table
    Uses solid white cards (premium corporate style).
    """

    # Clean UI-ish font
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10

    import numpy as np
    import textwrap
    from matplotlib.patches import FancyBboxPatch

    # --------------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------------

    def _hex_to_rgb_tuple(hex_color: str):
        """Convert #rrggbb → (r,g,b)."""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def _add_gradient_background(ax):
        """Soft pastel gradient (pink → lavender → aqua)."""
        top = _hex_to_rgb_tuple("#ffe6f7")
        mid = _hex_to_rgb_tuple("#f3f1ff")
        bottom = _hex_to_rgb_tuple("#e7fcff")

        n = 512
        grad_top_mid = np.linspace(top, mid, n // 2)
        grad_mid_bottom = np.linspace(mid, bottom, n // 2)
        gradient = np.vstack([grad_top_mid, grad_mid_bottom])
        gradient = gradient.reshape(gradient.shape[0], 1, 3)

        ax.imshow(
            gradient,
            extent=[0, 1, 0, 1],
            origin="lower",
            aspect="auto",
            zorder=0,
        )

    def _draw_wrapped(ax, text, x, y, max_chars, line_h, fontsize=10, color="#111827", weight=None):
        """Draw wrapped text block and return updated y-coordinate."""
        for line in textwrap.wrap(text, max_chars):
            ax.text(
                x, y, line,
                fontsize=fontsize,
                color=color,
                ha="left",
                va="top",
                fontweight=weight if weight else "normal",
                zorder=3,
            )
            y -= line_h
        return y

    # --------------------------------------------------------------
    # Solid White Premium Card
    # --------------------------------------------------------------
    def _white_card(ax, x, y, w, h):
        """
        Draw a premium-style white card:
        - Solid white
        - Soft grey border #e5e7eb
        - Gentle shadow
        """
        # Main white rectangle
        card = FancyBboxPatch(
            (x, y - h), w, h,
            boxstyle="round,pad=0.02, rounding_size=0.03",
            linewidth=1,
            edgecolor=(0.90, 0.92, 0.96, 1.0),  # #e5e7eb
            facecolor=(1, 1, 1, 1.0),
            zorder=2,
        )
        ax.add_patch(card)

        # Slight drop shadow
        shadow = FancyBboxPatch(
            (x + 0.003, y - h - 0.003), w, h,
            boxstyle="round,pad=0.02, rounding_size=0.03",
            linewidth=0,
            facecolor=(0, 0, 0, 0.10),
            zorder=1,
        )
        ax.add_patch(shadow)

    # --------------------------------------------------------------
    # UNPACK SUMMARY
    # --------------------------------------------------------------
    meta = summary["meta"]
    total_sampled = summary["total_sampled"]
    dom_label = summary["dominant_label"]
    dom_upper = str(dom_label).upper()
    dom_pct = summary["dominant_pct"]
    avg_conf = summary["avg_conf"]

    pos_share = summary["pos_share"]
    neg_share = summary["neg_share"]
    neu_share = summary["neu_share"]
    comment_count = meta.get("comment_count", None)

    # --------------------------------------------------------------
    # METRIC CALCULATIONS (MATCH TAB 3 LOGIC)
    # --------------------------------------------------------------
    # Sentiment Severity
    severity_raw = float(neg_share * avg_conf)
    severity_score = round(severity_raw * 100, 1)

    if severity_raw >= 0.45:
        severity_level = "Very High"
    elif severity_raw >= 0.30:
        severity_level = "High"
    elif severity_raw >= 0.15:
        severity_level = "Moderate"
    elif severity_raw > 0:
        severity_level = "Low"
    else:
        severity_level = "None"

    # Risk Score
    risk_score = round(severity_raw * 100, 1)
    if neg_share >= 0.40 and avg_conf >= 0.60:
        risk_band = "High Risk"
    elif neg_share >= 0.25:
        risk_band = "Moderate Risk"
    elif neg_share > 0:
        risk_band = "Low Risk"
    else:
        risk_band = "Minimal Risk"

    # Engagement Heat
    emotional_share = pos_share + neg_share
    if comment_count and comment_count > 0:
        coverage = min(1.0, total_sampled / comment_count)
    else:
        coverage = 0.0 if total_sampled == 0 else 1.0

    heat_raw = 0.6 * emotional_share + 0.4 * coverage
    heat_score = round(heat_raw * 100, 1)

    if heat_raw >= 0.70:
        heat_level = "Very Hot"
    elif heat_raw >= 0.45:
        heat_level = "Warm"
    elif heat_raw > 0:
        heat_level = "Mild"
    else:
        heat_level = "Cold"

    # Emotion Balance Index
    ebi_raw = float(pos_share - neg_share)
    ebi_score = round(ebi_raw * 100, 1)

    if ebi_raw > 0.15:
        ebi_label = "Leaning Positive"
    elif ebi_raw < -0.15:
        ebi_label = "Leaning Negative"
    elif abs(ebi_raw) <= 0.05:
        ebi_label = "Balanced / Mixed"
    else:
        ebi_label = "Slightly Tilted"

    # --------------------------------------------------------------
    # RECOMMENDATIONS LOGIC (MATCH TAB 3)
    # --------------------------------------------------------------
    rec = []

    if risk_band == "High Risk":
        rec.append("Address concerns directly and provide clarifications.")
        rec.append("Post an explanatory follow-up or pinned comment.")
    elif risk_band == "Moderate Risk":
        rec.append("Monitor negative themes and prepare responses.")
    elif risk_band == "Low Risk":
        rec.append("No urgent action needed — continue monitoring.")

    if dom_upper == "NEGATIVE":
        rec.append("Respond empathetically to repeated concerns.")
    if dom_upper == "POSITIVE":
        rec.append("Leverage positive momentum to strengthen community loyalty.")

    if heat_level in ["Very Hot", "Warm"]:
        rec.append("Engage actively with viewers through replies and likes.")
    else:
        rec.append("Encourage engagement using calls-to-action.")

    if "Negative" in ebi_label:
        rec.append("Investigate root causes of dissatisfaction.")
    elif "Positive" in ebi_label:
        rec.append("Focus on well-received content themes.")

    # --------------------------------------------------------------
    # START PDF BUFFER
    # --------------------------------------------------------------
    buf = BytesIO()

    with PdfPages(buf) as pdf:

        # ============================================================
        # PAGE 1 — Video Context + Core Sentiment (Tab 2)
        # ============================================================
        fig1 = plt.figure(figsize=(8.27, 11.69))
        ax1 = fig1.add_axes([0, 0, 1, 1])
        ax1.axis("off")
        _add_gradient_background(ax1)

        ax1.text(
            0.5, 0.95, "YouTube Emotion Insights — Full Report",
            fontsize=22, fontweight="bold", color="#0f172a",
            ha="center", va="top", zorder=5
        )

        y = 0.84
        line_h = 0.028

        # --- WHITE CARD: Video Context ---
        _white_card(ax1, x=0.05, y=0.84, w=0.90, h=0.22)

        ax1.text(0.07, y, "Video Context (Tab 2)",
                 fontsize=13, fontweight="bold", color="#0f172a")
        y -= 0.04

        ctx_lines = [
            f"Title: {meta['title']}",
            f"Channel: {meta['channel']}",
            f"Published: {meta['published_at']}",
            f"Total public comments: {comment_count}",
            f"Sampled comments analysed: {total_sampled}",
        ]
        for line in ctx_lines:
            y = _draw_wrapped(ax1, line, 0.07, y, 90, line_h)

        y -= 0.03

        # --- WHITE CARD: Core Sentiment Summary ---
        _white_card(ax1, x=0.05, y=y+0.04, w=0.90, h=0.22)
        ax1.text(0.07, y, "Core Sentiment Summary (Tab 2)",
                 fontsize=13, fontweight="bold", color="#0f172a")
        y -= 0.04

        summary_lines = [
            f"Dominant sentiment: {dom_upper} ({dom_pct*100:.1f}%)",
            f"Average model confidence: {avg_conf:.2f}",
            f"Positive share: {pos_share*100:.1f}%",
            f"Neutral share: {neu_share*100:.1f}%",
            f"Negative share: {neg_share*100:.1f}%",
        ]
        for line in summary_lines:
            y = _draw_wrapped(ax1, line, 0.07, y, 95, line_h)

        pdf.savefig(fig1)
        plt.close(fig1)

        # ============================================================
        # PAGE 2 — Emotional Climate Explanation (Tab 2)
        # ============================================================
        fig2 = plt.figure(figsize=(8.27, 11.69))
        ax2 = fig2.add_axes([0, 0, 1, 1])
        ax2.axis("off")
        _add_gradient_background(ax2)

        ax2.text(
            0.5, 0.95,
            "Understanding the Emotional Climate (Tab 2)",
            fontsize=18, fontweight="bold", color="#0f172a",
            ha="center", va="top"
        )

        y = 0.88
        para1 = (
            "This section summarises how people feel in the comment section. "
            "Instead of manually reading hundreds of comments, the model automatically "
            "identifies whether each comment is Positive, Neutral, or Negative."
        )
        y = _draw_wrapped(ax2, para1, 0.07, y, 95, 0.028)
        y -= 0.02

        # --- WHITE CARD: Dominant Sentiment ---
        _white_card(ax2, x=0.05, y=y+0.02, w=0.90, h=0.14)

        y = _draw_wrapped(
            ax2, f"Dominant sentiment: {dom_upper} ({dom_pct*100:.1f}%).",
            0.07, y, 95, 0.028, fontsize=11, weight="bold", color="#0f172a"
        )
        y -= 0.01

        for b in [
            "Positive — viewers enjoyed or supported the content.",
            "Negative — users expressed frustration or criticism.",
            "Neutral — comments are factual, short, or spam-like.",
        ]:
            y = _draw_wrapped(ax2, "• " + b, 0.07, y, 95, 0.028)

        y -= 0.02

        # --- WHITE CARD: P/N/N Mix ---
        _white_card(ax2, x=0.05, y=y+0.04, w=0.90, h=0.26)

        ax2.text(0.07, y,
                 "How to read the Positive–Neutral–Negative mix:",
                 fontsize=11, fontweight="bold", color="#0f172a")
        y -= 0.035

        for b in [
            "High Positive → strong approval and supportive community.",
            "High Negative → potential dissatisfaction or controversy.",
            "High Neutral → many timestamp or emotionless comments.",
            "Balanced mix → diverse reactions; discussion may be intense.",
        ]:
            y = _draw_wrapped(ax2, "• " + b, 0.07, y, 95, 0.028)

        y -= 0.02

        # --- WHITE CARD: Average Confidence ---
        _white_card(ax2, x=0.05, y=y+0.04, w=0.90, h=0.22)

        ax2.text(0.07, y,
                 f"Average confidence score: {avg_conf:.2f}",
                 fontsize=11, fontweight="bold", color="#0f172a")
        y -= 0.035

        for b in [
            "High (0.75–1.00) → strong, expressive emotion.",
            "Medium (0.50–0.74) → mild emotion or unclear tone.",
            "Low (<0.50) → ambiguous, short, or code-switched comments.",
        ]:
            y = _draw_wrapped(ax2, "• " + b, 0.07, y, 95, 0.028)

        pdf.savefig(fig2)
        plt.close(fig2)

        # ============================================================
        # PAGE 3 — Advanced Metrics Explanation (Tab 3)
        # ============================================================
        fig3 = plt.figure(figsize=(8.27, 11.69))
        ax3 = fig3.add_axes([0, 0, 1, 1])
        ax3.axis("off")
        _add_gradient_background(ax3)

        ax3.text(
            0.5, 0.95,
            "Advanced Metrics for Creators (Tab 3)",
            fontsize=18, fontweight="bold", color="#0f172a",
            ha="center", va="top"
        )

        y = 0.88
        line_h = 0.028

        sections = [
            (
                "Sentiment Severity",
                f"Severity = Negative Share × Avg Confidence. Measures how clear the negativity is. "
                f"Current level: {severity_level} (score {severity_score}).",
            ),
            (
                "Risk Score",
                f"Risk Score = Severity × 100. Translates severity into a 0–100 scale. "
                f"Current classification: {risk_band} (score {risk_score}).",
            ),
            (
                "Engagement Heat",
                f"Heat = 0.6 × Emotional Share + 0.4 × Coverage. Shows emotional activity. "
                f"Current heat: {heat_level} (score {heat_score}).",
            ),
            (
                "Emotion Balance Index (EBI)",
                f"EBI = (Positive – Negative) × 100. Shows whether conversation is supportive or critical. "
                f"Current leaning: {ebi_label} (score {ebi_score}).",
            ),
        ]

        for title, text in sections:

            _white_card(ax3, x=0.05, y=y+0.04, w=0.90, h=0.18)

            ax3.text(
                0.07, y, title,
                fontsize=12, fontweight="bold", color="#0f172a"
            )
            y -= 0.032

            y = _draw_wrapped(ax3, text, 0.07, y, 95, line_h)
            y -= 0.02

        pdf.savefig(fig3)
        plt.close(fig3)

        # ============================================================
        # PAGE 4 — Sentiment Mix Chart + Interpretation + Recommendations
        # ============================================================
        fig4 = plt.figure(figsize=(8.27, 11.69))
        ax4 = fig4.add_axes([0, 0, 1, 1])
        ax4.axis("off")
        _add_gradient_background(ax4)

        ax4.text(
            0.5, 0.95,
            "Sentiment Mix & Actionable Guidance (Tabs 2 & 3)",
            fontsize=18, fontweight="bold", color="#0f172a",
            ha="center", va="top"
        )

        # Chart (top half) — same as Tab 3 visual
        chart_ax = fig4.add_axes([0.12, 0.55, 0.76, 0.30])
        sentiments = ["Positive", "Neutral", "Negative"]
        shares = [pos_share, neu_share, neg_share]
        colors = ["#7cd9a5", "#9fc9e8", "#e49aa3"]

        bars = chart_ax.bar(sentiments, shares, color=colors)

        for bar, v in zip(bars, shares):
            chart_ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(1.02, (v + 0.03 if v > 0.02 else 0.05)),
                f"{v * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#334155",
            )

        chart_ax.set_ylim(0, 1.05)
        chart_ax.set_ylabel("Proportion of comments")
        chart_ax.set_title("Sentiment Mix Among Analysed Comments", fontweight="bold")
        chart_ax.grid(axis="y", alpha=0.3)
        chart_ax.set_axisbelow(True)

        # Lower metric explanation
        y = 0.48
        line_h = 0.026

        # WHITE CARD: Interpretation
        _white_card(ax4, x=0.05, y=y+0.04, w=0.90, h=0.32)

        ax4.text(0.07, y,
                 "How to interpret these metrics:",
                 fontsize=11, fontweight="bold", color="#0f172a")
        y -= 0.032

        for b in [
            "High Positive → strong approval and support.",
            "High Negative → dissatisfaction or controversy.",
            "High Neutral → emotionless or short comments.",
            f"Current risk band: {risk_band}.",
            f"Engagement Heat: {heat_score}.",
        ]:
            y = _draw_wrapped(ax4, "• " + b, 0.07, y, 95, line_h)

        y -= 0.02

        # WHITE CARD: Recommendations
        _white_card(ax4, x=0.05, y=y+0.04, w=0.90, h=0.28)

        ax4.text(0.07, y,
                 "Recommended actions for creators / stakeholders:",
                 fontsize=11, fontweight="bold", color="#0f172a")
        y -= 0.032

        for b in rec:
            y = _draw_wrapped(ax4, "• " + b, 0.07, y, 95, line_h)

        pdf.savefig(fig4)
        plt.close(fig4)

        # ============================================================
        # PAGE 5+ — COMMENT TABLE (NO CARDS) — Tab 2 comment-level
        # ============================================================
        df_clean = df_comments.copy()

        if "No." in df_clean.columns:
            df_clean = df_clean.set_index("No.")
        else:
            df_clean.index = range(1, len(df_clean) + 1)
            df_clean.index.name = "No."

        indices = df_clean.index.tolist()
        total_rows = len(indices)
        idx = 0
        line_h = 0.024

        while idx < total_rows:

            fig_t = plt.figure(figsize=(8.27, 11.69))
            ax_t = fig_t.add_axes([0, 0, 1, 1])
            ax_t.axis("off")
            _add_gradient_background(ax_t)

            ax_t.text(
                0.5, 0.95,
                "Detailed Comment-Level Sentiment (Tab 2)",
                fontsize=18, fontweight="bold", color="#0f172a",
                ha="center", va="top"
            )

            y = 0.88

            while idx < total_rows and y > 0.12:

                no = indices[idx]
                row = df_clean.loc[no]
                idx += 1

                comment = row.get("Comment", "")
                primary = row.get("Primary Emotion", "")
                conf = float(row.get("Confidence", 0.0))

                header_text = f"{no}. Sentiment: {primary} | Confidence: {conf:.2f}"

                wrapped = textwrap.wrap(str(comment), 110)
                needed_lines = 1 + len(wrapped)
                needed_height = needed_lines * line_h + 0.01

                if y - needed_height < 0.10:
                    # Not enough space on this page, roll back index and break
                    idx -= 1
                    break

                # Header
                ax_t.text(
                    0.07, y, header_text,
                    fontsize=10, fontweight="bold", color="#111827"
                )
                y -= line_h

                # Comment lines
                for line in wrapped:
                    ax_t.text(0.09, y, line, fontsize=9, color="#111827")
                    y -= line_h

                y -= line_h * 0.5

            pdf.savefig(fig_t)
            plt.close(fig_t)

    buf.seek(0)
    return buf.getvalue()

# ------------------------------------------------------------
# Helper: YouTube tools
# ------------------------------------------------------------
def extract_video_id(url: str):
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    if "youtube.com/watch" in url and "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    return None


def _safe_get(url: str, params: dict, timeout: int = 10):
    """
    Safe wrapper around requests.get for HF Spaces.
    Returns parsed JSON or None on failure.
    """
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def fetch_youtube_data(video_url: str, max_comments: int):
    """
    Uses YouTube Data API v3.
    In HF Spaces, set YOUTUBE_API_KEY in Secrets.
    Returns (video_details_dict, list_of_comment_texts) or (None, None) on error.
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        st.error("YouTube API key is not set (YOUTUBE_API_KEY).")
        return None, None

    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("Could not detect a valid YouTube video ID.")
        return None, None

    base = "https://www.googleapis.com/youtube/v3"

    # video details
    params_details = {
        "part": "snippet,statistics",
        "id": video_id,
        "key": api_key,
    }
    data_details = _safe_get(f"{base}/videos", params_details)
    if not data_details or not data_details.get("items"):
        st.error("Unable to fetch video details. Please check the URL or API key.")
        return None, None

    item = data_details["items"][0]
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})

    details = {
        "video_id": video_id,
        "title": snippet.get("title", "Not available"),
        "channel": snippet.get("channelTitle", "Not available"),
        "published_at": snippet.get("publishedAt", "Not available"),
        "thumbnail_url": snippet.get("thumbnails", {})
        .get("high", {})
        .get("url", snippet.get("thumbnails", {}).get("default", {}).get("url", "")),
        "comment_count": int(stats.get("commentCount", 0)) if stats.get("commentCount") else None,
    }

    # comments
    comments = []
    page_token = None
    remaining = max_comments

    while remaining > 0:
        params_comments = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(50, remaining),
            "order": "relevance",
            "textFormat": "plainText",
            "key": api_key,
        }
        if page_token:
            params_comments["pageToken"] = page_token

        data_comments = _safe_get(f"{base}/commentThreads", params_comments)
        if not data_comments:
            break

        for item in data_comments.get("items", []):
            try:
                text_c = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            except KeyError:
                continue
            comments.append(text_c)
            remaining -= 1
            if remaining <= 0:
                break

        page_token = data_comments.get("nextPageToken")
        if not page_token:
            break

    return details, comments

# ------------------------------------------------------------
# Table styling helpers
# ------------------------------------------------------------
def _sentiment_color(label: str) -> str:
    lab = str(label).upper()
    if lab == "POSITIVE":
        return "#dcfce7"
    if lab == "NEGATIVE":
        return "#fee2e2"
    return "#e0f2fe"


def style_comment_table(df):
    def row_color(row):
        color = _sentiment_color(row.get("Primary Emotion", "NEUTRAL"))
        return [f"background-color: {color}"] * len(row)

    return (
        df.style.apply(row_color, axis=1)
        .set_properties(subset=["Comment"], **{"white-space": "normal"})
        .set_properties(**{"font-size": "0.9rem"})
    )


def style_session_table(df):
    def row_color(row):
        color = _sentiment_color(row.get("Primary Emotion", "NEUTRAL"))
        return [f"background-color: {color}"] * len(row)

    return (
        df.style.apply(row_color, axis=1)
        .set_properties(subset=["Text"], **{"white-space": "normal"})
        .set_properties(**{"font-size": "0.9rem"})
    )

# ------------------------------------------------------------
# Example sentiment texts (Positive / Negative / Neutral)
# ------------------------------------------------------------
EMO_EXAMPLES = {
    "Positive": [
        "Alhamdulillah, result exam aku sangat bagus hari ini, rasa bangga dan lega gila.",
        "I really love this video, the explanation is clear and very helpful.",
        "Service dekat kedai ni memang tiptop, staff semua friendly dan sangat membantu.",
        "Today was such a good day, everything went better than expected.",
        "Workout session tadi padu, badan penat tapi hati super happy dan bersemangat.",
        "I finally got the internship I wanted, I’m so excited to start.",
        "Seronok dapat lepak dengan kawan-kawan malam ni, rasa sangat dihargai.",
        "The feedback from my lecturer was very positive and motivating.",
        "Project group kami jalan smooth, semua orang cooperate dengan baik.",
        "Rasa bersyukur sangat bila tengok progress diri sendiri makin improve.",
    ],
    "Negative": [
        "Service macam ni memang mengecewakan, tunggu lama tapi masalah langsung tak selesai.",
        "I am really disappointed with this service, waited so long and still no solution.",
        "Aku betul-betul frust, dah cuba yang terbaik tapi masih kena reject.",
        "Traffic jam hampir dua jam, memang rosakkan mood satu hari.",
        "Customer support langsung tak respon, rasa geram dan tak dihargai.",
        "Hari ni rasa teruk, asyik kena marah walaupun aku dah cuba yang terbaik.",
        "This product is not worth the money at all, totally regret buying it.",
        "Service lambat, order salah, memang pengalaman yang sangat teruk.",
        "Komen macam ni memang toxic, langsung tak membantu dan hanya menjatuhkan orang.",
        "Aku betul-betul sedih dan marah dengan cara benda ni dihandle.",
    ],
    "Neutral": [
        "I’ll be attending the meeting later today.",
        "Please send the file when you’re ready.",
        "The weather today is normal, not too hot or cold.",
        "It's almost 3 PM now. I'm still working on my tasks.",
        "Let me know if you need anything else.",
        "Today's schedule is the same as usual.",
        "I'm updating the document now. Will notify once done.",
        "The store opens at 10 AM. Just for your information.",
        "Saya akan hantar dokumen itu esok pagi.",
        "Barang itu baru sampai tadi petang.",
        "Saya tengah tunggu giliran, masih dalam queue.",
        "There is no update yet from the team.",
        "Dokumen sudah dimuat naik ke folder seperti diminta.",
        "Saya sudah terima mesej anda.",
        "Maklumat itu telah dikemaskini dalam sistem.",
        "Data sedang diproses, sila tunggu sebentar.",
        "Pendaftaran dibuka sehingga jam 6 petang.",
        "Nota mesyuarat telah dihantar kepada semua ahli.",
        "Saya sedang semak senarai tugasan sekarang.",
        "Jumlah peserta untuk acara itu ialah 32 orang.",
        "Kertas kerja ini mempunyai 12 halaman.",
        "Peranti itu sedang menjalankan proses biasa.",
        "Permohonan anda sedang diproses buat masa ini.",
    ],
}

EMO_ORDER = ["Positive", "Negative", "Neutral"]
EMO_EMOJI = {
    "Positive": "😊",
    "Negative": "😠",
    "Neutral": "😐",
}

# ------------------------------------------------------------
# Session State
# ------------------------------------------------------------
if "single_logs" not in st.session_state:
    st.session_state.single_logs = []

if "example_idx" not in st.session_state:
    st.session_state.example_idx = {k: 0 for k in EMO_ORDER}

if "yt_comments_df" not in st.session_state:
    st.session_state.yt_comments_df = None

if "yt_summary" not in st.session_state:
    st.session_state.yt_summary = None

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown(
    "<div class='app-main-title'>💬 Emotion Detection — Malay BERT (Sentiment)</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='app-subtitle'>Single-text sentiment and YouTube comment analytics using a fine-tuned three-class Malay–English BERT model (Negative / Neutral / Positive).</div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab_single, tab_post, tab_yt_adv, tab_session = st.tabs(
    [
        "🧠 Single Text Analyzer",
        "📺 Post / Comment Analyzer",
        "🔍 Advanced Insights & Analytics for Creators",
        "📊 Session Insights (Single Text)",
    ]
)

# ============================================================
# TAB 1 — Single Text Analyzer (Sentiment)
# ============================================================
with tab_single:
    left_col, right_col = st.columns([0.45, 0.55])

    # LEFT: chips + textarea
    with left_col:
        st.subheader("Example Inputs")

        st.markdown('<div id="emotion-chips">', unsafe_allow_html=True)
        chip_cols = st.columns(3)
        for i, emo in enumerate(EMO_ORDER):
            label = f"{EMO_EMOJI[emo]} {emo}"
            if chip_cols[i].button(label, key=f"emo_{emo}"):
                idx = st.session_state.example_idx[emo]
                st.session_state.example_idx[emo] = (idx + 1) % len(EMO_EXAMPLES[emo])
                st.session_state["single_text"] = EMO_EXAMPLES[emo][idx]
        st.markdown("</div>", unsafe_allow_html=True)

        text_value = st.text_area(
            "Type anything you want and watch the sentiment engine work its magic. A quick, intelligent read of the emotional tone behind your text!",
            key="single_text",
            height=160,
        )

        st.markdown('<div id="analyze-btn-wrap">', unsafe_allow_html=True)
        run_single = st.button("✨ Analyze Sentiment", key="btn_analyze_single")
        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT: prediction panel (main card + probabilities card)
    with right_col:
        st.subheader("Prediction Result")

        prediction_done = False  # safety flag
        top_label = None
        probs = None

        if run_single and text_value and text_value.strip():
            result = predict_single(text_value)
            top_label = result["top_label"]
            top_conf = result["top_conf"]
            probs = result["probs"]

            if top_label is None:
                st.warning("No sentiment could be predicted. Please try with different text.")
            else:
                prediction_done = True

                lab_upper = top_label.upper()

                if lab_upper == "POSITIVE":
                    color = "#16a34a"
                    emoji = "😊"
                elif lab_upper == "NEGATIVE":
                    color = "#dc2626"
                    emoji = "😠"
                else:
                    color = "#2563eb"
                    emoji = "😐"

                display_name = EMOTION_DISPLAY.get(lab_upper, lab_upper.title())

                # MAIN RESULT CARD
                st.markdown(
                    f"""
<div class="card primary-result-card">
    <div style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; color:#6b7280; margin-bottom:0.2rem;">
        Model output (single-label sentiment)
    </div>
    <div style="font-size:1.0rem; color:#4b5563; margin-bottom:0.3rem;">
        Predicted Sentiment:
    </div>
    <div style="font-size:1.35rem; font-weight:750; color:{color}; margin-bottom:0.15rem;">
        {emoji} {display_name}
    </div>
    <div style="font-size:0.95rem; color:#059669; font-weight:600;">
        Confidence: {top_conf*100:.1f}%
    </div>
</div>
                    """,
                    unsafe_allow_html=True,
                )

                # TOP PROBABILITIES AS ONE CARD
                ordered_idx = np.argsort(probs)[::-1]
                top_idx = ordered_idx[: len(EMOTION_LABELS)]
                top_names = [
                    EMOTION_DISPLAY.get(EMOTION_LABELS[i], EMOTION_LABELS[i].title())
                    for i in top_idx
                ]
                top_scores = [float(probs[i]) for i in top_idx]
                max_score = max(top_scores) if top_scores else 0.0

                rows_html = ""
                for sent, p in zip(top_names, top_scores):
                    bold_style = "font-weight:700; color:#111827;" if p == max_score else ""
                    rows_html += f"""
<tr>
  <td style="{bold_style}">{sent}</td>
  <td style="{bold_style}">{p*100:.1f}%</td>
</tr>
"""

                prob_card_html = f"""
<div class="explainer prob-card">
<div class="explainer-title">Sentiment Probabilities</div>

<p style="font-size:0.87rem; color:#4b5563; margin-bottom:0.4rem;">
Each percentage reflects how strongly the model leans toward that emotion based on the input sentence.
The bold row marks the sentiment chosen as the final prediction.
</p>

<table style="width:100%; border-collapse:collapse;">
<thead>
<tr><th>Sentiment</th><th>Probability</th></tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>

</div>
"""
                st.markdown(prob_card_html, unsafe_allow_html=True)

                # LOGGING
                st.session_state.single_logs.append(
                    {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "text": text_value,
                        "primary": lab_upper,
                        "confidence": top_conf,
                    }
                )

    # SECOND ROW — Chart + Interpretation (ONLY runs when prediction_done=True)
    if "run_single" in locals() and prediction_done:
        chart_col, expl_col = st.columns([0.45, 0.55])

        # CHART COLUMN
        with chart_col:
            fig, ax = plt.subplots(figsize=(5.2, 5.8))
            fig.subplots_adjust(top=0.90, bottom=0.18, left=0.12, right=0.95)

            labels = [EMOTION_DISPLAY.get(lbl, lbl.title()) for lbl in EMOTION_LABELS]
            colors = [EMOTION_COLORS.get(lbl, "#6b7280") for lbl in EMOTION_LABELS]

            # Ensure probs is a valid vector
            if probs is None or len(probs) != len(EMOTION_LABELS):
                probs_vec = np.zeros(len(EMOTION_LABELS), dtype=float)
            else:
                probs_vec = probs

            bars = ax.bar(labels, probs_vec, color=colors)

            for bar, p in zip(bars, probs_vec):
                y_val = min(1.02, (p + 0.03 if p > 0.02 else 0.05))
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_val,
                    f"{p*100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="700",
                    color="#334155",
                )

            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Probability", fontsize=11)
            ax.set_title("Model Confidence per Sentiment", fontsize=14, weight="700")
            ax.grid(axis="y")
            ax.set_axisbelow(True)

            st.pyplot(fig)

        # EXPLAINER COLUMN
        with expl_col:
            if lab_upper == "POSITIVE":
                explanation_text = f"""
<b style="color:{color};">{emoji} Positive ({top_conf*100:.1f}%)</b><br>
The text expresses satisfaction, appreciation, or a generally pleasant experience.
From an analysis perspective, this usually means the wording reflects praise, gratitude, or supportive comments.<br><br>

<b>Real-world interpretation:</b><br>
• Users feel good about the situation or outcome.<br>
• Positive tone supports marketing messages and brand trust.<br>
• Customer satisfaction is high, therefore useful for customer reviews, social media praise, performance feedback.<br><br>

<b>How this chart helps you read this</b><br>
• A tall Positive bar means the emotional intent is very clear.<br>
• Very small Negative/Neutral bars show almost no sign of frustration or factual tone.
"""
            elif lab_upper == "NEGATIVE":
                explanation_text = f"""
<b style="color:{color};">{emoji} Negative ({top_conf*100:.1f}%)</b><br>
The sentence carries clear signals of frustration, dissatisfaction, or criticism.
Words expressing complaints, anger, disappointment or stress strongly influence this sentiment.<br><br>

<b>Real-world interpretation:</b><br>
• This may indicate an unhappy customer, a service issue, or negative audience reaction (criticism).<br>
• Useful for: customer service escalation, product defect analysis, social media crisis alerts.<br>
• High Negative confidence signals an urgent or emotionally strong reaction.<br><br>

<b>How this chart helps you read this:</b><br>
• A tall Negative bar reflects a strong emotional complaint or rejection.<br>
• Low Positive/Neutral bars indicate little to no mixed feelings or factual reporting.
"""
            else:
                explanation_text = f"""
<b style="color:{color};">{emoji} Neutral ({top_conf*100:.1f}%)</b><br>
Neutral sentences generally provide information, updates, instructions, or emotionless statements.
The sentence does not show strong satisfaction or dissatisfaction because it usually states information without emotional weight.<br><br>

<b>Real-world interpretation:</b><br>
• Common in general office communication, announcements, plain facts, or automated messages.<br>
• Useful for analysing conversational patterns where emotion is not the focus (e.g., operations, reporting).<br>
• Neutral tone suggests the user is simply stating something without emotional intent.<br><br>

<b>How this chart helps you read this:</b><br>
• A moderately high Neutral bar means the model detects no clear emotional polarity.<br>
• Positive/Negative bars remain low because the text lacks expressive cues.
"""

            st.markdown(
                f"""
<div class="explainer">
<div class="explainer-title">How to interpret this prediction:</div>
{explanation_text}
</div>
""",
                unsafe_allow_html=True,
            )

# ============================================================
# TAB 2 — Post / Comment Analyzer (YouTube) — Sentiment
# ============================================================
with tab_post:
    header_left, header_right = st.columns([0.55, 0.45])

    with header_left:
        st.subheader("Analyze YouTube Video Comments")

    with header_right:
        st.markdown(
            '<div class="secondary-action" style="text-align:right;">',
            unsafe_allow_html=True,
        )
        fetch_clicked = st.button("📥 Analyze Now!", key="btn_fetch_comments")
        st.markdown("</div>", unsafe_allow_html=True)

    # Input fields
    video_url = st.text_input(
        "Curious how viewers are reacting to your content? Enter a YouTube URL and click Fetch & Analyze Comments to uncover audience sentiment at scale.",
        placeholder="https://www.youtube.com/watch?v=...",
        key="yt_url",
    )
    max_comments = st.slider("Choose the number of comments to analyze", 10, 50, 30, step=5)

    details = None
    comments = None

    # Fetch comments
    if fetch_clicked:
        if not video_url or not video_url.strip():
            st.warning("Please paste a valid YouTube video URL first.")
        else:
            with st.spinner("Hold tight! Analysing emotions…"):
                details, comments = fetch_youtube_data(video_url.strip(), max_comments)

                if details and comments:
                    rows = []
                    for idx, c in enumerate(comments, start=1):
                        result = predict_single(c)
                        lab = result["top_label"]
                        conf = result["top_conf"]

                        rows.append(
                            {
                                "No.": idx,
                                "Comment": c,
                                "Primary Emotion": lab if lab else "NEUTRAL",
                                "Confidence": conf,
                            }
                        )

                    df_comments = pd.DataFrame(rows)

                    counts = df_comments["Primary Emotion"].value_counts()
                    dominant_label = counts.idxmax()
                    dominant_share = counts.max() / len(df_comments)
                    avg_conf = float(df_comments["Confidence"].mean().round(3))

                    total = len(df_comments)
                    pos_share = counts.get("POSITIVE", 0) / total if total else 0.0
                    neg_share = counts.get("NEGATIVE", 0) / total if total else 0.0
                    neu_share = counts.get("NEUTRAL", 0) / total if total else 0.0

                    st.session_state.yt_comments_df = df_comments
                    st.session_state.yt_summary = {
                        "meta": details,
                        "total_sampled": len(df_comments),
                        "dominant_label": dominant_label,
                        "dominant_pct": dominant_share,
                        "avg_conf": avg_conf,
                        "pos_share": pos_share,
                        "neg_share": neg_share,
                        "neu_share": neu_share,
                    }

                elif details and not comments:
                    st.warning("No public comments found or comments are disabled.")
                else:
                    st.error(
                        "Unable to fetch comments. This may be due to quota limits, network issues, "
                        "or restricted video settings."
                    )

    df_comments = st.session_state.yt_comments_df
    summary = st.session_state.yt_summary

    # Show results
    if df_comments is not None and summary is not None:
        meta = summary["meta"]

        # VIDEO DETAILS
        st.markdown("### Video Details")

        vid_left, vid_right = st.columns([0.55, 0.45])

        with vid_left:
            if meta.get("thumbnail_url"):
                st.image(meta["thumbnail_url"])
            st.markdown(f"**Title:** {meta['title']}")
            st.markdown(f"**Channel:** {meta['channel']}")
            st.markdown(f"**Published:** {meta['published_at']}")

        with vid_right:
            fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
            fig2.subplots_adjust(top=0.88, bottom=0.20)

            counts = df_comments["Primary Emotion"].value_counts()
            ordered = [counts.get(lbl, 0) for lbl in EMOTION_LABELS]
            colors = [EMOTION_COLORS.get(lbl, "#6b7280") for lbl in EMOTION_LABELS]
            labels = [EMOTION_DISPLAY.get(lbl, lbl.title()) for lbl in EMOTION_LABELS]

            bars = ax2.bar(labels, ordered, color=colors)

            if any(ordered):
                offset = max(ordered) * 0.03
            else:
                offset = 0.2

            for bar, v in zip(bars, ordered):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + offset,
                    f"{v}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="700",
                    color="#334155",
                )

            ax2.set_ylabel("Count")
            ax2.set_title("Sentiment Distribution (Comments)", weight="700")
            ax2.grid(axis="y")
            ax2.set_axisbelow(True)

            st.pyplot(fig2)

        # SUMMARY CARDS
        total_sampled = summary["total_sampled"]
        dom_label = summary["dominant_label"]
        dom_pct = summary["dominant_pct"]
        avg_conf = summary["avg_conf"]

        pos_share = summary["pos_share"]
        neg_share = summary["neg_share"]
        neu_share = summary["neu_share"]

        dom_upper = str(dom_label).upper()
        if dom_upper == "POSITIVE":
            dom_class = "summary-card summary-positive"
            dom_emoji = "😊"
        elif dom_upper == "NEGATIVE":
            dom_class = "summary-card summary-negative"
            dom_emoji = "😠"
        else:
            dom_class = "summary-card summary-neutral"
            dom_emoji = "😐"

        st.markdown("### Summary at a Glance")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f"""
<div class="summary-card summary-total overview-card">
<h4>Total Comments Analyzed</h4>
<div class="value">{total_sampled}</div>
<p style="font-size:0.8rem; margin-top:0.2rem; color:#4b5563;">
A count of all public comments successfully retrieved and analyzed from this video.
</p>
</div>
""",
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
<div class="{dom_class} overview-card">
<h4>Dominant Sentiment</h4>
<div class="value">{dom_emoji} {dom_upper} ({dom_pct*100:.1f}%)</div>
<p style="font-size:0.8rem; margin-top:0.2rem; color:#4b5563;">
This is the sentiment most frequently detected across the comments which represents the audience’s overall mood.
</p>
</div>
""",
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                f"""
<div class="summary-card summary-confidence overview-card">
<h4>Avg Confidence & Mix</h4>
<div class="value">{avg_conf:.2f}</div>
<div style="font-size:0.8rem;margin-top:0.1rem;">
Pos: {pos_share*100:.1f}% | Neu: {neu_share*100:.1f}% | Neg: {neg_share*100:.1f}%
</div>
<p style="font-size:0.8rem; margin-top:0.2rem; color:#4b5563;">
Shows how clear the emotional expressions were in the comments, and how emotions are distributed.
</p>
</div>
""",
                unsafe_allow_html=True,
            )

        # INTERPRETATION CARD
        st.markdown(
            f"""
<div class="explainer">
<div class="explainer-title">Understanding the Emotional Climate of This Video</div>

This section summarises how people feel in the comment section. Instead of manually reading hundreds of comments, the model automatically identifies whether each comment is <b>Positive</b>, <b>Neutral</b>, or <b>Negative</b>.<br><br>

<b>1. What does the dominant sentiment mean? ({dom_emoji} {dom_upper}, {dom_pct*100:.1f}%)</b><br>
The dominant sentiment reflects the <b>overall mood</b> of the audience:<br><br>

• <b>POSITIVE</b> 😊 — Viewers enjoyed the content, found it helpful, or expressed support.<br>
&nbsp;&nbsp;Good sign for brand trust, creator reputation, and marketing campaigns.<br><br>
• <b>NEGATIVE</b> 😠 — Users expressed frustration, disagreement, or criticism.<br>
&nbsp;&nbsp;Useful for identifying service issues, controversial topics, or potential PR risks.<br><br>
• <b>NEUTRAL</b> 😐 — Comments are factual, emotionless, or spam-like.<br>
&nbsp;&nbsp;Typical in tutorial videos, large channels, or comment sections full of short replies.<br><br>

<b>2. How do we read and interpret the Positive–Neutral–Negative mix?</b><br>
This breakdown shows how varied or intense audience reactions are:<br><br>

• <b>High Positive</b> → Strong approval, supportive community.<br>
• <b>High Negative</b> → Potential dissatisfaction or backlash.<br>
• <b>High Neutral</b> → Many comments with no emotional cues (e.g., timestamps, updates, “first”).<br>
• <b>Balanced mix</b> → Diverse reactions; audience discussions may be active or opinion-splitting.<br><br>

<b>3. Let's take a closer look at the average confidence ({avg_conf:.2f}) and try understanding it. </b><br>
This score shows how clearly the model could detect emotion:<br><br>

• <b>High (0.75–1.00)</b> → Emotionally strong or expressive comments.<br>
• <b>Medium (0.50–0.74)</b> → Mild emotion, unclear tone, or mixed wording.<br>
• <b>Low (&lt;0.50)</b> → Ambiguous, sarcastic, very short, or code-switched comments.<br><br>

<b>Realistically, why do YouTube's (or any other social media platforms) comments often lower the overall confidence?</b><br>
The root causes are often the presence of:<br><br>

• Heavy slang and abbreviations<br>
• Malay–English code-switching<br>
• Sarcasm or jokes without clarity<br>
• 1–3 word replies (“nice”, “why?”, “😂”)<br>
• Spam, bots, or low-effort comments<br><br>

<b>4. What do we understand from these results?</b><br>
This dashboard helps creators, marketers, and stakeholders quickly understand:<br><br>

✓ Whether viewers overall liked or disliked the video<br>
✓ Whether the audience expressed strong emotions or stayed neutral<br>
✓ If there is emerging negative sentiment (early risk detection)<br>
✓ How clear or noisy the comment section is<br><br>

This makes the analysis suitable for <b>brand monitoring, content strategy, campaign evaluation, community management, and academic insight generation</b>.
</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # DOWNLOAD + TABLE
        title_col, btn_col = st.columns([0.75, 0.25])
        with title_col:
            st.markdown("### Comment-Level Sentiment Predictions")

        with btn_col:
            st.markdown(
                "<div id='session-actions' style='text-align:right;'>",
                unsafe_allow_html=True,
            )
            csv_df = df_comments.copy()
            csv_df["video_id"] = meta["video_id"]
            csv_df["video_title"] = meta["title"]
            csv_df["channel"] = meta["channel"]
            csv_df["published_at"] = meta["published_at"]
            csv_df["sampled_comments"] = total_sampled
            csv_df["dominant_sentiment"] = dom_upper
            csv_df["dominant_fraction"] = dom_pct
            csv_df["avg_confidence"] = avg_conf
            csv_df["positive_share"] = pos_share
            csv_df["neutral_share"] = neu_share
            csv_df["negative_share"] = neg_share

            st.download_button(
                "⬇️ Download results (CSV)",
                csv_df.to_csv(index=False).encode("utf-8"),
                "youtube_comment_sentiment.csv",
                mime="text/csv",
                key="btn_dl_comments",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        df_display = df_comments.set_index("No.")
        styled_df = style_comment_table(df_display)
        st.dataframe(styled_df, use_container_width=True)

    else:
        st.info("Paste a YouTube URL above and click the button **Analyze Now!** to begin.")

# ============================================================
# TAB 3 — Advanced Insights & Analytics for Creators (YouTube)
# ============================================================
with tab_yt_adv:
    df_comments = st.session_state.yt_comments_df
    summary = st.session_state.yt_summary

    if df_comments is None or summary is None:
        st.info(
            "No YouTube analysis data found. "
            "Please analyse a video in the **📺 Post / Comment Analyzer** tab first."
        )

    else:
        # ------------------------------
        # Extract summary + meta values
        # ------------------------------
        meta = summary["meta"]
        total_sampled = summary["total_sampled"]
        dom_label = summary["dominant_label"]
        dom_upper = str(dom_label).upper()
        avg_conf = summary["avg_conf"]

        pos_share = summary["pos_share"]
        neg_share = summary["neg_share"]
        neu_share = summary["neu_share"]

        comment_count = meta.get("comment_count", None)

        # ------------------------------------------------------------
        # Derived Metrics (same logic as PDF)
        # ------------------------------------------------------------

        # Sentiment Severity
        severity_raw = float(neg_share * avg_conf)
        severity_score = round(severity_raw * 100, 1)

        if severity_raw >= 0.45:
            severity_level = "Very High"
        elif severity_raw >= 0.30:
            severity_level = "High"
        elif severity_raw >= 0.15:
            severity_level = "Moderate"
        elif severity_raw > 0:
            severity_level = "Low"
        else:
            severity_level = "None"

        # Risk Score
        risk_score = round(severity_raw * 100, 1)
        if neg_share >= 0.40 and avg_conf >= 0.60:
            risk_band = "High Risk"
        elif neg_share >= 0.25:
            risk_band = "Moderate Risk"
        elif neg_share > 0:
            risk_band = "Low Risk"
        else:
            risk_band = "Minimal Risk"

        # Engagement Heat
        emotional_share = pos_share + neg_share
        if comment_count and comment_count > 0:
            coverage = min(1.0, total_sampled / comment_count)
        else:
            coverage = 0.0 if total_sampled == 0 else 1.0

        heat_raw = 0.6 * emotional_share + 0.4 * coverage
        heat_score = round(heat_raw * 100, 1)

        if heat_raw >= 0.70:
            heat_level = "Very Hot"
        elif heat_raw >= 0.45:
            heat_level = "Warm"
        elif heat_raw > 0:
            heat_level = "Mild"
        else:
            heat_level = "Cold"

        # Emotion Balance Index
        ebi_raw = float(pos_share - neg_share)
        ebi_score = round(ebi_raw * 100, 1)

        if ebi_raw > 0.15:
            ebi_label = "Leaning Positive"
        elif ebi_raw < -0.15:
            ebi_label = "Leaning Negative"
        elif abs(ebi_raw) <= 0.05:
            ebi_label = "Balanced / Mixed"
        else:
            ebi_label = "Slightly Tilted"

        # ------------------------------------------------------------
        # HEADER ROW (left: title, right: PDF export button)
        # ------------------------------------------------------------
        header_col, export_col = st.columns([0.70, 0.30])

        with header_col:
            st.markdown("### Overview for Creators & Stakeholders")

        with export_col:
            st.markdown("<div style='text-align:right;'>", unsafe_allow_html=True)
            pdf_bytes = build_full_report_pdf(summary, df_comments)
            st.download_button(
                "📄 Export Advanced Insights Report (PDF)",
                data=pdf_bytes,
                file_name="YouTube Advanced Insights Report.pdf",
                mime="application/pdf",
                key="btn_tab3_pdf",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

        # Wrap Top Cards for Tab 3 Spacing
        st.markdown("<div class='tab3-top-cards'>", unsafe_allow_html=True)

        c1, c2 = st.columns([0.55, 0.45])

        # VIDEO CONTEXT CARD
        with c1:
            st.markdown(
                f"""
<div class="summary-card summary-total overview-card" style="padding: 1.35rem 1.45rem;">
  <h4>Video Context</h4>
  <div class="value" style="font-size:1.05rem; margin-bottom:0.55rem;">{meta['title']}</div>
  <p style="font-size:1.0rem; color:#4b5563; line-height:1.55;">
    Channel: <b>{meta['channel']}</b><br>
    Published: {meta['published_at']}<br>
    Sampled comments analysed: <b>{total_sampled}</b><br>
    {f"Total public comments on YouTube: {comment_count}" if comment_count else ""}
  </p>
</div>
""",
                unsafe_allow_html=True,
            )

        # HIGH-LEVEL EMOTION CLIMATE
        with c2:
            st.markdown(
                f"""
<div class="summary-card summary-confidence overview-card" style="padding: 1.35rem 1.45rem;">
  <h4>High-Level Emotional Climate</h4>
  <div class="value" style="font-size:1.1rem; margin-bottom:0.55rem;">
    {dom_upper} · {avg_conf:.2f} avg confidence
  </div>
  <p style="font-size:1.0rem; color:#4b5563; line-height:1.55;">
    The dominant sentiment across the analysed comments is <b>{dom_upper}</b>
    with an average confidence level of <b>{avg_conf:.2f}</b>.
  </p>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)  # End top-cards wrapper

        st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)

        # ------------------------------------------------------------
        # SECTION: Advanced Metrics
        # ------------------------------------------------------------
        st.markdown("### Advanced Metrics at a Glance")
        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

        st.markdown("<div class='tab3-metrics-block'>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)

        # Card 1 — Sentiment Severity
        with m1:
            st.markdown(
                f"""
<div class="summary-card summary-negative metric-card" style="padding: 1.3rem 1.4rem;">
  <h4>Sentiment Severity</h4>
  <div class="value">{severity_score}</div>

  <div style="font-size:0.8rem; color:#4b5563;">
    <b>Formula:</b><br>
    Severity = Negative Share × Avg Confidence
  </div>

  <div style="font-size:0.8rem; margin-top:0.65rem; color:#4b5563;">
    <b>Meaning:</b><br>
    Indicates strength and clarity of negative sentiment.<br>
    Current level: <b>{severity_level}</b>.
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        # Card 2 — Risk Score
        with m2:
            st.markdown(
                f"""
<div class="summary-card summary-neutral metric-card" style="padding: 1.3rem 1.4rem;">
  <h4>Risk Score</h4>
  <div class="value">{risk_score}</div>

  <div style="font-size:0.8rem; color:#4b5563;">
    <b>Formula:</b><br>
    Risk Score = Severity × 100
  </div>

  <div style="font-size:0.8rem; margin-top:0.65rem; color:#4b5563;">
    <b>Meaning:</b><br>
    Simplifies severity into a practical 0–100 risk scale.<br>
    Current classification: <b>{risk_band}</b>.
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        # Card 3 — Engagement Heat
        with m3:
            st.markdown(
                f"""
<div class="summary-card summary-total metric-card" style="padding: 1.3rem 1.4rem;">
  <h4>Engagement Heat</h4>
  <div class="value">{heat_score}</div>

  <div style="font-size:0.8rem; color:#4b5563;">
    <b>Formula:</b><br>
    Heat = 0.6 × Emotional Share + 0.4 × Coverage
  </div>

  <div style="font-size:0.8rem; margin-top:0.65rem; color:#4b5563;">
    <b>Meaning:</b><br>
    Measures emotional activity and audience responsiveness.<br>
    Current heat: <b>{heat_level}</b>.
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        # Card 4 — Emotion Balance
        with m4:
            st.markdown(
                f"""
<div class="summary-card summary-positive metric-card" style="padding: 1.3rem 1.4rem;">
  <h4>Emotion Balance Index</h4>
  <div class="value">{ebi_score}</div>

  <div style="font-size:0.8rem; color:#4b5563;">
    <b>Formula:</b><br>
    EBI = (Positive – Negative) × 100
  </div>

  <div style="font-size:0.8rem; margin-top:0.65rem; color:#4b5563;">
    <b>Meaning:</b><br>
    Shows whether conversation leans supportive or critical.<br>
    Current leaning: <b>{ebi_label}</b>.
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)  # end metrics wrapper

        st.markdown("<div style='height:35px;'></div>", unsafe_allow_html=True)

        # ------------------------------------------------------------
        # VISUAL PERSPECTIVE
        # ------------------------------------------------------------
        st.markdown(
            "<h3 class='tab3-visual-title'>Visual Perspective of Audience Reactions</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

        vc1, vc2 = st.columns([0.50, 0.50])

        # Left: Sentiment Mix Bar Chart
        with vc1:
            fig, ax = plt.subplots(figsize=(4.6, 4.6))
            fig.subplots_adjust(top=0.88, bottom=0.20)

            sentiments = ["Positive", "Neutral", "Negative"]
            shares = [pos_share, neu_share, neg_share]
            colors = [
                EMOTION_COLORS["POSITIVE"],
                EMOTION_COLORS["NEUTRAL"],
                EMOTION_COLORS["NEGATIVE"],
            ]

            bars = ax.bar(sentiments, shares, color=colors)

            for bar, v in zip(bars, shares):
                y_val = min(1.02, (v + 0.03 if v > 0.02 else 0.05))
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_val,
                    f"{v * 100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="700",
                    color="#334155",
                )

            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Proportion of comments")
            ax.set_title("Sentiment Mix Among Analysed Comments", weight="700")
            ax.grid(axis="y")
            ax.set_axisbelow(True)

            st.pyplot(fig)

        # Right: Interpretation
        with vc2:
            st.markdown(
                f"""
<div class="explainer" style="padding: 1.3rem 1.4rem;">
  <div class="explainer-title">How to Interpret These Metrics as a Creator</div>

  <b>Sentiment Severity</b> shows how strong negative reactions are.<br>
  Current severity: <b>{severity_level}</b>.<br><br>

  <b>Risk Score</b> converts severity into a 0–100 scale.<br>
  Status: <b>{risk_band}</b>.<br><br>

  <b>Engagement Heat</b> measures emotional activity.<br>
  Current level: <b>{heat_level}</b>.<br><br>

  <b>Emotion Balance Index</b> shows support vs criticism.<br>
  Leaning: <b>{ebi_label}</b>.<br><br>

  Together, the metrics show:<br>
  • Audience satisfaction<br>
  • Expressiveness<br>
  • Emotional risks<br>
  • Overall sentiment health<br>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:35px;'></div>", unsafe_allow_html=True)

        # ------------------------------------------------------------
        # RECOMMENDATIONS
        # ------------------------------------------------------------
        st.markdown("### Recommended Next Steps for Creators / Stakeholders")
        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

        rec = []

        if risk_band == "High Risk":
            rec.append("Address concerns directly and provide clarifications.")
            rec.append("Consider posting a follow-up explanation or pinned comment.")
        elif risk_band == "Moderate Risk":
            rec.append("Monitor negative themes and prepare responses.")
        elif risk_band == "Low Risk":
            rec.append("No urgent action needed — continue monitoring.")

        if dom_upper == "NEGATIVE":
            rec.append("Respond empathetically to repeated concerns.")
        if dom_upper == "POSITIVE":
            rec.append("Leverage positive momentum to strengthen community loyalty.")

        if heat_level in ["Very Hot", "Warm"]:
            rec.append("Engage actively with viewers through replies and likes.")
        else:
            rec.append("Encourage engagement using calls-to-action.")

        if "Negative" in ebi_label:
            rec.append("Investigate root causes of dissatisfaction.")
        elif "Positive" in ebi_label:
            rec.append("Keep focusing on well-received content themes.")

        rec_html = "".join(f"<li>{r}</li>" for r in rec)

        st.markdown(
            f"""
<div class="explainer" style="padding: 1.3rem 1.4rem;">
  <div class="explainer-title">Creator / Analyst Action Recommendations</div>
  <ul style="margin-top:0.4rem; padding-left:1.2rem;">
    {rec_html}
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )

# ============================================================
# TAB 4 — SESSION INSIGHTS (Sentiment)
# ============================================================
with tab_session:
    logs = st.session_state.single_logs

    if logs:
        df_logs = pd.DataFrame(logs)

        # Determine emotion column
        if "primary" in df_logs.columns:
            primary_col = "primary"
        elif "label" in df_logs.columns:
            primary_col = "label"
        else:
            primary_col = "primary"
            df_logs[primary_col] = "NEUTRAL"

        if "confidence" not in df_logs.columns:
            df_logs["confidence"] = 0.0

        # Clean df
        df_logs_disp = df_logs.rename(
            columns={
                "time": "Time",
                "text": "Text",
                primary_col: "Primary Emotion",
                "confidence": "Confidence",
            }
        )

        cols_order = ["Time", "Text", "Primary Emotion", "Confidence"]
        df_logs_disp = df_logs_disp[[c for c in cols_order if c in df_logs_disp.columns]]

        df_disp = df_logs_disp.copy()
        df_disp.insert(0, "No.", range(1, len(df_disp) + 1))
        df_disp = df_disp.set_index("No.")

        # ------------------------------------------------------------
        # Header (Title + actions)
        # ------------------------------------------------------------
        t1, t2, t3 = st.columns([0.55, 0.25, 0.20])

        with t1:
            st.markdown("### Session Records")

        with t2:
            st.markdown(
                "<div id='session-actions' style='text-align:right;'>",
                unsafe_allow_html=True,
            )
            st.download_button(
                "⬇️ Download Session Insights (CSV)",
                df_logs.to_csv(index=False).encode("utf-8"),
                "single_text_session_insights.csv",
                mime="text/csv",
                key="btn_dl_session",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with t3:
            st.markdown(
                "<div id='session-actions' style='text-align:right;'>",
                unsafe_allow_html=True,
            )
            if st.button("🧹 Clear Session Insights", key="btn_clear_session"):
                st.session_state.single_logs = []
                st.experimental_rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # ------------------------------------------------------------
        # Table
        # ------------------------------------------------------------
        styled_logs = style_session_table(df_disp)
        st.dataframe(styled_logs, use_container_width=True)

        # ------------------------------------------------------------
        # Visual Summary
        # ------------------------------------------------------------
        st.markdown("### Visual Summary")
        left, right = st.columns(2)

        # Sentiment distribution
        with left:
            fig, ax = plt.subplots(figsize=(4.6, 4.6))
            fig.subplots_adjust(top=0.88, bottom=0.20)

            counts = df_logs[primary_col].value_counts()
            ordered = [counts.get(lbl, 0) for lbl in EMOTION_LABELS]
            colors = [EMOTION_COLORS.get(lbl, "#6b7280") for lbl in EMOTION_LABELS]
            labels = [EMOTION_DISPLAY.get(lbl, lbl.title()) for lbl in EMOTION_LABELS]

            bars = ax.bar(labels, ordered, color=colors)

            offset = max(ordered) * 0.03 if any(ordered) else 0.2

            for bar, v in zip(bars, ordered):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + offset,
                    f"{v}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="700",
                    color="#334155",
                )

            ax.set_ylabel("Count")
            ax.set_title("Sentiment Distribution (Session)", weight="700")
            ax.grid(axis="y")
            ax.set_axisbelow(True)

            st.pyplot(fig)

        # Confidence line chart
        with right:
            fig2, ax2 = plt.subplots(figsize=(4.8, 4.2))
            fig2.subplots_adjust(top=0.88, bottom=0.20)

            ax2.plot(
                range(1, len(df_logs) + 1),
                df_logs["confidence"],
                marker="o",
                linewidth=2,
            )
            ax2.set_ylim(0, 1.05)
            ax2.set_xlabel("Prediction Index")
            ax2.set_ylabel("Confidence")
            ax2.set_title("Confidence Over Predictions (Session)", weight="700")

            ax2.grid(axis="y")
            ax2.set_axisbelow(True)

            st.pyplot(fig2)

        # ------------------------------------------------------------
        # Interpretation
        # ------------------------------------------------------------
        st.markdown(
            """
<div class="explainer">
<div class="explainer-title">How to understand these session insights:</div>

This section summarises every text you have tested during this session.  
It helps you evaluate patterns, check model behaviour, and understand how
sentiment shifts across multiple examples.<br><br>

<b>1. Sentiment Distribution (left chart)</b><br>
Shows how many inputs fall under Negative, Neutral, or Positive.  
Tall bars indicate dominant emotional tone across your tests.<br><br>

<b>2. Confidence Over Predictions (right chart)</b><br>
Shows how certain the model was for each prediction.  
Sudden drops may indicate unclear, sarcastic, or emotionless text.<br><br>

<b>3. Why this tab matters</b><br>
Stakeholders can use this to understand:<br>
• model consistency<br>
• confidence stability<br>
• emotional patterns<br>
• reliability across examples<br><br>

Together, these insights provide both a quantitative and qualitative perspective
on how the model behaves across your interactions.
</div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.info("No session insights yet. Run a few single-text analyses first.")

# ============================================================
# Friendly footer (author intro)
# ============================================================
st.markdown(
    """
<br><br>
<div class="app-footer">
<p style="margin-bottom:0.3rem;">
Made with 💛 by <b>Siobhan Goh</b> · Capstone Project 2025
</p>
<p style="font-size:0.85rem; color:#94a3b8;">
Powered by Streamlit &amp; Hugging Face Spaces 🚀
</p>
</div>
""",
    unsafe_allow_html=True,
)
