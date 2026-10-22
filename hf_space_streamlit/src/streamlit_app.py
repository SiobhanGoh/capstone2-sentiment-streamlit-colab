"""Malay–English sentiment demo using the verified evaluated checkpoint."""
import os
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
from sentiment_core import (
    LABELS, SCORE_NOTE, SAMPLE_NOTE, SentimentPredictor, YouTubeError,
    analyse_comments, csv_bytes, fetch_youtube, sentiment_summary,
)
from reports import build_report

st.set_page_config(page_title="Malay–English Sentiment Analysis", page_icon="💬", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background:linear-gradient(130deg,#fff3f8,#f6f3ff 55%,#effbff);}
h1,h2,h3 {color:#243248;}
div[data-testid="stMetric"] {background:#ffffffc9;border:1px solid #e4dfef;border-radius:14px;padding:16px;}
[data-testid="stTabs"] button {font-weight:600;}
[data-testid="stBaseButton-primary"] {background:#6d4dad;border-color:#6d4dad;}
</style>""", unsafe_allow_html=True)
st.title("💬 Malay–English Sentiment Analysis")
st.write("Explore positive, neutral and negative sentiment in text and social comments.")
st.caption("Degree capstone · 2025–2026 · Siobhan Goh")


@st.cache_resource(show_spinner="Loading the verified sentiment model…")
def get_predictor():
    return SentimentPredictor(os.getenv("SENTIMENT_MODEL_PATH") or None)


try:
    predictor = get_predictor()
except Exception:
    st.error("The verified model could not be loaded. Please try again later.")
    st.stop()
release = {**predictor.spec, "revision": predictor.revision}


def youtube_key():
    value = os.getenv("YOUTUBE_API_KEY", "")
    if not value:
        try:
            value = st.secrets.get("YOUTUBE_API_KEY", "")
        except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
            pass
    return value


for key, value in {"single_logs": [], "last_prediction": None, "comment_analysis": None,
                   "single_text": "", "example_index": {}}.items():
    if key not in st.session_state:
        st.session_state[key] = value
EXAMPLES = {
    "Positive": ["Service dekat kedai ni bagus, staff friendly dan sangat membantu.",
                 "I really enjoyed this video. Thank you for the clear explanation."],
    "Neutral": ["Saya akan hantar dokumen itu esok pagi.", "The store opens at 10 AM."],
    "Negative": ["Service lambat, order salah, memang mengecewakan.",
                 "I am disappointed. I waited a long time and the problem is still unresolved."],
}


def choose_example(label):
    index = st.session_state.example_index.get(label, 0)
    st.session_state.single_text = EXAMPLES[label][index % len(EXAMPLES[label])]
    st.session_state.example_index[label] = index + 1
    st.session_state.last_prediction = None


def clear_prediction():
    st.session_state.last_prediction = None


def clear_comments():
    st.session_state.comment_analysis = None


def summary_view(frame):
    summary = sentiment_summary(frame)
    cols = st.columns(4)
    cols[0].metric("Comments analysed", summary["total"])
    for col, label in zip(cols[1:], LABELS):
        col.metric(label.title(), f"{summary['shares'][label]:.1%}",
                   help=f"{summary['counts'][label]} of {summary['total']} analysed comments")
    distribution = pd.DataFrame({"Sentiment": [s.title() for s in LABELS],
                                 "Comments": [summary["counts"][s] for s in LABELS]})
    st.bar_chart(distribution.set_index("Sentiment"), color="#9582c9")
    st.write(("Joint most frequent labels: " if summary["tied"] else "Most frequent label: ")
             + summary["dominant"])
    st.caption(SCORE_NOTE)


single_tab, comments_tab, insights_tab, session_tab = st.tabs(
    ["Single text", "Comment analysis", "Comment insights & reports", "Session insights"])
with single_tab:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Try an example or enter your own text")
        for col, label in zip(st.columns(3), EXAMPLES):
            col.button(label + " example", on_click=choose_example, args=(label,), key="example_" + label)
        text = st.text_area("Malay, English or code-switched text", key="single_text",
                            max_chars=10000, height=170, on_change=clear_prediction)
        st.caption("Links and @mentions are removed before analysis. Longer inputs use the first 128 model tokens.")
        if st.button("Analyse sentiment", type="primary", key="analyse_single"):
            clear_prediction()
            try:
                result = predictor.predict([text])[0]
                st.session_state.last_prediction = result
                st.session_state.single_logs.append(
                    {"Time (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "Text": text, "Sentiment": result["sentiment"], "Model score": result["score"],
                     **{label.title() + " score": result["scores"][i] for i, label in enumerate(LABELS)},
                     "Truncated": result["truncated"]})
            except ValueError as exc:
                st.warning(str(exc))
            except Exception:
                st.error("Prediction could not be completed. Please try again with shorter text.")
    with right:
        st.subheader("Prediction")
        result = st.session_state.last_prediction
        if result:
            st.metric("Predicted sentiment", result["sentiment"].title())
            st.metric("Top-class model score", f"{result['score']:.1%}")
            scores = pd.DataFrame({"Sentiment": [label.title() for label in LABELS],
                                   "Model score": result["scores"]})
            st.bar_chart(scores.set_index("Sentiment"), color="#9582c9")
            st.caption(SCORE_NOTE)
            if result["truncated"]:
                st.warning("This input was truncated to the first 128 model tokens.")
            st.write("Treat this as a suggested label and check the original wording. "
                     "Sarcasm, slang, context and unfamiliar topics may change the intended meaning.")
        else:
            st.info("Choose an example or enter text, then select Analyse sentiment.")

with comments_tab:
    st.subheader("Analyse a group of comments")
    source = st.radio("Comment source", ["YouTube video", "Paste comments"],
                      horizontal=True, key="comment_source", on_change=clear_comments)
    if source == "YouTube video":
        video_url = st.text_input("YouTube URL or video ID", key="youtube_url", on_change=clear_comments,
                                  placeholder="https://www.youtube.com/watch?v=…")
        limit = st.slider("Maximum comments", min_value=10, max_value=50, value=30, step=5)
        st.caption("Retrieves public top-level comments ordered by relevance; replies are excluded.")
        if not youtube_key():
            st.info("YouTube retrieval is not configured here. Use Paste comments to try the full analysis and reports.")
    else:
        pasted = st.text_area("One comment per line", height=180, max_chars=50000,
                              key="pasted_comments", on_change=clear_comments)
        st.caption("Up to 50 non-empty lines. Use comments you are permitted to analyse.")
    if st.button("Analyse comments", type="primary", key="analyse_comments"):
        clear_comments()
        try:
            with st.spinner("Preparing and analysing comments…"):
                if source == "YouTube video":
                    meta, comments = fetch_youtube(video_url, limit, youtube_key())
                else:
                    comments = [line.strip() for line in pasted.splitlines() if line.strip()]
                    if len(comments) > 50:
                        raise ValueError("Please use no more than 50 comments per analysis.")
                    meta = {"source": "Pasted comments", "title": "Pasted comment analysis",
                            "sampling": "User-supplied comments; not a representative sample."}
                if not comments:
                    raise ValueError("No comments were found. Try another video or paste some comments.")
                frame, excluded = analyse_comments(comments, predictor)
                meta["excluded"] = excluded
                meta["analysed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                pdf = build_report(frame, meta, release)
                export = frame.assign(Source=meta["source"], Title=meta["title"],
                                      Model=release["repo_id"], Revision=release["revision"],
                                      **{"Model SHA256": release["model_sha256"]})
                st.session_state.comment_analysis = {"frame": frame, "meta": meta,
                                                     "pdf": pdf, "csv": csv_bytes(export)}
        except (ValueError, YouTubeError) as exc:
            st.warning(str(exc))
        except Exception:
            st.error("This analysis could not be completed. Please try again with fewer or shorter comments.")
    analysis = st.session_state.comment_analysis
    if analysis:
        st.subheader(analysis["meta"]["title"])
        st.caption(analysis["meta"]["sampling"])
        if analysis["meta"]["excluded"]:
            st.info(f"{analysis['meta']['excluded']} inputs were empty after preprocessing and were excluded.")
        summary_view(analysis["frame"])
        st.dataframe(analysis["frame"], hide_index=True, width="stretch")
        st.download_button("Download comments CSV", analysis["csv"], "comment_sentiment.csv",
                           "text/csv", key="comments_csv")

with insights_tab:
    analysis = st.session_state.comment_analysis
    if not analysis:
        st.info("Analyse a video or pasted comments in Comment analysis first.")
    else:
        st.subheader("Comment insights")
        summary_view(analysis["frame"])
        st.write("Review the individual positive and negative comments to identify recurring feedback. "
                 "The labels alone do not establish customer satisfaction, engagement or business risk.")
        st.caption(SAMPLE_NOTE if analysis["meta"]["source"] == "YouTube"
                   else "This is a summary of the supplied comments only.")
        summary = sentiment_summary(analysis["frame"])
        st.write(f"Mean top-class model score: {summary['mean_score']:.1%}. "
                 f"Inputs truncated to 128 tokens: {summary['truncated']}.")
        selected = st.selectbox("Review predicted sentiment", ["All"] + LABELS)
        frame = analysis["frame"]
        if selected != "All":
            frame = frame.loc[frame.Sentiment == selected]
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button("Download sentiment report PDF", analysis["pdf"],
                           "sentiment_report.pdf", "application/pdf", key="comments_pdf")

with session_tab:
    st.subheader("Your single-text predictions")
    st.caption("History is held in this browser session. Clearing it removes the displayed history and current prediction.")
    if st.session_state.single_logs:
        frame = pd.DataFrame(st.session_state.single_logs)
        st.metric("Predictions in this session", len(frame))
        st.dataframe(frame, hide_index=True, width="stretch")
        counts = frame.Sentiment.value_counts().reindex(LABELS, fill_value=0)
        st.bar_chart(counts.rename("Predictions"), color="#9582c9")
        st.line_chart(frame["Model score"].reset_index(drop=True))
        st.caption(SCORE_NOTE)
        st.download_button("Download session CSV", csv_bytes(frame), "session_sentiment.csv",
                           "text/csv", key="session_csv")
        if st.button("Clear session", key="clear_session"):
            st.session_state.single_logs = []
            st.session_state.last_prediction = None
            st.rerun()
    else:
        st.info("Your predictions will appear here after you analyse some text.")

with st.expander("About the model and its limits"):
    st.write("Fine-tuned multilingual BERT for three-class Malay–English sentiment. "
             "The training corpus differs from many YouTube topics; errors are possible.")
    st.write("The corrected internal test comparison used 981 held-out inputs: "
             f"accuracy {release['test_accuracy']:.2%}, macro-F1 {release['test_macro_f1']:.4f}. "
             "Earlier development used the broader corpus, so this is retrospective internal evidence, "
             "not new external validation or a guarantee for your text.")
    st.caption(SCORE_NOTE)
    st.write("Loaded model: " + release["repo_id"])
    st.write("Pinned revision: " + release["revision"])
    st.write("Verified weights SHA-256: " + release["model_sha256"])
