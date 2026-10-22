from io import BytesIO
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import pytest
import requests
from pypdf import PdfReader

SRC = Path(__file__).resolve().parents[1] / "hf_space_streamlit/src"
sys.path.insert(0, str(SRC))
from sentiment_core import (
    LABELS, YouTubeError, clean_text, extract_video_id, fetch_youtube,
    analyse_comments, sentiment_summary, csv_bytes,
)
from reports import build_report, printable_text
from training.protocol import clean_text as training_clean_text


class ExamplePredictor:
    def predict(self, texts):
        return [{"sentiment": LABELS[i % 3], "score": .8, "scores": [.8, .1, .1],
                 "truncated": False} for i, _ in enumerate(texts)]


def comments_frame():
    return analyse_comments(["Bagus, terima kasih!", "Maklumat diterima.", "Service lambat."],
                            ExamplePredictor())[0]


@pytest.mark.parametrize("text", [
    "  ＳＡＹＡ #Suka @name https://example.com 😀\nBagus ",
    "Straße Café www.example.com #Test", "", None, "hello\tWORLD",
])
def test_preprocessing_matches_training(text):
    assert clean_text(text) == training_clean_text(text)


@pytest.mark.parametrize("url", [
    "dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ?t=2",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=share",
    "youtube.com/shorts/dQw4w9WgXcQ", "https://m.youtube.com/live/dQw4w9WgXcQ",
    "https://www.youtube.com/embed/dQw4w9WgXcQ",
])
def test_video_urls(url):
    assert extract_video_id(url) == "dQw4w9WgXcQ"


@pytest.mark.parametrize("url", ["", "https://youtube.com.attacker.example/watch?v=dQw4w9WgXcQ",
                                      "https://example.com/dQw4w9WgXcQ", "https://youtube.com/watch?v=short"])
def test_invalid_video_urls(url):
    with pytest.raises(YouTubeError):
        extract_video_id(url)


class Response:
    def __init__(self, data, code=200):
        self.data, self.status_code = data, code
    def json(self):
        return self.data


def comment_item(number, text):
    return {"id": str(number), "snippet": {"topLevelComment": {
        "id": str(number), "snippet": {"textOriginal": text}}}}


def test_youtube_pagination_and_duplicates():
    replies = iter([
        Response({"items": [{"snippet": {"title": "Video", "channelTitle": "Channel"}}]}),
        Response({"items": [comment_item(1, "Bagus")], "nextPageToken": "next"}),
        Response({"items": [comment_item(1, "Bagus"), comment_item(2, "Biasa")]}),
    ])
    calls = []
    def get(url, **kwargs):
        calls.append(kwargs)
        return next(replies)
    meta, comments = fetch_youtube("dQw4w9WgXcQ", 3, "test-only-key", get)
    assert comments == ["Bagus", "Biasa"]
    assert meta["source"] == "YouTube"
    assert calls[-1]["params"]["pageToken"] == "next"
    assert calls[1]["params"]["order"] == "relevance"


@pytest.mark.parametrize("reason, message", [("commentsDisabled", "disabled"),
                                            ("quotaExceeded", "quota"),
                                            ("keyInvalid", "configured")])
def test_youtube_errors_are_specific_and_hide_key(reason, message):
    def get(*args, **kwargs):
        return Response({"error": {"errors": [{"reason": reason}]}}, 403)
    with pytest.raises(YouTubeError, match=message) as error:
        fetch_youtube("dQw4w9WgXcQ", 10, "secret-never-exposed", get)
    assert "secret-never-exposed" not in str(error.value)


def test_network_error_and_partial_fetch_are_not_success():
    replies = iter([Response({"items": [{}]}),
                    Response({"items": [comment_item(1, "Good")], "nextPageToken": "next"})])
    def get(*args, **kwargs):
        try:
            return next(replies)
        except StopIteration:
            raise requests.Timeout("private-key-in-url")
    with pytest.raises(YouTubeError, match="reached") as error:
        fetch_youtube("dQw4w9WgXcQ", 10, "secret", get)
    assert "private-key" not in str(error.value)


def test_missing_key_empty_video_and_no_comments():
    with pytest.raises(YouTubeError, match="not configured"):
        fetch_youtube("dQw4w9WgXcQ", 10, "")
    with pytest.raises(YouTubeError, match="unavailable"):
        fetch_youtube("dQw4w9WgXcQ", 10, "test", lambda *a, **k: Response({"items": []}))
    replies = iter([Response({"items": [{}]}), Response({"items": []})])
    _, comments = fetch_youtube("dQw4w9WgXcQ", 10, "test", lambda *a, **k: next(replies))
    assert comments == []


def test_empty_preprocessed_comments_are_excluded_not_neutral():
    frame, excluded = analyse_comments(["@only https://example.com", "Good"], ExamplePredictor())
    assert excluded == 1 and len(frame) == 1 and frame.iloc[0]["No."] == 2
    with pytest.raises(ValueError):
        analyse_comments(["@only"], ExamplePredictor())


def test_summary_handles_ties_and_counts():
    summary = sentiment_summary(comments_frame())
    assert summary["tied"] and summary["counts"] == dict.fromkeys(LABELS, 1)
    assert sum(summary["shares"].values()) == 1


def test_csv_preserves_unicode_and_neutralises_spreadsheet_formulas():
    frame = pd.DataFrame({"Text": ["café, bagus\nreally good", "=1+1", " @SUM(A1)", "普通"],
                          "Score": [.8, .7, .6, .5]})
    result = csv_bytes(frame)
    loaded = pd.read_csv(BytesIO(result))
    assert loaded.iloc[0].Text == frame.iloc[0].Text
    assert loaded.iloc[1].Text == "'=1+1" and loaded.iloc[2].Text.startswith("'")
    assert loaded.iloc[3].Text == "普通"


def test_pdf_contains_all_comments_and_release_and_pages_long_text():
    frame = comments_frame()
    frame.loc[0, "Comment"] = "<tag> café & " + "long comment " * 500
    spec = json.loads((SRC / "model_release.json").read_text())
    result = build_report(frame, {"source": "Pasted comments", "title": "Test report"}, spec)
    reader = PdfReader(BytesIO(result))
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert len(reader.pages) > 1
    assert "<tag> café &" in text and "Service lambat." in text and spec["model_sha256"] in text.replace("\n", "")
    assert "not probabilities of being correct" in text
    assert "business risk" in text


def fake_ui_predictor():
    predictor = ExamplePredictor()
    predictor.spec = json.loads((SRC / "model_release.json").read_text())
    predictor.revision = "test-fixture"
    return predictor


def test_pdf_unsupported_glyphs_have_explicit_lossless_fallbacks():
    assert printable_text('a💀', {ord('a'): 1}) == 'a[SKULL U+1F480]'
    assert printable_text('😭', {ord('😭'): 1}) == '[LOUDLY CRYING FACE U+1F62D]'
    frame = comments_frame()
    frame.loc[0, 'Comment'] = 'café 💀 普通'
    spec = json.loads((SRC / 'model_release.json').read_text())
    text = '\n'.join(p.extract_text() for p in PdfReader(BytesIO(build_report(frame, {}, spec))).pages)
    assert 'SKULL U+1F480' in text and 'U+666E' in text
    assert 'preserves the original text' in text


def test_streamlit_flows_persist_clear_and_reject_stale_results(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import sentiment_core
    monkeypatch.setattr(sentiment_core, "SentimentPredictor", lambda *a: fake_ui_predictor())
    app = AppTest.from_file(str(SRC / "streamlit_app.py"), default_timeout=30).run()
    assert not app.exception
    app.button(key="example_Positive").click().run()
    app.button(key="analyse_single").click().run()
    assert len(app.session_state["single_logs"]) == 1
    app.run()
    assert len(app.session_state["single_logs"]) == 1  # downloads/reruns do not create duplicate logs
    app.radio(key="comment_source").set_value("Paste comments").run()
    app.text_area(key="pasted_comments").set_value("Good\nNormal\nBad").run()
    app.button(key="analyse_comments").click().run()
    assert not app.exception and len(app.session_state["comment_analysis"]["frame"]) == 3
    assert app.session_state["comment_analysis"]["pdf"].startswith(b"%PDF")
    assert len(app.get("download_button")) == 3
    app.selectbox[0].set_value("NEGATIVE").run()
    assert not app.exception
    app.text_area(key="pasted_comments").set_value("").run()
    app.button(key="analyse_comments").click().run()
    assert app.session_state["comment_analysis"] is None and len(app.warning) > 0
    app.button(key="clear_session").click().run()
    assert app.session_state["single_logs"] == [] and app.session_state["last_prediction"] is None
    assert not app.exception
