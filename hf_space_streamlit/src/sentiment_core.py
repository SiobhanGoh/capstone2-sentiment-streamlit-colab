"""Inference and comment retrieval shared by the app and its checks."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import requests

LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
MAX_LENGTH = 128
SCORE_NOTE = ("Softmax scores are uncalibrated model outputs, not probabilities of being correct "
              "or measures of emotion intensity.")
SAMPLE_NOTE = ("These results describe only the analysed comments. YouTube retrieval uses top-level "
               "comments ordered by relevance, not a random or representative audience sample.")


def clean_text(value):
    """Keep identical to training.protocol.clean_text; parity is regression-tested."""
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    return re.sub(r"\s+", " ", text.replace("#", "")).strip()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SentimentPredictor:
    def __init__(self, local_path=None):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.spec = json.loads(Path(__file__).with_name("model_release.json").read_text())
        if local_path:
            model_dir = Path(local_path)
            self.revision = "local-verified-checkpoint"
        else:
            revision = self.spec["revision"]
            if not re.fullmatch(r"[a-f0-9]{40}", revision):
                raise ValueError("A pinned model release is required.")
            subfolder = self.spec["subfolder"]
            files = ["config.json", "model.safetensors", "tokenizer.json",
                     "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]
            prefix = subfolder + "/" if subfolder else ""
            snapshot = snapshot_download(self.spec["repo_id"], revision=revision,
                                         allow_patterns=[prefix + f for f in files])
            model_dir = Path(snapshot) / subfolder
            self.revision = revision
        if file_sha256(model_dir / "model.safetensors") != self.spec["model_sha256"]:
            raise ValueError("Model checksum does not match the evaluated release.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, fix_mistral_regex=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, use_safetensors=True)
        labels = [str(self.model.config.id2label[i]).upper() for i in range(3)]
        if labels != LABELS:
            raise ValueError("The model label order does not match the evaluation protocol.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Bound CPU thread usage for a small shared Space.
        if self.device == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        self.model.to(self.device).eval()

    def predict(self, texts):
        import torch
        prepared = [clean_text(t) for t in texts]
        if any(not t for t in prepared):
            raise ValueError("Please enter text that remains after links and mentions are removed.")
        results = []
        for start in range(0, len(prepared), 8):
            batch = prepared[start:start + 8]
            lengths = [len(self.tokenizer(t, add_special_tokens=True)["input_ids"]) for t in batch]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=MAX_LENGTH, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                probabilities = torch.softmax(self.model(**inputs).logits, dim=-1).cpu().numpy()
            for text, scores, length in zip(batch, probabilities, lengths):
                if scores.shape != (3,) or not np.isfinite(scores).all():
                    raise ValueError("The model returned invalid scores.")
                top = int(scores.argmax())
                results.append({"sentiment": LABELS[top], "score": float(scores[top]),
                                "scores": scores.tolist(), "cleaned_text": text,
                                "truncated": length > MAX_LENGTH})
        return results


def analyse_comments(comments, predictor):
    valid = [(i + 1, c) for i, c in enumerate(comments) if clean_text(c)]
    if not valid:
        raise ValueError("There are no usable comments after removing links and mentions.")
    results = predictor.predict([c for _, c in valid])
    rows = []
    for (number, comment), result in zip(valid, results):
        rows.append({"No.": number, "Comment": comment,
                     "Sentiment": result["sentiment"], "Model score": result["score"],
                     **{label.title() + " score": result["scores"][i] for i, label in enumerate(LABELS)},
                     "Truncated": result["truncated"]})
    return pd.DataFrame(rows), len(comments) - len(valid)


def sentiment_summary(frame):
    counts = frame["Sentiment"].value_counts().reindex(LABELS, fill_value=0)
    total = len(frame)
    if not total:
        raise ValueError("Cannot summarise an empty analysis.")
    maximum = counts.max()
    leaders = counts[counts == maximum].index.tolist()
    return {"total": total, "counts": counts.to_dict(),
            "shares": (counts / total).to_dict(),
            "dominant": " / ".join(label.title() for label in leaders),
            "tied": len(leaders) > 1,
            "mean_score": float(frame["Model score"].mean()),
            "truncated": int(frame["Truncated"].sum())}


class YouTubeError(ValueError):
    """Safe messages that never include the request URL or API credential."""


def extract_video_id(value):
    value = value.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    if not re.match(r"^https?://", value):
        value = "https://" + value
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    parts = parsed.path.strip("/").split("/")
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parts[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif len(parts) == 2 and parts[0] in {"shorts", "embed", "live"}:
            video_id = parts[1]
        else:
            video_id = ""
    else:
        video_id = ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise YouTubeError("Enter a valid YouTube video URL or its 11-character video ID.")
    return video_id


def _youtube_get(endpoint, params, get):
    try:
        response = get("https://www.googleapis.com/youtube/v3/" + endpoint,
                       params=params, timeout=(5, 20))
    except requests.RequestException:
        raise YouTubeError("YouTube could not be reached. Please try again shortly.") from None
    try:
        data = response.json()
    except (ValueError, TypeError):
        raise YouTubeError("YouTube returned an unreadable response. Please try again.") from None
    if response.status_code != 200:
        error = data.get("error", {}) if isinstance(data, dict) else {}
        reasons = {item.get("reason") for item in error.get("errors", []) if isinstance(item, dict)}
        if "commentsDisabled" in reasons:
            message = "Comments are disabled for this video."
        elif reasons & {"quotaExceeded", "dailyLimitExceeded"}:
            message = "The YouTube daily request quota has been reached. Try again later or paste comments."
        elif reasons & {"keyInvalid", "accessNotConfigured", "forbidden", "ipRefererBlocked"}:
            message = "YouTube access is not configured correctly. The app owner needs to check its API key restrictions."
        elif response.status_code == 404:
            message = "This video is unavailable or is not public."
        else:
            message = "YouTube could not complete this request. Try a public video or paste comments."
        raise YouTubeError(message)
    if not isinstance(data, dict) or not isinstance(data.get("items", []), list):
        raise YouTubeError("YouTube returned an unexpected response.")
    return data


def fetch_youtube(video_url, max_comments, api_key, get=requests.get):
    video_id = extract_video_id(video_url)
    if not api_key:
        raise YouTubeError("YouTube retrieval is not configured for this app. You can still analyse pasted comments.")
    if not isinstance(max_comments, int) or not 1 <= max_comments <= 50:
        raise YouTubeError("Choose between 1 and 50 comments.")
    details = _youtube_get("videos", {"part": "snippet,statistics", "id": video_id, "key": api_key}, get)
    if not details.get("items"):
        raise YouTubeError("This video is unavailable or is not public.")
    video = details["items"][0]
    snippet = video.get("snippet", {})
    meta = {"source": "YouTube", "video_id": video_id,
            "title": snippet.get("title", "Untitled video"),
            "channel": snippet.get("channelTitle", "Unknown channel"),
            "published_at": snippet.get("publishedAt", ""),
            "sampling": "Up to 50 top-level comments, ordered by relevance; replies excluded."}
    comments, seen, tokens = [], set(), set()
    page_token = None
    while len(comments) < max_comments:
        params = {"part": "snippet", "videoId": video_id, "key": api_key,
                  "maxResults": min(50, max_comments - len(comments)),
                  "order": "relevance", "textFormat": "plainText"}
        if page_token:
            params["pageToken"] = page_token
        data = _youtube_get("commentThreads", params, get)
        for item in data.get("items", []):
            top = item.get("snippet", {}).get("topLevelComment", {})
            comment_id = top.get("id") or item.get("id")
            text = top.get("snippet", {}).get("textOriginal") or top.get("snippet", {}).get("textDisplay")
            if isinstance(text, str) and text.strip() and comment_id not in seen:
                comments.append(text.strip())
                seen.add(comment_id)
                if len(comments) >= max_comments:
                    break
        page_token = data.get("nextPageToken")
        if not page_token or page_token in tokens:
            break
        tokens.add(page_token)
    # Failed later pages raise above; never present a partial fetch as a completed one.
    return meta, comments


def csv_bytes(frame):
    safe = frame.copy()
    for col in safe.columns:
        safe[col] = safe[col].map(
            lambda x: "'" + x if isinstance(x, str) and x.lstrip().startswith(("=", "+", "-", "@", "\t", "\r"))
            else x)
    return safe.to_csv(index=False).encode("utf-8-sig")
