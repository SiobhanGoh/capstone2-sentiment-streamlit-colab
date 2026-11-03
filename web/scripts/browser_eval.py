"""Run the actual web app's model worker in headless Chrome on the frozen test partition.

Confirms that the browser pipeline (JS cleaning, transformers.js tokenizer, onnxruntime-web)
gives the same labels as the Python ONNX evaluation. Needs an evaluation build served locally:

    VITE_EXPOSE_PREDICT=1 VITE_LOCAL_MODEL_PATH=/models/ npx vite build && npx vite preview --port 4174
    python scripts/browser_eval.py --data <prepared data> --python-logits <folder> --variant model_quantized.onnx

Only aggregate numbers are printed; test text stays in the private data folder.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.protocol import clean_text, load_prepared  # noqa: E402


async def run(url, texts, chunk=50):
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
        page = await browser.new_page()
        await page.goto(url)
        await page.wait_for_function("() => typeof window.__predict === 'function'", timeout=60000)
        scores = []
        for start in range(0, len(texts), chunk):
            batch = texts[start:start + chunk]
            result = await page.evaluate("async (t) => (await window.__predict(t)).map(r => r.scores)", batch)
            scores.extend(result)
        await browser.close()
        return np.array(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--python-logits", required=True)
    parser.add_argument("--variant", default="model_quantized.onnx")
    parser.add_argument("--url", default="http://localhost:4174/")
    parser.add_argument("--out")
    args = parser.parse_args()
    frame, _ = load_prepared(args.data)
    test = frame.loc[frame.split == "test"].reset_index(drop=True)
    texts = test.text.tolist()
    # Prepared text is already cleaned; the app cleans again, which must change nothing.
    assert all(clean_text(t) == t for t in texts), "cleaning is not idempotent on the test text"
    browser = asyncio.run(run(args.url, texts))
    logits = np.load(Path(args.python_logits) / f"{args.variant}.logits.npy")
    # Expected scores: the release's scoring (temperature, then blend with the TF-IDF model), else a plain softmax.
    release = json.loads((ROOT / "web/src/model_release.json").read_text())
    scoring = release.get("scoring", {})
    temperature = scoring.get("temperature", 1.0)
    z = logits / temperature
    python = np.exp(z - z.max(1, keepdims=True))
    python /= python.sum(1, keepdims=True)
    if scoring.get("tfidf"):
        from training.export_tfidf import reference_proba
        tf = json.loads((ROOT / "web/public" / scoring["tfidf"]["file"].lstrip("/")).read_text(encoding="utf-8"))
        vocab = {t: i for i, t in enumerate(tf["terms"])}
        tfidf = np.array([reference_proba(t, vocab, tf["idf"], tf["coef"], tf["intercept"]) for t in texts])
        w = scoring["blend_weight"]
        python = w * python + (1 - w) * tfidf
    y = test.label_id.to_numpy()
    report = {
        "variant": args.variant,
        "test_inputs": len(texts),
        "browser_accuracy": float(accuracy_score(y, browser.argmax(1))),
        "browser_macro_f1": float(f1_score(y, browser.argmax(1), average="macro")),
        "label_agreement_with_python": float((browser.argmax(1) == python.argmax(1)).mean()),
        "max_abs_score_diff_vs_python": float(np.abs(browser - python).max()),
        "scoring": scoring.get("method", "model only"),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
