# Rojak Sentiment web app

The live front end for the Malay–English sentiment model. It is a static React site: the model runs in the visitor's browser through [transformers.js](https://github.com/huggingface/transformers.js) and ONNX, so typed and uploaded text never leaves their device. The only server code is a small Cloudflare Pages Function that fetches public YouTube comments without exposing the API key.

## What it does

- **Analyse:** one piece of Malay, English or code-switched text. Shows the label, all three class scores, where the text sits on the sentiment triangle, and an optional "Why this label?" view that shades the words whose removal changes the score most. Recent texts are kept in this browser.
- **Comments:** paste up to 1,000 comments, pull up to 100 from a YouTube video, or upload a CSV. Shows the video card, the share of each label, how many comments are worth checking by hand, creator notes (most-liked negative comments, viewer questions, recurring words), three explorable maps (sentiment triangle, theme map, places map of Malaysian states), when comments were posted, and every comment in three columns with search and sort. Exports as a themed PDF report, a CSV or a short copyable summary.
- **Compare:** two YouTube videos side by side, with the difference in each label's share.
- **Wrong label?:** on any result, record the label you think is right. Corrections stay in this browser and download as CSV, ready to become a hand-checked test set for the next model.
- **About:** the model card, evaluation numbers and limits.

On phones and mobile data the 356 MB model waits until the first analysis instead of downloading as soon as the page opens.

## Run locally

```bash
cd web
npm ci
npm run dev          # http://localhost:5173
npm test             # cleaning parity, analysis, exports, YouTube function
```

`VITE_MOCK_MODEL=1 npm run dev` uses clearly labelled placeholder scores, so you can work on the interface without downloading the model.

## Deploy on Cloudflare Pages (free)

1. In the Cloudflare dashboard, open **Workers & Pages → Create → Pages → Connect to Git** and pick this repository.
2. Set **Root directory** to `web`, **Build command** to `npm ci && npm run build` and **Build output directory** to `dist`.
3. Under **Settings → Variables and Secrets**, add a secret named `YOUTUBE_API_KEY` (a YouTube Data API v3 key). Without it, everything except YouTube retrieval still works.
4. Deploy. Pages picks up `functions/api/youtube.js` automatically and serves it at `/api/youtube`.

## Model release

`src/model_release.json` pins the Hugging Face repo, the exact revision and the file's SHA-256. The ONNX files are produced by `.github/workflows/browser-model.yml` from the evaluated checkpoint, scored by `training/evaluate_browser.py` on the frozen test partition, and published by `.github/workflows/publish-browser-model.yml` only if the scored file's checksum matches.

`scripts/browser_eval.py` runs this app's own worker in headless Chrome over the test partition to confirm the browser pipeline gives the same labels as the Python evaluation.

## Third-party assets

The PDF export embeds DejaVu Sans (see `src/assets/fonts/DejaVu-LICENSE`). Interface fonts are Bricolage Grotesque and Plus Jakarta Sans (SIL Open Font License), installed from npm.
