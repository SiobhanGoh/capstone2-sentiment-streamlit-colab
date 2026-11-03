// Runs the sentiment model off the main thread so the interface stays responsive.
import { AutoModelForSequenceClassification, AutoTokenizer, env } from '@huggingface/transformers';
import release from './model_release.json';
import { cleanText } from './lib/clean.js';
import { BATCH_SIZE, MAX_LENGTH, checkLabelOrder, toResult } from './lib/inference.js';
import { blendScores, prepareTfidf, tfidfProba } from './lib/blend.js';

async function sha256(text) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

// The TF-IDF half of the blend; checked against its published checksum before use.
async function loadTfidf(spec) {
  if (!spec?.file) return null;
  const response = await fetch(new URL(spec.file, self.location.origin));
  if (!response.ok) throw new Error('The text-pattern model could not be downloaded.');
  const text = await response.text();
  if ((await sha256(text)) !== spec.sha256) throw new Error('The text-pattern model does not match the evaluated release.');
  return prepareTfidf(JSON.parse(text));
}

let loading = null;

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

function load() {
  if (loading) return loading;
  const spec = release.browser_model;
  const localPath = import.meta.env.VITE_LOCAL_MODEL_PATH;
  // Evaluation builds only: serve onnxruntime's files from the same npm version instead of the CDN.
  const wasmPath = import.meta.env.VITE_ORT_WASM_PATH;
  if (wasmPath) {
    env.backends.onnx.wasm.wasmPaths = {
      mjs: `${wasmPath}ort-wasm-simd-threaded.asyncify.mjs`,
      wasm: `${wasmPath}ort-wasm-simd-threaded.asyncify.wasm`,
    };
  }
  if (localPath) {
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = localPath;
  } else {
    env.allowLocalModels = false;
    if (!/^[a-f0-9]{40}$/.test(spec.revision)) {
      return Promise.reject(new Error('This build has no verified model release yet.'));
    }
  }
  const files = new Map();
  const progress_callback = (event) => {
    if (event.status === 'progress' && event.file) {
      files.set(event.file, { loaded: event.loaded ?? 0, total: event.total ?? 0 });
      let loaded = 0;
      let total = 0;
      for (const f of files.values()) { loaded += f.loaded; total += f.total; }
      post('progress', { loaded, total });
    }
  };
  const options = { revision: spec.revision || 'main', progress_callback };
  loading = (async () => {
    const id = localPath ? 'model' : spec.repo_id;
    const tokenizer = await AutoTokenizer.from_pretrained(id, options);
    const model = await AutoModelForSequenceClassification.from_pretrained(id, {
      ...options, dtype: spec.dtype, device: 'wasm',
    });
    checkLabelOrder(model.config.id2label);
    const tfidf = await loadTfidf(release.scoring?.tfidf);
    return { tokenizer, model, tfidf };
  })();
  loading.catch(() => { loading = null; });
  return loading;
}

async function predict(id, texts) {
  const { tokenizer, model, tfidf } = await load();
  const scoring = release.scoring ?? {};
  const prepared = texts.map(cleanText);
  if (prepared.some((t) => !t)) {
    throw new Error('Please enter text that remains after links and mentions are removed.');
  }
  const results = [];
  for (let start = 0; start < prepared.length; start += BATCH_SIZE) {
    const batch = prepared.slice(start, start + BATCH_SIZE);
    const counts = batch.map((t) => tokenizer(t).input_ids.dims.at(-1));
    const inputs = tokenizer(batch, { padding: true, truncation: true, max_length: MAX_LENGTH });
    const { logits } = await model(inputs);
    const rows = logits.tolist();
    batch.forEach((text, i) => {
      const scores = blendScores(rows[i], tfidf ? tfidfProba(tfidf, text) : null,
        { temperature: scoring.temperature ?? 1, weight: tfidf ? scoring.blend_weight ?? 1 : 1 });
      results.push(toResult(text, rows[i], counts[i], scores));
    });
    post('batch', { id, done: results.length, total: prepared.length });
  }
  return results;
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') {
      await load();
      post('ready');
    } else if (data.type === 'predict') {
      post('result', { id: data.id, results: await predict(data.id, data.texts) });
    }
  } catch (error) {
    post('error', { id: data.id, message: error?.message || 'The model could not run.' });
  }
};
