// TF-IDF baseline and the validated blend, matching training/export_tfidf.py and training/post_hoc.py.
// Input text must already be cleaned with cleanText, exactly as in training.
import { softmax } from './inference.js';

// scikit-learn's token pattern (?u)\b\w\w+\b, i.e. runs of two or more word characters.
const TOKEN = /[\p{L}\p{N}_]{2,}/gu;

export function terms(cleaned) {
  const tokens = cleaned.toLowerCase().match(TOKEN) ?? [];
  const out = [...tokens];
  for (let i = 1; i < tokens.length; i += 1) out.push(`${tokens[i - 1]} ${tokens[i]}`);
  return out;
}

export function prepareTfidf(json) {
  if (json?.format !== 'rojak-tfidf-v1') throw new Error('Unexpected TF-IDF model format.');
  const vocab = new Map(json.terms.map((t, i) => [t, i]));
  return { vocab, idf: json.idf, coef: json.coef, intercept: json.intercept };
}

export function tfidfProba(model, cleaned) {
  const counts = new Map();
  for (const t of terms(cleaned)) {
    const i = model.vocab.get(t);
    if (i !== undefined) counts.set(i, (counts.get(i) ?? 0) + 1);
  }
  let norm = 0;
  const weights = [];
  for (const [i, c] of counts) {
    const w = (1 + Math.log(c)) * model.idf[i];
    weights.push([i, w]);
    norm += w * w;
  }
  norm = Math.sqrt(norm) || 1;
  const scores = model.intercept.map((b, k) => weights.reduce((s, [i, w]) => s + model.coef[k][i] * (w / norm), b));
  return softmax(scores);
}

// Temperature-scaled model probabilities, mixed with the baseline when a blend weight below 1 is set.
export function blendScores(logits, tfidf, { temperature = 1, weight = 1 } = {}) {
  const p = softmax(logits.map((v) => v / temperature));
  if (!tfidf || weight >= 1) return p;
  return p.map((v, k) => weight * v + (1 - weight) * tfidf[k]);
}
