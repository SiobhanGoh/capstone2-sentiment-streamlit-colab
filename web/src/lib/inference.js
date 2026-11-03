// Pure helpers shared by the model worker and tests.
export const LABELS = ['NEGATIVE', 'NEUTRAL', 'POSITIVE'];
export const MAX_LENGTH = 128;
export const BATCH_SIZE = 8;

export function softmax(row) {
  const max = Math.max(...row);
  const exps = row.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export function checkLabelOrder(id2label) {
  const labels = LABELS.map((_, i) => String(id2label?.[i] ?? id2label?.[String(i)] ?? '').toUpperCase());
  if (labels.join() !== LABELS.join()) {
    throw new Error('The model label order does not match the evaluation protocol.');
  }
}

// `scores` are the final class probabilities (calibrated and blended); defaults to a plain softmax.
export function toResult(cleaned, logits, tokenCount, scores = softmax(logits)) {
  if (scores.length !== 3 || !scores.every(Number.isFinite)) {
    throw new Error('The model returned invalid scores.');
  }
  let top = 0;
  for (let i = 1; i < 3; i += 1) if (scores[i] > scores[top]) top = i;
  return {
    sentiment: LABELS[top],
    score: scores[top],
    scores,
    cleaned_text: cleaned,
    truncated: tokenCount > MAX_LENGTH,
  };
}
