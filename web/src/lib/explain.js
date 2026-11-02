// "Why this label?": leave-one-word-out. Each word is removed in turn and the text is scored
// again; a word matters more when removing it lowers the predicted label's score more.
// This is an approximation of influence, not the model's internal reasoning.
import { cleanText } from './clean.js';
import { LABELS } from './inference.js';

export const MAX_EXPLAIN_WORDS = 60;

export function splitWords(text) {
  return cleanText(text).split(' ').filter(Boolean);
}

export function variantsWithoutEachWord(words) {
  return words.map((_, i) => words.filter((__, j) => j !== i).join(' '));
}

// delta > 0: the word supports the predicted label. delta < 0: it pulls the other way.
export function wordImpacts(words, baseScores, variantScores, label) {
  const k = LABELS.indexOf(label);
  return words.map((word, i) => ({ word, delta: baseScores[k] - (variantScores[i]?.[k] ?? baseScores[k]) }));
}

export async function explain(text, base, predict) {
  const words = splitWords(text);
  if (!words.length) throw new Error('Add some words first.');
  if (words.length > MAX_EXPLAIN_WORDS) {
    throw new Error(`Word highlights work on texts up to ${MAX_EXPLAIN_WORDS} words. Try a shorter text.`);
  }
  if (words.length === 1) return [{ word: words[0], delta: base.score }];
  const results = await predict(variantsWithoutEachWord(words));
  return wordImpacts(words, base.scores, results.map((r) => r.scores), base.sentiment);
}
