import { cleanText } from './clean.js';
import { LABELS } from './inference.js';

export const SCORE_NOTE =
  'Scores are uncalibrated model outputs, not the chance of being correct or a measure of emotion intensity.';
export const SAMPLE_NOTE =
  'These results describe only the analysed comments. YouTube retrieval uses top-level comments ordered by relevance, not a representative audience sample.';
export const MAX_COMMENTS = 1000;

export function splitPasted(text) {
  return text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
}

// Keep only inputs that survive preprocessing, remembering their original position.
export function usableComments(comments) {
  const valid = [];
  comments.forEach((c, i) => { if (cleanText(c)) valid.push({ no: i + 1, comment: c }); });
  return { valid, excluded: comments.length - valid.length };
}

export function buildRows(valid, results) {
  return valid.map((v, i) => ({ ...v, ...results[i] }));
}

export function summarise(rows) {
  const total = rows.length;
  if (!total) throw new Error('Cannot summarise an empty analysis.');
  const counts = Object.fromEntries(LABELS.map((l) => [l, 0]));
  rows.forEach((r) => { counts[r.sentiment] += 1; });
  const max = Math.max(...Object.values(counts));
  const leaders = LABELS.filter((l) => counts[l] === max);
  return {
    total,
    counts,
    shares: Object.fromEntries(LABELS.map((l) => [l, counts[l] / total])),
    dominant: leaders.map(title).join(' / '),
    tied: leaders.length > 1,
    meanScore: rows.reduce((a, r) => a + r.score, 0) / total,
    truncated: rows.filter((r) => r.truncated).length,
  };
}

export function title(label) {
  return label.charAt(0) + label.slice(1).toLowerCase();
}

export function pct(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}
