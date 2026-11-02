// Fetch a video's comments and label them, shared by the Comments and Compare pages.
import { buildRows, usableComments } from './analysis.js';
import { fetchComments } from './youtube.js';
import { LABELS } from './inference.js';

export async function analyseComments(comments, meta, predict, onProgress, extras = {}) {
  const { valid, excluded } = usableComments(comments);
  if (!valid.length) throw new Error('None of these comments has text left after links and mentions are removed.');
  onProgress?.({ label: 'Analysing', done: 0, total: valid.length });
  const results = await predict(valid.map((v) => v.comment), (done, total) => onProgress?.({ label: 'Analysing', done, total }));
  const rows = buildRows(valid, results).map((r) => ({
    ...r,
    ...(extras.likes ? { likes: extras.likes[r.no - 1] ?? 0 } : {}),
    ...(extras.dates ? { date: extras.dates[r.no - 1] ?? null } : {}),
  }));
  return { rows, meta: { ...meta, excluded, analysed_at: new Date().toISOString() } };
}

export async function analyseYouTube(url, max, predict, onProgress) {
  onProgress?.({ label: 'Fetching comments from YouTube' });
  const data = await fetchComments(url, max);
  if (!data.comments?.length) throw new Error('No comments were found for this video.');
  return analyseComments(data.comments, data.meta, predict, onProgress, {
    likes: Array.isArray(data.likes) ? data.likes : null,
    dates: Array.isArray(data.dates) ? data.dates : null,
  });
}

// Comments where the top two class scores are within `margin` of each other.
export const NEAR_SPLIT = 0.2;
export function isNearSplit(r, margin = NEAR_SPLIT) {
  const [a, b] = [...r.scores].sort((x, y) => y - x);
  return a - b < margin;
}

export function shareDiffs(a, b) {
  return LABELS.map((l) => ({ label: l, a: a.shares[l], b: b.shares[l], points: Math.round((b.shares[l] - a.shares[l]) * 100) }));
}
