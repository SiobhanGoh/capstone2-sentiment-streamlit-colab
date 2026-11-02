// A short plain-text summary to paste into a message or post.
import { LABELS } from './inference.js';
import { pct, summarise, title } from './analysis.js';
import { recurringWords } from './insights.js';

export const SITE = 'capstone2-sentiment-streamlit-colab.pages.dev';

export function summaryText({ rows, meta }) {
  const s = summarise(rows);
  const head = meta.source === 'YouTube'
    ? `"${meta.title}" by ${meta.channel}: ${s.total} top comments analysed.`
    : `${meta.title}: ${s.total} comments analysed.`;
  const mix = LABELS.map((l) => `${title(l)} ${pct(s.shares[l], 0)}`).join(', ');
  const neg = recurringWords(rows, 'NEGATIVE', { limit: 3 }).map((w) => w.word);
  const pos = recurringWords(rows, 'POSITIVE', { limit: 3 }).map((w) => w.word);
  return [
    head,
    `${mix}.`,
    neg.length ? `Negative comments often mention ${neg.join(', ')}.` : '',
    pos.length ? `Positive comments often mention ${pos.join(', ')}.` : '',
    `Labels are model suggestions from Rojak Sentiment (${SITE}).`,
  ].filter(Boolean).join('\n');
}
