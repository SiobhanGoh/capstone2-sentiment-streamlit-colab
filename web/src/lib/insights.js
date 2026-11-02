// Creator notes: descriptive pointers built only from what the comments show
// (labels, likes, questions, repeated words). Nothing here is a validated score or prediction.
import { cleanText } from './clean.js';
import { pct } from './analysis.js';

const QUESTION_WORDS = [
  'bila', 'macam mana', 'camne', 'kenapa', 'apa', 'mana', 'siapa', 'berapa', 'boleh tak', 'ada tak', 'betul ke', 'ke?',
  'how', 'when', 'why', 'what', 'where', 'who', 'which', 'can you', 'could you', 'will you', 'is there', 'do you', 'pls', 'please',
];

// Common Malay and English words that say little about a topic on their own.
const STOP = new Set(`
a an the and or but if so to of in on at for with from by as is are was were be been am it its this that these those i me my
you your we our they them their he she his her us not no yes do does did done have has had can could will would should just
very really too also more most much many some any all about into than then there here what when why how which who whom
lah lahh la je jer ja ni tu itu ini yang dan atau tapi kalau untuk dari pada dengan ke di dah sudah tak tidak x nak hendak
ada boleh akan saya aku kau awak kita kami korang dia mereka dorang pun lagi juga sangat sgt dgn yg utk tu kot ke ye ya eh
oh haha hahaha lol omg like get got go one dont don't im i'm it's thats that's video vid
`.split(/\s+/).filter(Boolean));

export function isQuestion(text) {
  const t = cleanText(text);
  if (!t) return false;
  if (t.includes('?')) return true;
  return QUESTION_WORDS.some((w) => t.startsWith(`${w} `) || t === w);
}

function words(text) {
  return cleanText(text).replace(/[^\p{L}\p{N}\s'-]/gu, ' ').split(/\s+/)
    .map((w) => w.replace(/^['-]+|['-]+$/g, ''))
    .filter((w) => w.length >= 3 && !STOP.has(w) && !/^\d+$/.test(w));
}

// Words that appear in at least `min` different comments of one label, counted once per comment.
export function recurringWords(rows, label, { min = 2, limit = 6 } = {}) {
  const counts = new Map();
  for (const r of rows) {
    if (r.sentiment !== label) continue;
    for (const w of new Set(words(r.comment))) counts.set(w, (counts.get(w) ?? 0) + 1);
  }
  return [...counts.entries()].filter(([, n]) => n >= min)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, limit)
    .map(([word, comments]) => ({ word, comments }));
}

export function creatorNotes(rows) {
  const hasLikes = rows.some((r) => Number.isFinite(r.likes));
  const byLikes = (a, b) => (b.likes ?? 0) - (a.likes ?? 0) || b.score - a.score;
  const negative = rows.filter((r) => r.sentiment === 'NEGATIVE');
  const positive = rows.filter((r) => r.sentiment === 'POSITIVE');
  const questions = rows.filter((r) => isQuestion(r.comment));
  const readFirst = [...negative].sort(hasLikes ? byLikes : (a, b) => b.score - a.score).slice(0, 3);
  const highlights = [...positive].sort(hasLikes ? byLikes : (a, b) => b.score - a.score).slice(0, 3);
  const negWords = recurringWords(rows, 'NEGATIVE');
  const posWords = recurringWords(rows, 'POSITIVE');

  const notes = [];
  if (negative.length) {
    notes.push(`${negative.length} of ${rows.length} comments (${pct(negative.length / rows.length, 0)}) were labelled negative. `
      + `Start with the ${hasLikes ? 'most-liked' : 'highest-scoring'} ones below, since ${hasLikes ? 'other viewers agreed with them' : 'the model was most decided on them'}.`);
  } else {
    notes.push('No comments were labelled negative in this set. Skim the neutral ones in case the model missed sarcasm or mild complaints.');
  }
  if (negWords.length) {
    notes.push(`Words that come up across negative comments: ${negWords.map((w) => `“${w.word}” (${w.comments})`).join(', ')}. `
      + 'If one of these points to a fixable issue, a pinned reply can address it once for everyone.');
  }
  if (questions.length) {
    notes.push(`${questions.length} ${questions.length === 1 ? 'comment asks' : 'comments ask'} a question. Answering a few, or covering them in a follow-up, is an easy way to engage.`);
  }
  if (posWords.length) {
    notes.push(`Positive comments often mention ${posWords.slice(0, 4).map((w) => `“${w.word}”`).join(', ')}, which may be what viewers liked most.`);
  }
  return { notes, readFirst, highlights, questions: questions.slice(0, 5), negWords, posWords, hasLikes };
}

export const NOTES_DISCLAIMER = 'These notes only point to comments worth reading. They come from the labels, likes and repeated words, not from a validated measure of audience mood or business risk.';
