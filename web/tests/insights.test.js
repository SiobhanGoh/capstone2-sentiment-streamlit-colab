import { describe, expect, it } from 'vitest';
import { creatorNotes, isQuestion, recurringWords } from '../src/lib/insights.js';

const row = (no, comment, sentiment, score = 0.8, likes) => ({ no, comment, sentiment, score, scores: [0, 0, 0], truncated: false, likes });

describe('creator notes', () => {
  const rows = [
    row(1, 'Audio terlalu perlahan, susah dengar', 'NEGATIVE', 0.9, 2),
    row(2, 'audio teruk sangat', 'NEGATIVE', 0.7, 40),
    row(3, 'Bila nak upload part 2?', 'NEUTRAL', 0.6, 5),
    row(4, 'Best gila editing dia', 'POSITIVE', 0.95, 10),
    row(5, 'editing power, suka sangat', 'POSITIVE', 0.9, 1),
  ];
  it('spots questions in Malay and English', () => {
    expect(isQuestion('Bila nak upload part 2?')).toBe(true);
    expect(isQuestion('how did you edit this')).toBe(true);
    expect(isQuestion('macam mana nak beli')).toBe(true);
    expect(isQuestion('best video')).toBe(false);
  });
  it('counts recurring words once per comment and skips filler words', () => {
    expect(recurringWords(rows, 'NEGATIVE')).toEqual([{ word: 'audio', comments: 2 }]);
    expect(recurringWords(rows, 'POSITIVE').map((w) => w.word)).toEqual(['editing']);
  });
  it('puts the most-liked negative comment first when likes exist', () => {
    const n = creatorNotes(rows);
    expect(n.readFirst.map((r) => r.no)).toEqual([2, 1]);
    expect(n.questions.map((r) => r.no)).toEqual([3]);
    expect(n.notes.join(' ')).toMatch(/2 of 5 comments \(40%\) were labelled negative/);
    expect(n.notes.join(' ')).toMatch(/“audio” \(2\)/);
  });
  it('falls back to model score without likes, and handles no negatives', () => {
    const plain = rows.map(({ likes, ...r }) => r);
    expect(creatorNotes(plain).readFirst.map((r) => r.no)).toEqual([1, 2]);
    expect(creatorNotes([row(1, 'great', 'POSITIVE')]).notes[0]).toMatch(/No comments were labelled negative/);
  });
});
