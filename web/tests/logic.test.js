import { describe, expect, it } from 'vitest';
import { LABELS, checkLabelOrder, softmax, toResult } from '../src/lib/inference.js';
import { buildRows, splitPasted, summarise, usableComments } from '../src/lib/analysis.js';
import { guessTextColumn, safeCell, toCsv } from '../src/lib/csv.js';
import { extractVideoId } from '../src/lib/youtube.js';
import { printable } from '../src/lib/pdf.js';
import release from '../src/model_release.json';

describe('inference helpers', () => {
  it('softmax is stable and sums to one', () => {
    const s = softmax([1000, 1001, 999]);
    expect(s.reduce((a, b) => a + b)).toBeCloseTo(1, 12);
    expect(s[1]).toBeGreaterThan(s[0]);
  });
  it('flags truncation only above 128 tokens', () => {
    expect(toResult('x', [0, 0, 1], 128).truncated).toBe(false);
    expect(toResult('x', [0, 0, 1], 129).truncated).toBe(true);
    expect(toResult('x', [3, 0, 1], 5).sentiment).toBe('NEGATIVE');
  });
  it('rejects a model whose labels are in another order', () => {
    expect(() => checkLabelOrder({ 0: 'negative', 1: 'neutral', 2: 'positive' })).not.toThrow();
    expect(() => checkLabelOrder({ 0: 'POSITIVE', 1: 'NEUTRAL', 2: 'NEGATIVE' })).toThrow();
  });
  it('release labels match the protocol', () => {
    expect(release.labels).toEqual(LABELS);
    expect(release.max_length).toBe(128);
  });
});

describe('comment analysis', () => {
  it('splits pasted lines and drops blanks', () => {
    expect(splitPasted('a\r\n\n  b  \n')).toEqual(['a', 'b']);
  });
  it('excludes inputs that are empty after cleaning but keeps original numbering', () => {
    const { valid, excluded } = usableComments(['bagus', '@ali https://x.co', 'ok']);
    expect(excluded).toBe(1);
    expect(valid.map((v) => v.no)).toEqual([1, 3]);
  });
  it('summarises ties and shares', () => {
    const r = (sentiment, score) => ({ sentiment, score, scores: [0, 0, 0], truncated: false });
    const rows = buildRows([{ no: 1, comment: 'a' }, { no: 2, comment: 'b' }], [r('POSITIVE', 0.9), r('NEGATIVE', 0.7)]);
    const s = summarise(rows);
    expect(s.tied).toBe(true);
    expect(s.dominant).toBe('Negative / Positive');
    expect(s.shares.POSITIVE).toBe(0.5);
    expect(s.meanScore).toBeCloseTo(0.8);
    expect(() => summarise([])).toThrow();
  });
});

describe('exports', () => {
  it('neutralises spreadsheet formula prefixes', () => {
    for (const v of ['=SUM(A1)', '+1', '-2', '@x', '  =x']) expect(safeCell(v).startsWith("'")).toBe(true);
    expect(safeCell('fine')).toBe('fine');
    expect(safeCell(3)).toBe(3);
    expect(toCsv([{ a: '=1' }]).startsWith('﻿')).toBe(true);
  });
  it('guesses the text column', () => {
    expect(guessTextColumn(['id', 'Comment'], [])).toBe('Comment');
    expect(guessTextColumn(['id', 'body'], [{ id: '1', body: 'a long comment here' }])).toBe('body');
  });
  it('shows unsupported PDF characters as visible codes', () => {
    expect(printable('ok 😀')).toBe('ok [U+1F600]');
    expect(printable('Cafe é')).toBe('Cafe é');
  });
});

describe('YouTube video IDs', () => {
  it.each([
    ['dQw4w9WgXcQ', 'dQw4w9WgXcQ'],
    ['https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=3', 'dQw4w9WgXcQ'],
    ['youtu.be/dQw4w9WgXcQ', 'dQw4w9WgXcQ'],
    ['https://m.youtube.com/shorts/dQw4w9WgXcQ', 'dQw4w9WgXcQ'],
  ])('reads %s', (input, id) => expect(extractVideoId(input)).toBe(id));
  it.each(['https://evil.com/watch?v=dQw4w9WgXcQ', 'nope', 'https://youtube.com/channel/abc'])('rejects %s', (input) => {
    expect(() => extractVideoId(input)).toThrow(/valid YouTube/);
  });
});
