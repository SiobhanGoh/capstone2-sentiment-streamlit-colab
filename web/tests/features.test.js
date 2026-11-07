import { beforeEach, describe, expect, it } from 'vitest';
import { explain, splitWords, variantsWithoutEachWord, wordImpacts } from '../src/lib/explain.js';
import { addCorrection, clearCorrections, correctionsCsv, findCorrection, loadCorrections, removeCorrection } from '../src/lib/feedback.js';
import { bucketLabel, timeBuckets } from '../src/lib/timeline.js';
import { CHECK_GAP, isNearSplit, shareDiffs } from '../src/lib/pipeline.js';
import { summarise } from '../src/lib/analysis.js';
import { summaryText } from '../src/lib/summaryText.js';
import { shouldAutoLoad } from '../src/lib/useModel.js';

const row = (no, sentiment, scores, extra = {}) => ({ no, comment: `c${no}`, sentiment, score: Math.max(...scores), scores, truncated: false, ...extra });

describe('word highlights', () => {
  it('builds one variant per removed word from the cleaned text', () => {
    expect(splitWords('Best GILA video @ali https://x.co')).toEqual(['best', 'gila', 'video']);
    expect(variantsWithoutEachWord(['a', 'b', 'c'])).toEqual(['b c', 'a c', 'a b']);
  });
  it('scores each word by how much the predicted label drops without it', () => {
    const impacts = wordImpacts(['best', 'video'], [0.1, 0.1, 0.8], [[0.3, 0.4, 0.3], [0.05, 0.05, 0.9]], 'POSITIVE');
    expect(impacts[0].delta).toBeCloseTo(0.5);
    expect(impacts[1].delta).toBeCloseTo(-0.1);
  });
  it('runs through predict and refuses very long texts', async () => {
    const predict = async (texts) => texts.map(() => ({ scores: [0.2, 0.2, 0.6] }));
    const out = await explain('sedap gila', { sentiment: 'POSITIVE', score: 0.9, scores: [0.05, 0.05, 0.9] }, predict);
    expect(out.map((w) => w.word)).toEqual(['sedap', 'gila']);
    expect(out[0].delta).toBeCloseTo(0.3);
    await expect(explain(Array(61).fill('w').join(' '), { sentiment: 'POSITIVE', scores: [0, 0, 1] }, predict)).rejects.toThrow(/up to 60 words/);
  });
});

describe('label corrections', () => {
  beforeEach(() => { localStorage.clear(); clearCorrections(); });
  it('saves, replaces, finds and removes corrections per text and source', () => {
    addCorrection({ text: 'best gila', predicted: 'NEGATIVE', corrected: 'POSITIVE', scores: [0.6, 0.1, 0.3] });
    addCorrection({ text: 'best gila', predicted: 'NEGATIVE', corrected: 'NEUTRAL', scores: [0.6, 0.1, 0.3] });
    addCorrection({ text: 'best gila', predicted: 'NEGATIVE', corrected: 'POSITIVE', scores: [0.6, 0.1, 0.3], source: 'YouTube x' });
    const items = loadCorrections();
    expect(items).toHaveLength(2);
    expect(findCorrection(items, 'best gila', undefined).corrected).toBe('NEUTRAL');
    removeCorrection('best gila', 'YouTube x');
    expect(loadCorrections()).toHaveLength(1);
    expect(correctionsCsv(loadCorrections())).toMatch(/Your label/);
  });
});

describe('timeline', () => {
  it('groups by day for short spans and fills gaps', () => {
    const rows = [row(1, 'POSITIVE', [0, 0, 1], { date: '2026-10-01T05:00:00Z' }), row(2, 'NEGATIVE', [1, 0, 0], { date: '2026-10-01T09:00:00Z' }),
      row(3, 'POSITIVE', [0, 0, 1], { date: '2026-10-03T00:00:00Z' })];
    const t = timeBuckets(rows);
    expect(t.unit).toBe('day');
    expect(t.buckets.map((b) => b.total)).toEqual([2, 0, 1]);
    expect(t.buckets[0].NEGATIVE).toBe(1);
    expect(bucketLabel(t.buckets[0].start, 'day')).toBe('1 Oct');
  });
  it('switches to months for long spans and needs at least three dates', () => {
    const rows = ['2025-01-15', '2025-06-02', '2026-02-20'].map((d, i) => row(i + 1, 'NEUTRAL', [0, 1, 0], { date: `${d}T00:00:00Z` }));
    expect(timeBuckets(rows).unit).toBe('month');
    expect(timeBuckets(rows.slice(0, 2))).toBeNull();
  });
});

describe('comparison and summaries', () => {
  it('flags comments where the top two scores are close', () => {
    expect(isNearSplit({ scores: [0.45, 0.35, 0.2] })).toBe(true);
    expect(isNearSplit({ scores: [0.85, 0.1, 0.05] })).toBe(false);
    // The validated cut-off (abstain.json): a gap just under it needs checking, just over it does not.
    expect(CHECK_GAP).toBeGreaterThan(0.5);
    expect(CHECK_GAP).toBeLessThan(0.6);
    const at = (gap) => ({ scores: [(1 + gap) / 2, (1 - gap) / 2, 0] });
    expect(isNearSplit(at(CHECK_GAP - 0.01))).toBe(true);
    expect(isNearSplit(at(CHECK_GAP + 0.01))).toBe(false);
  });
  it('reports share differences in points', () => {
    const a = summarise([row(1, 'NEGATIVE', [1, 0, 0]), row(2, 'POSITIVE', [0, 0, 1])]);
    const b = summarise([row(1, 'NEGATIVE', [1, 0, 0]), row(2, 'NEGATIVE', [1, 0, 0]), row(3, 'POSITIVE', [0, 0, 1]), row(4, 'NEUTRAL', [0, 1, 0])]);
    expect(shareDiffs(a, b).map((d) => d.points)).toEqual([0, 25, -25]);
  });
  it('writes a short pasteable summary', () => {
    const text = summaryText({ rows: [row(1, 'POSITIVE', [0, 0, 1]), row(2, 'NEGATIVE', [1, 0, 0])], meta: { source: 'YouTube', title: 'T', channel: 'C' } });
    expect(text).toMatch(/^"T" by C: 2 top comments analysed\./);
    expect(text).toMatch(/Negative 50%, Neutral 0%, Positive 50%\./);
    expect(text).toMatch(/model suggestions/);
  });
});

describe('model download timing', () => {
  const win = (coarse, width) => ({ matchMedia: () => ({ matches: coarse }), innerWidth: width });
  it('waits on data saver, cellular, slow links and phones without connection info', () => {
    expect(shouldAutoLoad({ connection: { saveData: true } }, win(false, 1400))).toBe(false);
    expect(shouldAutoLoad({ connection: { type: 'cellular' } }, win(true, 400))).toBe(false);
    expect(shouldAutoLoad({ connection: { effectiveType: '3g' } }, win(false, 1400))).toBe(false);
    expect(shouldAutoLoad({}, win(true, 390))).toBe(false);
  });
  it('loads straight away on desktop or Wi-Fi', () => {
    expect(shouldAutoLoad({}, win(false, 1400))).toBe(true);
    expect(shouldAutoLoad({ connection: { type: 'wifi', effectiveType: '4g' } }, win(true, 390))).toBe(true);
  });
});

describe('period and brush filters', () => {
  const now = Date.UTC(2026, 10, 3);
  const rows = [5, 20, 40].map((d, i) => ({ no: i + 1, sentiment: 'NEUTRAL', scores: [0, 1, 0], date: new Date(now - d * 86400000).toISOString() }));
  it('keeps comments inside the chosen period', async () => {
    const { filterRows, periodCounts } = await import('../src/lib/timeline.js');
    expect(filterRows(rows, { period: '7', now }).map((r) => r.no)).toEqual([1]);
    expect(filterRows(rows, { period: '30', now }).map((r) => r.no)).toEqual([1, 2]);
    expect(periodCounts(rows, now)).toEqual({ all: 3, 30: 2, 7: 1 });
  });
  it('narrows to a brushed range and drops undated rows only when filtering', async () => {
    const { filterRows } = await import('../src/lib/timeline.js');
    const range = { start: now - 25 * 86400000, end: now - 10 * 86400000 };
    expect(filterRows(rows, { range, now }).map((r) => r.no)).toEqual([2]);
    expect(filterRows([...rows, { no: 9 }], { now })).toHaveLength(4);
    expect(filterRows([...rows, { no: 9 }], { period: '30', now })).toHaveLength(2);
  });
});
