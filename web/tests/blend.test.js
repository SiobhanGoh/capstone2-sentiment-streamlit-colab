import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { createHash } from 'node:crypto';
import fixtures from './tfidf_fixtures.json';
import release from '../src/model_release.json';
import { blendScores, prepareTfidf, terms, tfidfProba } from '../src/lib/blend.js';
import { softmax } from '../src/lib/inference.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const raw = readFileSync(path.join(here, '../public/tfidf.json'), 'utf8');
const model = prepareTfidf(JSON.parse(raw));

describe('TF-IDF baseline in the browser', () => {
  it('builds unigrams and adjacent bigrams like scikit-learn', () => {
    expect(terms('best gila, a video ni')).toEqual(['best', 'gila', 'video', 'ni', 'best gila', 'gila video', 'video ni']);
  });
  it.each(fixtures.map((f, i) => [i, f]))('matches scikit-learn probabilities, case %i', (_, f) => {
    const p = tfidfProba(model, f.text);
    p.forEach((v, k) => expect(v).toBeCloseTo(f.expected[k], 5));
  });
  it('ships the exact file the release pins', () => {
    expect(createHash('sha256').update(raw).digest('hex')).toBe(release.scoring.tfidf.sha256);
  });
});

describe('blend', () => {
  it('is a plain softmax with temperature 1 and no baseline', () => {
    expect(blendScores([1, 2, 3], null)).toEqual(softmax([1, 2, 3]));
  });
  it('softens with temperature and mixes by weight', () => {
    const p = blendScores([0, 0, 3], [1, 0, 0], { temperature: 1.5, weight: 0.75 });
    const m = softmax([0, 0, 2]);
    p.forEach((v, k) => expect(v).toBeCloseTo(0.75 * m[k] + (k === 0 ? 0.25 : 0), 12));
    expect(p.reduce((a, b) => a + b)).toBeCloseTo(1, 12);
  });
  it('uses the validated settings', () => {
    expect(release.scoring.method).toBe('blend');
    expect(release.scoring.blend_weight).toBeGreaterThan(0);
    expect(release.scoring.blend_weight).toBeLessThan(1);
    expect(release.scoring.temperature).toBeGreaterThan(1);
  });
});
