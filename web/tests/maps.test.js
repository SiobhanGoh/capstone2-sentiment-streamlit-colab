import { describe, expect, it } from 'vitest';
import { dominantLabel, placeCounts, placesIn } from '../src/lib/places.js';
import { themeMatrix } from '../src/lib/insights.js';
import { toXY } from '../src/components/maps.jsx';

const row = (no, comment, sentiment) => ({ no, comment, sentiment, score: 0.8, scores: [0.1, 0.1, 0.8], truncated: false });

describe('places', () => {
  it('matches states and towns as whole words only', () => {
    expect(placesIn('Best char kway teow in Penang!')).toEqual(['PNG']);
    expect(placesIn('dari JB ke KL naik bas')).toEqual(['KUL', 'JHR']);
    expect(placesIn('Ipoh lagi sedap')).toEqual(['PRK']);
    expect(placesIn('klasik gila, jbg')).toEqual([]);
    expect(placesIn('Kuching and Kota Kinabalu')).toEqual(['SBH', 'SWK']);
  });
  it('counts mentions per state with their labels', () => {
    const c = placeCounts([row(1, 'penang best', 'POSITIVE'), row(2, 'penang teruk', 'NEGATIVE'), row(3, 'penang sedap', 'POSITIVE')]);
    expect(c.PNG.total).toBe(3);
    expect(dominantLabel(c.PNG)).toBe('POSITIVE');
    expect(dominantLabel(c.JHR)).toBeNull();
  });
});

describe('theme matrix', () => {
  it('counts each word once per comment across labels', () => {
    const t = themeMatrix([row(1, 'audio audio teruk', 'NEGATIVE'), row(2, 'audio ok', 'NEUTRAL'), row(3, 'editing best', 'POSITIVE')]);
    expect(t).toEqual([{ word: 'audio', total: 2, NEGATIVE: 1, NEUTRAL: 1, POSITIVE: 0 }]);
  });
});

describe('sentiment triangle', () => {
  it('puts certain scores on the corners and even scores in the centre', () => {
    expect(toXY([1, 0, 0])).toEqual([30, 322]);
    expect(toXY([0, 0, 1])).toEqual([370, 322]);
    const [x, y] = toXY([1 / 3, 1 / 3, 1 / 3]);
    expect(x).toBeCloseTo(200);
    expect(y).toBeCloseTo((322 + 28 + 322) / 3);
  });
});
