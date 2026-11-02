import { describe, expect, it } from 'vitest';
import fixtures from './clean_fixtures.json';
import { cleanText } from '../src/lib/clean.js';

describe('cleanText matches the Python training preprocessing', () => {
  it.each(fixtures.map((f, i) => [i, f]))('case %i', (_, f) => {
    expect(cleanText(f.input)).toBe(f.expected);
  });
  it('returns empty for non-strings', () => {
    expect(cleanText(null)).toBe('');
    expect(cleanText(42)).toBe('');
  });
});
