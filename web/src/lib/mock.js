// Development-only stand-in used when VITE_MOCK_MODEL=1, so the interface can be built
// without downloading the model. The interface labels these results as placeholders.
import { cleanText } from './clean.js';
import { toResult } from './inference.js';

const POS = ['bagus', 'best', 'suka', 'good', 'great', 'love', 'enjoyed', 'friendly', 'membantu', 'terbaik', 'thank'];
const NEG = ['lambat', 'teruk', 'kecewa', 'mengecewakan', 'bad', 'worst', 'disappointed', 'salah', 'benci', 'hate'];

export async function mockPredict(texts, onBatch) {
  const out = [];
  for (const text of texts) {
    const cleaned = cleanText(text);
    if (!cleaned) throw new Error('Please enter text that remains after links and mentions are removed.');
    const words = cleaned.split(' ');
    const p = words.filter((w) => POS.some((k) => w.includes(k))).length;
    const n = words.filter((w) => NEG.some((k) => w.includes(k))).length;
    out.push(toResult(cleaned, [n * 1.6, 0.9, p * 1.6], words.length * 1.3));
    onBatch?.(out.length, texts.length);
    await new Promise((r) => setTimeout(r, 15));
  }
  return out;
}
