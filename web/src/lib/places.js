// Malaysian states and federal territories, with common town names, laid out as a tile map.
// A comment that names a place counts for that state; it says nothing about where the viewer lives.
import { cleanText } from './clean.js';
import { LABELS } from './inference.js';

export const STATES = [
  { id: 'PLS', name: 'Perlis', col: 1, row: 0, aliases: ['perlis', 'kangar'] },
  { id: 'KDH', name: 'Kedah', col: 1, row: 1, aliases: ['kedah', 'alor setar', 'alor star', 'langkawi', 'sungai petani'] },
  { id: 'KTN', name: 'Kelantan', col: 2, row: 1, aliases: ['kelantan', 'kota bharu', 'kota bahru'] },
  { id: 'PNG', name: 'Pulau Pinang', col: 0, row: 2, aliases: ['penang', 'pulau pinang', 'georgetown', 'george town', 'butterworth'] },
  { id: 'PRK', name: 'Perak', col: 1, row: 2, aliases: ['perak', 'ipoh', 'taiping', 'teluk intan'] },
  { id: 'PHG', name: 'Pahang', col: 2, row: 2, aliases: ['pahang', 'kuantan', 'cameron highlands', 'genting'] },
  { id: 'TRG', name: 'Terengganu', col: 3, row: 2, aliases: ['terengganu', 'kuala terengganu'] },
  { id: 'SGR', name: 'Selangor', col: 0, row: 3, aliases: ['selangor', 'shah alam', 'petaling jaya', 'pj', 'subang', 'klang', 'puchong', 'kajang', 'bangi', 'cyberjaya'] },
  { id: 'KUL', name: 'Kuala Lumpur', col: 1, row: 3, aliases: ['kuala lumpur', 'kl', 'klcc', 'bukit bintang', 'jalan alor', 'cheras', 'bangsar'] },
  { id: 'PJY', name: 'Putrajaya', col: 0, row: 4, aliases: ['putrajaya'] },
  { id: 'NSN', name: 'Negeri Sembilan', col: 1, row: 4, aliases: ['negeri sembilan', 'seremban', 'port dickson', 'n9'] },
  { id: 'MLK', name: 'Melaka', col: 1, row: 5, aliases: ['melaka', 'malacca'] },
  { id: 'JHR', name: 'Johor', col: 2, row: 5, aliases: ['johor', 'johor bahru', 'jb', 'muar', 'batu pahat', 'kluang'] },
  { id: 'LBN', name: 'Labuan', col: 5, row: 1, aliases: ['labuan'] },
  { id: 'SBH', name: 'Sabah', col: 6, row: 1, aliases: ['sabah', 'kota kinabalu', 'sandakan', 'tawau'] },
  { id: 'SWK', name: 'Sarawak', col: 5, row: 2, aliases: ['sarawak', 'kuching', 'miri', 'sibu', 'bintulu'] },
];

const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const PATTERNS = STATES.map((s) => ({
  id: s.id,
  re: new RegExp(`(?<![\\p{L}\\p{N}])(?:${s.aliases.map(escape).join('|')})(?![\\p{L}\\p{N}])`, 'u'),
}));

export function placesIn(text) {
  const t = cleanText(text);
  return PATTERNS.filter((p) => p.re.test(t)).map((p) => p.id);
}

export function placeCounts(rows) {
  const counts = Object.fromEntries(STATES.map((s) => [s.id, { total: 0, NEGATIVE: 0, NEUTRAL: 0, POSITIVE: 0, rows: [] }]));
  for (const r of rows) {
    for (const id of placesIn(r.comment)) {
      counts[id].total += 1;
      counts[id][r.sentiment] += 1;
      counts[id].rows.push(r);
    }
  }
  return counts;
}

export function dominantLabel(c) {
  let best = null;
  for (const l of LABELS) if (c[l] > 0 && (!best || c[l] > c[best])) best = l;
  return best;
}
