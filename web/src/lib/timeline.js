// Groups dated comments into days, weeks or months for the "when were these posted" chart.
const DAY = 86400000;

function startOfWeek(d) {
  const t = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
  const dow = (new Date(t).getUTCDay() + 6) % 7; // Monday = 0
  return t - dow * DAY;
}

export function timeBuckets(rows) {
  const dated = rows.filter((r) => r.date && !Number.isNaN(Date.parse(r.date)));
  if (dated.length < 3) return null;
  const times = dated.map((r) => Date.parse(r.date));
  const span = Math.max(...times) - Math.min(...times);
  const unit = span <= 21 * DAY ? 'day' : span <= 26 * 7 * DAY ? 'week' : 'month';
  const keyOf = (t) => {
    const d = new Date(t);
    if (unit === 'day') return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
    if (unit === 'week') return startOfWeek(d);
    return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), 1);
  };
  const map = new Map();
  dated.forEach((r, i) => {
    const k = keyOf(times[i]);
    const b = map.get(k) ?? { start: k, NEGATIVE: 0, NEUTRAL: 0, POSITIVE: 0, total: 0 };
    b[r.sentiment] += 1;
    b.total += 1;
    map.set(k, b);
  });
  // Fill empty periods so gaps show as gaps.
  const keys = [...map.keys()].sort((a, b) => a - b);
  const buckets = [];
  for (let k = keys[0]; k <= keys.at(-1);) {
    buckets.push(map.get(k) ?? { start: k, NEGATIVE: 0, NEUTRAL: 0, POSITIVE: 0, total: 0 });
    if (unit === 'day') k += DAY;
    else if (unit === 'week') k += 7 * DAY;
    else { const d = new Date(k); k = Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 1); }
    if (buckets.length > 120) break;
  }
  return { unit, buckets, dated: dated.length };
}

export function bucketLabel(start, unit) {
  const d = new Date(start);
  if (unit === 'month') return d.toLocaleDateString('en-GB', { month: 'short', year: '2-digit', timeZone: 'UTC' });
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', timeZone: 'UTC' });
}
