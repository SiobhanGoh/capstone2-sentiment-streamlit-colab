import { useId, useMemo, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { LABELS } from '../lib/inference.js';
import { title } from '../lib/analysis.js';
import { bucketEnd, bucketLabel, timeBuckets } from '../lib/timeline.js';
import { TONE } from './ui.jsx';

const W = 640;
const H = 220;
const PAD = { top: 12, bottom: 28, side: 8 };

// Smooth line through points using horizontal-tangent cubic segments.
function smooth(points) {
  if (!points.length) return '';
  let d = `M${points[0][0]},${points[0][1]}`;
  for (let i = 1; i < points.length; i += 1) {
    const [x0, y0] = points[i - 1];
    const [x1, y1] = points[i];
    const mx = (x0 + x1) / 2;
    d += ` C${mx},${y0} ${mx},${y1} ${x1},${y1}`;
  }
  return d;
}

export default function StreamTimeline({ rows, range, onBrush, focus }) {
  const reduce = useReducedMotion();
  const clip = useId().replace(/:/g, '');
  const t = useMemo(() => timeBuckets(rows), [rows]);
  const [hover, setHover] = useState(null);
  if (!t) return <p className="text-small text-ink-faint">Not enough dated comments to draw a timeline.</p>;
  const n = t.buckets.length;
  const max = Math.max(...t.buckets.map((b) => b.total), 1);
  const step = n > 1 ? (W - PAD.side * 2) / (n - 1) : 0;
  const x = (i) => (n > 1 ? PAD.side + i * step : W / 2);
  const y = (v) => H - PAD.bottom - (v / max) * (H - PAD.top - PAD.bottom);
  // Sparse data (few periods or many empty ones) reads better as bars than as a smooth stream.
  const zeros = t.buckets.filter((b) => b.total === 0).length;
  const bars = n < 20 || zeros / n > 0.1;
  const slot = (W - PAD.side * 2) / n;
  const colW = bars ? slot : n > 1 ? step : W - PAD.side * 2;
  const cx = (i) => (bars ? PAD.side + slot * (i + 0.5) : x(i));
  // Stack bottom to top: negative, neutral, positive.
  const layers = LABELS.map((l, k) => {
    const low = t.buckets.map((b) => LABELS.slice(0, k).reduce((a, m) => a + b[m], 0));
    const high = t.buckets.map((b, i) => low[i] + b[l]);
    const top = smooth(high.map((v, i) => [x(i), y(v)]));
    const bottom = smooth(low.map((v, i) => [x(i), y(v)]).reverse()).replace(/^M/, 'L');
    return { l, d: `${top} ${bottom} Z` };
  });
  const active = hover ?? (range ? t.buckets.findIndex((b) => b.start === range.start) : -1);
  const ab = active >= 0 ? t.buckets[active] : null;
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full overflow-visible" role="img"
        aria-label={`Comments per ${t.unit} by label, from ${bucketLabel(t.buckets[0].start, t.unit)} to ${bucketLabel(t.buckets.at(-1).start, t.unit)}`}>
        <defs>
          <clipPath id={clip}>
            <motion.rect x="0" y="0" height={H} initial={reduce ? false : { width: 0 }} animate={{ width: W }} transition={{ duration: 0.9, ease: 'easeOut' }} />
          </clipPath>
        </defs>
        <g clipPath={`url(#${clip})`}>
          {!bars ? layers.map(({ l, d }) => (
            <motion.path key={l} d={d} fill={TONE[l].hex} animate={{ opacity: focus && focus !== l ? 0.15 : 0.75 }} transition={{ duration: 0.24 }} />
          )) : t.buckets.map((b, i) => {
            const bw = Math.min(slot * 0.7, 56);
            return (
              <motion.g key={b.start} initial={reduce ? false : { scaleY: 0 }} animate={{ scaleY: 1 }}
                transition={{ delay: reduce ? 0 : i * 0.03, type: 'spring', stiffness: 140, damping: 18 }}
                style={{ transformOrigin: `${cx(i)}px ${H - PAD.bottom}px` }}>
                {LABELS.map((l, k) => {
                  const low = LABELS.slice(0, k).reduce((a, m) => a + b[m], 0);
                  if (!b[l]) return null;
                  return (
                    <motion.rect key={l} x={cx(i) - bw / 2} width={bw} y={y(low + b[l])} height={y(low) - y(low + b[l])} rx="4"
                      fill={TONE[l].hex} animate={{ opacity: focus && focus !== l ? 0.15 : 0.85 }} transition={{ duration: 0.24 }} />
                  );
                })}
              </motion.g>
            );
          })}
        </g>
        {range && active >= 0 && (
          <rect x={cx(active) - colW / 2} y={PAD.top} width={colW} height={H - PAD.top - PAD.bottom} rx="8" fill="currentColor" opacity="0.08" />
        )}
        {ab && !bars && <line x1={x(active)} x2={x(active)} y1={PAD.top} y2={H - PAD.bottom} stroke="currentColor" strokeOpacity="0.4" strokeDasharray="3 3" />}
        {t.buckets.map((b, i) => (
          <rect key={b.start} x={cx(i) - colW / 2} y="0" width={colW} height={H} fill={hover === i && bars ? 'currentColor' : 'transparent'} fillOpacity={hover === i && bars ? 0.05 : 0} className="cursor-pointer"
            onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
            onClick={() => onBrush(range?.start === b.start ? null : { start: b.start, end: bucketEnd(b.start, t.unit), label: bucketLabel(b.start, t.unit), unit: t.unit })}>
            <title>{`${bucketLabel(b.start, t.unit)}: ${b.total} comments`}</title>
          </rect>
        ))}
        {t.buckets.map((b, i) => (i % Math.max(1, Math.ceil(n / 7)) === 0 || i === n - 1) && (
          <text key={b.start} x={cx(i)} y={H - 8} textAnchor={bars ? 'middle' : i === 0 ? 'start' : i === n - 1 ? 'end' : 'middle'} className="fill-current text-[11px] opacity-50">{bucketLabel(b.start, t.unit)}</text>
        ))}
      </svg>
      <div className="mt-2 flex min-h-[1.5rem] flex-wrap items-center gap-x-4 gap-y-1 text-small">
        {ab ? (
          <>
            <span className="font-semibold">{bucketLabel(ab.start, t.unit)}{t.unit === 'week' ? ' week' : ''}: {ab.total}</span>
            {LABELS.map((l) => <span key={l} className={TONE[l].text}>{ab[l]} {title(l).toLowerCase()}</span>)}
          </>
        ) : <span className="text-ink-faint">Grouped by {t.unit}. Click a {t.unit} to show only its comments.</span>}
      </div>
    </div>
  );
}
