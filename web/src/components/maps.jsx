import { useMemo, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { MapPin, ThumbsUp } from 'lucide-react';
import { LABELS } from '../lib/inference.js';
import { pct, title } from '../lib/analysis.js';
import { themeMatrix } from '../lib/insights.js';
import { STATES, dominantLabel, placeCounts } from '../lib/places.js';
import { Badge, Card, TONE } from './ui.jsx';
import { InfoTip } from './extras.jsx';

// ---------- Sentiment map: a triangle with one corner per label ----------
const V = { NEGATIVE: [30, 322], NEUTRAL: [200, 28], POSITIVE: [370, 322] };
const CENTRE = [(V.NEGATIVE[0] + V.NEUTRAL[0] + V.POSITIVE[0]) / 3, (V.NEGATIVE[1] + V.NEUTRAL[1] + V.POSITIVE[1]) / 3];
const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];

export function toXY(scores) {
  return [0, 1].map((k) => LABELS.reduce((sum, l, i) => sum + scores[i] * V[l][k], 0));
}

function regions() {
  const m = { NN: mid(V.NEGATIVE, V.NEUTRAL), NP: mid(V.NEGATIVE, V.POSITIVE), UP: mid(V.NEUTRAL, V.POSITIVE) };
  return {
    NEGATIVE: [V.NEGATIVE, m.NN, CENTRE, m.NP],
    NEUTRAL: [V.NEUTRAL, m.UP, CENTRE, m.NN],
    POSITIVE: [V.POSITIVE, m.NP, CENTRE, m.UP],
  };
}
const pts = (poly) => poly.map((p) => p.join(',')).join(' ');

export function SentimentTriangle({ rows, selected, onSelect, single = false }) {
  const reduce = useReducedMotion();
  const maxLikes = Math.max(1, ...rows.map((r) => r.likes ?? 0));
  const reg = regions();
  return (
    <svg viewBox="0 0 400 352" className="w-full" role="img"
      aria-label={single ? 'Where this text sits between negative, neutral and positive' : `${rows.length} comments placed between negative, neutral and positive by their scores`}>
      {LABELS.map((l) => <polygon key={l} points={pts(reg[l])} fill={TONE[l].hex} opacity="0.13" />)}
      <polygon points={pts([V.NEGATIVE, V.NEUTRAL, V.POSITIVE])} fill="none" stroke="currentColor" strokeOpacity="0.18" strokeWidth="1.5" strokeLinejoin="round" />
      {Object.values(reg).map((poly, i) => <polyline key={i} points={pts([poly[1], CENTRE])} stroke="currentColor" strokeOpacity="0.12" strokeDasharray="3 4" fill="none" />)}
      <circle cx={CENTRE[0]} cy={CENTRE[1]} r="34" fill="none" stroke="currentColor" strokeOpacity="0.25" strokeDasharray="2 4" />
      <text x={CENTRE[0]} y={CENTRE[1] + 48} textAnchor="middle" className="fill-current text-[10px] opacity-50">unsure</text>
      {LABELS.map((l) => {
        const [x, y] = V[l];
        const dy = l === 'NEUTRAL' ? -10 : 22;
        return <text key={l} x={x} y={y + dy} textAnchor={l === 'NEGATIVE' ? 'start' : l === 'POSITIVE' ? 'end' : 'middle'} fill={TONE[l].hex} className="text-[13px] font-bold">{title(l)}</text>;
      })}
      {rows.map((r, i) => {
        const [x, y] = toXY(r.scores);
        const radius = single ? 9 : 4 + Math.sqrt((r.likes ?? 0) / maxLikes) * 8;
        const active = selected?.no === r.no;
        return (
          <g key={r.no ?? i}>
            {(single || active) && !reduce && (
              <motion.circle cx={x} cy={y} r={radius} fill="none" stroke={TONE[r.sentiment].hex} strokeWidth="2"
                initial={{ r: radius, opacity: 0.8 }} animate={{ r: radius + 14, opacity: 0 }} transition={{ repeat: Infinity, duration: 1.6 }} />
            )}
            <motion.circle cx={x} cy={y} fill={TONE[r.sentiment].hex} fillOpacity={active || single ? 1 : 0.7}
              stroke="white" strokeWidth={active ? 2.5 : 1}
              initial={reduce ? false : { r: 0 }} animate={{ r: active ? radius + 2 : radius }}
              transition={{ delay: reduce ? 0 : Math.min(i * 0.012, 0.8), type: 'spring', stiffness: 260, damping: 18 }}
              className={onSelect ? 'cursor-pointer' : undefined}
              onMouseEnter={() => onSelect?.(r)} onClick={() => onSelect?.(r)} />
          </g>
        );
      })}
    </svg>
  );
}

function SentimentMap({ rows }) {
  const [selected, setSelected] = useState(null);
  const hasLikes = rows.some((r) => Number.isFinite(r.likes));
  return (
    <div className="grid gap-5 lg:grid-cols-[1.2fr_1fr] lg:items-center">
      <div className="text-ink dark:text-ube-50"><SentimentTriangle rows={rows} selected={selected} onSelect={setSelected} /></div>
      <div>
        <p className="text-sm text-ink-soft dark:text-ube-100">
          Each dot is a comment, placed by its three scores. Dots near a corner are ones the model was sure about; dots near the middle are the ones worth checking.{hasLikes ? ' Bigger dots got more likes.' : ''}
        </p>
        <AnimatePresence mode="wait">
          {selected ? (
            <motion.div key={selected.no} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              className={`mt-4 rounded-xl2 p-4 ${TONE[selected.sentiment].soft}`}>
              <div className="flex flex-wrap items-center gap-2 text-xs text-ink-soft dark:text-ube-100">
                <Badge label={selected.sentiment} /><span className="tabular-nums">#{selected.no}</span>
                {Number.isFinite(selected.likes) && <span className="inline-flex items-center gap-1 tabular-nums"><ThumbsUp size={12} aria-hidden />{selected.likes.toLocaleString()}</span>}
              </div>
              <p className="mt-2 line-clamp-6 whitespace-pre-wrap break-words">{selected.comment}</p>
              <p className="mt-2 text-xs tabular-nums text-ink-soft dark:text-ube-100">{LABELS.map((l, i) => `${title(l)} ${pct(selected.scores[i], 0)}`).join(', ')}</p>
            </motion.div>
          ) : (
            <motion.p key="hint" className="mt-4 rounded-xl2 border border-dashed border-ink-faint/40 p-4 text-sm text-ink-faint">
              Hover over or tap a dot to read the comment.
            </motion.p>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

// ---------- Theme map: recurring words against the three labels ----------
function ThemeMap({ rows, onTheme }) {
  const themes = useMemo(() => themeMatrix(rows), [rows]);
  if (!themes.length) return <p className="text-sm text-ink-faint">No word comes up in two or more comments yet. Try a video with more comments.</p>;
  const max = Math.max(...themes.flatMap((t) => LABELS.map((l) => t[l])), 1);
  return (
    <div>
      <p className="mb-4 text-sm text-ink-soft dark:text-ube-100">The words that come up most, and which label the comments using them got. Tap a word to see those comments.</p>
      <div className="grid max-w-2xl grid-cols-[minmax(0,1fr)_repeat(3,minmax(56px,90px))] items-center gap-x-2 gap-y-1.5">
        <span />
        {LABELS.map((l) => <span key={l} className={`text-center text-xs font-semibold ${TONE[l].text}`}>{title(l)}</span>)}
        {themes.map((t, i) => (
          <motion.div key={t.word} className="contents" initial={false}>
            <button onClick={() => onTheme(t.word)} className="truncate rounded-full px-3 py-1.5 text-left text-sm font-semibold hover:bg-ube-100 dark:hover:bg-night-raised">
              {t.word} <span className="font-normal text-ink-faint">{t.total}</span>
            </button>
            {LABELS.map((l) => {
              const n = t[l];
              const size = n ? 14 + (n / max) * 30 : 0;
              return (
                <button key={l} onClick={() => onTheme(t.word)} aria-label={`${t.word}: ${n} ${title(l).toLowerCase()} comments`}
                  className="grid h-12 place-items-center">
                  {n > 0 ? (
                    <motion.span className="grid place-items-center rounded-full text-[11px] font-bold text-ink"
                      style={{ background: TONE[l].hex, width: size, height: size }}
                      initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: i * 0.04, type: 'spring', stiffness: 220, damping: 16 }}>
                      {n}
                    </motion.span>
                  ) : <span className="h-1.5 w-1.5 rounded-full bg-ink-faint/30" />}
                </button>
              );
            })}
          </motion.div>
        ))}
      </div>
    </div>
  );
}

// ---------- Places map: Malaysian states named in comments ----------
function PlacesMap({ rows }) {
  const counts = useMemo(() => placeCounts(rows), [rows]);
  const mentioned = STATES.filter((s) => counts[s.id].total > 0);
  const [pick, setPick] = useState(null);
  const active = pick ?? mentioned.sort((a, b) => counts[b.id].total - counts[a.id].total)[0] ?? null;
  const max = Math.max(1, ...STATES.map((s) => counts[s.id].total));
  const cell = 58;
  const gap = 6;
  const width = 7 * (cell + gap);
  const height = 6 * (cell + gap) + 16;
  return (
    <div className="grid gap-5 lg:grid-cols-[1.3fr_1fr] lg:items-start">
      <div>
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label="Tile map of Malaysian states, coloured by how comments that name them were labelled">
          <text x={(cell + gap) * 1.5} y={height - 2} textAnchor="middle" className="fill-current text-[11px] opacity-40">Peninsular</text>
          <text x={(cell + gap) * 6} y={(cell + gap) * 3 + 14} textAnchor="middle" className="fill-current text-[11px] opacity-40">Borneo</text>
          {STATES.map((s, i) => {
            const c = counts[s.id];
            const dom = dominantLabel(c);
            const x = s.col * (cell + gap);
            const y = s.row * (cell + gap);
            const on = c.total > 0;
            const selectedTile = active?.id === s.id;
            return (
              <motion.g key={s.id} className={on ? 'cursor-pointer' : undefined} onClick={() => on && setPick(s)}
                initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.03 }}
                style={{ transformOrigin: `${x + cell / 2}px ${y + cell / 2}px` }}>
                <title>{`${s.name}: ${c.total ? `${c.total} ${c.total === 1 ? 'comment' : 'comments'}` : 'not named'}`}</title>
                <rect x={x} y={y} width={cell} height={cell} rx="14"
                  fill={on ? TONE[dom].hex : 'currentColor'} fillOpacity={on ? 0.35 + (c.total / max) * 0.6 : 0.06}
                  stroke={selectedTile ? 'currentColor' : 'none'} strokeWidth="2.5" />
                <text x={x + cell / 2} y={y + cell / 2 - 2} textAnchor="middle" className="fill-current text-[12px] font-bold">{s.id}</text>
                <text x={x + cell / 2} y={y + cell / 2 + 14} textAnchor="middle" className="fill-current text-[11px] opacity-70">{on ? c.total : ''}</text>
              </motion.g>
            );
          })}
        </svg>
        <div className="mt-2 flex flex-wrap gap-3 text-xs text-ink-soft dark:text-ube-100">
          {LABELS.map((l) => <span key={l} className="flex items-center gap-1.5"><span className="h-3 w-3 rounded" style={{ background: TONE[l].hex }} />Mostly {title(l).toLowerCase()}</span>)}
          <span className="flex items-center gap-1.5"><span className="h-3 w-3 rounded bg-ink-faint/20" />Not named</span>
        </div>
      </div>
      <div>
        <p className="text-sm text-ink-soft dark:text-ube-100">
          States lit up when a comment names them or a well-known town in them, such as Penang, Ipoh, JB or Kuching. A mention isn't where the viewer is from.
        </p>
        {active ? (
          <div className="mt-4">
            <p className="flex items-center gap-1.5 font-display text-xl font-bold"><MapPin size={18} aria-hidden />{active.name}</p>
            <p className="mt-1 text-sm text-ink-soft dark:text-ube-100">
              {counts[active.id].total} {counts[active.id].total === 1 ? 'comment' : 'comments'}: {LABELS.map((l) => `${counts[active.id][l]} ${title(l).toLowerCase()}`).join(', ')}
            </p>
            <ul className="mt-3 max-h-64 space-y-2 overflow-y-auto pr-1">
              {counts[active.id].rows.slice(0, 20).map((r) => (
                <li key={r.no} className={`rounded-xl2 px-3 py-2 text-sm ${TONE[r.sentiment].soft}`}>
                  <p className="line-clamp-3 break-words">{r.comment}</p>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <p className="mt-4 rounded-xl2 border border-dashed border-ink-faint/40 p-4 text-sm text-ink-faint">No Malaysian states or major towns are named in these comments.</p>
        )}
      </div>
    </div>
  );
}

const TABS = [
  { id: 'sentiment', name: 'Sentiment map' },
  { id: 'themes', name: 'Theme map' },
  { id: 'places', name: 'Places map' },
];

export default function ExploreMaps({ rows, onTheme }) {
  const [tab, setTab] = useState('sentiment');
  return (
    <Card>
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="mr-auto flex items-center gap-1 text-2xl font-semibold">
          Explore the comments
          <InfoTip label="About these maps">These maps only rearrange the labelled comments so patterns are easier to spot. Positions come from uncalibrated model scores, and word and place matches are simple text matches.</InfoTip>
        </h2>
        <div role="tablist" aria-label="Map" className="neu-inset inline-flex flex-wrap gap-1 rounded-full p-1">
          {TABS.map((t) => (
            <button key={t.id} role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}
              className={`relative rounded-full px-4 py-2 text-sm font-semibold ${tab === t.id ? 'text-ink dark:text-ube-50' : 'text-ink-soft dark:text-ube-300'}`}>
              {tab === t.id && <motion.span layoutId="map-pill" className="neu-raised absolute inset-0 rounded-full" transition={{ type: 'spring', stiffness: 300, damping: 28 }} />}
              <span className="relative">{t.name}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="mt-5">
        {tab === 'sentiment' && <SentimentMap rows={rows} />}
        {tab === 'themes' && <ThemeMap rows={rows} onTheme={onTheme} />}
        {tab === 'places' && <PlacesMap rows={rows} />}
      </div>
    </Card>
  );
}
