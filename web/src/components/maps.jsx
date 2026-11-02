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

export function SentimentTriangle({ rows, selected, onSelect, onPick, focus, single = false }) {
  const reduce = useReducedMotion();
  const maxLikes = Math.max(1, ...rows.map((r) => r.likes ?? 0));
  const reg = regions();
  return (
    <svg viewBox="0 0 400 352" className="w-full" role="img"
      aria-label={single ? 'Where this text sits between negative, neutral and positive' : `${rows.length} comments placed between negative, neutral and positive by their scores`}>
      {LABELS.map((l) => (
        <motion.polygon key={l} points={pts(reg[l])} fill={TONE[l].hex} animate={{ opacity: focus ? (focus === l ? 0.24 : 0.05) : 0.13 }} transition={{ duration: 0.24 }} />
      ))}
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
        const dim = focus && focus !== r.sentiment;
        return (
          <g key={r.no ?? i}>
            {(single || active) && !reduce && (
              <motion.circle cx={x} cy={y} r={radius} fill="none" stroke={TONE[r.sentiment].hex} strokeWidth="2"
                initial={{ r: radius, opacity: 0.8 }} animate={{ r: radius + 14, opacity: 0 }} transition={{ repeat: Infinity, duration: 1.6 }} />
            )}
            <motion.circle fill={TONE[r.sentiment].hex}
              stroke="white" strokeWidth={active ? 2.5 : 1}
              initial={reduce ? false : { r: 0, cx: CENTRE[0], cy: CENTRE[1] }}
              animate={{ r: active ? radius + 2 : radius, cx: x, cy: y, fillOpacity: dim ? 0.12 : active || single ? 1 : 0.72 }}
              transition={{ delay: reduce ? 0 : Math.min(i * 0.01, 0.6), type: 'spring', stiffness: 170, damping: 20 }}
              className={onSelect ? 'cursor-pointer' : undefined}
              onMouseEnter={() => onSelect?.(r)} onClick={() => { onSelect?.(r); onPick?.(r); }} />
          </g>
        );
      })}
    </svg>
  );
}

function SentimentMap({ rows, focus, onPick }) {
  const [selected, setSelected] = useState(null);
  const hasLikes = rows.some((r) => Number.isFinite(r.likes));
  return (
    <div className="grid gap-5 lg:grid-cols-[1.2fr_1fr] lg:items-center">
      <div className="text-ink dark:text-ube-50"><SentimentTriangle rows={rows} selected={selected} onSelect={setSelected} focus={focus} /></div>
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
              {onPick && <button className="mt-3 text-small font-semibold text-ube-700 underline-offset-2 hover:underline dark:text-ube-100" onClick={() => onPick(selected)}>Find it in the list</button>}
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

// ---------- Theme flow: recurring words streaming into the labels that used them ----------
function ThemeFlow({ rows, onTheme, focus }) {
  const reduce = useReducedMotion();
  const themes = useMemo(() => themeMatrix(rows, { limit: 8 }), [rows]);
  const [hot, setHot] = useState(null);
  if (!themes.length) return <p className="text-small text-ink-faint">No word comes up in two or more comments yet. Try a video with more comments.</p>;
  const rowH = 40;
  const H = Math.max(themes.length * rowH + 24, 200);
  const leftX = 150;
  const rightX = 520;
  const totals = Object.fromEntries(LABELS.map((l) => [l, themes.reduce((a, t) => a + t[l], 0)]));
  const all = LABELS.reduce((a, l) => a + totals[l], 0) || 1;
  // Right-hand label nodes, stacked with gaps, sized by how many theme mentions flow into them.
  let cursor = 12;
  const usable = H - 24 - 2 * 16;
  const nodes = Object.fromEntries(LABELS.map((l) => {
    const h = Math.max(18, (totals[l] / all) * usable);
    const n = { y: cursor, h, used: 0 };
    cursor += h + 16;
    return [l, n];
  }));
  const maxFlow = Math.max(1, ...themes.flatMap((t) => LABELS.map((l) => t[l])));
  const links = [];
  themes.forEach((t, i) => {
    const y0 = 12 + i * rowH + rowH / 2;
    LABELS.forEach((l) => {
      if (!t[l]) return;
      const node = nodes[l];
      const share = totals[l] ? t[l] / totals[l] : 0;
      const y1 = node.y + node.used + (share * node.h) / 2;
      node.used += share * node.h;
      links.push({ key: `${t.word}-${l}`, word: t.word, l, n: t[l], d: `M${leftX + 8},${y0} C${(leftX + rightX) / 2},${y0} ${(leftX + rightX) / 2},${y1} ${rightX - 8},${y1}`, w: 2 + (t[l] / maxFlow) * 14 });
    });
  });
  const lit = (link) => (hot ? link.word === hot || link.l === hot : focus ? link.l === focus : true);
  return (
    <div>
      <p className="mb-3 text-small text-ink-soft dark:text-ube-100">Words that come up in several comments, flowing into the label those comments got. Thicker streams mean more comments. Tap a word to see them.</p>
      <svg viewBox={`0 0 640 ${H}`} className="w-full" role="img" aria-label="Flow from recurring words to the labels of the comments that use them">
        {links.map((k, i) => (
          <motion.path key={k.key} d={k.d} fill="none" stroke={TONE[k.l].hex} strokeLinecap="round" strokeWidth={k.w}
            initial={reduce ? false : { pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: lit(k) ? 0.55 : 0.08 }}
            transition={{ pathLength: { duration: 0.8, delay: i * 0.03, ease: 'easeOut' }, opacity: { duration: 0.24 } }} />
        ))}
        {!reduce && links.filter(lit).map((k) => (
          <motion.path key={`${k.key}-dash`} d={k.d} fill="none" stroke="white" strokeOpacity="0.7" strokeWidth="1.5" strokeLinecap="round"
            strokeDasharray="2 14" initial={{ strokeDashoffset: 0 }} animate={{ strokeDashoffset: -64 }}
            transition={{ repeat: Infinity, duration: 2.4, ease: 'linear' }} />
        ))}
        {themes.map((t, i) => {
          const y0 = 12 + i * rowH + rowH / 2;
          return (
            <g key={t.word} className="cursor-pointer" onMouseEnter={() => setHot(t.word)} onMouseLeave={() => setHot(null)} onClick={() => onTheme(t.word)}>
              <rect x="0" y={y0 - rowH / 2 + 3} width={leftX + 14} height={rowH - 6} rx="10" fill="currentColor" opacity={hot === t.word ? 0.08 : 0} />
              <text x={leftX - 6} y={y0 + 5} textAnchor="end" className="fill-current text-[14px] font-semibold">
                {t.word}<tspan dx="6" className="text-[11px] font-normal" opacity="0.5">{t.total}</tspan>
              </text>
              <circle cx={leftX + 6} cy={y0} r="5" fill="currentColor" opacity="0.5" />
            </g>
          );
        })}
        {LABELS.map((l) => {
          const nd = nodes[l];
          return (
            <g key={l} className="cursor-default" onMouseEnter={() => setHot(l)} onMouseLeave={() => setHot(null)}>
              <rect x={rightX - 8} y={nd.y} width="16" height={nd.h} rx="8" fill={TONE[l].hex} />
              <text x={rightX + 16} y={nd.y + nd.h / 2 + 5} fill={TONE[l].hex} className="text-[14px] font-bold">{title(l)}</text>
            </g>
          );
        })}
      </svg>
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
  { id: 'themes', name: 'Theme flow' },
  { id: 'places', name: 'Places map' },
];

export default function ExploreMaps({ rows, onTheme, onPick, focus }) {
  const [tab, setTab] = useState('sentiment');
  return (
    <Card>
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="mr-auto flex items-center gap-1 text-2xl font-semibold">
          Explore the comments
          <InfoTip label="About these maps">These maps only rearrange the labelled comments so patterns are easier to spot. Positions come from uncalibrated model scores, and word and place matches are simple text matches.</InfoTip>
        </h2>
        <div role="tablist" aria-label="Map" className="neu-inset inline-flex max-w-full gap-1 overflow-x-auto rounded-full p-1">
          {TABS.map((t) => (
            <button key={t.id} role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}
              className={`relative shrink-0 whitespace-nowrap rounded-full px-4 py-2 text-sm font-semibold ${tab === t.id ? 'text-ink dark:text-ube-50' : 'text-ink-soft dark:text-ube-300'}`}>
              {tab === t.id && <motion.span layoutId="map-pill" className="neu-raised absolute inset-0 rounded-full" transition={{ type: 'spring', stiffness: 300, damping: 28 }} />}
              <span className="relative">{t.name}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="mt-5">
        {tab === 'sentiment' && <SentimentMap rows={rows} focus={focus} onPick={onPick} />}
        {tab === 'themes' && <ThemeFlow rows={rows} onTheme={onTheme} focus={focus} />}
        {tab === 'places' && <PlacesMap rows={rows} />}
      </div>
    </Card>
  );
}
