import { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { Check, Copy, Download, ExternalLink, FileText, HelpCircle, Lightbulb, Search, ThumbsUp, X } from 'lucide-react';
import { LABELS } from '../lib/inference.js';
import { SAMPLE_NOTE, SCORE_NOTE, pct, summarise, title } from '../lib/analysis.js';
import { cleanText } from '../lib/clean.js';
import { download, rowsToRecords, toCsv } from '../lib/csv.js';
import { NOTES_DISCLAIMER, creatorNotes, isQuestion } from '../lib/insights.js';
import { CHECK_EVIDENCE, CHECK_GAP, isNearSplit, topGap } from '../lib/pipeline.js';
import { summaryText } from '../lib/summaryText.js';
import { PERIODS, filterRows, periodCounts, timeBuckets } from '../lib/timeline.js';
import { Ripple, TONE } from './ui.jsx';
import { InfoTip, WrongLabel } from './extras.jsx';
import { Count } from './motion.jsx';
import ExploreMaps from './maps.jsx';
import StreamTimeline from './timeline.jsx';

const THUMB_HOST = /^https:\/\/i\d?\.ytimg\.com\//;
const compact = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 });
const likesText = (n) => `${n.toLocaleString()} ${n === 1 ? 'like' : 'likes'}`;
const Panel = ({ className = '', children, ...p }) => <section className={`glass rounded-panel p-6 sm:p-8 ${className}`} {...p}>{children}</section>;

export function VideoCard({ meta, small = false }) {
  const facts = [
    meta.views != null && `${compact.format(meta.views)} views`,
    meta.likes != null && `${compact.format(meta.likes)} likes`,
    meta.comment_count != null && `${compact.format(meta.comment_count)} comments`,
  ].filter(Boolean);
  const thumb = THUMB_HOST.test(meta.thumbnail_url || '');
  return (
    <a href={meta.url} target="_blank" rel="noreferrer" className={`group relative block overflow-hidden rounded-card ${small ? 'aspect-video' : 'aspect-[4/3]'} bg-ube-100 dark:bg-night-raised`}>
      {thumb && <img src={meta.thumbnail_url} alt="" referrerPolicy="no-referrer" className="absolute inset-0 h-full w-full object-cover transition-transform duration-[600ms] ease-out group-hover:scale-105" />}
      <div className="absolute inset-0 bg-gradient-to-t from-[#1c1830ee] via-[#1c183044] to-transparent" />
      <div className="absolute inset-x-0 bottom-0 p-5 text-white">
        <p className="text-small opacity-80">{meta.channel}{meta.published_at ? `, ${new Date(meta.published_at).toLocaleDateString()}` : ''}</p>
        <p className={`mt-1 font-display font-semibold leading-tight ${small ? 'line-clamp-2 text-lg' : 'line-clamp-3 text-h2'}`}>{meta.title}</p>
        {!small && facts.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {facts.map((f) => <span key={f} className="rounded-full bg-white/20 px-3 py-1 text-micro backdrop-blur">{f}</span>)}
          </div>
        )}
        <span className="mt-3 inline-flex items-center gap-1 text-micro font-semibold opacity-90">Open on YouTube <ExternalLink size={12} aria-hidden /></span>
      </div>
    </a>
  );
}

export function ShareBar({ s, height = 'h-6', focus, onFocus }) {
  return (
    <div className={`neu-inset flex ${height} overflow-hidden rounded-full`} role="img"
      aria-label={LABELS.map((l) => `${title(l)} ${pct(s.shares[l])}`).join(', ')}>
      {LABELS.map((l) => (
        <motion.div key={l} className={`${TONE[l].bar} ${onFocus ? 'cursor-pointer' : ''}`}
          onMouseEnter={() => onFocus?.(l)} onMouseLeave={() => onFocus?.(null)}
          animate={{ width: `${s.shares[l] * 100}%`, opacity: focus && focus !== l ? 0.35 : 1 }}
          transition={{ width: { type: 'spring', stiffness: 110, damping: 20 }, opacity: { duration: 0.24 } }} />
      ))}
    </div>
  );
}

function Segmented({ value, onChange, counts }) {
  return (
    <div role="tablist" aria-label="Period" className="neu-inset inline-flex gap-1 rounded-full p-1">
      {PERIODS.map((p) => {
        const empty = counts[p.id] === 0;
        return (
          <button key={p.id} role="tab" aria-selected={value === p.id} disabled={empty} onClick={() => onChange(p.id)}
            title={empty ? 'No comments in this period' : `${counts[p.id]} comments`}
            className={`relative rounded-full px-4 py-2 text-small font-semibold transition-colors duration-hover disabled:cursor-not-allowed disabled:opacity-40 ${value === p.id ? 'text-white' : 'text-ink-soft hover:text-ink dark:text-ube-100'}`}>
            {value === p.id && <motion.span layoutId="period-pill" className="absolute inset-0 rounded-full bg-ube-600" transition={{ type: 'spring', stiffness: 380, damping: 32 }} />}
            <span className="relative">{p.name}</span>
          </button>
        );
      })}
    </div>
  );
}

function Hero({ s, focus, setFocus }) {
  const leaders = LABELS.filter((l) => s.counts[l] === Math.max(...LABELS.map((k) => s.counts[k])));
  const lead = leaders[0];
  const others = LABELS.filter((l) => l !== lead);
  return (
    <Panel>
      <div className="grid gap-8 md:grid-cols-[1.4fr_1fr] md:items-end">
        <div onMouseEnter={() => setFocus(lead)} onMouseLeave={() => setFocus(null)}>
          <AnimatePresence mode="wait">
            <motion.p key={lead} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.24 }}
              className={`font-display text-[5.5rem] font-bold leading-[0.85] tracking-[-0.04em] tabular-nums sm:text-hero ${TONE[lead].text}`}>
              <Count value={Math.round(s.shares[lead] * 100)} suffix="%" />
            </motion.p>
          </AnimatePresence>
          <p className="mt-4 text-h2 font-semibold">
            {leaders.length > 1
              ? `each read as ${leaders.map((l) => title(l).toLowerCase()).join(' and ')}, a tie`
              : `of comments read as ${title(lead).toLowerCase()}`}
          </p>
        </div>
        <div className="space-y-3">
          {others.map((l) => (
            <button key={l} onMouseEnter={() => setFocus(l)} onMouseLeave={() => setFocus(null)} onFocus={() => setFocus(l)} onBlur={() => setFocus(null)}
              className={`flex w-full items-baseline justify-between border-b border-ink/10 pb-3 text-left transition-opacity duration-state dark:border-white/10 ${focus && focus !== l ? 'opacity-40' : ''}`}>
              <span className={`text-body font-semibold ${TONE[l].text}`}>{title(l)}</span>
              <span className="font-display text-num font-bold tabular-nums"><Count value={Math.round(s.shares[l] * 100)} suffix="%" /></span>
            </button>
          ))}
        </div>
      </div>
      <div className="mt-8"><ShareBar s={s} height="h-4" focus={focus} onFocus={setFocus} /></div>
      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-small text-ink-soft dark:text-ube-100">
        {LABELS.map((l) => (
          <span key={l} className="flex items-center gap-1.5" onMouseEnter={() => setFocus(l)} onMouseLeave={() => setFocus(null)}>
            <span className={`h-2.5 w-2.5 rounded-full ${TONE[l].bar}`} />{s.counts[l]} {title(l).toLowerCase()}
          </span>
        ))}
        <InfoTip>{SCORE_NOTE}</InfoTip>
      </div>
    </Panel>
  );
}

function Stat({ label, value, hint, onClick, active }) {
  const Tag = onClick ? 'button' : 'div';
  return (
    <Tag onClick={onClick} className={`glass rounded-card px-5 py-4 text-left transition-[transform,box-shadow] duration-hover ${onClick ? 'hover:-translate-y-0.5 hover:shadow-lg' : ''} ${active ? 'ring-2 ring-ube-500' : ''}`}>
      <p className="flex items-center gap-1 text-small text-ink-soft dark:text-ube-300">{label}{hint}</p>
      <p className="mt-1 truncate font-display text-[2rem] font-bold leading-tight tabular-nums">{typeof value === 'number' ? <Count value={value} /> : value}</p>
    </Tag>
  );
}

function Quote({ row, tone }) {
  return (
    <li className={`rounded-tile px-4 py-3 ${tone}`}>
      <p className="line-clamp-3 break-words text-body leading-snug">{row.comment}</p>
      <p className="mt-1 flex items-center gap-3 text-micro text-ink-soft dark:text-ube-100">
        <span className="tabular-nums">#{row.no}</span>
        {Number.isFinite(row.likes) && <span className="inline-flex items-center gap-1 tabular-nums"><ThumbsUp size={11} aria-hidden />{row.likes.toLocaleString()}</span>}
      </p>
    </li>
  );
}

function SidePanel({ analysis, rows, isYouTube, exporting, onPdf, onTheme }) {
  const n = useMemo(() => creatorNotes(rows), [rows]);
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    const text = summaryText({ rows, meta: analysis.meta });
    try { await navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 2200); } catch { window.prompt('Copy this summary:', text); }
  };
  return (
    <aside className="glass space-y-6 rounded-panel p-4 sm:p-5">
      {isYouTube ? <VideoCard meta={analysis.meta} /> : (
        <div className="rounded-card bg-gradient-to-br from-ube-100 to-jambu-100 p-5 dark:from-ube-700/30 dark:to-jambu-700/20">
          <p className="text-small text-ink-soft dark:text-ube-100">{analysis.meta.source}</p>
          <p className="mt-1 font-display text-h2 font-semibold leading-tight">{analysis.meta.title}</p>
          <p className="mt-2 text-micro text-ink-faint">{analysis.meta.sampling}</p>
        </div>
      )}
      <div className="space-y-4 px-1">
        <h2 className="flex items-center gap-2 text-h2 font-semibold">
          <Lightbulb size={18} aria-hidden className="text-kaya-700 dark:text-kaya-300" />{isYouTube ? 'For the creator' : 'What to look at'}
          <InfoTip label="How these notes are made">{NOTES_DISCLAIMER}</InfoTip>
        </h2>
        <ul className="space-y-2 text-small text-ink-soft dark:text-ube-100">
          {n.notes.map((t) => <li key={t} className="flex gap-2"><span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-ube-300" />{t}</li>)}
        </ul>
        {[...n.negWords.map((w) => ({ ...w, l: 'NEGATIVE' })), ...n.posWords.slice(0, 3).map((w) => ({ ...w, l: 'POSITIVE' }))].length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {[...n.negWords.map((w) => ({ ...w, l: 'NEGATIVE' })), ...n.posWords.slice(0, 3).map((w) => ({ ...w, l: 'POSITIVE' }))].map((w) => (
              <Ripple key={w.l + w.word} onClick={() => onTheme(w.word)} className={`chip relative cursor-pointer overflow-hidden ${TONE[w.l].soft} ${TONE[w.l].text}`}>{w.word} {w.comments}</Ripple>
            ))}
          </div>
        )}
        {n.readFirst.length > 0 && (
          <div>
            <h3 className="mb-2 font-sans text-small font-semibold text-jambu-700 dark:text-jambu-300">Read these first</h3>
            <ul className="space-y-2">{n.readFirst.slice(0, 2).map((r) => <Quote key={r.no} row={r} tone={TONE.NEGATIVE.soft} />)}</ul>
          </div>
        )}
        {n.questions.length > 0 && (
          <div>
            <h3 className="mb-2 flex items-center gap-1 font-sans text-small font-semibold text-ube-700 dark:text-ube-100"><HelpCircle size={13} aria-hidden /> Viewer questions</h3>
            <ul className="space-y-2">{n.questions.slice(0, 2).map((r) => <Quote key={r.no} row={r} tone="bg-ube-100 dark:bg-ube-700/30" />)}</ul>
          </div>
        )}
        {n.highlights.length > 0 && (
          <div>
            <h3 className="mb-2 font-sans text-small font-semibold text-pandan-700 dark:text-pandan-300">What viewers enjoyed</h3>
            <ul className="space-y-2">{n.highlights.slice(0, 2).map((r) => <Quote key={r.no} row={r} tone={TONE.POSITIVE.soft} />)}</ul>
          </div>
        )}
      </div>
      <div className="grid gap-2 px-1 pb-1">
        <Ripple className="btn-primary w-full" onClick={onPdf} disabled={exporting}>
          <FileText size={15} aria-hidden /> {exporting ? 'Building PDF…' : 'Download PDF report'}
        </Ripple>
        <div className="grid grid-cols-2 gap-2">
          <Ripple className="btn-soft" onClick={() => download('comment_sentiment.csv', toCsv(rowsToRecords(rows, analysis.meta)), 'text/csv;charset=utf-8')}>
            <Download size={15} aria-hidden /> CSV
          </Ripple>
          <Ripple className="btn-soft" onClick={copy} aria-live="polite">
            {copied ? <Check size={15} aria-hidden /> : <Copy size={15} aria-hidden />} {copied ? 'Copied' : 'Summary'}
          </Ripple>
        </div>
      </div>
    </aside>
  );
}

function Column({ label, rows, total, query, sort, source, focus, flash, onlyQuestions }) {
  const [limit, setLimit] = useState(30);
  const list = useMemo(() => {
    const q = cleanText(query);
    let l = rows.filter((r) => (!q || cleanText(r.comment).includes(q)) && (!onlyQuestions || isQuestion(r.comment)));
    if (sort === 'score') l = [...l].sort((a, b) => b.score - a.score);
    if (sort === 'likes') l = [...l].sort((a, b) => (b.likes ?? 0) - (a.likes ?? 0));
    if (sort === 'unsure') l = [...l].sort((a, b) => topGap(a) - topGap(b));
    return l;
  }, [rows, query, sort, onlyQuestions]);
  useEffect(() => {
    if (flash && rows.some((r) => r.no === flash)) {
      const i = list.findIndex((r) => r.no === flash);
      if (i >= limit) setLimit(i + 10);
    }
  }, [flash]); // eslint-disable-line react-hooks/exhaustive-deps
  const t = TONE[label];
  return (
    <motion.section aria-label={`${title(label)} comments`} animate={{ opacity: focus && focus !== label ? 0.45 : 1 }} transition={{ duration: 0.24 }}
      className={`flex min-w-0 flex-col rounded-panel p-3 ${t.soft}`}>
      <header className="flex items-end justify-between gap-2 px-2 pb-3 pt-1">
        <div>
          <h3 className={`font-display text-xl font-bold ${t.text}`}>{title(label)}</h3>
          <p className="text-micro text-ink-soft dark:text-ube-300">{query || onlyQuestions ? `${list.length} matching, ` : ''}{rows.length.toLocaleString()} of {total.toLocaleString()}</p>
        </div>
        <span className={`font-display text-4xl font-bold tabular-nums ${t.text}`}><Count value={total ? Math.round((rows.length / total) * 100) : 0} suffix="%" /></span>
      </header>
      <ul className="max-h-[620px] space-y-2 overflow-y-auto pr-1">
        <AnimatePresence initial={false}>
          {list.slice(0, limit).map((r) => (
            <motion.li key={r.no} id={`comment-${r.no}`} layout="position" initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }} className={`group glass scroll-mt-24 rounded-card px-4 py-3 ${flash === r.no ? 'flash-ring' : ''}`}>
              <p className="whitespace-pre-wrap break-words text-body leading-relaxed">{r.comment}</p>
              <div className="mt-2 flex flex-wrap items-center gap-2 text-micro text-ink-faint">
                <span className="tabular-nums">#{r.no}</span>
                {Number.isFinite(r.likes) && <span className="inline-flex items-center gap-1 tabular-nums" title={likesText(r.likes)}><ThumbsUp size={12} aria-hidden />{r.likes.toLocaleString()}</span>}
                {r.truncated && <span className="chip bg-kaya-100 text-kaya-700">truncated</span>}
                {isNearSplit(r) && <span className="chip bg-ube-100 text-ube-700 dark:bg-ube-700/30 dark:text-ube-100">check by hand</span>}
                <span className="transition-opacity duration-hover [@media(hover:hover)]:opacity-0 [@media(hover:hover)]:focus-within:opacity-100 [@media(hover:hover)]:group-hover:opacity-100">
                  <WrongLabel compact text={r.comment} predicted={r.sentiment} scores={r.scores} source={source} />
                </span>
                <span className="ml-auto flex items-center gap-1.5 tabular-nums" title="Top-class model score">
                  <span className="neu-inset block h-1.5 w-12 overflow-hidden rounded-full"><span className={`block h-full ${t.bar}`} style={{ width: `${r.score * 100}%` }} /></span>
                  {pct(r.score, 0)}
                </span>
              </div>
            </motion.li>
          ))}
        </AnimatePresence>
        {!list.length && <li className="px-2 py-6 text-center text-small text-ink-faint">{query || onlyQuestions ? 'No matches here.' : 'None in this set.'}</li>}
      </ul>
      {list.length > limit && (
        <button onClick={() => setLimit((k) => k + 30)} className="mt-3 rounded-full py-2 text-small font-semibold text-ink-soft hover:text-ink dark:text-ube-100">
          Show {Math.min(30, list.length - limit)} more of {list.length - limit}
        </button>
      )}
    </motion.section>
  );
}

function FilterChips({ period, range, query, onlyQuestions, onClear }) {
  const chips = [
    period !== 'all' && { k: 'period', text: PERIODS.find((p) => p.id === period).name },
    range && { k: 'range', text: `${range.unit === 'week' ? 'Week of ' : ''}${range.label}` },
    query && { k: 'query', text: `“${query}”` },
    onlyQuestions && { k: 'questions', text: 'Questions only' },
  ].filter(Boolean);
  if (!chips.length) return null;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-small text-ink-faint">Showing</span>
      <AnimatePresence>
        {chips.map((c) => (
          <motion.button key={c.k} layout initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.9 }}
            onClick={() => onClear(c.k)} className="chip inline-flex items-center gap-1 bg-ube-600 text-white">
            {c.text} <X size={12} aria-label="Remove filter" />
          </motion.button>
        ))}
      </AnimatePresence>
      {chips.length > 1 && <button onClick={() => onClear('all')} className="text-small text-ink-soft underline underline-offset-2 dark:text-ube-100">Clear all</button>}
    </div>
  );
}

export default function CommentResults({ analysis, exporting, onPdf }) {
  const reduce = useReducedMotion();
  const isYouTube = analysis.meta.source === 'YouTube';
  const [period, setPeriod] = useState('all');
  const [range, setRange] = useState(null);
  const [focus, setFocus] = useState(null);
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState('order');
  const [onlyQuestions, setOnlyQuestions] = useState(false);
  const [flash, setFlash] = useState(null);
  const listTop = useRef(null);

  const dated = useMemo(() => analysis.rows.filter((r) => r.date).length >= 3, [analysis.rows]);
  const counts = useMemo(() => periodCounts(analysis.rows), [analysis.rows]);
  const inPeriod = useMemo(() => filterRows(analysis.rows, { period }), [analysis.rows, period]);
  const rows = useMemo(() => filterRows(inPeriod, { range }), [inPeriod, range]);
  const hasRows = rows.length > 0;
  const s = hasRows ? summarise(rows) : null;
  const n = useMemo(() => (hasRows ? creatorNotes(rows) : null), [rows, hasRows]);
  const source = isYouTube ? `YouTube ${analysis.meta.video_id}` : analysis.meta.source;

  const jumpToList = () => listTop.current?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
  const theme = (w) => { setQuery(w); setOnlyQuestions(false); jumpToList(); };
  const pick = (r) => {
    setQuery(''); setOnlyQuestions(false); setFlash(r.no);
    setTimeout(() => document.getElementById(`comment-${r.no}`)?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'center' }), 80);
    setTimeout(() => setFlash(null), 2200);
  };
  const clear = (k) => {
    if (k === 'period' || k === 'all') setPeriod('all');
    if (k === 'range' || k === 'all') setRange(null);
    if (k === 'query' || k === 'all') setQuery('');
    if (k === 'questions' || k === 'all') setOnlyQuestions(false);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="min-w-0">
          <p className="text-small text-ink-soft dark:text-ube-300">Results for</p>
          <h2 className="line-clamp-2 font-display text-[1.75rem] font-bold leading-tight sm:text-h1">{analysis.meta.title}</h2>
        </div>
        {isYouTube && dated && <Segmented value={period} onChange={(p) => { setPeriod(p); setRange(null); }} counts={counts} />}
      </div>
      <FilterChips period={period} range={range} query={query} onlyQuestions={onlyQuestions} onClear={clear} />

      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-12 space-y-6 lg:col-span-8">
          {hasRows ? (
            <>
              <Hero s={s} focus={focus} setFocus={setFocus} />
              <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                <Stat label="Comments" value={s.total} hint={analysis.meta.excluded ? <InfoTip>{analysis.meta.excluded} were empty after removing links and mentions.</InfoTip> : null} />
                <Stat label="Worth checking" value={rows.filter((r) => isNearSplit(r)).length} active={sort === 'unsure'}
                  hint={<InfoTip label="What counts as worth checking">The model's top two scores were less than {Math.round(CHECK_GAP * 100)} points apart. In testing, labels like these were right {Math.round(CHECK_EVIDENCE.not_sure_accuracy * 100)}% of the time, against {Math.round(CHECK_EVIDENCE.answered_accuracy * 100)}% for the rest. Click to list these first.</InfoTip>}
                  onClick={() => { setSort('unsure'); setQuery(''); jumpToList(); }} />
                <Stat label="Questions" value={n.questions.length} active={onlyQuestions} onClick={() => { setOnlyQuestions((q) => !q); setQuery(''); jumpToList(); }} />
                <Stat label="Top theme" value={n.posWords[0]?.word ?? n.negWords[0]?.word ?? '–'}
                  onClick={(n.posWords[0] || n.negWords[0]) ? () => theme((n.posWords[0] ?? n.negWords[0]).word) : undefined} />
              </div>
              <ExploreMaps rows={rows} onTheme={theme} onPick={pick} focus={focus} />
              {isYouTube && dated && timeBuckets(inPeriod) && (
                <Panel>
                  <div className="flex items-center gap-1">
                    <h2 className="text-h2 font-semibold">When these comments were posted</h2>
                    <InfoTip label="About this chart">{SAMPLE_NOTE} This shows when the sampled comments were posted.</InfoTip>
                  </div>
                  <div className="mt-4 text-ink dark:text-ube-50"><StreamTimeline rows={inPeriod} range={range} onBrush={setRange} focus={focus} /></div>
                </Panel>
              )}
            </>
          ) : (
            <Panel className="text-center">
              <p className="font-display text-h2 font-semibold">No comments in this period.</p>
              <button className="btn-soft mt-4" onClick={() => clear('all')}>Show all comments</button>
            </Panel>
          )}
        </div>
        <div className="col-span-12 lg:col-span-4">
          {hasRows && <SidePanel analysis={analysis} rows={rows} isYouTube={isYouTube} exporting={exporting} onPdf={() => onPdf(rows)} onTheme={theme} />}
        </div>
      </div>

      {hasRows && (
        <Panel>
          <div ref={listTop} className="flex scroll-mt-6 flex-wrap items-center gap-3">
            <h2 className="mr-auto text-h2 font-semibold">Every comment, by label</h2>
            <label className="neu-inset flex items-center gap-2 rounded-full px-3 py-1.5">
              <Search size={15} aria-hidden className="text-ink-faint" />
              <span className="sr-only">Search comments</span>
              <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search all three" className="w-40 bg-transparent text-small outline-none" />
              {query && <button aria-label="Clear search" onClick={() => setQuery('')}><X size={14} aria-hidden /></button>}
            </label>
            <label className="text-small">
              <span className="sr-only">Sort</span>
              <select value={sort} onChange={(e) => setSort(e.target.value)} className="neu-inset rounded-full px-3 py-1.5 text-small">
                <option value="order">Original order</option>
                {rows.some((r) => Number.isFinite(r.likes)) && <option value="likes">Most liked first</option>}
                <option value="score">Most confident first</option>
                <option value="unsure">Least confident first</option>
              </select>
            </label>
          </div>
          <div className="mt-5 grid gap-4 lg:grid-cols-3">
            {['NEGATIVE', 'NEUTRAL', 'POSITIVE'].map((l) => (
              <Column key={l} label={l} rows={rows.filter((r) => r.sentiment === l)} total={rows.length} query={query} sort={sort}
                source={source} focus={focus} flash={flash} onlyQuestions={onlyQuestions} />
            ))}
          </div>
        </Panel>
      )}
    </div>
  );
}
