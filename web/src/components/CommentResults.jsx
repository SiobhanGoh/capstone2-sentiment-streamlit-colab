import { useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Check, Copy, Download, ExternalLink, FileText, HelpCircle, Lightbulb, Search, ThumbsUp, X } from 'lucide-react';
import { LABELS } from '../lib/inference.js';
import { SAMPLE_NOTE, SCORE_NOTE, pct, summarise, title } from '../lib/analysis.js';
import { cleanText } from '../lib/clean.js';
import { download, rowsToRecords, toCsv } from '../lib/csv.js';
import { NOTES_DISCLAIMER, creatorNotes } from '../lib/insights.js';
import { NEAR_SPLIT, isNearSplit } from '../lib/pipeline.js';
import { summaryText } from '../lib/summaryText.js';
import { bucketLabel, timeBuckets } from '../lib/timeline.js';
import { Card, Ripple, TONE, Tilt } from './ui.jsx';
import { InfoTip, WrongLabel } from './extras.jsx';
import ExploreMaps from './maps.jsx';

const THUMB_HOST = /^https:\/\/i\d?\.ytimg\.com\//;
const compact = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 });
const likesText = (n) => `${n.toLocaleString()} ${n === 1 ? 'like' : 'likes'}`;

export function VideoCard({ meta, small = false }) {
  const facts = [
    meta.views != null && `${compact.format(meta.views)} views`,
    meta.likes != null && `${compact.format(meta.likes)} likes`,
    meta.comment_count != null && `${compact.format(meta.comment_count)} comments`,
  ].filter(Boolean);
  return (
    <Card className={`grid gap-5 overflow-hidden ${small ? 'grid-cols-[minmax(0,150px)_1fr] items-center !p-4' : 'sm:grid-cols-[minmax(0,320px)_1fr] sm:items-center'}`}>
      {THUMB_HOST.test(meta.thumbnail_url || '') ? (
        <a href={meta.url} target="_blank" rel="noreferrer" className="block overflow-hidden rounded-xl2">
          <img src={meta.thumbnail_url} alt={`Thumbnail for ${meta.title}`} className="aspect-video w-full object-cover transition-transform duration-300 hover:scale-[1.03]" referrerPolicy="no-referrer" />
        </a>
      ) : <div className="aspect-video rounded-xl2 bg-ube-100 dark:bg-night-raised" aria-hidden />}
      <div className="min-w-0">
        <h2 className={`${small ? 'line-clamp-2 text-base' : 'text-2xl'} font-semibold leading-tight`}>{meta.title}</h2>
        <p className={`mt-1 text-ink-soft dark:text-ube-300 ${small ? 'text-sm' : ''}`}>{meta.channel}{meta.published_at ? `, published ${new Date(meta.published_at).toLocaleDateString()}` : ''}</p>
        {facts.length > 0 && !small && (
          <div className="mt-3 flex flex-wrap gap-2">
            {facts.map((f) => <span key={f} className="chip neu-raised text-ink-soft dark:text-ube-100">{f}</span>)}
          </div>
        )}
        {!small && <p className="mt-3 text-sm text-ink-faint">{meta.sampling}</p>}
        <a href={meta.url} target="_blank" rel="noreferrer" className="mt-3 inline-flex items-center gap-1.5 text-sm font-semibold text-ube-600 underline-offset-2 hover:underline dark:text-ube-300">
          Open on YouTube <ExternalLink size={14} aria-hidden />
        </a>
      </div>
    </Card>
  );
}

export function ShareBar({ s, height = 'h-6' }) {
  return (
    <div className={`neu-inset flex ${height} overflow-hidden rounded-full`} role="img"
      aria-label={LABELS.map((l) => `${title(l)} ${pct(s.shares[l])}`).join(', ')}>
      {LABELS.map((l) => (
        <motion.div key={l} className={TONE[l].bar} initial={{ width: 0 }} animate={{ width: `${s.shares[l] * 100}%` }}
          transition={{ type: 'spring', stiffness: 80, damping: 18 }} />
      ))}
    </div>
  );
}

function Summary({ analysis, s, unsure, onShowUnsure }) {
  return (
    <div className="grid grid-cols-3 gap-3 sm:gap-5 md:grid-cols-6">
      <Tilt className="col-span-3 md:col-span-3" max={4}>
        <Card className="flex h-full flex-col">
          <p className="flex items-center gap-1 text-sm text-ink-soft dark:text-ube-300">
            Worth checking by hand
            <InfoTip label="What counts as worth checking">
              Comments where the model's top two scores were within {Math.round(NEAR_SPLIT * 100)} points of each other, so it was nearly split between two labels. These are the likeliest to be wrong.
            </InfoTip>
          </p>
          <p className="mt-1 font-display text-4xl font-bold tabular-nums">{unsure} <span className="text-lg font-semibold text-ink-soft dark:text-ube-300">of {s.total}</span></p>
          <div className="mt-auto flex flex-wrap items-center gap-3 pt-3">
            {unsure > 0 && <button className="btn-soft !px-4 !py-1.5 text-xs" onClick={onShowUnsure}>Show them first</button>}
            <span className="text-xs text-ink-faint">
              {s.total.toLocaleString()} analysed{analysis.meta.excluded ? `, ${analysis.meta.excluded} empty after cleaning` : ''}{s.truncated ? `, ${s.truncated} truncated` : ''}
            </span>
          </div>
        </Card>
      </Tilt>
      {LABELS.map((l) => (
        <Tilt key={l} className="col-span-1" max={6}>
          <Card className={`h-full ${TONE[l].soft} !border-transparent max-sm:!p-4`}>
            <p className={`text-sm font-semibold ${TONE[l].text}`}>{title(l)}</p>
            <p className="mt-2 font-display text-2xl font-bold tabular-nums sm:text-3xl">{pct(s.shares[l], 0)}</p>
            <p className="mt-1 text-sm text-ink-soft dark:text-ube-300">{s.counts[l].toLocaleString()} {s.counts[l] === 1 ? 'comment' : 'comments'}</p>
          </Card>
        </Tilt>
      ))}
      <Card className="col-span-3 md:col-span-6">
        <div className="mb-3 flex items-center gap-1">
          <p className="font-semibold">Share of comments</p>
          <InfoTip>{analysis.meta.source === 'YouTube' ? SAMPLE_NOTE : 'A summary of the supplied comments only.'} {SCORE_NOTE}</InfoTip>
        </div>
        <ShareBar s={s} />
        <div className="mt-3 flex flex-wrap gap-4 text-sm">
          {LABELS.map((l) => (
            <span key={l} className="flex items-center gap-2"><span className={`h-3 w-3 rounded-full ${TONE[l].bar}`} />{title(l)} {pct(s.shares[l], 0)}</span>
          ))}
        </div>
      </Card>
    </div>
  );
}

function Timeline({ rows }) {
  const t = useMemo(() => timeBuckets(rows), [rows]);
  if (!t) return null;
  const max = Math.max(...t.buckets.map((b) => b.total), 1);
  const step = Math.max(1, Math.ceil(t.buckets.length / 8));
  return (
    <Card>
      <div className="flex items-center gap-1">
        <h2 className="text-2xl font-semibold">When these comments were posted</h2>
        <InfoTip label="About this chart">
          Comments are fetched by relevance, not all at once, so this shows when the sampled comments were posted, grouped by {t.unit}. It is not every comment on the video.
        </InfoTip>
      </div>
      <p className="mt-1 text-sm text-ink-faint">{t.dated} dated comments, by {t.unit}.</p>
      <div className="mt-5 flex h-48 items-end gap-1" role="img"
        aria-label={`Comments per ${t.unit}, coloured by label, from ${bucketLabel(t.buckets[0].start, t.unit)} to ${bucketLabel(t.buckets.at(-1).start, t.unit)}`}>
        {t.buckets.map((b) => (
          <div key={b.start} className="group relative flex h-full min-w-0 flex-1 flex-col justify-end">
            <div className="flex flex-col-reverse overflow-hidden rounded-t-lg" style={{ height: `${(b.total / max) * 100}%` }}>
              {LABELS.map((l) => b[l] > 0 && <div key={l} className={TONE[l].bar} style={{ height: `${(b[l] / b.total) * 100}%` }} />)}
            </div>
            <span className="glass pointer-events-none absolute bottom-full left-1/2 z-20 mb-1 hidden w-max -translate-x-1/2 rounded-lg px-2 py-1 text-xs group-hover:block">
              {bucketLabel(b.start, t.unit)}: {b.total} ({LABELS.map((l) => `${b[l]} ${title(l).toLowerCase()}`).join(', ')})
            </span>
          </div>
        ))}
      </div>
      <div className="mt-2 flex gap-1 text-[11px] text-ink-faint">
        {t.buckets.map((b, i) => <span key={b.start} className="min-w-0 flex-1 truncate text-center">{i % step === 0 ? bucketLabel(b.start, t.unit) : ''}</span>)}
      </div>
    </Card>
  );
}

function Quote({ row }) {
  return (
    <li className={`rounded-xl2 px-4 py-3 ${TONE[row.sentiment].soft}`}>
      <p className="line-clamp-4 whitespace-pre-wrap break-words text-[15px]">{row.comment}</p>
      <p className="mt-1.5 flex items-center gap-3 text-xs text-ink-soft dark:text-ube-300">
        <span className="tabular-nums">#{row.no}</span>
        {Number.isFinite(row.likes) && <span className="inline-flex items-center gap-1 tabular-nums"><ThumbsUp size={12} aria-hidden />{row.likes.toLocaleString()}</span>}
      </p>
    </li>
  );
}

function CreatorNotes({ rows, isYouTube, onTheme }) {
  const n = useMemo(() => creatorNotes(rows), [rows]);
  const themes = [...n.negWords.map((w) => ({ ...w, label: 'NEGATIVE' })), ...n.posWords.slice(0, 4).map((w) => ({ ...w, label: 'POSITIVE' }))];
  return (
    <Card>
      <div className="flex items-center gap-2">
        <Lightbulb size={20} aria-hidden className="text-kaya-700 dark:text-kaya-300" />
        <h2 className="text-2xl font-semibold">{isYouTube ? 'Notes for the creator' : 'What to look at'}</h2>
        <InfoTip label="How these notes are made">{NOTES_DISCLAIMER}</InfoTip>
      </div>
      <ul className="mt-4 max-w-[75ch] list-disc space-y-2 pl-5 text-ink-soft dark:text-ube-100">
        {n.notes.map((t) => <li key={t}>{t}</li>)}
      </ul>
      {themes.length > 0 && (
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <span className="text-sm text-ink-faint">Show comments mentioning</span>
          {themes.map((w) => (
            <Ripple key={w.label + w.word} onClick={() => onTheme(w.word)}
              className={`chip relative cursor-pointer overflow-hidden px-3 py-1 text-sm ${TONE[w.label].soft} ${TONE[w.label].text}`}>
              {w.word} <span className="opacity-70">{w.comments}</span>
            </Ripple>
          ))}
        </div>
      )}
      <div className="mt-6 grid gap-5 lg:grid-cols-3">
        <div>
          <h3 className="mb-3 font-sans text-sm font-semibold text-jambu-700 dark:text-jambu-300">Read these first</h3>
          {n.readFirst.length ? <ul className="space-y-2">{n.readFirst.map((r) => <Quote key={r.no} row={r} />)}</ul>
            : <p className="text-sm text-ink-faint">No negative comments in this set.</p>}
        </div>
        <div>
          <h3 className="mb-3 flex items-center gap-1.5 font-sans text-sm font-semibold text-ube-700 dark:text-ube-300"><HelpCircle size={15} aria-hidden /> Viewer questions</h3>
          {n.questions.length ? <ul className="space-y-2">{n.questions.slice(0, 3).map((r) => <Quote key={r.no} row={r} />)}</ul>
            : <p className="text-sm text-ink-faint">No questions spotted.</p>}
        </div>
        <div>
          <h3 className="mb-3 font-sans text-sm font-semibold text-pandan-700 dark:text-pandan-300">What viewers enjoyed</h3>
          {n.highlights.length ? <ul className="space-y-2">{n.highlights.map((r) => <Quote key={r.no} row={r} />)}</ul>
            : <p className="text-sm text-ink-faint">No positive comments in this set.</p>}
        </div>
      </div>
    </Card>
  );
}

function ReportCard({ analysis, exporting, onPdf }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(summaryText(analysis));
      setCopied(true);
      setTimeout(() => setCopied(false), 2200);
    } catch {
      setCopied(false);
      window.prompt('Copy this summary:', summaryText(analysis));
    }
  };
  return (
    <Card className="flex flex-wrap items-center gap-3 bg-gradient-to-r from-ube-100/70 to-transparent dark:from-ube-700/20">
      <div className="mr-auto">
        <h2 className="text-xl font-semibold">Take the report with you</h2>
        <p className="mt-1 max-w-[56ch] text-sm text-ink-soft dark:text-ube-300">
          The PDF has the summary, {analysis.meta.source === 'YouTube' ? 'video details, creator notes' : 'notes'} and every comment. The CSV keeps the original text and all three scores.
        </p>
      </div>
      <Ripple className="btn-primary" onClick={onPdf} disabled={exporting}>
        <FileText size={15} aria-hidden /> {exporting ? 'Building PDF…' : 'Download PDF report'}
      </Ripple>
      <Ripple className="btn-soft" onClick={() => download('comment_sentiment.csv', toCsv(rowsToRecords(analysis.rows, analysis.meta)), 'text/csv;charset=utf-8')}>
        <Download size={15} aria-hidden /> CSV
      </Ripple>
      <Ripple className="btn-soft" onClick={copy} aria-live="polite">
        {copied ? <Check size={15} aria-hidden /> : <Copy size={15} aria-hidden />} {copied ? 'Copied' : 'Copy summary'}
      </Ripple>
    </Card>
  );
}

function Column({ label, rows, total, query, sort, source }) {
  const [limit, setLimit] = useState(30);
  const list = useMemo(() => {
    const q = cleanText(query);
    let l = rows.filter((r) => !q || cleanText(r.comment).includes(q));
    if (sort === 'score') l = [...l].sort((a, b) => b.score - a.score);
    if (sort === 'likes') l = [...l].sort((a, b) => (b.likes ?? 0) - (a.likes ?? 0));
    if (sort === 'unsure') l = [...l].sort((a, b) => a.score - b.score);
    return l;
  }, [rows, query, sort]);
  const t = TONE[label];
  return (
    <section aria-label={`${title(label)} comments`} className={`flex min-w-0 flex-col rounded-xl3 p-3 ${t.soft}`}>
      <header className="flex items-end justify-between gap-2 px-2 pb-3 pt-1">
        <div>
          <h3 className={`font-display text-xl font-bold ${t.text}`}>{title(label)}</h3>
          <p className="text-xs text-ink-soft dark:text-ube-300">{query ? `${list.length} matching, ` : ''}{rows.length.toLocaleString()} of {total.toLocaleString()}</p>
        </div>
        <span className={`font-display text-4xl font-bold tabular-nums ${t.text}`}>{pct(total ? rows.length / total : 0, 0)}</span>
      </header>
      <ul className="max-h-[620px] space-y-2 overflow-y-auto pr-1">
        {list.slice(0, limit).map((r) => (
          <motion.li key={r.no} layout="position" className="group glass rounded-xl2 px-4 py-3">
            <p className="whitespace-pre-wrap break-words text-[15px] leading-relaxed">{r.comment}</p>
            <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-ink-faint">
              <span className="tabular-nums">#{r.no}</span>
              {Number.isFinite(r.likes) && <span className="inline-flex items-center gap-1 tabular-nums" title={likesText(r.likes)}><ThumbsUp size={12} aria-hidden />{r.likes.toLocaleString()}</span>}
              {r.truncated && <span className="chip bg-kaya-100 text-kaya-700">truncated</span>}
              {isNearSplit(r) && <span className="chip bg-ube-100 text-ube-700 dark:bg-ube-700/30 dark:text-ube-100">nearly split</span>}
              <span className="transition-opacity [@media(hover:hover)]:opacity-0 [@media(hover:hover)]:focus-within:opacity-100 [@media(hover:hover)]:group-hover:opacity-100">
                <WrongLabel compact text={r.comment} predicted={r.sentiment} scores={r.scores} source={source} />
              </span>
              <span className="ml-auto flex items-center gap-1.5 tabular-nums" title="Top-class model score">
                <span className="neu-inset block h-1.5 w-12 overflow-hidden rounded-full"><span className={`block h-full ${t.bar}`} style={{ width: `${r.score * 100}%` }} /></span>
                {pct(r.score, 0)}
              </span>
            </div>
          </motion.li>
        ))}
        {!list.length && <li className="px-2 py-6 text-center text-sm text-ink-faint">{query ? 'No matches here.' : 'None in this set.'}</li>}
      </ul>
      {list.length > limit && (
        <button onClick={() => setLimit((n) => n + 30)} className="mt-3 rounded-full py-2 text-sm font-semibold text-ink-soft hover:text-ink dark:text-ube-100">
          Show {Math.min(30, list.length - limit)} more of {list.length - limit}
        </button>
      )}
    </section>
  );
}

const SentimentColumns = ({ rows, query, setQuery, sort, setSort, anchor, source }) => {
  const hasLikes = rows.some((r) => Number.isFinite(r.likes));
  return (
    <Card>
      <div ref={anchor} className="flex scroll-mt-6 flex-wrap items-center gap-3">
        <h2 className="mr-auto text-2xl font-semibold">Every comment, by label</h2>
        <label className="neu-inset flex items-center gap-2 rounded-full px-3 py-1.5">
          <Search size={15} aria-hidden className="text-ink-faint" />
          <span className="sr-only">Search comments</span>
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search all three" className="w-40 bg-transparent text-sm outline-none" />
          {query && <button aria-label="Clear search" onClick={() => setQuery('')}><X size={14} aria-hidden /></button>}
        </label>
        <label className="text-sm">
          <span className="sr-only">Sort</span>
          <select value={sort} onChange={(e) => setSort(e.target.value)} className="neu-inset rounded-full px-3 py-1.5 text-sm">
            <option value="order">Original order</option>
            {hasLikes && <option value="likes">Most liked first</option>}
            <option value="score">Most confident first</option>
            <option value="unsure">Least confident first</option>
          </select>
        </label>
      </div>
      <div className="mt-5 grid gap-4 lg:grid-cols-3">
        {['NEGATIVE', 'NEUTRAL', 'POSITIVE'].map((l) => (
          <Column key={l} label={l} rows={rows.filter((r) => r.sentiment === l)} total={rows.length} query={query} sort={sort} source={source} />
        ))}
      </div>
    </Card>
  );
};

export default function CommentResults({ analysis, exporting, onPdf }) {
  const s = summarise(analysis.rows);
  const isYouTube = analysis.meta.source === 'YouTube';
  const unsure = analysis.rows.filter((r) => isNearSplit(r)).length;
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState('order');
  const anchor = useRef(null);
  const jump = () => anchor.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  const source = isYouTube ? `YouTube ${analysis.meta.video_id}` : analysis.meta.source;
  return (
    <div className="space-y-5">
      {isYouTube ? <VideoCard meta={analysis.meta} /> : (
        <div>
          <h2 className="text-2xl font-semibold">{analysis.meta.title}</h2>
          <p className="text-sm text-ink-soft dark:text-ube-300">{analysis.meta.sampling}</p>
        </div>
      )}
      <Summary analysis={analysis} s={s} unsure={unsure} onShowUnsure={() => { setQuery(''); setSort('unsure'); jump(); }} />
      <CreatorNotes rows={analysis.rows} isYouTube={isYouTube} onTheme={(w) => { setQuery(w); jump(); }} />
      <ExploreMaps rows={analysis.rows} onTheme={(w) => { setQuery(w); jump(); }} />
      {isYouTube && <Timeline rows={analysis.rows} />}
      <ReportCard analysis={analysis} exporting={exporting} onPdf={onPdf} />
      <SentimentColumns rows={analysis.rows} query={query} setQuery={setQuery} sort={sort} setSort={setSort} anchor={anchor} source={source} />
    </div>
  );
}
