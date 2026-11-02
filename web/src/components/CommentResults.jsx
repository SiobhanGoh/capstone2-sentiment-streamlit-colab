import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { Download, ExternalLink, FileText, HelpCircle, Lightbulb, Search, ThumbsUp } from 'lucide-react';
import { LABELS } from '../lib/inference.js';
import { SAMPLE_NOTE, SCORE_NOTE, pct, summarise, title } from '../lib/analysis.js';
import { download, rowsToRecords, toCsv } from '../lib/csv.js';
import { NOTES_DISCLAIMER, creatorNotes } from '../lib/insights.js';
import { Card, Ripple, ScoreBars, TONE, Tilt } from './ui.jsx';

const THUMB_HOST = /^https:\/\/i\d?\.ytimg\.com\//;
const compact = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 });

function VideoCard({ meta }) {
  const facts = [
    meta.views != null && `${compact.format(meta.views)} views`,
    meta.likes != null && `${compact.format(meta.likes)} likes`,
    meta.comment_count != null && `${compact.format(meta.comment_count)} comments`,
  ].filter(Boolean);
  return (
    <Card className="grid gap-5 overflow-hidden sm:grid-cols-[minmax(0,320px)_1fr] sm:items-center">
      {THUMB_HOST.test(meta.thumbnail_url || '') ? (
        <a href={meta.url} target="_blank" rel="noreferrer" className="block overflow-hidden rounded-xl2">
          <img src={meta.thumbnail_url} alt={`Thumbnail for ${meta.title}`} className="aspect-video w-full object-cover transition-transform duration-300 hover:scale-[1.03]" referrerPolicy="no-referrer" />
        </a>
      ) : <div className="aspect-video rounded-xl2 bg-ube-100 dark:bg-night-raised" aria-hidden />}
      <div>
        <h2 className="text-2xl font-semibold leading-tight">{meta.title}</h2>
        <p className="mt-1 text-ink-soft dark:text-ube-300">{meta.channel}{meta.published_at ? `, published ${new Date(meta.published_at).toLocaleDateString()}` : ''}</p>
        {facts.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {facts.map((f) => <span key={f} className="chip neu-raised text-ink-soft dark:text-ube-100">{f}</span>)}
          </div>
        )}
        <p className="mt-3 text-sm text-ink-faint">{meta.sampling}</p>
        <a href={meta.url} target="_blank" rel="noreferrer" className="mt-3 inline-flex items-center gap-1.5 text-sm font-semibold text-ube-600 underline-offset-2 hover:underline dark:text-ube-300">
          Open on YouTube <ExternalLink size={14} aria-hidden />
        </a>
      </div>
    </Card>
  );
}

function Summary({ analysis, s }) {
  const meanScores = LABELS.map((_, i) => analysis.rows.reduce((a, r) => a + r.scores[i], 0) / s.total);
  return (
    <div className="grid grid-cols-3 gap-3 sm:gap-5 md:grid-cols-6">
      <Tilt className="col-span-3 md:col-span-2" max={5}>
        <Card className="h-full">
          <p className="text-sm text-ink-soft dark:text-ube-300">{s.tied ? 'Joint most frequent' : 'Most frequent label'}</p>
          <p className="mt-1 font-display text-4xl font-bold">{s.dominant}</p>
          <p className="mt-4 text-sm text-ink-soft dark:text-ube-300">
            {s.total.toLocaleString()} analysed{analysis.meta.excluded ? `, ${analysis.meta.excluded} empty after cleaning` : ''}
            {s.truncated ? `, ${s.truncated} truncated to 128 tokens` : ''}.
          </p>
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
      <Card className="col-span-3 md:col-span-1">
        <p className="text-sm text-ink-soft dark:text-ube-300">Mean top score</p>
        <p className="mt-2 font-display text-3xl font-bold tabular-nums">{pct(s.meanScore, 0)}</p>
        <p className="mt-1 text-xs text-ink-faint">Not accuracy</p>
      </Card>
      <Card className="col-span-3">
        <p className="mb-3 font-semibold">Share of comments</p>
        <div className="neu-inset flex h-6 overflow-hidden rounded-full" role="img"
          aria-label={LABELS.map((l) => `${title(l)} ${pct(s.shares[l])}`).join(', ')}>
          {LABELS.map((l) => (
            <motion.div key={l} className={TONE[l].bar} initial={{ width: 0 }} animate={{ width: `${s.shares[l] * 100}%` }}
              transition={{ type: 'spring', stiffness: 80, damping: 18 }} />
          ))}
        </div>
        <div className="mt-3 flex flex-wrap gap-4 text-sm">
          {LABELS.map((l) => (
            <span key={l} className="flex items-center gap-2"><span className={`h-3 w-3 rounded-full ${TONE[l].bar}`} />{title(l)}</span>
          ))}
        </div>
        <p className="mt-4 text-xs text-ink-faint">{analysis.meta.source === 'YouTube' ? SAMPLE_NOTE : 'A summary of the supplied comments only.'}</p>
      </Card>
      <Card className="col-span-3">
        <p className="mb-3 font-semibold">Average score per class</p>
        <ScoreBars scores={meanScores} compact />
        <p className="mt-4 text-xs text-ink-faint">{SCORE_NOTE}</p>
      </Card>
    </div>
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

function CreatorNotes({ rows, isYouTube }) {
  const n = useMemo(() => creatorNotes(rows), [rows]);
  return (
    <Card>
      <div className="flex items-center gap-2">
        <Lightbulb size={20} aria-hidden className="text-kaya-700 dark:text-kaya-300" />
        <h2 className="text-2xl font-semibold">{isYouTube ? 'Notes for the creator' : 'What to look at'}</h2>
      </div>
      <ul className="mt-4 max-w-[75ch] list-disc space-y-2 pl-5 text-ink-soft dark:text-ube-100">
        {n.notes.map((t) => <li key={t}>{t}</li>)}
      </ul>
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
      <p className="mt-5 text-xs text-ink-faint">{NOTES_DISCLAIMER}</p>
    </Card>
  );
}

function ReportCard({ analysis, exporting, onPdf }) {
  return (
    <Card className="flex flex-wrap items-center gap-4 bg-gradient-to-r from-ube-100/70 to-transparent dark:from-ube-700/20">
      <div className="mr-auto">
        <h2 className="text-xl font-semibold">Take the report with you</h2>
        <p className="mt-1 max-w-[60ch] text-sm text-ink-soft dark:text-ube-300">
          The PDF has the summary, {analysis.meta.source === 'YouTube' ? 'video details, creator notes' : 'notes'} and every comment. The CSV keeps the original text and all three scores for your own analysis.
        </p>
      </div>
      <Ripple className="btn-primary" onClick={onPdf} disabled={exporting}>
        <FileText size={15} aria-hidden /> {exporting ? 'Building PDF…' : 'Download PDF report'}
      </Ripple>
      <Ripple className="btn-soft" onClick={() => download('comment_sentiment.csv', toCsv(rowsToRecords(analysis.rows, analysis.meta)), 'text/csv;charset=utf-8')}>
        <Download size={15} aria-hidden /> Download CSV
      </Ripple>
    </Card>
  );
}

function Column({ label, rows, total, query, sort }) {
  const [limit, setLimit] = useState(30);
  const list = useMemo(() => {
    const q = query.trim().toLowerCase();
    let l = rows.filter((r) => !q || r.comment.toLowerCase().includes(q));
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
          <p className="text-xs text-ink-soft dark:text-ube-300">{rows.length.toLocaleString()} of {total.toLocaleString()}</p>
        </div>
        <span className={`font-display text-4xl font-bold tabular-nums ${t.text}`}>{pct(total ? rows.length / total : 0, 0)}</span>
      </header>
      <ul className="max-h-[620px] space-y-2 overflow-y-auto pr-1">
        {list.slice(0, limit).map((r) => (
          <motion.li key={r.no} layout="position" className="glass rounded-xl2 px-4 py-3">
            <p className="whitespace-pre-wrap break-words text-[15px] leading-relaxed">{r.comment}</p>
            <div className="mt-2 flex items-center gap-2 text-xs text-ink-faint">
              <span className="tabular-nums">#{r.no}</span>
              {Number.isFinite(r.likes) && <span className="inline-flex items-center gap-1 tabular-nums"><ThumbsUp size={12} aria-hidden />{r.likes.toLocaleString()}</span>}
              {r.truncated && <span className="chip bg-kaya-100 text-kaya-700">truncated</span>}
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

function SentimentColumns({ rows }) {
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState('order');
  const hasLikes = rows.some((r) => Number.isFinite(r.likes));
  return (
    <Card>
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="mr-auto text-2xl font-semibold">Every comment, by label</h2>
        <label className="neu-inset flex items-center gap-2 rounded-full px-3 py-1.5">
          <Search size={15} aria-hidden className="text-ink-faint" />
          <span className="sr-only">Search comments</span>
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search all three" className="w-40 bg-transparent text-sm outline-none" />
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
      <p className="mt-2 text-sm text-ink-faint">Least confident comments are the best ones to check by hand.</p>
      <div className="mt-5 grid gap-4 lg:grid-cols-3">
        {['NEGATIVE', 'NEUTRAL', 'POSITIVE'].map((l) => (
          <Column key={l} label={l} rows={rows.filter((r) => r.sentiment === l)} total={rows.length} query={query} sort={sort} />
        ))}
      </div>
    </Card>
  );
}

export default function CommentResults({ analysis, exporting, onPdf }) {
  const s = summarise(analysis.rows);
  const isYouTube = analysis.meta.source === 'YouTube';
  return (
    <div className="space-y-5">
      {isYouTube ? <VideoCard meta={analysis.meta} /> : (
        <div>
          <h2 className="text-2xl font-semibold">{analysis.meta.title}</h2>
          <p className="text-sm text-ink-soft dark:text-ube-300">{analysis.meta.sampling}</p>
        </div>
      )}
      <Summary analysis={analysis} s={s} />
      <CreatorNotes rows={analysis.rows} isYouTube={isYouTube} />
      <ReportCard analysis={analysis} exporting={exporting} onPdf={onPdf} />
      <SentimentColumns rows={analysis.rows} />
    </div>
  );
}
