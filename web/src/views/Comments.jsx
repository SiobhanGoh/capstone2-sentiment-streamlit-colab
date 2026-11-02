import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { Download, FileText, Search, Upload } from 'lucide-react';
import { useModel } from '../lib/useModel.js';
import { MAX_COMMENTS, SAMPLE_NOTE, SCORE_NOTE, buildRows, pct, splitPasted, summarise, title, usableComments } from '../lib/analysis.js';
import { download, guessTextColumn, parseCsvFile, rowsToRecords, toCsv } from '../lib/csv.js';
import { fetchComments } from '../lib/youtube.js';
import { LABELS } from '../lib/inference.js';
import { Badge, Card, Notice, Ripple, ScoreBars, TONE, Tilt } from '../components/ui.jsx';

const SOURCES = [
  { id: 'paste', name: 'Paste comments' },
  { id: 'youtube', name: 'YouTube video' },
  { id: 'file', name: 'Upload a CSV' },
];

function SourceTabs({ value, onChange }) {
  return (
    <div role="tablist" aria-label="Comment source" className="neu-inset inline-flex flex-wrap gap-1 rounded-full p-1">
      {SOURCES.map((s) => (
        <button key={s.id} role="tab" aria-selected={value === s.id} onClick={() => onChange(s.id)}
          className={`relative rounded-full px-4 py-2 text-sm font-semibold transition-colors ${value === s.id ? 'text-ink dark:text-ube-50' : 'text-ink-soft dark:text-ube-300'}`}>
          {value === s.id && <motion.span layoutId="source-pill" className="neu-raised absolute inset-0 rounded-full" transition={{ type: 'spring', stiffness: 300, damping: 28 }} />}
          <span className="relative">{s.name}</span>
        </button>
      ))}
    </div>
  );
}

function Summary({ analysis }) {
  const s = summarise(analysis.rows);
  const meanScores = LABELS.map((_, i) => analysis.rows.reduce((a, r) => a + r.scores[i], 0) / s.total);
  return (
    <div className="grid gap-5 md:grid-cols-6">
      <Tilt className="md:col-span-2" max={5}>
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
        <Tilt key={l} className="md:col-span-1 max-md:col-span-1" max={6}>
          <Card className={`h-full ${TONE[l].soft} !border-transparent`}>
            <p className={`text-sm font-semibold ${TONE[l].text}`}>{title(l)}</p>
            <p className="mt-2 font-display text-3xl font-bold tabular-nums">{pct(s.shares[l], 0)}</p>
            <p className="mt-1 text-sm text-ink-soft dark:text-ube-300">{s.counts[l].toLocaleString()} {s.counts[l] === 1 ? 'comment' : 'comments'}</p>
          </Card>
        </Tilt>
      ))}
      <Card className="md:col-span-1">
        <p className="text-sm text-ink-soft dark:text-ube-300">Mean top score</p>
        <p className="mt-2 font-display text-3xl font-bold tabular-nums">{pct(s.meanScore, 0)}</p>
        <p className="mt-1 text-xs text-ink-faint">Not accuracy</p>
      </Card>
      <Card className="md:col-span-3">
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
      <Card className="md:col-span-3">
        <p className="mb-3 font-semibold">Average score per class</p>
        <ScoreBars scores={meanScores} compact />
        <p className="mt-4 text-xs text-ink-faint">{SCORE_NOTE}</p>
      </Card>
    </div>
  );
}

function ResultsTable({ rows }) {
  const [filter, setFilter] = useState('ALL');
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState('order');
  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    let list = rows.filter((r) => (filter === 'ALL' || r.sentiment === filter) && (!q || r.comment.toLowerCase().includes(q)));
    if (sort === 'high') list = [...list].sort((a, b) => b.score - a.score);
    if (sort === 'low') list = [...list].sort((a, b) => a.score - b.score);
    return list;
  }, [rows, filter, query, sort]);
  return (
    <Card>
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="mr-auto text-xl font-semibold">Every comment</h2>
        <label className="neu-inset flex items-center gap-2 rounded-full px-3 py-1.5">
          <Search size={15} aria-hidden className="text-ink-faint" />
          <span className="sr-only">Search comments</span>
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search"
            className="w-36 bg-transparent text-sm outline-none" />
        </label>
        <label className="text-sm">
          <span className="sr-only">Filter by sentiment</span>
          <select value={filter} onChange={(e) => setFilter(e.target.value)} className="neu-inset rounded-full px-3 py-1.5 text-sm">
            <option value="ALL">All labels</option>
            {LABELS.map((l) => <option key={l} value={l}>{title(l)}</option>)}
          </select>
        </label>
        <label className="text-sm">
          <span className="sr-only">Sort</span>
          <select value={sort} onChange={(e) => setSort(e.target.value)} className="neu-inset rounded-full px-3 py-1.5 text-sm">
            <option value="order">Original order</option>
            <option value="high">Highest score first</option>
            <option value="low">Lowest score first</option>
          </select>
        </label>
      </div>
      <p className="mt-3 text-sm text-ink-faint">Showing {shown.length.toLocaleString()} of {rows.length.toLocaleString()}. Low scores are a good place to start checking by hand.</p>
      <ul className="mt-4 max-h-[560px] space-y-2 overflow-y-auto pr-1">
        {shown.slice(0, 500).map((r) => (
          <li key={r.no} className="rounded-xl2 bg-white/50 px-4 py-3 dark:bg-night-raised/70">
            <div className="flex flex-wrap items-center gap-2 text-xs text-ink-faint">
              <span className="tabular-nums">#{r.no}</span><Badge label={r.sentiment} />
              <span className="tabular-nums">{pct(r.score)}</span>
              {r.truncated && <span className="chip bg-kaya-100 text-kaya-700">truncated</span>}
            </div>
            <p className="mt-1.5 whitespace-pre-wrap break-words text-[15px]">{r.comment}</p>
          </li>
        ))}
      </ul>
      {shown.length > 500 && <p className="mt-3 text-sm text-ink-faint">The first 500 matches are shown here. Downloads include everything.</p>}
    </Card>
  );
}

export default function Comments() {
  const model = useModel();
  const [source, setSource] = useState('paste');
  const [pasted, setPasted] = useState('');
  const [video, setVideo] = useState('');
  const [limit, setLimit] = useState(50);
  const [file, setFile] = useState(null);
  const [column, setColumn] = useState('');
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(null);
  const [message, setMessage] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [exporting, setExporting] = useState(false);

  const choose = (id) => { setSource(id); setMessage(''); };

  const readFile = async (f) => {
    setMessage('');
    setFile(null);
    if (!f) return;
    try {
      const parsed = await parseCsvFile(f);
      setFile({ name: f.name, ...parsed });
      setColumn(guessTextColumn(parsed.fields, parsed.data));
    } catch (e) {
      setMessage(e.message);
    }
  };

  const run = async () => {
    setMessage('');
    setAnalysis(null);
    setBusy(true);
    setProgress(null);
    try {
      let comments;
      let meta;
      if (source === 'paste') {
        comments = splitPasted(pasted);
        meta = { source: 'Pasted comments', title: 'Pasted comment analysis', sampling: 'User-supplied comments; not a representative sample.' };
      } else if (source === 'youtube') {
        setProgress({ label: 'Fetching comments from YouTube' });
        const data = await fetchComments(video, limit);
        comments = data.comments;
        meta = data.meta;
      } else {
        if (!file) throw new Error('Choose a CSV file first.');
        comments = file.data.map((r) => String(r[column] ?? ''));
        meta = { source: 'Uploaded file', title: file.name, sampling: `Column "${column}" from the uploaded file; not a representative sample.` };
      }
      if (!comments.length) throw new Error('No comments were found. Try another source or add one comment per line.');
      if (comments.length > MAX_COMMENTS) throw new Error(`Please use up to ${MAX_COMMENTS.toLocaleString()} comments at a time.`);
      const { valid, excluded } = usableComments(comments);
      if (!valid.length) throw new Error('None of these comments has text left after links and mentions are removed.');
      setProgress({ label: 'Analysing', done: 0, total: valid.length });
      const results = await model.predict(valid.map((v) => v.comment), (done, total) => setProgress({ label: 'Analysing', done, total }));
      setAnalysis({ rows: buildRows(valid, results), meta: { ...meta, excluded, analysed_at: new Date().toISOString() } });
    } catch (e) {
      setMessage(e.message || 'This analysis could not be completed. Try fewer or shorter comments.');
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  const exportPdf = async () => {
    setExporting(true);
    try {
      const { buildReport } = await import('../lib/pdf.js');
      download('sentiment_report.pdf', await buildReport(analysis.rows, analysis.meta), 'application/pdf');
    } catch {
      setMessage('The PDF could not be created. The CSV download still has every result.');
    } finally {
      setExporting(false);
    }
  };

  const lineCount = splitPasted(pasted).length;

  return (
    <div className="space-y-5">
      <Card>
        <h1 className="text-3xl font-bold sm:text-4xl">Read a whole comment section.</h1>
        <p className="mt-2 max-w-[62ch] text-ink-soft dark:text-ube-100">
          Paste comments, pull them from a YouTube video, or upload a spreadsheet export. Up to {MAX_COMMENTS.toLocaleString()} at a time, analysed on your device.
        </p>
        <div className="mt-5"><SourceTabs value={source} onChange={choose} /></div>

        <div className="mt-5">
          {source === 'paste' && (
            <>
              <label htmlFor="pasted" className="mb-2 block text-sm font-medium">One comment per line</label>
              <textarea id="pasted" rows={8} value={pasted} onChange={(e) => setPasted(e.target.value)}
                placeholder={'Best gila video ni!\nThe audio is too soft\nBila nak upload part 2?'}
                className="field neu-inset min-h-[200px] resize-y" />
              <p className="mt-2 text-xs text-ink-faint">{lineCount.toLocaleString()} {lineCount === 1 ? 'comment' : 'comments'}. Use comments you are permitted to analyse.</p>
            </>
          )}
          {source === 'youtube' && (
            <div className="grid gap-4 sm:grid-cols-[1fr_auto] sm:items-end">
              <label className="block">
                <span className="mb-2 block text-sm font-medium">YouTube link or video ID</span>
                <input value={video} onChange={(e) => setVideo(e.target.value)} placeholder="https://www.youtube.com/watch?v=…"
                  className="field neu-inset" inputMode="url" />
              </label>
              <label className="block">
                <span className="mb-2 block text-sm font-medium">Up to {limit} comments</span>
                <input type="range" min={10} max={100} step={10} value={limit} onChange={(e) => setLimit(Number(e.target.value))}
                  className="w-48 accent-ube-600" />
              </label>
              <p className="text-xs text-ink-faint sm:col-span-2">Public top-level comments, ordered by relevance. Replies are left out. Only the video link is sent to our server; analysis still runs here.</p>
            </div>
          )}
          {source === 'file' && (
            <div className="space-y-4">
              <label className="neu-inset flex cursor-pointer flex-col items-center gap-2 rounded-xl2 px-6 py-8 text-center">
                <Upload aria-hidden className="text-ube-600 dark:text-ube-300" />
                <span className="font-medium">{file ? file.name : 'Choose a CSV file'}</span>
                <span className="text-xs text-ink-faint">The first row must hold column names. The file stays in your browser.</span>
                <input type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => readFile(e.target.files?.[0])} />
              </label>
              {file && (
                <label className="block text-sm">
                  <span className="mb-2 block font-medium">Column with the comments ({file.data.length.toLocaleString()} rows)</span>
                  <select value={column} onChange={(e) => setColumn(e.target.value)} className="neu-inset rounded-full px-4 py-2">
                    {file.fields.map((f) => <option key={f} value={f}>{f}</option>)}
                  </select>
                </label>
              )}
            </div>
          )}
        </div>

        <div className="mt-6 flex flex-wrap items-center gap-4">
          <Ripple className="btn-primary" onClick={run} disabled={busy}>{busy ? 'Working…' : 'Analyse comments'}</Ripple>
          {progress && (
            <div className="min-w-[220px] flex-1" role="status" aria-live="polite">
              <p className="mb-1.5 text-xs text-ink-soft dark:text-ube-300">
                {progress.label}{progress.total ? ` ${progress.done.toLocaleString()} of ${progress.total.toLocaleString()}` : '…'}
              </p>
              <div className="neu-inset h-2 overflow-hidden rounded-full">
                <div className="h-full rounded-full bg-ube-500 transition-[width]" style={{ width: progress.total ? `${(progress.done / progress.total) * 100}%` : '12%' }} />
              </div>
            </div>
          )}
        </div>
        {message && <div className="mt-4"><Notice tone="warn">{message}</Notice></div>}
      </Card>

      {analysis && (
        <>
          <div className="flex flex-wrap items-end gap-3">
            <div className="mr-auto">
              <h2 className="text-2xl font-semibold">{analysis.meta.title}</h2>
              <p className="text-sm text-ink-soft dark:text-ube-300">{analysis.meta.channel ? `${analysis.meta.channel} · ` : ''}{analysis.meta.sampling}</p>
            </div>
            <Ripple className="btn-soft" onClick={() => download('comment_sentiment.csv', toCsv(rowsToRecords(analysis.rows, analysis.meta)), 'text/csv;charset=utf-8')}>
              <Download size={15} aria-hidden /> Download CSV
            </Ripple>
            <Ripple className="btn-soft" onClick={exportPdf} disabled={exporting}>
              <FileText size={15} aria-hidden /> {exporting ? 'Building PDF…' : 'Download PDF report'}
            </Ripple>
          </div>
          <Summary analysis={analysis} />
          <ResultsTable rows={analysis.rows} />
        </>
      )}
    </div>
  );
}
