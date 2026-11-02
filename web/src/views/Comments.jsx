import { useState } from 'react';
import { motion } from 'framer-motion';
import { Upload } from 'lucide-react';
import { useModel } from '../lib/useModel.js';
import { MAX_COMMENTS, splitPasted } from '../lib/analysis.js';
import { analyseComments, analyseYouTube } from '../lib/pipeline.js';
import { download, guessTextColumn, parseCsvFile } from '../lib/csv.js';
import { Card, Notice, Ripple } from '../components/ui.jsx';
import CommentResults from '../components/CommentResults.jsx';

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
      if (source === 'youtube') {
        setAnalysis(await analyseYouTube(video, limit, model.predict, setProgress));
        return;
      }
      let comments;
      let meta;
      if (source === 'paste') {
        comments = splitPasted(pasted);
        meta = { source: 'Pasted comments', title: 'Pasted comment analysis', sampling: 'User-supplied comments; not a representative sample.' };
      } else {
        if (!file) throw new Error('Choose a CSV file first.');
        comments = file.data.map((r) => String(r[column] ?? ''));
        meta = { source: 'Uploaded file', title: file.name, sampling: `Column "${column}" from the uploaded file; not a representative sample.` };
      }
      if (!comments.length) throw new Error('No comments were found. Try another source or add one comment per line.');
      if (comments.length > MAX_COMMENTS) throw new Error(`Please use up to ${MAX_COMMENTS.toLocaleString()} comments at a time.`);
      setAnalysis(await analyseComments(comments, meta, model.predict, setProgress));
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

      {analysis && <CommentResults analysis={analysis} exporting={exporting} onPdf={exportPdf} />}
    </div>
  );
}
