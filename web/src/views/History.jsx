import { Download, Trash2 } from 'lucide-react';
import { LABELS } from '../lib/inference.js';
import { pct, title } from '../lib/analysis.js';
import { download, toCsv } from '../lib/csv.js';
import { Badge, Card, Ripple, TONE } from '../components/ui.jsx';

export default function History({ items, onClear, stored }) {
  if (!items.length) {
    return (
      <Card className="text-center">
        <h1 className="text-3xl font-bold">No predictions yet.</h1>
        <p className="mx-auto mt-2 max-w-[48ch] text-ink-soft dark:text-ube-100">Analyse some text and each result is kept here, on this device only.</p>
        <a href="#/" className="btn-primary mt-5">Analyse text</a>
      </Card>
    );
  }
  const counts = Object.fromEntries(LABELS.map((l) => [l, items.filter((i) => i.sentiment === l).length]));
  const max = Math.max(...Object.values(counts), 1);
  const exportCsv = () => download('session_sentiment.csv', toCsv(items.map((i) => ({
    'Time (UTC)': i.time, Text: i.text, Sentiment: i.sentiment, 'Model score': i.score,
    ...Object.fromEntries(LABELS.map((l, k) => [`${title(l)} score`, i.scores[k]])), Truncated: i.truncated,
  }))), 'text/csv;charset=utf-8');
  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_2fr]">
      <Card className="h-fit">
        <h1 className="text-3xl font-bold">Your predictions</h1>
        <p className="mt-1 text-sm text-ink-soft dark:text-ube-300">
          {items.length} saved {stored ? 'in this browser' : 'for this visit only, because browser storage is unavailable'}.
        </p>
        <div className="mt-6 flex h-36 items-end gap-4" role="img" aria-label={LABELS.map((l) => `${title(l)} ${counts[l]}`).join(', ')}>
          {LABELS.map((l) => (
            <div key={l} className="flex flex-1 flex-col items-center gap-2">
              <span className="text-sm font-semibold tabular-nums">{counts[l]}</span>
              <div className="neu-inset flex h-24 w-full items-end overflow-hidden rounded-xl2">
                <div className={`w-full rounded-xl2 ${TONE[l].bar}`} style={{ height: `${(counts[l] / max) * 100}%` }} />
              </div>
              <span className="text-xs text-ink-soft dark:text-ube-300">{title(l)}</span>
            </div>
          ))}
        </div>
        <div className="mt-6 flex flex-wrap gap-3">
          <Ripple className="btn-soft" onClick={exportCsv}><Download size={15} aria-hidden /> Download CSV</Ripple>
          <Ripple className="btn-soft" onClick={() => { if (confirm('Clear every saved prediction on this device?')) onClear(); }}>
            <Trash2 size={15} aria-hidden /> Clear history
          </Ripple>
        </div>
      </Card>
      <Card>
        <ul className="space-y-2">
          {items.map((i) => (
            <li key={i.time + i.text} className="rounded-xl2 bg-white/50 px-4 py-3 dark:bg-night-raised/70">
              <div className="flex flex-wrap items-center gap-2 text-xs text-ink-faint">
                <time dateTime={i.time}>{new Date(i.time).toLocaleString()}</time>
                <Badge label={i.sentiment} /><span className="tabular-nums">{pct(i.score)}</span>
              </div>
              <p className="mt-1.5 whitespace-pre-wrap break-words">{i.text}</p>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}
