import { useState } from 'react';
import { ArrowRightLeft } from 'lucide-react';
import { useModel } from '../lib/useModel.js';
import { LABELS } from '../lib/inference.js';
import { pct, summarise, title } from '../lib/analysis.js';
import { recurringWords } from '../lib/insights.js';
import { analyseYouTube, isNearSplit, shareDiffs } from '../lib/pipeline.js';
import { Card, Notice, Ripple, TONE } from '../components/ui.jsx';
import { InfoTip } from '../components/extras.jsx';
import { ShareBar, VideoCard } from '../components/CommentResults.jsx';

function Side({ name, a }) {
  const s = summarise(a.rows);
  const neg = recurringWords(a.rows, 'NEGATIVE', { limit: 4 });
  const pos = recurringWords(a.rows, 'POSITIVE', { limit: 4 });
  return (
    <div className="min-w-0 space-y-4">
      <p className="font-display text-sm font-bold text-ube-600 dark:text-ube-300">Video {name}</p>
      <VideoCard meta={a.meta} small />
      <Card>
        <div className="grid grid-cols-3 gap-2">
          {LABELS.map((l) => (
            <div key={l} className={`rounded-xl2 p-3 ${TONE[l].soft}`}>
              <p className={`text-xs font-semibold ${TONE[l].text}`}>{title(l)}</p>
              <p className="font-display text-2xl font-bold tabular-nums">{pct(s.shares[l], 0)}</p>
              <p className="text-xs text-ink-soft dark:text-ube-300">{s.counts[l]}</p>
            </div>
          ))}
        </div>
        <div className="mt-4"><ShareBar s={s} height="h-4" /></div>
        <p className="mt-3 text-sm text-ink-soft dark:text-ube-300">
          {s.total} comments analysed, {a.rows.filter((r) => isNearSplit(r)).length} worth checking by hand.
        </p>
        {[['NEGATIVE', neg, 'Negative comments mention'], ['POSITIVE', pos, 'Positive comments mention']].map(([l, words, label]) => (
          <div key={l} className="mt-3">
            <p className="text-xs text-ink-faint">{label}</p>
            <div className="mt-1 flex flex-wrap gap-1.5">
              {words.length ? words.map((w) => <span key={w.word} className={`chip ${TONE[l].soft} ${TONE[l].text}`}>{w.word} {w.comments}</span>)
                : <span className="text-xs text-ink-faint">No repeated words</span>}
            </div>
          </div>
        ))}
      </Card>
    </div>
  );
}

function Differences({ a, b }) {
  const sa = summarise(a.rows);
  const sb = summarise(b.rows);
  const diffs = shareDiffs(sa, sb);
  const small = Math.min(sa.total, sb.total) < 30;
  return (
    <Card>
      <div className="flex items-center gap-1">
        <h2 className="text-2xl font-semibold">What's different</h2>
        <InfoTip label="Reading the differences">
          Each video's comments are a sample ordered by relevance, so small gaps can come from which comments were picked. Treat differences under about 10 points as noise, especially with fewer than 50 comments each.
        </InfoTip>
      </div>
      <ul className="mt-4 space-y-3">
        {diffs.map((d) => {
          const same = Math.abs(d.points) < 5;
          return (
            <li key={d.label} className={`flex flex-wrap items-center gap-3 rounded-xl2 px-4 py-3 ${TONE[d.label].soft}`}>
              <span className={`w-20 font-semibold ${TONE[d.label].text}`}>{title(d.label)}</span>
              <span className="tabular-nums">A {pct(d.a, 0)}</span>
              <span className="text-ink-faint">→</span>
              <span className="tabular-nums">B {pct(d.b, 0)}</span>
              <span className="ml-auto font-display text-lg font-bold tabular-nums">
                {same ? 'About the same' : `B is ${Math.abs(d.points)} points ${d.points > 0 ? 'higher' : 'lower'}`}
              </span>
            </li>
          );
        })}
      </ul>
      {small && <p className="mt-3 text-sm text-ink-faint">At least one video has fewer than 30 analysed comments, so these differences are rough.</p>}
    </Card>
  );
}

export default function Compare() {
  const model = useModel();
  const [urls, setUrls] = useState(['', '']);
  const [limit, setLimit] = useState(50);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(null);
  const [message, setMessage] = useState('');
  const [result, setResult] = useState(null);

  const run = async () => {
    setMessage('');
    setResult(null);
    setBusy(true);
    try {
      const out = [];
      for (const [i, url] of urls.entries()) {
        const name = i === 0 ? 'A' : 'B';
        try {
          out.push(await analyseYouTube(url, limit, model.predict, (p) => setProgress({ ...p, name })));
        } catch (e) {
          throw new Error(`Video ${name}: ${e.message}`);
        }
      }
      setResult(out);
    } catch (e) {
      setMessage(e.message);
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  return (
    <div className="space-y-5">
      <Card>
        <h1 className="text-3xl font-bold sm:text-4xl">Compare two videos.</h1>
        <p className="mt-2 max-w-[62ch] text-ink-soft dark:text-ube-100">
          See how the comment mood differs between two uploads, for example this week's video and last week's.
        </p>
        <div className="mt-5 grid gap-4 md:grid-cols-[1fr_auto_1fr] md:items-end">
          {[0, 1].map((i) => (
            <label key={i} className={`block ${i === 1 ? 'md:order-3' : ''}`}>
              <span className="mb-2 block text-sm font-medium">Video {i === 0 ? 'A' : 'B'}</span>
              <input value={urls[i]} inputMode="url" placeholder="https://www.youtube.com/watch?v=…" className="field neu-inset"
                onChange={(e) => setUrls((u) => u.map((v, k) => (k === i ? e.target.value : v)))} />
            </label>
          ))}
          <button className="neu-raised hidden h-11 w-11 place-items-center rounded-full md:order-2 md:grid" aria-label="Swap videos"
            onClick={() => setUrls(([x, y]) => [y, x])}>
            <ArrowRightLeft size={17} aria-hidden />
          </button>
        </div>
        <div className="mt-5 flex flex-wrap items-center gap-5">
          <Ripple className="btn-primary" onClick={run} disabled={busy || !urls[0].trim() || !urls[1].trim()}>{busy ? 'Working…' : 'Compare videos'}</Ripple>
          <label className="flex items-center gap-3 text-sm">
            Up to {limit} comments each
            <input type="range" min={20} max={100} step={10} value={limit} onChange={(e) => setLimit(Number(e.target.value))} className="w-40 accent-ube-600" />
          </label>
          {progress && (
            <p className="text-sm text-ink-soft dark:text-ube-300" role="status" aria-live="polite">
              Video {progress.name}: {progress.label}{progress.total ? ` ${progress.done} of ${progress.total}` : '…'}
            </p>
          )}
        </div>
        {message && <div className="mt-4"><Notice tone="warn">{message}</Notice></div>}
      </Card>
      {result && (
        <>
          <Differences a={result[0]} b={result[1]} />
          <div className="grid gap-5 lg:grid-cols-2">
            <Side name="A" a={result[0]} />
            <Side name="B" a={result[1]} />
          </div>
        </>
      )}
    </div>
  );
}
