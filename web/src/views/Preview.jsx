// Design preview only (VITE_PREVIEW=1): an asymmetric results layout with sample data.
import { useEffect, useMemo, useState } from 'react';
import { animate, motion, useReducedMotion } from 'framer-motion';
import { Copy, Download, FileText, HelpCircle, Lightbulb, ThumbsUp } from 'lucide-react';
import { LABELS } from '../lib/inference.js';
import { pct, summarise, title } from '../lib/analysis.js';
import { creatorNotes } from '../lib/insights.js';
import { isNearSplit } from '../lib/pipeline.js';
import { bucketLabel, timeBuckets } from '../lib/timeline.js';
import { TONE } from '../components/ui.jsx';
import { SentimentTriangle } from '../components/maps.jsx';

const TEXTS = [
  ['Best gila video ni! editing terbaik', 'POSITIVE'], ['The audio is too soft, susah nak dengar', 'NEGATIVE'], ['Bila nak upload part 2?', 'NEUTRAL'],
  ['audio lambat sync dengan video, teruk', 'NEGATIVE'], ['Saya tengok dari Johor', 'NEUTRAL'], ['love the editing, suka sangat', 'POSITIVE'],
  ['How did you film the drone shots?', 'NEUTRAL'], ['ok je, nothing special', 'NEUTRAL'], ['Best char kway teow in Penang, sedap gila!', 'POSITIVE'],
  ['terbaik! terima kasih sebab share', 'POSITIVE'], ['worst audio ever, please fix the mic', 'NEGATIVE'], ['Ipoh punya lagi best tbh', 'POSITIVE'],
  ['Datang dari JB semata-mata nak cuba, worth it', 'POSITIVE'], ['KL traffic teruk sangat nak pergi sana', 'NEGATIVE'], ['nak tahu harga dia berapa?', 'NEUTRAL'],
  ['music background tu sedap', 'POSITIVE'], ['first time tengok channel ni', 'NEUTRAL'], ['editing makin power', 'POSITIVE'],
];
const SAMPLE = Array.from({ length: 54 }, (_, i) => {
  const [comment, sentiment] = TEXTS[i % TEXTS.length];
  const k = LABELS.indexOf(sentiment);
  const sure = 0.5 + ((i * 37) % 45) / 100;
  const rest = (1 - sure) * (((i * 13) % 10) / 10);
  const scores = LABELS.map((_, j) => (j === k ? sure : j === (k + 1) % 3 ? rest : 1 - sure - rest));
  const day = (i * 11) % 34;
  return { no: i + 1, comment, sentiment, score: sure, scores, truncated: false, likes: (i * 53) % 140, date: new Date(Date.UTC(2026, 9, 1 + day, 6 + (i % 12))).toISOString() };
});
const NOW = Date.UTC(2026, 10, 3);
const PERIODS = [{ id: 'all', name: 'All time', days: null }, { id: '30', name: 'Last 30 days', days: 30 }, { id: '7', name: 'Last 7 days', days: 7 }];

function Count({ value, suffix = '' }) {
  const reduce = useReducedMotion();
  const [shown, setShown] = useState(value);
  useEffect(() => {
    if (reduce) { setShown(value); return undefined; }
    const c = animate(shown, value, { duration: 0.24, ease: 'easeOut', onUpdate: (v) => setShown(v) });
    return () => c.stop();
  }, [value]); // eslint-disable-line react-hooks/exhaustive-deps
  return <>{Math.round(shown)}{suffix}</>;
}

function Segmented({ value, onChange }) {
  return (
    <div role="tablist" aria-label="Period" className="neu-inset inline-flex gap-1 rounded-full p-1">
      {PERIODS.map((p) => (
        <button key={p.id} role="tab" aria-selected={value === p.id} onClick={() => onChange(p.id)}
          className={`relative rounded-full px-4 py-2 text-[13px] font-semibold transition-colors duration-150 ${value === p.id ? 'text-white' : 'text-ink-soft hover:text-ink dark:text-ube-100'}`}>
          {value === p.id && <motion.span layoutId="period" className="absolute inset-0 rounded-full bg-ube-600" transition={{ type: 'spring', stiffness: 380, damping: 32 }} />}
          <span className="relative">{p.name}</span>
        </button>
      ))}
    </div>
  );
}

export default function Preview() {
  const [period, setPeriod] = useState('all');
  const rows = useMemo(() => {
    const days = PERIODS.find((p) => p.id === period).days;
    return days ? SAMPLE.filter((r) => NOW - Date.parse(r.date) <= days * 86400000) : SAMPLE;
  }, [period]);
  const s = summarise(rows);
  const n = creatorNotes(rows);
  const lead = [...LABELS].sort((a, b) => s.shares[b] - s.shares[a])[0];
  const others = LABELS.filter((l) => l !== lead);
  const t = timeBuckets(rows);
  const max = t ? Math.max(...t.buckets.map((b) => b.total), 1) : 1;
  const stats = [
    ['Comments', s.total], ['Worth checking', rows.filter((r) => isNearSplit(r)).length], ['Questions', n.questions.length], ['Top theme', n.posWords[0]?.word ?? '–'],
  ];

  return (
    <div className="grid grid-cols-12 gap-6">
      <div className="col-span-12 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-[13px] text-ink-soft dark:text-ube-300">Results for</p>
          <h1 className="text-[40px] font-bold leading-tight">Makan-makan di Kuala Lumpur</h1>
        </div>
        <Segmented value={period} onChange={setPeriod} />
      </div>

      {/* Main column */}
      <div className="col-span-12 space-y-6 lg:col-span-8">
        <section className="glass rounded-[28px] p-8">
          <div className="grid gap-8 md:grid-cols-[1.5fr_1fr] md:items-end">
            <div>
              <motion.p key={lead} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.24 }}
                className={`font-display text-[136px] font-bold leading-[0.85] tracking-[-0.04em] tabular-nums ${TONE[lead].text}`}>
                <Count value={Math.round(s.shares[lead] * 100)} suffix="%" />
              </motion.p>
              <p className="mt-3 text-[24px] font-semibold">of comments read as {title(lead).toLowerCase()}</p>
            </div>
            <div className="space-y-4">
              {others.map((l) => (
                <div key={l} className="flex items-baseline justify-between border-b border-ink/10 pb-3 dark:border-white/10">
                  <span className={`text-[15px] font-semibold ${TONE[l].text}`}>{title(l)}</span>
                  <span className="font-display text-[56px] font-bold leading-none tabular-nums"><Count value={Math.round(s.shares[l] * 100)} suffix="%" /></span>
                </div>
              ))}
            </div>
          </div>
          <div className="neu-inset mt-8 flex h-4 overflow-hidden rounded-full">
            {LABELS.map((l) => <motion.div key={l} className={TONE[l].bar} animate={{ width: `${s.shares[l] * 100}%` }} transition={{ type: 'spring', stiffness: 120, damping: 20 }} />)}
          </div>
        </section>

        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          {stats.map(([k, v]) => (
            <div key={k} className="glass rounded-[20px] px-5 py-4">
              <p className="text-[13px] text-ink-soft dark:text-ube-300">{k}</p>
              <p className="mt-1 font-display text-[32px] font-bold leading-tight tabular-nums">{typeof v === 'number' ? <Count value={v} /> : v}</p>
            </div>
          ))}
        </div>

        <div className="grid gap-6 md:grid-cols-5">
          <section className="glass rounded-[28px] p-6 md:col-span-3">
            <h2 className="text-[24px] font-semibold">Sentiment map</h2>
            <div className="mt-2 text-ink dark:text-ube-50"><SentimentTriangle rows={rows} /></div>
          </section>
          <section className="glass flex flex-col rounded-[28px] p-6 md:col-span-2">
            <h2 className="text-[24px] font-semibold">Posted over time</h2>
            <p className="text-[13px] text-ink-faint">{t ? `By ${t.unit}` : 'Not enough dates'}</p>
            {t && (
              <div className="mt-auto flex h-56 items-end gap-1 pt-4">
                {t.buckets.map((b) => (
                  <motion.div key={b.start} layout className="flex h-full min-w-0 flex-1 flex-col justify-end" title={`${bucketLabel(b.start, t.unit)}: ${b.total}`}>
                    <motion.div className="flex flex-col-reverse overflow-hidden rounded-t-md" animate={{ height: `${(b.total / max) * 100}%` }} transition={{ duration: 0.24 }}>
                      {LABELS.map((l) => b[l] > 0 && <div key={l} className={TONE[l].bar} style={{ height: `${(b[l] / b.total) * 100}%` }} />)}
                    </motion.div>
                  </motion.div>
                ))}
              </div>
            )}
          </section>
        </div>
      </div>

      {/* Right vertical panel */}
      <aside className="col-span-12 lg:col-span-4">
        <div className="glass space-y-6 overflow-hidden rounded-[28px] lg:sticky lg:top-6">
          <div className="relative">
            <img src="/api/thumb?id=sample" alt="" className="aspect-[4/3] w-full object-cover" />
            <div className="absolute inset-0 bg-gradient-to-t from-ink/80 via-ink/10 to-transparent" />
            <div className="absolute inset-x-0 bottom-0 p-6 text-white">
              <p className="text-[13px] opacity-80">Jalan Jalan Cari Makan</p>
              <p className="mt-1 text-[24px] font-semibold leading-tight">Street food tour, Kuala Lumpur</p>
              <div className="mt-3 flex gap-2 text-[12px]">
                {['184K views', '9.1K likes', '612 comments'].map((f) => <span key={f} className="rounded-full bg-white/20 px-3 py-1 backdrop-blur">{f}</span>)}
              </div>
            </div>
          </div>
          <div className="space-y-4 px-6">
            <h2 className="flex items-center gap-2 text-[20px] font-semibold"><Lightbulb size={18} className="text-kaya-700 dark:text-kaya-300" aria-hidden /> For the creator</h2>
            {n.readFirst.slice(0, 2).map((r) => (
              <div key={r.no} className={`rounded-[14px] px-4 py-3 ${TONE.NEGATIVE.soft}`}>
                <p className="text-[15px]">{r.comment}</p>
                <p className="mt-1 flex items-center gap-1 text-[12px] text-ink-soft dark:text-ube-100"><ThumbsUp size={12} aria-hidden />{r.likes}</p>
              </div>
            ))}
            {n.questions[0] && (
              <div className="rounded-[14px] bg-ube-100 px-4 py-3 dark:bg-ube-700/30">
                <p className="flex items-center gap-1 text-[12px] font-semibold text-ube-700 dark:text-ube-100"><HelpCircle size={12} aria-hidden /> Viewer question</p>
                <p className="mt-1 text-[15px]">{n.questions[0].comment}</p>
              </div>
            )}
          </div>
          <div className="grid gap-2 px-6 pb-6">
            <button className="btn-primary w-full"><FileText size={15} aria-hidden /> Download PDF report</button>
            <div className="grid grid-cols-2 gap-2">
              <button className="btn-soft"><Download size={15} aria-hidden /> CSV</button>
              <button className="btn-soft"><Copy size={15} aria-hidden /> Copy summary</button>
            </div>
          </div>
        </div>
      </aside>
    </div>
  );
}
