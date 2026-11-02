import { useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { Download, Sparkles, Trash2, Wand2 } from 'lucide-react';
import { useModel } from '../lib/useModel.js';
import { cleanText } from '../lib/clean.js';
import { LABELS } from '../lib/inference.js';
import { SCORE_NOTE, pct, title } from '../lib/analysis.js';
import { download, toCsv } from '../lib/csv.js';
import { explain } from '../lib/explain.js';
import { clearCorrections, correctionNote, correctionsCsv, useCorrections } from '../lib/feedback.js';
import { Badge, Card, Notice, Ripple, ScoreBars, TONE, Tilt } from '../components/ui.jsx';
import { InfoTip, WordHighlights, WrongLabel } from '../components/extras.jsx';
import ModelStatus from '../components/ModelStatus.jsx';
import { Count } from '../components/motion.jsx';
import { SentimentTriangle } from '../components/maps.jsx';

const EXAMPLES = {
  POSITIVE: ['Service dekat kedai ni bagus, staff friendly dan sangat membantu.', 'I really enjoyed this video. Thank you for the clear explanation.'],
  NEUTRAL: ['Saya akan hantar dokumen itu esok pagi.', 'The store opens at 10 AM.'],
  NEGATIVE: ['Service lambat, order salah, memang mengecewakan.', 'I am disappointed. I waited a long time and the problem is still unresolved.'],
};

// A gentle wave across the three sentiment dots; faster while the model is reading.
function Dots({ busy }) {
  const reduce = useReducedMotion();
  return (
    <div className="flex gap-2.5" aria-hidden>
      {['bg-pandan-300', 'bg-kaya-300', 'bg-jambu-300'].map((c, i) => (
        <motion.span key={c} className={`h-10 w-10 rounded-full ${c}`}
          animate={reduce ? undefined : { y: [0, busy ? -16 : -9, 0], scale: [1, busy ? 1.08 : 1.03, 1] }}
          transition={reduce ? undefined : { repeat: Infinity, duration: busy ? 0.6 : 1.4, delay: i * (busy ? 0.12 : 0.22), ease: 'easeInOut' }} />
      ))}
    </div>
  );
}

function Explanation({ text, result }) {
  const model = useModel();
  const [state, setState] = useState({ status: 'idle' });
  const run = async () => {
    setState({ status: 'busy' });
    try {
      setState({ status: 'done', impacts: await explain(text, result, model.predict) });
    } catch (e) {
      setState({ status: 'error', message: e.message });
    }
  };
  if (state.status === 'done') {
    return (
      <div className="mt-5 rounded-xl2 bg-white/50 p-4 dark:bg-night-raised/60">
        <p className="mb-1 flex items-center gap-1 text-sm font-semibold">
          Why {title(result.sentiment)}?
          <InfoTip label="How word highlights work">
            Each word is removed in turn and the text is scored again. Darker words lowered the {title(result.sentiment).toLowerCase()} score most when removed; dotted words pushed the other way. It approximates influence; it is not the model's internal reasoning.
          </InfoTip>
        </p>
        <WordHighlights impacts={state.impacts} label={result.sentiment} />
      </div>
    );
  }
  return (
    <div className="mt-5">
      <Ripple className="btn-soft" onClick={run} disabled={state.status === 'busy'}>
        <Wand2 size={15} aria-hidden /> {state.status === 'busy' ? 'Checking each word…' : 'Why this label?'}
      </Ripple>
      {state.status === 'error' && <p className="mt-2 text-sm text-jambu-700 dark:text-jambu-300">{state.message}</p>}
    </div>
  );
}

function ResultCard({ result, busy, text }) {
  const reduce = useReducedMotion();
  const tone = result ? TONE[result.sentiment] : null;
  return (
    <Tilt className="h-full" max={4}>
      <Card className="flex h-full min-h-[360px] flex-col">
        <AnimatePresence mode="wait">
          {result ? (
            <motion.div key={result.cleaned_text + result.score} className="flex h-full flex-col"
              initial={reduce ? false : { rotateY: -90, opacity: 0 }} animate={{ rotateY: 0, opacity: 1 }}
              exit={reduce ? undefined : { rotateY: 90, opacity: 0 }} transition={{ duration: 0.32 }}>
              <div className="flex items-start justify-between gap-2">
                <p className="text-sm text-ink-soft dark:text-ube-300">Predicted sentiment</p>
                <WrongLabel text={text} predicted={result.sentiment} scores={result.scores} source="Analyse" />
              </div>
              <div className="mt-1 flex flex-wrap items-end gap-x-4 gap-y-2">
                <p className={`font-display text-6xl font-bold sm:text-7xl ${tone.text}`}>{tone.word}</p>
                <div className={`mb-2 inline-flex items-center gap-2 rounded-xl2 px-4 py-2 ${tone.soft}`}>
                  <span className="font-display text-2xl font-semibold tabular-nums"><Count value={result.score * 100} decimals={1} suffix="%" /></span>
                  <span className="text-sm text-ink-soft dark:text-ube-100">score</span>
                  <InfoTip>{SCORE_NOTE}</InfoTip>
                </div>
              </div>
              {result.truncated && <div className="mt-4"><Notice tone="warn">Long input: only the first 128 model tokens were read.</Notice></div>}
              <div className="mt-5 grid items-center gap-4 sm:grid-cols-[1fr_170px]">
                <ScoreBars scores={result.scores} />
                <div className="mx-auto w-44 text-ink sm:w-full dark:text-ube-50" title="Where this text sits between the three labels">
                  <SentimentTriangle rows={[{ ...result, no: 0 }]} single />
                </div>
              </div>
              <Explanation text={text} result={result} />
            </motion.div>
          ) : (
            <motion.div key="empty" className="flex h-full flex-col items-start justify-center gap-4" initial={false}>
              <Dots busy={busy} />
              <p className="max-w-[26ch] font-display text-2xl font-semibold">{busy ? 'Reading the mood…' : 'Your result lands here.'}</p>
              <p className="text-sm text-ink-soft dark:text-ube-300">Pick an example or write your own, then analyse.</p>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>
    </Tilt>
  );
}

function Recent({ items, onClear, stored, onPick }) {
  if (!items.length) return null;
  const exportCsv = () => download('recent_texts.csv', toCsv(items.map((i) => ({
    'Time (UTC)': i.time, Text: i.text, Sentiment: i.sentiment, 'Model score': i.score,
    ...Object.fromEntries(LABELS.map((l, k) => [`${title(l)} score`, i.scores[k]])), Truncated: i.truncated,
  }))), 'text/csv;charset=utf-8');
  return (
    <Card>
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="mr-auto text-xl font-semibold">Your recent texts</h2>
        <button className="btn-soft !px-3 !py-1.5 text-xs" onClick={exportCsv}><Download size={13} aria-hidden /> CSV</button>
        <button className="btn-soft !px-3 !py-1.5 text-xs" onClick={() => { if (confirm('Clear your recent texts on this device?')) onClear(); }}>
          <Trash2 size={13} aria-hidden /> Clear
        </button>
      </div>
      <p className="mt-1 text-xs text-ink-faint">{stored ? 'Kept in this browser only.' : 'Browser storage is unavailable, so these last for this visit only.'}</p>
      <ul className="mt-3 max-h-72 space-y-2 overflow-y-auto pr-1">
        {items.slice(0, 50).map((i) => (
          <li key={i.time + i.text}>
            <button onClick={() => onPick(i.text)} className="w-full rounded-xl2 bg-white/50 px-4 py-2.5 text-left hover:bg-white/80 dark:bg-night-raised/60">
              <span className="flex items-center gap-2 text-xs text-ink-faint">
                <Badge label={i.sentiment} /><span className="tabular-nums">{pct(i.score, 0)}</span>
                <time className="ml-auto" dateTime={i.time}>{new Date(i.time).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}</time>
              </span>
              <span className="mt-1 line-clamp-2 block break-words text-sm">{i.text}</span>
            </button>
          </li>
        ))}
      </ul>
    </Card>
  );
}

function Corrections() {
  const items = useCorrections();
  return (
    <Card>
      <h2 className="text-xl font-semibold">Labels you corrected</h2>
      {items.length ? (
        <>
          <p className="mt-1 text-sm text-ink-soft dark:text-ube-300">
            {items.length} {items.length === 1 ? 'correction' : 'corrections'}. Kept in this browser. Download them to build a hand-checked test set for the next model.
          </p>
          <ul className="mt-3 max-h-52 space-y-1.5 overflow-y-auto pr-1 text-sm">
            {items.slice(0, 30).map((c) => (
              <li key={c.time + c.text} className="flex gap-2 rounded-xl2 bg-white/50 px-3 py-2 dark:bg-night-raised/60">
                <span className="shrink-0 text-xs font-semibold text-ink-soft dark:text-ube-100">{correctionNote(c)}</span>
                <span className="line-clamp-1 break-all text-ink-faint">{c.text}</span>
              </li>
            ))}
          </ul>
          <div className="mt-3 flex flex-wrap gap-2">
            <button className="btn-soft !px-3 !py-1.5 text-xs" onClick={() => download('label_corrections.csv', correctionsCsv(items), 'text/csv;charset=utf-8')}>
              <Download size={13} aria-hidden /> Download CSV
            </button>
            <button className="btn-soft !px-3 !py-1.5 text-xs" onClick={() => { if (confirm('Delete every saved correction on this device?')) clearCorrections(); }}>
              <Trash2 size={13} aria-hidden /> Clear
            </button>
          </div>
        </>
      ) : (
        <p className="mt-1 text-sm text-ink-soft dark:text-ube-300">
          When the model gets one wrong, tap <span className="font-semibold">Wrong label?</span> on the result or on any comment. Your corrections stay on this device and can be downloaded later.
        </p>
      )}
    </Card>
  );
}

export default function Analyse({ onPredicted, history, onClearHistory, stored }) {
  const model = useModel();
  const [text, setText] = useState('');
  const [analysedText, setAnalysedText] = useState('');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const turns = useRef({});
  const box = useRef(null);
  const cleaned = cleanText(text);

  const pick = (label) => {
    const i = turns.current[label] ?? 0;
    setText(EXAMPLES[label][i % EXAMPLES[label].length]);
    turns.current[label] = i + 1;
    setResult(null);
    setMessage('');
  };

  const analyse = async () => {
    setMessage('');
    if (!cleaned) { setMessage('Add some words first. Links and @mentions are removed before analysis.'); return; }
    setBusy(true);
    setResult(null);
    try {
      const [r] = await model.predict([text]);
      setResult(r);
      setAnalysedText(text);
      onPredicted({ time: new Date().toISOString(), text, ...r });
    } catch (e) {
      setMessage(e.message || 'The prediction could not be completed. Try shorter text.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-5">
      <div className="grid gap-5 lg:grid-cols-[1.2fr_1fr]">
        <Card className="flex flex-col">
          <h1 className="text-4xl font-bold leading-[1.05] sm:text-[3rem]">Read the mood of rojak text.</h1>
          <p className="mt-3 max-w-[60ch] text-ink-soft dark:text-ube-100">
            Malay, English or both in one sentence. A fine-tuned multilingual BERT runs right here in your browser, so what you type stays on your device.
          </p>
          <div className="mt-6 flex flex-wrap gap-2">
            {Object.keys(EXAMPLES).map((label) => (
              <Ripple key={label} className={`chip relative cursor-pointer overflow-hidden px-4 py-1.5 text-sm ${TONE[label].soft} ${TONE[label].text}`}
                onClick={() => pick(label)}>
                {TONE[label].word} example
              </Ripple>
            ))}
          </div>
          <label htmlFor="single" className="sr-only">Text to analyse</label>
          <textarea id="single" ref={box} value={text} maxLength={10000} rows={6}
            onChange={(e) => { setText(e.target.value); setResult(null); }}
            onKeyDown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') analyse(); }}
            placeholder="Contoh: Makanan sedap tapi service slow sikit lah…"
            className="field neu-inset mt-4 min-h-[160px] resize-y" />
          <div className="mt-2 flex flex-wrap justify-between gap-2 text-xs text-ink-faint">
            <span>Links and @mentions are removed. Ctrl or ⌘ + Enter to analyse.</span>
            <span className="tabular-nums">{text.length.toLocaleString()} / 10,000</span>
          </div>
          {cleaned && cleaned !== text.trim() && (
            <details className="mt-3 text-sm">
              <summary className="cursor-pointer text-ink-soft dark:text-ube-300">What the model reads</summary>
              <p className="mt-2 rounded-xl2 bg-ube-50 px-4 py-3 text-ink-soft dark:bg-night-raised dark:text-ube-100">{cleaned}</p>
            </details>
          )}
          <div className="mt-5 flex flex-wrap items-center gap-4">
            <Ripple className="btn-primary" onClick={analyse} disabled={busy}>
              <Sparkles size={16} aria-hidden /> {busy ? 'Analysing…' : 'Analyse sentiment'}
            </Ripple>
            <ModelStatus />
          </div>
          {message && <div className="mt-4"><Notice tone="warn">{message}</Notice></div>}
          {model.status === 'error' && !message && <div className="mt-4"><Notice tone="error">{model.error}</Notice></div>}
        </Card>
        <ResultCard result={result} busy={busy} text={analysedText} />
      </div>
      <div className="grid gap-5 lg:grid-cols-2">
        <Recent items={history} onClear={onClearHistory} stored={stored}
          onPick={(t) => { setText(t); setResult(null); box.current?.focus(); window.scrollTo({ top: 0, behavior: 'smooth' }); }} />
        <Corrections />
      </div>
    </div>
  );
}
