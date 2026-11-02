import { useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { Sparkles } from 'lucide-react';
import { useModel } from '../lib/useModel.js';
import { cleanText } from '../lib/clean.js';
import { SCORE_NOTE, pct } from '../lib/analysis.js';
import { Card, Notice, Ripple, ScoreBars, TONE, Tilt } from '../components/ui.jsx';
import ModelStatus from '../components/ModelStatus.jsx';

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

function ResultCard({ result, busy }) {
  const reduce = useReducedMotion();
  const tone = result ? TONE[result.sentiment] : null;
  return (
    <Tilt className="h-full">
      <Card className="flex h-full min-h-[360px] flex-col">
        <AnimatePresence mode="wait">
          {result ? (
            <motion.div key={result.cleaned_text + result.score} className="flex h-full flex-col"
              initial={reduce ? false : { rotateY: -90, opacity: 0 }} animate={{ rotateY: 0, opacity: 1 }}
              exit={reduce ? undefined : { rotateY: 90, opacity: 0 }} transition={{ duration: 0.32 }}>
              <p className="text-sm text-ink-soft dark:text-ube-300">Predicted sentiment</p>
              <div className="mt-1 flex flex-wrap items-end gap-x-4 gap-y-2">
                <p className={`font-display text-6xl font-bold sm:text-7xl ${tone.text}`}>{tone.word}</p>
                <div className={`mb-2 inline-flex items-baseline gap-2 rounded-xl2 px-4 py-2 ${tone.soft}`}>
                  <span className="font-display text-2xl font-semibold tabular-nums">{pct(result.score)}</span>
                  <span className="text-sm text-ink-soft dark:text-ube-100">top-class score</span>
                </div>
              </div>
              {result.truncated && <div className="mt-4"><Notice tone="warn">Long input: only the first 128 model tokens were read.</Notice></div>}
              <div className="mt-6">
                <p className="mb-3 text-sm font-semibold">How the model split its scores</p>
                <ScoreBars scores={result.scores} />
              </div>
              <p className="mt-auto pt-6 text-xs leading-relaxed text-ink-soft dark:text-ube-300">
                A suggested label, not a verdict. Sarcasm, slang and unfamiliar topics can change the meaning. {SCORE_NOTE}
              </p>
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

export default function Analyse({ onPredicted }) {
  const model = useModel();
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const turns = useRef({});
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
      onPredicted({ time: new Date().toISOString(), text, ...r });
    } catch (e) {
      setMessage(e.message || 'The prediction could not be completed. Try shorter text.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid gap-5 lg:grid-cols-[1.25fr_1fr]">
      <Card className="flex flex-col">
        <h1 className="text-4xl font-bold leading-[1.05] sm:text-5xl">
          Read the mood of rojak text.
        </h1>
        <p className="mt-3 max-w-[60ch] text-ink-soft dark:text-ube-100">
          Malay, English or both in one sentence. A fine-tuned multilingual BERT runs right here in your browser, so what you type stays on your device.
        </p>
        <div className="mt-6 flex flex-wrap gap-2">
          {Object.keys(EXAMPLES).map((label) => (
            <Ripple key={label} className={`chip cursor-pointer ${TONE[label].soft} ${TONE[label].text} relative overflow-hidden px-4 py-1.5 text-sm`}
              onClick={() => pick(label)}>
              {TONE[label].word} example
            </Ripple>
          ))}
        </div>
        <label htmlFor="single" className="sr-only">Text to analyse</label>
        <textarea id="single" value={text} maxLength={10000} rows={6}
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
      <ResultCard result={result} busy={busy} />
    </div>
  );
}
