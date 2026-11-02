import { useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import release from '../model_release.json';
import { SCORE_NOTE, pct } from '../lib/analysis.js';
import { Card, Tilt } from '../components/ui.jsx';

const REPO = 'https://github.com/SiobhanGoh/capstone2-sentiment-streamlit-colab';

// A card that flips on click or Enter to show a short explanation on its back.
function FactCard({ front, value, back }) {
  const reduce = useReducedMotion();
  const [open, setOpen] = useState(false);
  return (
    <Tilt max={6}>
      <button onClick={() => setOpen((o) => !o)} aria-pressed={open} className="flip-scene block h-44 w-full text-left">
        <motion.div className="flip-inner" animate={{ rotateY: open ? 180 : 0 }} transition={reduce ? { duration: 0 } : { type: 'spring', stiffness: 130, damping: 16 }}>
          <div className="flip-face glass flex flex-col justify-between rounded-xl3 p-5">
            <span className="text-sm text-ink-soft dark:text-ube-300">{front}</span>
            <span className="font-display text-4xl font-bold tabular-nums">{value}</span>
            <span className="text-xs text-ink-faint">Tap for what this means</span>
          </div>
          <div className="flip-face flip-back glass rounded-xl3 p-5 text-sm leading-relaxed">{back}</div>
        </motion.div>
      </button>
    </Tilt>
  );
}

export default function About() {
  const ev = release.evaluation;
  const verified = ev.browser_accuracy != null;
  const spec = release.browser_model;
  return (
    <div className="space-y-5">
      <Card>
        <h1 className="text-3xl font-bold sm:text-4xl">About the model</h1>
        <p className="mt-3 max-w-[68ch] text-ink-soft dark:text-ube-100">
          A multilingual BERT (bert-base-multilingual-cased) fine-tuned to label Malay, English and code-switched social text as negative, neutral or positive. It was built for a 2025–26 degree capstone and runs in your browser through ONNX, so typed and uploaded text is not sent anywhere.
        </p>
      </Card>
      <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <FactCard front="Test accuracy, original model" value={pct(ev.original_accuracy)}
          back={`Measured once on ${ev.test_inputs} held-out inputs after choosing settings on a separate validation split. Retrospective internal evidence, not new external validation.`} />
        <FactCard front="Macro-F1, original model" value={ev.original_macro_f1.toFixed(3)}
          back="The average F1 across the three classes, so the smaller neutral class counts as much as the others." />
        <FactCard front="Browser model on the same test" value={verified ? pct(ev.browser_accuracy) : 'Pending'}
          back={verified
            ? `The compressed copy that runs here scored ${pct(ev.browser_accuracy)} accuracy and agreed with the original model's label on ${pct(ev.label_agreement_with_original)} of the ${ev.test_inputs} test inputs.`
            : 'The compressed copy for browsers is checked on the same test inputs before it is published.'} />
        <FactCard front="Longest input read" value={`${release.max_length} tokens`}
          back="Longer text is cut to its first 128 model tokens, and the result says when that happened." />
      </div>
      <div className="grid gap-5 lg:grid-cols-2">
        <Card>
          <h2 className="text-xl font-semibold">Use it with care</h2>
          <ul className="mt-3 list-disc space-y-2 pl-5 text-ink-soft dark:text-ube-100">
            <li>The training comments differ from many YouTube topics, so errors are expected on new subjects.</li>
            <li>Sarcasm, slang and heavy code-switching are hard cases. Check the original wording before acting.</li>
            <li>{SCORE_NOTE}</li>
            <li>Labels describe polarity only. They do not measure customer satisfaction, business risk or anyone's mental health.</li>
          </ul>
        </Card>
        <Card>
          <h2 className="text-xl font-semibold">Release details</h2>
          <dl className="mt-3 space-y-2 break-all text-sm">
            <div><dt className="text-ink-faint">Browser model</dt><dd>{spec.repo_id} · {spec.dtype}</dd></div>
            <div><dt className="text-ink-faint">Pinned revision</dt><dd>{spec.revision || 'Not yet published'}</dd></div>
            <div><dt className="text-ink-faint">Converted from</dt><dd>{release.source_checkpoint.repo_id} @ {release.source_checkpoint.revision.slice(0, 12)}</dd></div>
            <div><dt className="text-ink-faint">Original weights SHA-256</dt><dd>{release.source_checkpoint.model_sha256}</dd></div>
          </dl>
          <a className="btn-soft mt-5" href={REPO} target="_blank" rel="noreferrer">Source and evaluation protocol</a>
        </Card>
      </div>
    </div>
  );
}
