import { useEffect, useId, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Check, Flag, Info } from 'lucide-react';
import { LABELS } from '../lib/inference.js';
import { title } from '../lib/analysis.js';
import { addCorrection, findCorrection, removeCorrection, useCorrections } from '../lib/feedback.js';
import { TONE } from './ui.jsx';

// A small ⓘ that shows a note on hover, focus or tap, so caveats are one tap away instead of repeated.
export function InfoTip({ children, label = 'About this number' }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const ref = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const close = (e) => { if (!ref.current?.contains(e.target)) setOpen(false); };
    document.addEventListener('pointerdown', close);
    return () => document.removeEventListener('pointerdown', close);
  }, [open]);
  return (
    <span ref={ref} className="relative inline-flex align-middle" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button type="button" aria-label={label} aria-describedby={open ? id : undefined} onClick={() => setOpen((o) => !o)}
        onFocus={() => setOpen(true)} onBlur={() => setOpen(false)} onKeyDown={(e) => { if (e.key === 'Escape') setOpen(false); }}
        className="grid h-5 w-5 place-items-center rounded-full text-ink-faint hover:text-ink dark:hover:text-ube-50">
        <Info size={14} aria-hidden />
      </button>
      <AnimatePresence>
        {open && (
          <motion.span id={id} role="tooltip" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="glass absolute bottom-7 left-1/2 z-30 w-64 -translate-x-1/2 rounded-xl2 px-3 py-2 text-left text-xs font-normal leading-relaxed text-ink-soft dark:text-ube-100">
            {children}
          </motion.span>
        )}
      </AnimatePresence>
    </span>
  );
}

// "Wrong label?" lets the person record the label they think is right. Saved only in this browser.
export function WrongLabel({ text, predicted, scores, source, compact = false }) {
  const items = useCorrections();
  const saved = findCorrection(items, text, source);
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const close = (e) => { if (!ref.current?.contains(e.target)) setOpen(false); };
    document.addEventListener('pointerdown', close);
    return () => document.removeEventListener('pointerdown', close);
  }, [open]);
  if (saved) {
    return (
      <span className={`inline-flex items-center gap-1.5 ${compact ? 'text-xs' : 'text-sm'} text-ink-soft dark:text-ube-100`}>
        <Check size={compact ? 12 : 14} aria-hidden className="text-pandan-700 dark:text-pandan-300" />
        You said {title(saved.corrected)}
        <button className="underline underline-offset-2" onClick={() => removeCorrection(text, source)}>Undo</button>
      </span>
    );
  }
  return (
    <span ref={ref} className="relative inline-flex">
      <button type="button" aria-expanded={open} onClick={() => setOpen((o) => !o)}
        className={`inline-flex items-center gap-1 rounded-full ${compact ? 'px-1.5 py-0.5 text-xs' : 'px-3 py-1.5 text-sm'} text-ink-faint hover:text-ink dark:hover:text-ube-50`}>
        <Flag size={compact ? 11 : 14} aria-hidden /> Wrong label?
      </button>
      {open && (
        <span className="glass absolute right-0 top-full z-30 mt-1 flex w-max flex-col gap-1 rounded-xl2 p-2 text-sm" role="menu">
          <span className="px-2 pb-1 text-xs text-ink-faint">It should be</span>
          {LABELS.filter((l) => l !== predicted).map((l) => (
            <button key={l} role="menuitem" onClick={() => { addCorrection({ text, predicted, corrected: l, scores, source }); setOpen(false); }}
              className={`rounded-full px-3 py-1.5 text-left font-semibold ${TONE[l].soft} ${TONE[l].text}`}>
              {title(l)}
            </button>
          ))}
        </span>
      )}
    </span>
  );
}

// Words shaded by how much removing them lowers the predicted label's score.
export function WordHighlights({ impacts, label }) {
  const max = Math.max(...impacts.map((w) => Math.abs(w.delta)), 1e-6);
  const hex = TONE[label].hex;
  return (
    <p className="text-[17px] leading-[2.1]">
      {impacts.map((w, i) => {
        const strength = Math.min(Math.abs(w.delta) / max, 1);
        const supports = w.delta > 0;
        const against = !supports && strength >= 0.15;
        const alpha = Math.round((0.12 + strength * 0.7) * 255).toString(16).padStart(2, '0');
        return (
          <span key={`${w.word}-${i}`} title={`${supports ? 'Supports' : 'Pulls against'} ${title(label)}: ${(w.delta * 100).toFixed(1)} points`}
            className={`mx-0.5 rounded-md px-1 py-0.5 ${against ? 'underline decoration-dotted decoration-2 underline-offset-4' : ''}`}
            style={{ background: supports && strength >= 0.05 ? `${hex}${alpha}` : 'transparent' }}>
            {w.word}
          </span>
        );
      })}
    </p>
  );
}
