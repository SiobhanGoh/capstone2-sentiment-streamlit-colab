import { useRef, useState } from 'react';
import { motion, useMotionValue, useReducedMotion, useSpring, useTransform } from 'framer-motion';
import { LABELS } from '../lib/inference.js';
import { pct, title } from '../lib/analysis.js';

export const TONE = {
  POSITIVE: { bar: 'bg-pandan-300', text: 'text-pandan-700 dark:text-pandan-300', soft: 'bg-pandan-100 dark:bg-pandan-700/25', word: 'Positive', hex: '#5DB585' },
  NEUTRAL: { bar: 'bg-kaya-300', text: 'text-kaya-700 dark:text-kaya-300', soft: 'bg-kaya-100 dark:bg-kaya-700/25', word: 'Neutral', hex: '#D9B44A' },
  NEGATIVE: { bar: 'bg-jambu-300', text: 'text-jambu-700 dark:text-jambu-300', soft: 'bg-jambu-100 dark:bg-jambu-700/25', word: 'Negative', hex: '#E07892' },
};

export function Ripple({ as: Tag = 'button', className = '', children, onPointerDown, ...props }) {
  const [waves, setWaves] = useState([]);
  const add = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const size = Math.max(r.width, r.height);
    const wave = { id: performance.now(), x: e.clientX - r.left - size / 2, y: e.clientY - r.top - size / 2, size };
    setWaves((w) => [...w, wave]);
    setTimeout(() => setWaves((w) => w.filter((v) => v.id !== wave.id)), 650);
    onPointerDown?.(e);
  };
  return (
    <Tag className={className} onPointerDown={add} {...props}>
      {children}
      {waves.map((w) => (
        <span key={w.id} className="ripple" style={{ left: w.x, top: w.y, width: w.size, height: w.size }} />
      ))}
    </Tag>
  );
}

// Gentle 3D tilt that follows the pointer. Off for touch and reduced motion.
export function Tilt({ children, className = '', max = 7 }) {
  const reduce = useReducedMotion();
  const ref = useRef(null);
  const px = useMotionValue(0.5);
  const py = useMotionValue(0.5);
  const rx = useSpring(useTransform(py, [0, 1], [max, -max]), { stiffness: 180, damping: 18 });
  const ry = useSpring(useTransform(px, [0, 1], [-max, max]), { stiffness: 180, damping: 18 });
  const move = (e) => {
    if (reduce || e.pointerType !== 'mouse') return;
    const r = ref.current.getBoundingClientRect();
    px.set((e.clientX - r.left) / r.width);
    py.set((e.clientY - r.top) / r.height);
  };
  const leave = () => { px.set(0.5); py.set(0.5); };
  return (
    <motion.div ref={ref} onPointerMove={move} onPointerLeave={leave} className={className}
      style={reduce ? undefined : { rotateX: rx, rotateY: ry, transformPerspective: 1100 }}>
      {children}
    </motion.div>
  );
}

export function Card({ className = '', children, ...props }) {
  return <section className={`glass rounded-xl3 p-5 sm:p-6 ${className}`} {...props}>{children}</section>;
}

export function ScoreBars({ scores, compact = false }) {
  return (
    <ul className={compact ? 'space-y-2' : 'space-y-3.5'}>
      {LABELS.map((label, i) => (
        <li key={label}>
          <div className="mb-1.5 flex justify-between text-sm">
            <span className="font-medium">{title(label)}</span>
            <span className="tabular-nums text-ink-soft dark:text-ube-300">{pct(scores[i])}</span>
          </div>
          <div className={`neu-inset overflow-hidden rounded-full ${compact ? 'h-2.5' : 'h-3.5'}`}>
            <motion.div className={`h-full rounded-full ${TONE[label].bar}`}
              initial={{ width: 0 }} animate={{ width: `${Math.max(scores[i] * 100, 1.5)}%` }}
              transition={{ type: 'spring', stiffness: 90, damping: 18 }} />
          </div>
        </li>
      ))}
    </ul>
  );
}

export function Badge({ label }) {
  return <span className={`chip ${TONE[label].soft} ${TONE[label].text}`}>{title(label)}</span>;
}

export function Notice({ tone = 'info', children }) {
  const styles = {
    info: 'bg-ube-100/70 text-ube-700 dark:bg-ube-700/25 dark:text-ube-100',
    warn: 'bg-kaya-100 text-kaya-700 dark:bg-kaya-700/25 dark:text-kaya-100',
    error: 'bg-jambu-100 text-jambu-700 dark:bg-jambu-700/25 dark:text-jambu-100',
  };
  return <p role={tone === 'error' ? 'alert' : 'status'} className={`rounded-xl2 px-4 py-3 text-sm ${styles[tone]}`}>{children}</p>;
}
