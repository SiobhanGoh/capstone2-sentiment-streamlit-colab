import { useEffect, useRef, useState } from 'react';
import { animate, useReducedMotion } from 'framer-motion';

// Counts from the previous value to the new one in 240 ms; jumps straight there with reduced motion.
export function Count({ value, suffix = '', decimals = 0 }) {
  const reduce = useReducedMotion();
  const [shown, setShown] = useState(value);
  const from = useRef(value);
  useEffect(() => {
    if (reduce) { setShown(value); from.current = value; return undefined; }
    const c = animate(from.current, value, { duration: 0.24, ease: 'easeOut', onUpdate: (v) => { setShown(v); from.current = v; } });
    return () => c.stop();
  }, [value, reduce]);
  return <>{shown.toFixed(decimals)}{suffix}</>;
}
