import { createContext, createElement, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { mockPredict } from './mock.js';

export const MOCK = import.meta.env.VITE_MOCK_MODEL === '1';
export const MODEL_MB = 356;

// Start the one-time download straight away only when it is unlikely to cost mobile data.
export function shouldAutoLoad(nav = typeof navigator !== 'undefined' ? navigator : {}, win = typeof window !== 'undefined' ? window : null) {
  const c = nav.connection;
  if (c?.saveData) return false;
  if (c?.type === 'cellular') return false;
  if (['slow-2g', '2g', '3g'].includes(c?.effectiveType)) return false;
  if (!c?.type && win?.matchMedia?.('(pointer: coarse)').matches && win.innerWidth < 900) return false;
  return true;
}
const ModelContext = createContext(null);

export function ModelProvider({ children }) {
  const worker = useRef(null);
  const pending = useRef(new Map());
  const nextId = useRef(1);
  const [state, setState] = useState({ status: MOCK ? 'ready' : 'idle', loaded: 0, total: 0, error: '', waiting: !MOCK && !shouldAutoLoad() });

  const ensureWorker = useCallback(() => {
    if (MOCK || worker.current) return worker.current;
    const w = new Worker(new URL('../model.worker.js', import.meta.url), { type: 'module' });
    w.onmessage = ({ data }) => {
      if (data.type === 'progress') setState((s) => ({ ...s, status: 'loading', loaded: data.loaded, total: data.total }));
      else if (data.type === 'ready') setState((s) => ({ ...s, status: 'ready', error: '' }));
      else if (data.type === 'batch') pending.current.get(data.id)?.onBatch?.(data.done, data.total);
      else if (data.type === 'result') {
        pending.current.get(data.id)?.resolve(data.results);
        pending.current.delete(data.id);
        setState((s) => ({ ...s, status: 'ready', error: '' }));
      } else if (data.type === 'error') {
        const job = pending.current.get(data.id);
        if (job) { job.reject(new Error(data.message)); pending.current.delete(data.id); }
        else setState((s) => ({ ...s, status: 'error', error: data.message }));
      }
    };
    worker.current = w;
    return w;
  }, []);

  const load = useCallback(() => {
    if (MOCK) return;
    setState((s) => (s.status === 'ready' ? s : { ...s, status: 'loading', error: '', waiting: false }));
    ensureWorker().postMessage({ type: 'load' });
  }, [ensureWorker]);

  // On desktop or Wi-Fi, start downloading shortly after the page settles; on mobile data, wait to be asked.
  useEffect(() => {
    if (MOCK || !shouldAutoLoad()) return undefined;
    const t = setTimeout(load, 600);
    return () => clearTimeout(t);
  }, [load]);

  const predict = useCallback((texts, onBatch) => {
    if (MOCK) return mockPredict(texts, onBatch);
    const id = nextId.current++;
    setState((s) => (s.status === 'idle' ? { ...s, status: 'loading', waiting: false } : s));
    return new Promise((resolve, reject) => {
      pending.current.set(id, { resolve, reject, onBatch });
      ensureWorker().postMessage({ type: 'predict', id, texts });
    });
  }, [ensureWorker]);

  // Evaluation builds only: lets scripts/browser_eval.py run the real worker on test inputs.
  useEffect(() => {
    if (import.meta.env.VITE_EXPOSE_PREDICT === '1') window.__predict = predict;
  }, [predict]);

  const value = useMemo(() => ({ ...state, predict, retry: load }), [state, predict, load]);
  return createElement(ModelContext.Provider, { value }, children);
}

export function useModel() {
  return useContext(ModelContext);
}
