import { useModel, MOCK, MODEL_MB } from '../lib/useModel.js';

function mb(bytes) {
  return `${(bytes / 1e6).toFixed(0)} MB`;
}

export default function ModelStatus() {
  const m = useModel();
  if (MOCK) {
    return <p className="chip bg-kaya-100 text-kaya-700">Placeholder scores for interface development</p>;
  }
  if (m.status === 'ready') {
    return (
      <p className="flex items-center gap-2 text-xs font-medium text-ink-soft dark:text-ube-300">
        <span className="h-2 w-2 rounded-full bg-pandan-500" aria-hidden /> Model ready on this device
      </p>
    );
  }
  if (m.status === 'error') {
    return (
      <p className="flex flex-wrap items-center gap-2 text-xs text-jambu-700 dark:text-jambu-300">
        {m.error}
        <button className="underline underline-offset-2" onClick={m.retry}>Try again</button>
      </p>
    );
  }
  if (m.status === 'idle' && m.waiting) {
    return (
      <p className="max-w-xs text-xs text-ink-soft dark:text-ube-300">
        The model ({MODEL_MB} MB) downloads once, when you first analyse. Wi-Fi is best.{' '}
        <button className="font-semibold underline underline-offset-2" onClick={m.retry}>Download now</button>
      </p>
    );
  }
  const share = m.total ? m.loaded / m.total : 0;
  return (
    <div className="w-full max-w-xs" role="status" aria-live="polite">
      <p className="mb-1.5 text-xs font-medium text-ink-soft dark:text-ube-300">
        {m.total ? `Downloading the model, ${mb(m.loaded)} of ${mb(m.total)}` : 'Preparing the model'}
        <span className="block font-normal text-ink-faint">One-time download. Later visits load from your browser cache.</span>
      </p>
      <div className="neu-inset h-2 overflow-hidden rounded-full">
        <div className="h-full rounded-full bg-ube-500 transition-[width] duration-300" style={{ width: `${Math.max(share * 100, 3)}%` }} />
      </div>
    </div>
  );
}
