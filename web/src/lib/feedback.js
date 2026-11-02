// Labels the person says are wrong, kept only in this browser. Exported as CSV, they can
// become a hand-checked test set for the next model.
import { useEffect, useState } from 'react';
import { toCsv } from './csv.js';
import { title } from './analysis.js';

const KEY = 'rojak-corrections-v1';
const EVENT = 'rojak-corrections-changed';
const LIMIT = 2000;

export function loadCorrections() {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) ?? '[]');
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}

function save(items) {
  try {
    localStorage.setItem(KEY, JSON.stringify(items.slice(0, LIMIT)));
  } catch { /* storage unavailable: corrections last for this visit only */ }
  window.dispatchEvent(new CustomEvent(EVENT, { detail: items }));
}

let memory = null;
const current = () => memory ?? loadCorrections();

const same = (c, text, source) => c.text === text && c.source === (source ?? 'Analyse');

export function addCorrection({ text, predicted, corrected, scores, source }) {
  const src = source ?? 'Analyse';
  memory = [{ time: new Date().toISOString(), text, predicted, corrected, scores, source: src },
    ...current().filter((c) => !same(c, text, src))];
  save(memory);
}

export function removeCorrection(text, source) {
  memory = current().filter((c) => !same(c, text, source));
  save(memory);
}

export function findCorrection(items, text, source) {
  return items.find((c) => same(c, text, source));
}

export function clearCorrections() {
  memory = [];
  save(memory);
}

export function correctionsCsv(items) {
  return toCsv(items.map((c) => ({
    'Time (UTC)': c.time, Text: c.text, 'Model label': c.predicted, 'Your label': c.corrected,
    'Negative score': c.scores?.[0], 'Neutral score': c.scores?.[1], 'Positive score': c.scores?.[2], Source: c.source,
  })));
}

export function useCorrections() {
  const [items, setItems] = useState(current);
  useEffect(() => {
    const on = (e) => setItems(e.detail);
    window.addEventListener(EVENT, on);
    return () => window.removeEventListener(EVENT, on);
  }, []);
  return items;
}

export const correctionNote = (c) => `${title(c.predicted)} → ${title(c.corrected)}`;
