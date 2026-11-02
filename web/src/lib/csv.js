import Papa from 'papaparse';
import release from '../model_release.json';
import { LABELS } from './inference.js';
import { title } from './analysis.js';

const FORMULA = /^[=+\-@\t\r]/;

// Stop spreadsheet apps from treating user text as a formula.
export function safeCell(value) {
  return typeof value === 'string' && FORMULA.test(value.trimStart()) ? `'${value}` : value;
}

export function toCsv(records) {
  const safe = records.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, safeCell(v)])));
  return `﻿${Papa.unparse(safe)}`;
}

export function rowsToRecords(rows, meta) {
  const spec = release.browser_model;
  return rows.map((r) => ({
    'No.': r.no,
    Comment: r.comment,
    Sentiment: r.sentiment,
    'Model score': r.score,
    ...Object.fromEntries(LABELS.map((l, i) => [`${title(l)} score`, r.scores[i]])),
    Truncated: r.truncated,
    ...(Number.isFinite(r.likes) ? { Likes: r.likes } : {}),
    Source: meta.source,
    Title: meta.title,
    Model: spec.repo_id,
    Revision: spec.revision,
    Precision: spec.dtype,
  }));
}

export function download(name, content, type) {
  const blob = content instanceof Blob ? content : new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Parse an uploaded CSV and return its header names and rows.
export function parseCsvFile(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: 'greedy',
      complete: ({ data, meta, errors }) => {
        if (!meta.fields?.length) reject(new Error('That file has no header row. Add column names in the first row.'));
        else resolve({ fields: meta.fields, data, errors: errors.length });
      },
      error: () => reject(new Error('That file could not be read as CSV.')),
    });
  });
}

// Guess the text column: a known name first, then the column with the longest average text.
export function guessTextColumn(fields, data) {
  const known = ['comment', 'comments', 'text', 'tweet', 'review', 'message', 'content'];
  const hit = fields.find((f) => known.includes(f.trim().toLowerCase()));
  if (hit) return hit;
  let best = fields[0];
  let bestLen = -1;
  for (const f of fields) {
    const sample = data.slice(0, 50);
    const len = sample.reduce((a, r) => a + String(r[f] ?? '').length, 0) / Math.max(sample.length, 1);
    if (len > bestLen) { best = f; bestLen = len; }
  }
  return best;
}
