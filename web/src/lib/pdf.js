// Comment report PDF built in the browser. Loaded only when someone exports.
import { jsPDF } from 'jspdf';
import regularUrl from '../assets/fonts/DejaVuSans.ttf?url';
import boldUrl from '../assets/fonts/DejaVuSans-Bold.ttf?url';
import glyphs from './pdf_glyphs.json';
import release from '../model_release.json';
import { LABELS } from './inference.js';
import { SAMPLE_NOTE, SCORE_NOTE, pct, summarise, title } from './analysis.js';

function supported(cp) {
  let lo = 0;
  let hi = glyphs.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (cp < glyphs[mid][0]) hi = mid - 1;
    else if (cp > glyphs[mid][1]) lo = mid + 1;
    else return true;
  }
  return false;
}

// Show characters the font cannot draw as visible codes instead of empty boxes.
export function printable(text) {
  let out = '';
  for (const ch of String(text)) {
    const cp = ch.codePointAt(0);
    out += ch === '\n' || ch === '\t' || supported(cp) ? ch : `[U+${cp.toString(16).toUpperCase().padStart(4, '0')}]`;
  }
  return out;
}

async function fontBase64(url) {
  const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

export async function buildReport(rows, meta) {
  const doc = new jsPDF({ unit: 'pt', format: 'a4' });
  doc.addFileToVFS('DejaVuSans.ttf', await fontBase64(regularUrl));
  doc.addFont('DejaVuSans.ttf', 'Sans', 'normal');
  doc.addFileToVFS('DejaVuSans-Bold.ttf', await fontBase64(boldUrl));
  doc.addFont('DejaVuSans-Bold.ttf', 'Sans', 'bold');
  doc.setProperties({ title: 'Sentiment analysis report', author: 'Siobhan Goh' });

  const margin = 42;
  const width = doc.internal.pageSize.getWidth() - margin * 2;
  const bottom = doc.internal.pageSize.getHeight() - margin;
  let y = margin;

  const write = (text, { size = 9, bold = false, gap = 6, color = [43, 37, 64] } = {}) => {
    doc.setFont('Sans', bold ? 'bold' : 'normal');
    doc.setFontSize(size);
    doc.setTextColor(...color);
    const lineHeight = size * 1.45;
    for (const line of doc.splitTextToSize(printable(text), width)) {
      if (y + lineHeight > bottom) { doc.addPage(); y = margin; }
      doc.text(line, margin, y + size);
      y += lineHeight;
    }
    y += gap;
  };

  const s = summarise(rows);
  const spec = release.browser_model;
  write('Sentiment analysis report', { size: 18, bold: true, gap: 10 });
  write(meta.title || 'Comment analysis', { size: 11, bold: true });
  write(`Source: ${meta.source}${meta.channel ? ` · ${meta.channel}` : ''}`);
  write(meta.sampling || 'User-supplied comments; not a representative sample.');
  write(`Analysed comments: ${s.total}. Excluded empty inputs: ${meta.excluded ?? 0}. Inputs truncated to 128 tokens: ${s.truncated}.`);
  write(SCORE_NOTE);
  if (rows.some((r) => printable(r.comment) !== String(r.comment))) {
    write('Characters this PDF font cannot draw are shown as Unicode codes in brackets. The CSV download keeps the original text.');
  }
  if (meta.source === 'YouTube') write(SAMPLE_NOTE);

  y += 4;
  const cols = [margin, margin + 250, margin + 360];
  doc.setFillColor(236, 231, 248);
  doc.rect(margin, y, 470, 22, 'F');
  doc.setFont('Sans', 'bold');
  doc.setFontSize(9);
  ['Predicted sentiment', 'Count', 'Share'].forEach((h, i) => doc.text(h, cols[i] + 8, y + 15));
  y += 22;
  doc.setFont('Sans', 'normal');
  LABELS.forEach((l) => {
    doc.setDrawColor(217, 217, 217);
    doc.rect(margin, y, 470, 22);
    [title(l), String(s.counts[l]), pct(s.shares[l])].forEach((v, i) => doc.text(v, cols[i] + 8, y + 15));
    y += 22;
  });
  y += 14;
  write(`Mean top-class model score: ${pct(s.meanScore)}. This is not accuracy or sentiment intensity.`);
  write('Review individual comments before acting. Sarcasm, code-switching and new topics may produce errors. This classifier does not assess business risk or mental health.');
  write('Comment details', { size: 12, bold: true, gap: 8 });
  rows.forEach((r) => {
    write(`${r.no}. ${title(r.sentiment)} | Model score ${pct(r.score)}${r.truncated ? ' | truncated' : ''}`, { bold: true, gap: 2 });
    write(r.comment, { size: 8.5, gap: 10, color: [74, 68, 96] });
  });
  write('Model release', { size: 12, bold: true, gap: 8 });
  write(`${spec.repo_id} | Revision ${spec.revision || 'unreleased'} | ${spec.dtype}`, { size: 7 });
  write(`Converted from ${release.source_checkpoint.repo_id} (weights SHA-256 ${release.source_checkpoint.model_sha256}).`, { size: 7 });
  write('Degree capstone 2025–2026. Corrected retrospective internal evaluation; external generalisation and calibration are not established.', { size: 7 });

  const pages = doc.getNumberOfPages();
  for (let p = 1; p <= pages; p += 1) {
    doc.setPage(p);
    doc.setFont('Sans', 'normal');
    doc.setFontSize(8);
    doc.setTextColor(120, 112, 140);
    doc.text(`Page ${p} of ${pages}`, doc.internal.pageSize.getWidth() - margin, doc.internal.pageSize.getHeight() - 22, { align: 'right' });
  }
  return doc.output('blob');
}
