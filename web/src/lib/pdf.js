// Comment report PDF in the app's pastel theme, built in the browser.
// Loaded only when someone exports.
import { jsPDF } from 'jspdf';
import regularUrl from '../assets/fonts/DejaVuSans.ttf?url';
import boldUrl from '../assets/fonts/DejaVuSans-Bold.ttf?url';
import displayUrl from '../assets/fonts/BricolageGrotesque-Bold.ttf?url';
import glyphs from './pdf_glyphs.json';
import release from '../model_release.json';
import { LABELS } from './inference.js';
import { SCORE_NOTE, pct, summarise, title } from './analysis.js';
import { NOTES_DISCLAIMER, creatorNotes } from './insights.js';
import { isNearSplit } from './pipeline.js';

const C = {
  ink: [43, 37, 64], soft: [90, 83, 115], faint: [138, 132, 160], mist: [244, 241, 251], white: [255, 255, 255],
  ube: [117, 98, 180], ube100: [233, 227, 250], ube300: [191, 176, 236],
};
const TONE = {
  NEGATIVE: { soft: [252, 228, 234], bar: [244, 175, 192], deep: [169, 53, 86] },
  NEUTRAL: { soft: [251, 243, 214], bar: [242, 222, 151], deep: [138, 106, 18] },
  POSITIVE: { soft: [221, 243, 229], bar: [168, 221, 189], deep: [47, 122, 84] },
};
const DOTS = [TONE.POSITIVE.bar, TONE.NEUTRAL.bar, TONE.NEGATIVE.bar];
const W = 595.28;
const H = 841.89;
const M = 40;
const CW = W - M * 2;
const BOTTOM = H - 52;

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

const EMOJI = /\p{Extended_Pictographic}/u;
const JOINERS = new Set([0x200d, 0xfe0f, 0xfe0e]);

// Characters the PDF font can't draw are shown as [emoji] or a Unicode code, never as empty boxes.
export function printable(text) {
  let out = '';
  let lastEmoji = false;
  for (const ch of String(text)) {
    const cp = ch.codePointAt(0);
    if (JOINERS.has(cp) || (cp >= 0x1f3fb && cp <= 0x1f3ff)) continue;
    if (ch === '\n' || ch === '\t' || supported(cp)) { out += ch; lastEmoji = false; continue; }
    if (EMOJI.test(ch)) {
      if (!lastEmoji) out += '[emoji]';
      lastEmoji = true;
      continue;
    }
    out += `[U+${cp.toString(16).toUpperCase().padStart(4, '0')}]`;
    lastEmoji = false;
  }
  return out;
}

async function fontBase64(url) {
  const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

async function thumbnail(meta) {
  if (!meta.video_id) return null;
  try {
    const res = await fetch(`/api/thumb?id=${encodeURIComponent(meta.video_id)}`);
    if (!res.ok) return null;
    const blob = await res.blob();
    return await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => resolve(null);
      reader.readAsDataURL(blob);
    });
  } catch {
    return null;
  }
}

const compact = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 });
const likesText = (n) => `${n.toLocaleString()} ${n === 1 ? 'like' : 'likes'}`;
const clip = (s, n) => (s.length > n ? `${s.slice(0, n - 1).trimEnd()}…` : s);

export async function buildReport(rows, meta) {
  const doc = new jsPDF({ unit: 'pt', format: 'a4' });
  const [reg, bold, disp, thumb] = await Promise.all([fontBase64(regularUrl), fontBase64(boldUrl), fontBase64(displayUrl), thumbnail(meta)]);
  doc.addFileToVFS('DejaVuSans.ttf', reg);
  doc.addFont('DejaVuSans.ttf', 'Sans', 'normal');
  doc.addFileToVFS('DejaVuSans-Bold.ttf', bold);
  doc.addFont('DejaVuSans-Bold.ttf', 'Sans', 'bold');
  doc.addFileToVFS('Bricolage-Bold.ttf', disp);
  doc.addFont('Bricolage-Bold.ttf', 'Display', 'bold');
  doc.setProperties({ title: `Sentiment report: ${meta.title || 'comments'}`, author: 'Rojak Sentiment' });

  const s = summarise(rows);
  const n = creatorNotes(rows);
  const isYouTube = meta.source === 'YouTube';
  let y = M;

  const fill = (c) => doc.setFillColor(...c);
  const alpha = (a) => doc.setGState(new doc.GState({ opacity: a }));
  const font = (f, size, color = C.ink) => {
    doc.setFont(f === 'display' ? 'Display' : 'Sans', f === 'regular' ? 'normal' : 'bold');
    doc.setFontSize(size);
    doc.setTextColor(...color);
  };
  const lines = (text, width, f, size) => { font(f, size); return doc.splitTextToSize(printable(text), width); };

  const background = () => {
    fill(C.mist);
    doc.rect(0, 0, W, H, 'F');
    alpha(0.45);
    fill(C.ube300); doc.circle(-30, -10, 170, 'F');
    fill(TONE.POSITIVE.bar); doc.circle(W + 40, 330, 150, 'F');
    fill(TONE.NEGATIVE.bar); doc.circle(W * 0.38, H + 60, 170, 'F');
    alpha(1);
  };
  const card = (x, top, w, h, color = C.white, opacity = 0.82) => {
    alpha(opacity); fill(color); doc.roundedRect(x, top, w, h, 14, 14, 'F'); alpha(1);
  };
  const newPage = () => { doc.addPage(); background(); y = M; };
  const ensure = (h) => { if (y + h > BOTTOM) newPage(); };
  const dots = (x, cy, r) => DOTS.forEach((c, i) => { fill(c); doc.circle(x + r + i * r * 1.55, cy, r, 'F'); });
  const chip = (x, top, text, bg, color) => {
    font('bold', 7.5, color);
    const w = doc.getTextWidth(text) + 14;
    fill(bg); doc.roundedRect(x, top, w, 15, 7.5, 7.5, 'F');
    doc.text(text, x + 7, top + 10.3);
    return w;
  };

  // ---------- Cover ----------
  background();
  dots(M, y + 6, 6);
  font('display', 13); doc.text('Rojak Sentiment', M + 34, y + 10.5);
  font('regular', 8.5, C.faint);
  doc.text(`Generated ${new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' })}`, W - M, y + 10, { align: 'right' });
  y += 44;
  font('display', 30); doc.text('Sentiment report', M, y + 24);
  y += 40;
  font('regular', 10.5, C.soft);
  const intro = isYouTube ? 'What the comments on this video say, read by a Malay–English sentiment model.' : 'What these comments say, read by a Malay–English sentiment model.';
  doc.text(intro, M, y + 10);
  y += 28;

  // Video or source card
  if (isYouTube) {
    const th = thumb ? 101 : 0;
    const textX = M + 16 + (thumb ? 180 + 16 : 0);
    const textW = CW - (textX - M) - 16;
    const tl = lines(meta.title || 'Untitled video', textW, 'bold', 12.5).slice(0, 3);
    const h = Math.max(th + 32, 24 + tl.length * 16 + 64);
    card(M, y, CW, h);
    if (thumb) {
      fill(C.white); doc.roundedRect(M + 14, y + 14, 184, th + 4, 8, 8, 'F');
      doc.addImage(thumb, 'JPEG', M + 16, y + 16, 180, th);
    }
    let ty = y + 26;
    font('bold', 12.5); tl.forEach((l) => { doc.text(l, textX, ty); ty += 16; });
    font('regular', 9, C.soft);
    doc.text(printable(`${meta.channel || ''}${meta.published_at ? `, published ${meta.published_at.slice(0, 10)}` : ''}`), textX, ty + 2);
    ty += 14;
    let cx = textX;
    [meta.views != null && `${compact.format(meta.views)} views`, meta.likes != null && `${compact.format(meta.likes)} likes`,
      meta.comment_count != null && `${compact.format(meta.comment_count)} comments`].filter(Boolean)
      .forEach((t) => { cx += chip(cx, ty, t, C.ube100, C.ube) + 6; });
    ty += 28;
    font('bold', 8.5, C.ube);
    if (meta.url) doc.textWithLink('Open on YouTube', textX, ty, { url: meta.url });
    y += h + 14;
  } else {
    const tl = lines(meta.title || 'Comment analysis', CW - 32, 'bold', 12.5).slice(0, 2);
    const h = 30 + tl.length * 16 + 14;
    card(M, y, CW, h);
    let ty = y + 26;
    font('bold', 12.5); tl.forEach((l) => { doc.text(l, M + 16, ty); ty += 16; });
    font('regular', 9, C.soft); doc.text(printable(meta.source), M + 16, ty + 2);
    y += h + 14;
  }

  // KPI tiles
  const gap = 10;
  const tileH = 78;
  const wide = CW * 0.31;
  const narrow = (CW - wide - gap * 3) / 3;
  card(M, y, wide, tileH);
  const unsure = rows.filter((r) => isNearSplit(r)).length;
  font('regular', 8.5, C.soft); doc.text('Worth checking by hand', M + 14, y + 20);
  font('display', 22); doc.text(String(unsure), M + 14, y + 46);
  const uw = doc.getTextWidth(String(unsure));
  font('regular', 10, C.soft); doc.text(`of ${s.total}`, M + 20 + uw, y + 46);
  font('regular', 7.5, C.faint); doc.text('Top two scores within 20 points', M + 14, y + 64);
  LABELS.forEach((l, i) => {
    const x = M + wide + gap + i * (narrow + gap);
    card(x, y, narrow, tileH, TONE[l].soft, 1);
    font('bold', 8.5, TONE[l].deep); doc.text(title(l), x + 14, y + 20);
    font('display', 24); doc.text(pct(s.shares[l], 0), x + 14, y + 49);
    font('regular', 8, C.soft); doc.text(`${s.counts[l]} ${s.counts[l] === 1 ? 'comment' : 'comments'}`, x + 14, y + 65);
  });
  y += tileH + 12;

  // Share bar
  const boxH = 86;
  card(M, y, CW, boxH);
  font('bold', 10); doc.text('Share of comments', M + 14, y + 22);
  let bx = M + 14;
  const bw = CW - 28;
  LABELS.forEach((l, i) => {
    const w = bw * s.shares[l];
    if (w <= 0) return;
    fill(TONE[l].bar);
    const first = bx === M + 14;
    const last = LABELS.slice(i + 1).every((k) => s.shares[k] === 0);
    doc.roundedRect(bx, y + 32, w, 14, first || last ? 7 : 0, first || last ? 7 : 0, 'F');
    if (first && !last) doc.rect(bx + Math.min(7, w / 2), y + 32, w - Math.min(7, w / 2), 14, 'F');
    if (last && !first) doc.rect(bx, y + 32, Math.max(w - 7, 0), 14, 'F');
    if (!first && !last) doc.rect(bx, y + 32, w, 14, 'F');
    bx += w;
  });
  let lx = M + 14;
  LABELS.forEach((l) => {
    fill(TONE[l].bar); doc.circle(lx + 4, y + 61, 4, 'F');
    font('regular', 8.5, C.soft); doc.text(`${title(l)} ${pct(s.shares[l], 0)}`, lx + 12, y + 64);
    lx += doc.getTextWidth(`${title(l)} ${pct(s.shares[l], 0)}`) + 26;
  });
  font('regular', 7, C.faint);
  doc.text(isYouTube ? 'Top-level comments ordered by relevance; not a representative audience sample.' : 'A summary of the supplied comments only.', M + 14, y + 78);
  y += boxH + 14;

  // Creator notes
  const noteLines = n.notes.map((t) => lines(t, CW - 46, 'regular', 9));
  const colW = (CW - 28 - 16) / 3;
  const groups = [
    { name: 'Read these first', list: n.readFirst, tone: TONE.NEGATIVE.soft, color: TONE.NEGATIVE.deep, empty: 'No negative comments.' },
    { name: 'Viewer questions', list: n.questions.slice(0, 3), tone: C.ube100, color: C.ube, empty: 'No questions spotted.' },
    { name: 'What viewers enjoyed', list: n.highlights, tone: TONE.POSITIVE.soft, color: TONE.POSITIVE.deep, empty: 'No positive comments.' },
  ].map((g) => ({ ...g, quotes: g.list.map((r) => { const all = lines(clip(r.comment, 220), colW - 18, 'regular', 8); const l = all.slice(0, 5); if (all.length > 5) l[4] = `${l[4].replace(/\s*\S*$/, '')}…`; return { r, l }; }) }));
  const colH = (g) => 18 + (g.quotes.length ? g.quotes.reduce((a, q) => a + q.l.length * 10.5 + 26, 0) : 16);
  const notesH = 56 + noteLines.reduce((a, l) => a + l.length * 12.5 + 5, 0);
  ensure(notesH);
  card(M, y, CW, notesH);
  fill(TONE.NEUTRAL.bar); doc.circle(M + 21, y + 22, 6, 'F');
  font('display', 15); doc.text(isYouTube ? 'Notes for the creator' : 'What to look at', M + 34, y + 27);
  let ny = y + 48;
  noteLines.forEach((l) => {
    fill(C.ube300); doc.circle(M + 21, ny - 3, 2.2, 'F');
    font('regular', 9, C.soft); l.forEach((line) => { doc.text(line, M + 30, ny); ny += 12.5; });
    ny += 5;
  });
  y += notesH + 12;

  const quotesH = 30 + Math.max(...groups.map(colH)) + 26;
  ensure(quotesH);
  card(M, y, CW, quotesH);
  groups.forEach((g, i) => {
    const gx = M + 14 + i * (colW + 8);
    let gy = y + 24;
    font('bold', 8.5, g.color); doc.text(g.name, gx, gy); gy += 10;
    if (!g.quotes.length) { font('regular', 8, C.faint); doc.text(g.empty, gx, gy + 8); return; }
    g.quotes.forEach(({ r, l }) => {
      const qh = l.length * 10.5 + 20;
      fill(g.tone); doc.roundedRect(gx, gy, colW, qh, 8, 8, 'F');
      font('regular', 8, C.ink); l.forEach((line, k) => doc.text(line, gx + 9, gy + 13 + k * 10.5));
      font('regular', 7, C.soft);
      doc.text(`#${r.no}${Number.isFinite(r.likes) ? `   ${likesText(r.likes)}` : ''}`, gx + 9, gy + qh - 6);
      gy += qh + 6;
    });
  });
  font('regular', 6.8, C.faint);
  doc.text(doc.splitTextToSize(NOTES_DISCLAIMER, CW - 28), M + 14, y + quotesH - 16);
  y += quotesH + 16;

  // Comments by label
  for (const label of LABELS) {
    const group = rows.filter((r) => r.sentiment === label);
    const t = TONE[label];
    const firstH = group.length ? Math.min(34 + lines(group[0].comment, CW - 40, 'regular', 9).length * 12.5, 160) : 24;
    ensure(50 + firstH + 8);
    fill(t.soft); doc.roundedRect(M, y, CW, 40, 12, 12, 'F');
    font('display', 16, t.deep); doc.text(`${title(label)} comments`, M + 16, y + 26);
    font('display', 16, t.deep); doc.text(pct(s.shares[label], 0), W - M - 16, y + 26, { align: 'right' });
    font('regular', 8.5, C.soft); doc.text(`${group.length} of ${s.total}`, W - M - 60, y + 25, { align: 'right' });
    y += 50;
    if (!group.length) { font('regular', 9, C.faint); doc.text('None in this set.', M + 16, y + 6); y += 24; continue; }
    for (const r of group) {
      const body = lines(r.comment, CW - 44, 'regular', 9);
      const ch = 34 + body.length * 12.5;
      if (ch > BOTTOM - M) {
        // Very long comments flow across pages without a card.
        ensure(40);
        font('bold', 8, t.deep); doc.text(`#${r.no}  ${pct(r.score)}`, M + 16, y + 10); y += 22;
        font('regular', 9, C.ink);
        for (const line of body) { ensure(13); doc.text(line, M + 16, y); y += 12.5; }
        y += 10;
        continue;
      }
      ensure(ch + 8);
      card(M, y, CW, ch, C.white, 0.85);
      fill(t.bar); doc.roundedRect(M + 7, y + 9, 4, ch - 18, 2, 2, 'F');
      let cx = M + 20;
      font('bold', 8, C.faint); doc.text(`#${r.no}`, cx, y + 17); cx += doc.getTextWidth(`#${r.no}`) + 8;
      cx += chip(cx, y + 7, `${pct(r.score)} score`, t.soft, t.deep) + 5;
      if (Number.isFinite(r.likes)) cx += chip(cx, y + 7, likesText(r.likes), C.ube100, C.ube) + 5;
      if (r.truncated) chip(cx, y + 7, 'truncated', TONE.NEUTRAL.soft, TONE.NEUTRAL.deep);
      font('regular', 9, C.ink);
      body.forEach((line, k) => doc.text(line, M + 20, y + 38 + k * 12.5));
      y += ch + 8;
    }
    y += 10;
  }

  // About this report
  const spec = release.browser_model;
  const about = [
    SCORE_NOTE,
    `Analysed comments: ${s.total}. Excluded empty inputs: ${meta.excluded ?? 0}. Inputs truncated to 128 tokens: ${s.truncated}.`,
    meta.sampling || 'User-supplied comments; not a representative sample.',
    'Review individual comments before acting. Sarcasm, slang, code-switching and new topics may produce errors. This classifier does not assess business risk or mental health.',
    rows.some((r) => printable(r.comment) !== String(r.comment)) && 'Emoji and characters this PDF font cannot draw appear as [emoji] or a Unicode code. The CSV download keeps the original text.',
    `Model: ${spec.repo_id} (${spec.dtype}), revision ${spec.revision || 'unreleased'}, converted from ${release.source_checkpoint.repo_id}. Internal test accuracy ${pct(release.evaluation.scoring_accuracy ?? release.evaluation.original_accuracy, 2)} on ${release.evaluation.test_inputs} inputs${release.scoring?.method === 'blend' ? ' (model blended 3:1 with a TF-IDF baseline, chosen on validation)' : ''}; retrospective, not external validation.`,
  ].filter(Boolean).map((t) => lines(t, CW - 32, 'regular', 8));
  const aboutH = 40 + about.reduce((a, l) => a + l.length * 11 + 5, 0);
  ensure(aboutH);
  card(M, y, CW, aboutH, C.ube100, 0.7);
  font('display', 12); doc.text('About this report', M + 16, y + 24);
  let ay = y + 42;
  about.forEach((l) => { font('regular', 8, C.soft); l.forEach((line) => { doc.text(line, M + 16, ay); ay += 11; }); ay += 5; });

  // Footer on every page
  const pages = doc.getNumberOfPages();
  for (let p = 1; p <= pages; p += 1) {
    doc.setPage(p);
    dots(M, H - 26, 3);
    font('bold', 7.5, C.faint); doc.text('Rojak Sentiment', M + 18, H - 23.5);
    font('regular', 7.5, C.faint); doc.text(`Page ${p} of ${pages}`, W - M, H - 23.5, { align: 'right' });
  }
  return doc.output('blob');
}
