"""Readable PDF exports with bounded paragraphs and automatic pagination."""
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape
import unicodedata

import matplotlib
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from sentiment_core import LABELS, SCORE_NOTE, SAMPLE_NOTE, sentiment_summary


def printable_text(text, supported):
    """Preserve unsupported code points visibly instead of drawing missing-glyph boxes."""
    # ReportLab's font mapping is BMP-oriented; escape supplementary characters
    # even when the source font has a glyph so PDF text extraction stays valid.
    return ''.join(character if (ord(character) <= 0xFFFF and ord(character) in supported) or character in '\n\t' else
                   '[' + unicodedata.name(character, 'CHARACTER') + f' U+{ord(character):04X}]'
                   for character in str(text))


def build_report(frame, meta, release):
    font_dir = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, filename in [("SentimentSans", "DejaVuSans.ttf"), ("SentimentSansBold", "DejaVuSans-Bold.ttf")]:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "SentimentSans"
    styles["Title"].fontName = "SentimentSansBold"
    styles["Title"].fontSize = 18
    styles["Heading2"].fontName = "SentimentSansBold"
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 13
    styles["BodyText"].spaceAfter = 8
    styles.add(ParagraphStyle("Comment", fontName="SentimentSans", fontSize=8.5,
                              leading=12, spaceAfter=10, splitLongWords=True))
    styles.add(ParagraphStyle("Foot", fontName="SentimentSans", fontSize=7, leading=10))
    summary = sentiment_summary(frame)
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=42, leftMargin=42,
                            topMargin=42, bottomMargin=42, title="Sentiment analysis report",
                            author="Siobhan Goh")
    story = [Paragraph("Sentiment analysis report", styles["Title"])]
    supported = pdfmetrics.getFont('SentimentSans').face.charToGlyph
    def paragraph(text, style="BodyText"):
        return Paragraph(escape(printable_text(text, supported)).replace("\n", "<br/>"), styles[style])
    story += [paragraph(meta.get("title", "Comment analysis")),
              paragraph("Source: " + meta.get("source", "Pasted comments")),
              paragraph(meta.get("sampling", "User-supplied comments; not a representative sample.")),
              paragraph(f"Analysed comments: {summary['total']}. Excluded empty inputs: {meta.get('excluded', 0)}. "
                        f"Inputs truncated to 128 tokens: {summary['truncated']}."),
              paragraph(SCORE_NOTE)]
    if any(printable_text(value, supported) != str(value) for value in frame.Comment):
        story.append(paragraph("Characters unavailable in this PDF font are shown by their Unicode names "
                               "and codes in brackets. The CSV download preserves the original text."))
    if meta.get("source") == "YouTube":
        story.append(paragraph(SAMPLE_NOTE))
    table = [["Predicted sentiment", "Count", "Share"]]
    table.extend([[label.title(), str(summary["counts"][label]),
                   f"{summary['shares'][label]:.1%}"] for label in LABELS])
    table = Table(table, colWidths=[250, 110, 110], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "SentimentSans"), ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e6f5")),
        ("GRID", (0, 0), (-1, -1), .5, colors.HexColor("#d9d9d9")),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [Spacer(1, 10), table, Spacer(1, 12),
              paragraph(f"Mean top-class model score: {summary['mean_score']:.1%}. "
                        "This is not accuracy or sentiment intensity."),
              paragraph("Review individual comments before acting. Sarcasm, code switching and new topics "
                        "may produce errors. This classifier does not assess business risk or mental health."),
              paragraph("Comment details", "Heading2")]
    for _, row in frame.iterrows():
        story.append(paragraph(f"{row['No.']}. {row['Sentiment'].title()} | Model score {row['Model score']:.1%}",
                               "BodyText"))
        # Separate paragraphs make even very long comments flow cleanly across pages.
        text = str(row["Comment"])
        for start in range(0, len(text), 1400):
            story.append(paragraph(text[start:start + 1400], "Comment"))
    story += [Spacer(1, 12), paragraph("Model release", "Heading2"),
              paragraph(f"{release['repo_id']} | Revision: {release['revision']}", "Foot"),
              paragraph("Weights SHA-256: " + release["model_sha256"], "Foot"),
              paragraph("Degree capstone 2025–2026. Corrected retrospective internal evaluation; "
                        "external generalisation and calibration are not established.", "Foot")]
    def footer(canvas, document):
        canvas.setFont("SentimentSans", 8)
        canvas.drawRightString(A4[0] - 42, 25, f"Page {document.page}")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
