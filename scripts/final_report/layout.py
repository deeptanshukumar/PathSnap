"""Typography, page template and small flowable helpers for the final report."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

INK = colors.HexColor("#163d35")
TEAL = colors.HexColor("#087f6a")
GOLD = colors.HexColor("#b9842b")
PALE = colors.HexColor("#eef3f1")
CREAM = colors.HexColor("#f6f1e6")
RULE = colors.HexColor("#c9d3ce")
RED = colors.HexColor("#b4472f")

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm
TEXT_W = PAGE_W - 2 * MARGIN


def register_fonts() -> None:
    lib = Path("/usr/share/fonts/truetype/liberation")
    dv = Path("/usr/share/fonts/truetype/dejavu")
    faces = {
        "Body": lib / "LiberationSerif-Regular.ttf", "Body-Bold": lib / "LiberationSerif-Bold.ttf",
        "Body-Italic": lib / "LiberationSerif-Italic.ttf", "Body-BoldItalic": lib / "LiberationSerif-BoldItalic.ttf",
        "Sans": lib / "LiberationSans-Regular.ttf", "Sans-Bold": lib / "LiberationSans-Bold.ttf",
        "Sans-Italic": lib / "LiberationSans-Italic.ttf", "Sans-BoldItalic": lib / "LiberationSans-BoldItalic.ttf",
        "Mono": dv / "DejaVuSansMono.ttf", "Mono-Bold": dv / "DejaVuSansMono-Bold.ttf",
    }
    for name, path in faces.items():
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
                                  boldItalic="Body-BoldItalic")
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic",
                                  boldItalic="Sans-BoldItalic")
    pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-Bold", italic="Mono", boldItalic="Mono-Bold")


register_fonts()

S = {
    "body": ParagraphStyle("body", fontName="Body", fontSize=10.2, leading=14, alignment=TA_JUSTIFY,
                           spaceAfter=5, textColor=colors.black),
    "bodyleft": ParagraphStyle("bodyleft", fontName="Body", fontSize=10.2, leading=14, alignment=TA_LEFT,
                               spaceAfter=5),
    "small": ParagraphStyle("small", fontName="Body", fontSize=8.8, leading=11.5, spaceAfter=3),
    "caption": ParagraphStyle("caption", fontName="Sans-Italic", fontSize=8, leading=10.2, textColor=INK,
                              spaceBefore=2, spaceAfter=9, alignment=TA_LEFT),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=17, leading=21, textColor=INK, spaceBefore=4,
                         spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=12.4, leading=15.5, textColor=TEAL, spaceBefore=11,
                         spaceAfter=5),
    "h3": ParagraphStyle("h3", fontName="Sans-Bold", fontSize=10.4, leading=13, textColor=INK, spaceBefore=8,
                         spaceAfter=3),
    "q": ParagraphStyle("q", fontName="Sans-Bold", fontSize=9.6, leading=12.5, textColor=colors.white),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=7.7, leading=9.4),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=7.7, leading=9.4),
    "head": ParagraphStyle("head", fontName="Sans-Bold", fontSize=7.7, leading=9.4, textColor=colors.white),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=7.6, leading=10, textColor=INK,
                           backColor=colors.HexColor("#f4f6f5"), borderPadding=(4, 5, 4, 5), spaceBefore=3,
                           spaceAfter=8),
    "eq": ParagraphStyle("eq", fontName="Body-Italic", fontSize=10.4, leading=15, alignment=TA_CENTER,
                         spaceBefore=2, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=10.2, leading=13.6, leftIndent=13,
                             bulletIndent=3, spaceAfter=2.5, alignment=TA_JUSTIFY),
    "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=30, leading=34, textColor=INK),
    "subtitle": ParagraphStyle("subtitle", fontName="Sans", fontSize=13.5, leading=18, textColor=TEAL),
    "toc1": ParagraphStyle("toc1", fontName="Sans-Bold", fontSize=9.8, leading=13, leftIndent=0, spaceBefore=3),
    "toc2": ParagraphStyle("toc2", fontName="Sans", fontSize=8.8, leading=11.2, leftIndent=14),
}


class ReportDoc(BaseDocTemplate):
    """Registers H1/H2 headings in the table of contents and the PDF outline."""

    def __init__(self, path, **kw):
        super().__init__(str(path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=2.1 * cm,
                         bottomMargin=1.9 * cm, title="PathSnap - Final Technical Report (Team PT-088)",
                         author="Team PT-088", subject="UE24CS352A PathSnap map matching", **kw)
        frame = Frame(MARGIN, 1.9 * cm, TEXT_W, PAGE_H - 4.0 * cm, id="main")
        self.addPageTemplates([PageTemplate("cover", [frame], onPage=_cover),
                               PageTemplate("page", [frame], onPage=_page)])
        self._outline_seq = 0

    def afterFlowable(self, flowable):
        level = getattr(flowable, "toc_level", None)
        if level is None:
            return
        text = flowable.toc_text
        key = "h%08x" % (hash((level, text)) & 0xFFFFFFFF)
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
        self.notify("TOCEntry", (level, text, self.page, key))


def _cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(TEAL)
    canvas.rect(0, PAGE_H - 1.1 * cm, PAGE_W, 1.1 * cm, stroke=0, fill=1)
    canvas.setFillColor(GOLD)
    canvas.rect(0, 0, PAGE_W, 0.5 * cm, stroke=0, fill=1)
    canvas.restoreState()


def _page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, PAGE_H - 1.45 * cm, PAGE_W - MARGIN, PAGE_H - 1.45 * cm)
    canvas.setFont("Sans", 7.6)
    canvas.setFillColor(INK)
    canvas.drawString(MARGIN, PAGE_H - 1.25 * cm, "PathSnap  |  Team PT-088  |  Final technical report")
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 1.25 * cm, "UE24CS352A Machine Learning, PES University")
    canvas.line(MARGIN, 1.35 * cm, PAGE_W - MARGIN, 1.35 * cm)
    canvas.drawString(MARGIN, 1.0 * cm, "All figures and tables are generated from artifacts/ in the repository.")
    canvas.drawRightString(PAGE_W - MARGIN, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


def toc() -> TableOfContents:
    t = TableOfContents()
    t.levelStyles = [S["toc1"], S["toc2"]]
    t.dotsMinLevel = 0
    return t


def heading(text: str, level: int):
    style = S["h1"] if level == 0 else S["h2"] if level == 1 else S["h3"]
    p = Paragraph(text, style)
    if level <= 1:
        p.toc_level = level
        p.toc_text = text
    return p


def H1(text):
    return [CondPageBreak(9 * cm), heading(text, 0)]


def H2(text):
    return [CondPageBreak(4.5 * cm), heading(text, 1)]


def H3(text):
    return [CondPageBreak(3 * cm), heading(text, 2)]


def P(text, style="body"):
    if style == "body" and "Mono" in text:
        return Paragraph(text, S["bodyleft"])  # long monospaced tokens make justified lines gappy
    return Paragraph(text, S[style])


def bullets(items, style="bullet"):
    return [Paragraph(item, S[style], bulletText="•") for item in items]


def numbered(items):
    return [Paragraph(item, S["bullet"], bulletText=f"{i}.") for i, item in enumerate(items, 1)]


def code(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
    text = text.replace(" ", "&nbsp;")
    return Paragraph(text, S["code"])


def eq(text):
    return Paragraph(text, S["eq"])


def _cell(value, bold=False, size=None):
    if isinstance(value, Paragraph):
        return value
    style = S["cellb" if bold else "cell"]
    if size:
        style = ParagraphStyle(style.name + str(size), parent=style, fontSize=size, leading=size * 1.22)
    return Paragraph(str(value), style)


def table(rows, widths, header_rows=1, bold_first_col=False, zebra=True, highlight_rows=(), align_right_from=None,
          font_size=None):
    data = []
    for r, row in enumerate(rows):
        cells = []
        for c, v in enumerate(row):
            if r < header_rows:
                head = S["head"] if not font_size else ParagraphStyle("h%s" % font_size, parent=S["head"],
                                                                      fontSize=font_size, leading=font_size * 1.22)
                cells.append(Paragraph(str(v), head))
            else:
                cells.append(_cell(v, bold=(bold_first_col and c == 0) or r in highlight_rows, size=font_size))
        data.append(cells)
    total = sum(widths)
    t = Table(data, colWidths=[w * TEXT_W / total for w in widths], repeatRows=header_rows, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.8, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
    ]
    if zebra:
        for r in range(header_rows, len(rows)):
            if (r - header_rows) % 2 == 1:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f4f7f5")))
    for r in highlight_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#e2efe9")))
    t.setStyle(TableStyle(style))
    return t


def figure(path: Path, caption: str, width=1.0, max_height=None):
    from PIL import Image as PILImage
    w_px, h_px = PILImage.open(path).size
    w = TEXT_W * width
    h = w * h_px / w_px
    if max_height and h > max_height:
        h = max_height
        w = h * w_px / h_px
    img = Image(str(path), width=w, height=h)
    img.hAlign = "CENTER"
    return KeepTogether([img, Paragraph(caption, S["caption"])])


def question(label: str, text: str):
    """Coloured bar quoting a question from the organiser documents."""
    t = Table([[Paragraph(f"{label}&nbsp;&nbsp;{text}", S["q"])]], colWidths=[TEXT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TEAL), ("TOPPADDING", (0, 0), (-1, -1), 4),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5), ("LEFTPADDING", (0, 0), (-1, -1), 7)]))
    return [CondPageBreak(3.2 * cm), Spacer(1, 3), t, Spacer(1, 4)]


def callout(paragraphs, title=None, colour=PALE, edge=TEAL):
    content = []
    if title:
        content.append(Paragraph(f"<b>{title}</b>", ParagraphStyle("ct", parent=S["small"], fontName="Sans-Bold",
                                                                   textColor=INK, fontSize=8.8)))
    for p in paragraphs:
        content.append(Paragraph(p, S["small"]) if isinstance(p, str) else p)
    t = Table([[content]], colWidths=[TEXT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colour), ("LINEBEFORE", (0, 0), (0, -1), 2.5, edge),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    return [Spacer(1, 2), t, Spacer(1, 7)]


def pending_box(title, lines):
    return callout(lines, title=title, colour=colors.HexColor("#fbeeea"), edge=RED)


def gap(h=6):
    return Spacer(1, h)
