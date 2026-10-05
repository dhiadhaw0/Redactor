"""Word, Excel, PowerPoint, email and text files: turned into a PDF, then redacted like any other PDF.

Why convert instead of editing the file in place: a PDF contains only what a reader sees, so the same detection,
redaction, verification and grading apply, and there are far fewer places left for personal data to hide.

What is kept (the visible content) and what is deliberately dropped, and reported, because it is where Office
documents leak: comments, tracked deletions, hidden text, speaker notes, hidden slides, hidden sheets, hidden rows,
document properties (author, last modified by, company), routing headers of an email, macros and embedded objects.

Embedded pictures are carried into the PDF as pictures, so text inside them is read by the OCR of mixed pages.
Files are parsed with the standard library only (zipfile + ElementTree), with limits against zip bombs.
"""
import codecs
import email
import email.policy
import html
import io
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import pymupdf as fitz
from PIL import Image

OFFICE_SUFFIXES = {".docx", ".docm", ".xlsx", ".xlsm", ".pptx", ".pptm", ".eml", ".txt", ".csv", ".md", ".html", ".htm"}
MAX_UNCOMPRESSED = 400 * 1024 * 1024       # refuse zip bombs
MAX_PART = 60 * 1024 * 1024
MAX_IMAGE_WIDTH_PT = 440
MIN_IMAGE_PIXELS = 24                      # smaller pictures are bullets and lines, not content

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"


@dataclass
class Block:
    kind: str                  # "heading" | "para" | "row" | "image"
    text: str = ""
    image: bytes | None = None


@dataclass
class Extracted:
    blocks: list[Block] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)   # what the original carried that the PDF does not

    def para(self, text: str) -> None:
        if text.strip():
            self.blocks.append(Block("para", text.strip()))

    def heading(self, text: str) -> None:
        if text.strip():
            self.blocks.append(Block("heading", text.strip()))

    def row(self, cells: list[str]) -> None:
        cells = [c.strip() for c in cells if c and c.strip()]
        if cells:
            self.blocks.append(Block("row", " | ".join(cells)))

    def image(self, data: bytes) -> None:
        self.blocks.append(Block("image", image=data))

    def drop(self, what: str) -> None:
        self.dropped.append(what)


# ------------------------------------------------------------------ helpers
def _open_zip(path: str) -> zipfile.ZipFile:
    z = zipfile.ZipFile(path)
    if sum(i.file_size for i in z.infolist()) > MAX_UNCOMPRESSED or any(i.file_size > MAX_PART for i in z.infolist()):
        raise ValueError("The file expands to an unreasonable size and was refused (possible zip bomb).")
    return z


def _xml(z: zipfile.ZipFile, name: str):
    try:
        return ET.fromstring(z.read(name))
    except KeyError:
        return None


def _names(z, prefix: str, suffix: str = ".xml") -> list[str]:
    key = lambda n: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", n)]
    return sorted((n for n in z.namelist() if n.startswith(prefix) and n.endswith(suffix) and "/_rels/" not in n), key=key)


def _rels(z, part: str) -> dict[str, str]:
    """relationship id -> zip path, for a part such as 'word/document.xml'."""
    folder, _, name = part.rpartition("/")
    root = _xml(z, f"{folder}/_rels/{name}.rels") if folder else _xml(z, f"_rels/{name}.rels")
    out = {}
    for rel in (root if root is not None else []):
        target = rel.get("Target", "")
        out[rel.get("Id")] = target.lstrip("/") if target.startswith("/") else _norm(f"{folder}/{target}")
    return out


def _norm(path: str) -> str:
    parts = []
    for p in path.split("/"):
        if p == "..":
            parts and parts.pop()
        elif p and p != ".":
            parts.append(p)
    return "/".join(parts)


def _pictures_ok(data: bytes) -> bytes | None:
    """PNG bytes for a picture big enough to hold content, or None (unreadable or tiny)."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        if min(img.size) < MIN_IMAGE_PIXELS:
            return None
        buf = io.BytesIO()
        img.convert("RGB" if img.mode not in ("RGB", "L", "RGBA") else img.mode).save(buf, "PNG")
        return buf.getvalue()
    except Exception:
        return None


def _properties(z: zipfile.ZipFile) -> list[str]:
    """Document properties that identify people: author, last editor, company, manager..."""
    fields = []
    for name in ("docProps/core.xml", "docProps/app.xml", "docProps/custom.xml"):
        root = _xml(z, name)
        if root is None:
            continue
        for el in root.iter():
            tag = el.tag.rsplit("}", 1)[-1]
            if tag in ("creator", "lastModifiedBy", "Company", "Manager", "title", "subject", "keywords", "description") \
                    and (el.text or "").strip():
                fields.append(tag)
    return sorted(set(fields))


def _common_drops(z: zipfile.ZipFile, ex: Extracted) -> None:
    props = _properties(z)
    if props:
        ex.drop("document properties (" + ", ".join(props) + ")")
    if any(n.endswith("vbaProject.bin") for n in z.namelist()):
        ex.drop("macros (VBA project)")
    embedded = [n for n in z.namelist() if "/embeddings/" in n]
    if embedded:
        ex.drop(f"{len(embedded)} embedded object(s)")
    if any("/customXml/" in n or n.startswith("customXml/") for n in z.namelist()):
        ex.drop("custom XML data")


def _carry_images(z, names, ex: Extracted) -> None:
    unreadable = 0
    for name in names:
        png = _pictures_ok(z.read(name))
        if png:
            ex.image(png)
        else:
            unreadable += 1
    if unreadable:
        ex.drop(f"{unreadable} picture(s) that are too small or in a format that cannot be carried over")


# ------------------------------------------------------------------ DOCX
def _walk_text(el, state: dict) -> str:
    """The visible text of a Word element, and a tally of what is not visible."""
    out = []
    for ch in el:
        tag = ch.tag
        if tag in (W + "pPr", W + "rPr", W + "tblPr", MC + "Fallback", W + "moveFrom"):
            continue
        if tag == W + "del":
            state["deleted"] += sum(1 for _ in ch.iter(W + "delText"))
            continue
        if tag == W + "r":
            if ch.find(f"{W}rPr/{W}vanish") is not None:
                state["hidden"] += 1
                continue
            for sub in ch:
                if sub.tag == W + "t":
                    out.append(sub.text or "")
                elif sub.tag == W + "tab":
                    out.append("\t")
                elif sub.tag in (W + "br", W + "cr"):
                    out.append("\n")
                elif sub.tag in (W + "drawing", W + "pict", MC + "AlternateContent", W + "object"):
                    for blip in sub.iter(A + "blip"):
                        state["images"].append(blip.get(R + "embed"))
                    out.append(_walk_text(sub, state))
                elif sub.tag == W + "txbxContent":
                    out.append(_walk_text(sub, state))
            continue
        out.append(_walk_text(ch, state))
    return "".join(out)


def _docx_part_paragraphs(root, state) -> list[str]:
    return [t for t in (_walk_text(p, state).strip() for p in root.iter(W + "p")) if t]


def _docx(path: str) -> Extracted:
    ex = Extracted()
    with _open_zip(path) as z:
        doc = _xml(z, "word/document.xml")
        if doc is None:
            raise ValueError("Not a Word document (word/document.xml is missing).")
        rels = _rels(z, "word/document.xml")
        state = {"deleted": 0, "hidden": 0, "images": []}

        headers = {t for n in _names(z, "word/header") for t in _docx_part_paragraphs(_xml(z, n), state)}
        for t in sorted(headers):
            ex.para("[Header] " + t)

        def body(parent):
            for ch in parent:
                if ch.tag == W + "p":
                    before = len(state["images"])
                    text = _walk_text(ch, state)
                    style = ch.find(f"{W}pPr/{W}pStyle")
                    is_head = style is not None and (style.get(W + "val") or "").lower().startswith(("heading", "title"))
                    (ex.heading if is_head else ex.para)(text)
                    for rid in state["images"][before:]:
                        if rid in rels and rels[rid] in z.namelist():
                            png = _pictures_ok(z.read(rels[rid]))
                            if png:
                                ex.image(png)
                elif ch.tag == W + "tbl":
                    for tr in ch.iter(W + "tr"):
                        ex.row([" ".join(t for t in (_walk_text(p, state).strip() for p in tc.iter(W + "p")) if t)
                                for tc in tr.findall(W + "tc")])
                elif ch.tag in (W + "sdt", W + "sdtContent", MC + "AlternateContent", MC + "Choice"):
                    body(ch)
                elif ch.tag == W + "body":
                    body(ch)

        body(doc.find(W + "body") if doc.find(W + "body") is not None else doc)

        notes = []
        for n in ("word/footnotes.xml", "word/endnotes.xml"):
            root = _xml(z, n)
            if root is not None:
                for note in list(root):
                    if note.get(W + "type") in ("separator", "continuationSeparator", "continuationNotice"):
                        continue
                    notes += [t for t in (_walk_text(p, state).strip() for p in note.iter(W + "p")) if t]
        if notes:
            ex.heading("Notes")
            for t in notes:
                ex.para(t)
        for t in sorted({t for n in _names(z, "word/footer") for t in _docx_part_paragraphs(_xml(z, n), state)}):
            ex.para("[Footer] " + t)

        comments = _xml(z, "word/comments.xml")
        n_comments = len(comments.findall(W + "comment")) if comments is not None else 0
        if n_comments:
            ex.drop(f"{n_comments} comment(s)")
        if state["deleted"]:
            ex.drop(f"tracked deletions ({state['deleted']} piece(s) of deleted text)")
        if state["hidden"]:
            ex.drop(f"hidden text ({state['hidden']} run(s))")
        _common_drops(z, ex)
    return ex


# ------------------------------------------------------------------ XLSX
def _shared_strings(z) -> list[str]:
    root = _xml(z, "xl/sharedStrings.xml")
    if root is None:
        return []
    return ["".join(t.text or "" for t in si.iter(S + "t")) for si in root.findall(S + "si")]


def _xlsx(path: str) -> Extracted:
    ex = Extracted()
    with _open_zip(path) as z:
        book = _xml(z, "xl/workbook.xml")
        if book is None:
            raise ValueError("Not an Excel workbook (xl/workbook.xml is missing).")
        rels = _rels(z, "xl/workbook.xml")
        shared = _shared_strings(z)
        hidden_sheets = hidden_rows = 0
        for sheet in book.iter(S + "sheet"):
            name, rid, state = sheet.get("name", "Sheet"), sheet.get(R + "id"), sheet.get("state", "visible")
            part = rels.get(rid)
            if part is None or part not in z.namelist():
                continue
            if state != "visible":
                hidden_sheets += 1
                continue
            ex.heading(f"Sheet: {name}")
            root = _xml(z, part)
            for row in root.iter(S + "row"):
                if row.get("hidden") in ("1", "true"):
                    hidden_rows += 1
                    continue
                cells = []
                for c in row.findall(S + "c"):
                    kind, v = c.get("t", "n"), c.find(S + "v")
                    if kind == "s" and v is not None and (v.text or "").isdigit() and int(v.text) < len(shared):
                        cells.append(shared[int(v.text)])
                    elif kind == "inlineStr":
                        cells.append("".join(t.text or "" for t in c.iter(S + "t")))
                    elif v is not None and v.text is not None:
                        cells.append("TRUE" if kind == "b" and v.text == "1" else "FALSE" if kind == "b" else v.text)
                ex.row(cells)
        if hidden_sheets:
            ex.drop(f"{hidden_sheets} hidden sheet(s)")
        if hidden_rows:
            ex.drop(f"{hidden_rows} hidden row(s)")
        n_comments = sum(len(list(_xml(z, n).iter(S + "comment"))) for n in _names(z, "xl/comments"))
        if n_comments or _names(z, "xl/threadedComments"):
            ex.drop("cell comments")
        if _names(z, "xl/externalLinks"):
            ex.drop("links to external workbooks")
        _carry_images(z, [n for n in z.namelist() if n.startswith("xl/media/")], ex)
        _common_drops(z, ex)
    return ex


# ------------------------------------------------------------------ PPTX
def _slide_text(root) -> tuple[list[tuple[str, str]], list[str]]:
    """[(kind, text)] for the shapes of a slide, and the picture relationship ids."""
    items, pics = [], []
    for el in root.iter():
        if el.tag == P + "sp":
            ph = el.find(f"{P}nvSpPr/{P}nvPr/{P}ph")
            is_title = ph is not None and ph.get("type") in ("title", "ctrTitle")
            for para in el.iter(A + "p"):
                text = "".join(t.text or "" for t in para.iter(A + "t")).strip()
                if text:
                    items.append(("heading" if is_title else "para", text))
        elif el.tag == A + "tbl":
            for tr in el.iter(A + "tr"):
                items.append(("row", " | ".join("".join(t.text or "" for t in tc.iter(A + "t")).strip()
                                                  for tc in tr.findall(A + "tc"))))
        elif el.tag == P + "pic":
            for blip in el.iter(A + "blip"):
                pics.append(blip.get(R + "embed"))
    return items, pics


def _pptx(path: str) -> Extracted:
    ex = Extracted()
    with _open_zip(path) as z:
        pres = _xml(z, "ppt/presentation.xml")
        if pres is None:
            raise ValueError("Not a PowerPoint file (ppt/presentation.xml is missing).")
        rels = _rels(z, "ppt/presentation.xml")
        order = [rels[s.get(R + "id")] for s in pres.iter(P + "sldId") if s.get(R + "id") in rels]
        hidden = notes = 0
        for n, part in enumerate(order or _names(z, "ppt/slides/slide"), 1):
            root = _xml(z, part)
            if root is None:
                continue
            if root.get("show") == "0":
                hidden += 1
                continue
            ex.heading(f"Slide {n}")
            items, pics = _slide_text(root)
            for kind, text in items:
                {"heading": ex.heading, "para": ex.para}.get(kind, lambda t: ex.row([t]))(text)
            srels = _rels(z, part)
            for rid in pics:
                if rid in srels and srels[rid] in z.namelist():
                    png = _pictures_ok(z.read(srels[rid]))
                    if png:
                        ex.image(png)
        for name in _names(z, "ppt/notesSlides/notesSlide"):
            if any((t.text or "").strip() for t in _xml(z, name).iter(A + "t")):
                notes += 1
        if hidden:
            ex.drop(f"{hidden} hidden slide(s)")
        if notes:
            ex.drop(f"speaker notes on {notes} slide(s)")
        if _names(z, "ppt/comments/") or _names(z, "ppt/comments/modernComment"):
            ex.drop("slide comments")
        _common_drops(z, ex)
    return ex


# ------------------------------------------------------------------ e-mail and text
class _HtmlText(HTMLParser):
    BREAKS = {"br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self.skip += 1
        elif tag in self.BREAKS:
            self.out.append("\n")
        elif tag in ("td", "th"):
            self.out.append(" | ")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self.skip = max(0, self.skip - 1)
        elif tag in self.BREAKS:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_to_text(source: str) -> str:
    parser = _HtmlText()
    parser.feed(source)
    return re.sub(r"\n\s*\n+", "\n", "".join(parser.out)).strip()


def _decode(data: bytes) -> str:
    """UTF-8 (with or without BOM), UTF-16 only when it announces itself with a BOM, else Windows-1252. UTF-16
    must not be tried blindly: nearly any even-length byte string decodes as UTF-16, into garbage."""
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16")
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def _text_file(path: str, kind: str) -> Extracted:
    ex = Extracted()
    raw = _decode(Path(path).read_bytes())
    text = html_to_text(raw) if kind == "html" else raw
    for line in text.splitlines():
        ex.para(line)
    return ex


ROUTING = ("received", "x-originating-ip", "x-mailer", "user-agent", "message-id", "return-path", "x-received",
           "authentication-results", "dkim-signature", "received-spf", "x-google-smtp-source")


def _eml(path: str) -> Extracted:
    ex = Extracted()
    msg = email.message_from_binary_file(open(path, "rb"), policy=email.policy.default)
    for name in ("From", "To", "Cc", "Reply-To", "Date", "Subject"):
        if msg.get(name):
            ex.para(f"{name}: {msg.get(name)}")
    # A message can carry the same body twice (plain + HTML), and the plain part is sometimes a stub
    # ("your mail client does not support HTML"): keep whichever version has more text.
    versions = []
    for kind in ("plain", "html"):
        part = msg.get_body(preferencelist=(kind,))
        if part is not None:
            content = part.get_content()
            versions.append(html_to_text(content) if kind == "html" else content)
    text = max(versions, key=lambda t: len("".join(t.split())), default="")
    for line in text.splitlines():
        if line.strip():
            ex.para(line)
    attachments, inline = [], 0
    for part in msg.iter_attachments():
        data = part.get_payload(decode=True) or b""
        if part.get_content_maintype() == "image":
            png = _pictures_ok(data)
            if png:
                ex.image(png)
                inline += 1
                continue
        attachments.append(f"{part.get_filename() or 'unnamed'} ({part.get_content_type()}, {len(data)} bytes)")
    for a in attachments:
        ex.para("Attachment: " + a)
    if attachments:
        ex.drop(f"{len(attachments)} attachment(s) not included (process them separately): their names are kept in the text")
    routing = [h for h in msg.keys() if h.lower() in ROUTING]
    if routing or msg.get("Bcc"):
        ex.drop(f"{len(routing)} routing / tracking header(s)" + (" and Bcc" if msg.get("Bcc") else ""))
    return ex


# ------------------------------------------------------------------ dispatch and rendering
EXTRACTORS = {".docx": _docx, ".docm": _docx, ".xlsx": _xlsx, ".xlsm": _xlsx, ".pptx": _pptx, ".pptm": _pptx, ".eml": _eml,
              ".txt": lambda p: _text_file(p, "text"), ".csv": lambda p: _text_file(p, "text"),
              ".md": lambda p: _text_file(p, "text"), ".html": lambda p: _text_file(p, "html"),
              ".htm": lambda p: _text_file(p, "html")}


def extract(path: str) -> Extracted:
    suffix = Path(path).suffix.lower()
    if suffix not in EXTRACTORS:
        raise ValueError(f"Unsupported file type: {suffix}")
    try:
        return EXTRACTORS[suffix](path)
    except zipfile.BadZipFile as e:
        raise ValueError(f"The {suffix} file is damaged or not a real Office file: {e}") from e


def _html_for(blocks: list[Block], archive: "fitz.Archive") -> str:
    parts = ['<body style="font-family: sans-serif; font-size: 10pt">']
    for i, b in enumerate(blocks):
        if b.kind == "heading":
            parts.append(f"<h3>{html.escape(b.text)}</h3>")
        elif b.kind == "image" and b.image:
            name = f"img{i}.png"
            archive.add(b.image, name)
            width = min(MAX_IMAGE_WIDTH_PT, Image.open(io.BytesIO(b.image)).width * 0.75)
            parts.append(f'<p><img src="{name}" width="{round(width)}"/></p>')
        else:
            parts.append(f'<p style="margin: 0 0 3pt 0">{html.escape(b.text)}</p>')
    parts.append("</body>")
    return "".join(parts)


def render_pdf(extracted: Extracted, dst: str) -> None:
    """Flow the extracted blocks onto A4 pages."""
    archive = fitz.Archive()
    story = fitz.Story(html=_html_for(extracted.blocks or [Block("para", " ")], archive), archive=archive)
    writer = fitz.DocumentWriter(dst)
    mediabox = fitz.paper_rect("a4")
    where = mediabox + (50, 50, -50, -50)
    more = 1
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()


def convert(src: str, dst: str) -> list[str]:
    """Write `src` (Word, Excel, PowerPoint, e-mail, text) as a PDF at `dst`. Returns what the original carried
    that the PDF does not (comments, notes, hidden sheets, properties...), in plain words."""
    ex = extract(src)
    render_pdf(ex, dst)
    return ex.dropped
