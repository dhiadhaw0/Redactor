"""Hidden-content inspection and post-redaction verification.

A redaction tool that only *claims* success is dangerous, so after saving we re-open the
output and check it independently of the code that produced it.
"""
import re
from dataclasses import dataclass, field

import pymupdf as fitz

from .imagemeta import pdf_image_metadata, strip_pdf_images
from .ocr import ocr_page, page_text

BLACK_LEVEL = 40          # gray value (0-255) counted as black
MIN_BLACK_FRACTION = 0.97  # share of a redaction box that must render black
EDGE = 1.0                 # points ignored at the border of a box when checking it is black
DIGIT_WINDOW = 8          # a run of this many digits from a term, still visible, is a partial leak
MIN_DISTINCT_DIGITS = 3   # '00000000' is a coincidence (a capital of 24 000 000), not a leak


@dataclass
class Verification:
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


# ------------------------------------------------------------------ hidden content

def inspect_hidden(path: str) -> dict:
    """What the file carries besides the visible page. Each of these can leak personal data."""
    r = {}
    with fitz.open(path) as doc:
        r["annotations"] = sum(len(list(p.annots() or [])) for p in doc)
        r["form_fields"] = sum(len(list(p.widgets() or [])) for p in doc)
        r["links"] = sum(len(p.get_links()) for p in doc)
        r["attachments"] = len(doc.embfile_names())
        r["bookmarks"] = len(doc.get_toc())
        r["metadata_fields"] = [k for k, v in (doc.metadata or {}).items()
                                if v and k not in ("format", "encryption")]
        r["xmp_metadata"] = bool(doc.xref_xml_metadata())
        hidden = 0
        for p in doc:
            try:
                hidden += sum(1 for t in p.get_texttrace() if t.get("type") == 3 and str(t.get("chars", "")).strip())
            except Exception:  # texttrace is best-effort
                pass
        r["invisible_text_spans"] = hidden
        r["photo_metadata"] = pdf_image_metadata(doc)   # EXIF / GPS / XMP inside embedded JPEG photos
    return r


def describe_hidden(report: dict) -> list[str]:
    """Human-readable lines for the non-empty items of an inspect_hidden() report."""
    labels = {
        "annotations": "comments / annotations", "form_fields": "form fields", "links": "hyperlinks",
        "attachments": "embedded file attachments", "bookmarks": "bookmarks",
        "invisible_text_spans": "invisible text spans (e.g. a hidden OCR layer)",
    }
    out = [f"{report[k]} {label}" for k, label in labels.items() if report.get(k)]
    if report.get("metadata_fields"):
        out.append("document metadata (" + ", ".join(report["metadata_fields"]) + ")")
    if report.get("xmp_metadata"):
        out.append("XMP metadata")
    if report.get("photo_metadata"):
        kinds = ", ".join(report["photo_metadata"])
        out.append(f"photo metadata in {max(report['photo_metadata'].values())} embedded image(s) ({kinds})")
    return out


def strip_hidden(doc) -> None:
    """Remove everything inspect_hidden() reports. Call after the redactions are applied."""
    for page in doc:
        for annot in list(page.annots() or []):
            page.delete_annot(annot)
        for widget in list(page.widgets() or []):
            page.delete_widget(widget)
        for link in page.get_links():
            page.delete_link(link)
    doc.scrub(  # attachments, embedded files, JS, thumbnails, hidden text, metadata, XMP, ...
        attached_files=True, clean_pages=True, embedded_files=True, hidden_text=True,
        javascript=True, metadata=True, redactions=True, remove_links=True,
        reset_fields=True, reset_responses=True, thumbnails=True, xml_metadata=True,
    )
    doc.set_toc([])
    doc.set_metadata({})
    doc.del_xml_metadata()
    strip_pdf_images(doc)   # EXIF / GPS / XMP / comments cut out of embedded JPEGs, losslessly


# ------------------------------------------------------------------ verification

def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def _digits(text: str) -> str:
    """The digits of each line, lines separated by '|': a run of digits can never span two lines."""
    return "|".join(re.sub(r"\D", "", line) for line in text.splitlines())


def _leaks(term: str, squashed: str, digits: str) -> str | None:
    t = _squash(term)
    if t and t in squashed:
        return "still readable"
    td = re.sub(r"\D", "", term)
    if len(td) >= 2 * DIGIT_WINDOW:
        for i in range(len(td) - DIGIT_WINDOW + 1):
            window = td[i:i + DIGIT_WINDOW]
            if len(set(window)) >= MIN_DISTINCT_DIGITS and window in digits:
                return "partly readable (a run of its digits is still visible)"
    return None


def _non_page_text(doc) -> str:
    """Text that lives outside the page content: metadata, annotations, bookmarks."""
    parts = [str(v) for v in (doc.metadata or {}).values() if v]
    parts += [t[1] for t in doc.get_toc()]
    for page in doc:
        for annot in page.annots() or []:
            parts += [annot.info.get("content", ""), annot.info.get("title", "")]
    return "\n".join(parts)


def verify_output(path: str, terms: list[str], rects: list[tuple[int, "fitz.Rect"]],
                  ocr_pages: set[int] = frozenset(), lang: str = "en", code_payloads: set[str] = frozenset()) -> Verification:
    """Independent re-check of a redacted PDF.

    1. every redaction box renders (almost) fully black;
    2. no approved term survives in the text layer, metadata, annotations or bookmarks;
    3. pages that needed OCR are OCR'd again: no approved term may be readable in the pixels;
    4. no QR code / barcode whose payload was redacted can still be decoded.
    """
    v = Verification()
    with fitz.open(path) as doc:
        for pno, rect in rects:
            if pno >= len(doc) or rect.width < 1 or rect.height < 1:
                continue
            # rects are in unrotated page space; rendering clips are in the rotated view
            view = fitz.Rect(rect * doc[pno].rotation_matrix).normalize()
            # sample just inside the box: its outermost pixels are anti-aliased against the page
            inner = fitz.Rect(view.x0 + EDGE, view.y0 + EDGE, view.x1 - EDGE, view.y1 - EDGE)
            if inner.width < 1 or inner.height < 1:
                inner = view
            pix = doc[pno].get_pixmap(clip=inner, dpi=100, colorspace=fitz.csGRAY)
            samples = pix.samples
            if samples and sum(1 for b in samples if b < BLACK_LEVEL) / len(samples) < MIN_BLACK_FRACTION:
                v.problems.append(f"Page {pno + 1}: a redaction box is not fully black "
                                  f"(at {rect.x0:.0f},{rect.y0:.0f}) - content may still show through.")

        blob = "\n".join([_non_page_text(doc)] + [page_text(p) for p in doc])
        squashed, digits = _squash(blob), _digits(blob)
        for term in terms:
            why = _leaks(term, squashed, digits)
            if why:
                v.problems.append(f"'{term}' is {why} in the text layer.")

        if code_payloads:
            from .codes import detect_codes
            for page in doc:
                for hit in detect_codes(page):
                    if hit.payload in code_payloads:
                        v.problems.append(f"Page {page.number + 1}: a QR code / barcode that was meant to be redacted "
                                          "can still be decoded.")

        for pno in sorted(ocr_pages):
            if pno >= len(doc):
                continue
            text = ocr_page(doc[pno], lang).text
            sq, dg = _squash(text), _digits(text)
            for term in terms:
                why = _leaks(term, sq, dg)
                if why:
                    v.problems.append(f"Page {pno + 1}: '{term}' is {why} when the redacted page is OCR'd again.")
    return v
