"""QR codes and barcodes.

They are not text, so no text search can find them, yet they usually carry the holder's data: a payment QR on an
invoice (name and IBAN), a vCard, a boarding pass, the PDF417 on a driving licence, a ticket number. A decoded
code becomes a BARCODE finding whose text is its payload; redaction maps that payload back to the rectangle where
it was found and blacks the rectangle out (and the pixels under it).

Decoding uses zxing-cpp. If it is not installed the feature is simply off.
"""
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pymupdf as fitz

try:
    import zxingcpp
except ImportError:  # optional dependency
    zxingcpp = None

CODE_DPI = 200
PAD_RATIO = 0.12   # margin around the decoded corners, as a share of the code's size: covers the quiet zone
MIN_PAD = 6.0      # points


def available() -> bool:
    return zxingcpp is not None


RETAIL_LENGTHS = (8, 12, 13, 14)   # EAN-8, UPC-A, EAN-13, ITF-14
PERSONAL_KINDS = {"IBAN", "CREDIT_CARD", "EMAIL", "PHONE", "ID", "MRZ", "URL", "PERSON", "ADDRESS"}


def payload_score(payload: str) -> int:
    """How sure are we that a code's data is personal? 85 for any decoded code (they usually are); 97 when its
    payload itself contains an IBAN, an email, a phone number, an ID... that the detector recognises."""
    from . import scoring as sc
    from .detect import detect_regex

    if payload.isdigit() and len(payload) in RETAIL_LENGTHS:
        return sc.BARCODE  # EAN / UPC product code: it can pass the credit-card checksum by chance, it is not a card
    if any(f.kind in PERSONAL_KINDS and f.score >= 85 for f in detect_regex(payload)):
        return 97
    return sc.BARCODE


@dataclass(frozen=True)
class CodeHit:
    page: int
    rect: tuple          # (x0, y0, x1, y1) in PyMuPDF's unrotated page space, padded
    format: str
    payload: str

    @property
    def box(self) -> "fitz.Rect":
        return fitz.Rect(self.rect)


def detect_codes(page, dpi: int = CODE_DPI) -> list[CodeHit]:
    """Every readable QR code / barcode on the page, with its rectangle."""
    if zxingcpp is None:
        return []
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]
    scale = 72.0 / dpi
    hits: list[CodeHit] = []
    for r in zxingcpp.read_barcodes(np.ascontiguousarray(gray)):
        payload = r.text or (r.bytes.hex() if r.bytes else "")
        if not payload:
            continue
        pos = r.position
        corners = [pos.top_left, pos.top_right, pos.bottom_right, pos.bottom_left]
        xs, ys = [c.x * scale for c in corners], [c.y * scale for c in corners]
        view = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
        pad = max(MIN_PAD, PAD_RATIO * max(view.width, view.height))
        view = fitz.Rect(view.x0 - pad, view.y0 - pad, view.x1 + pad, view.y1 + pad) & page.rect
        if view.is_empty:
            continue
        box = fitz.Rect(view * page.derotation_matrix).normalize()   # the viewer's coordinates -> unrotated page space
        hits.append(CodeHit(page.number, tuple(box), str(r.format), payload))
    return hits


@lru_cache(maxsize=256)
def _cached(path: str, mtime_ns: int, size: int, pno: int) -> tuple:
    with fitz.open(path) as doc:
        return tuple(detect_codes(doc[pno]))


def codes_for_page(page) -> tuple:
    """detect_codes, remembered per unchanged file: the scan, the preview and the redaction all ask."""
    path = page.parent.name
    try:
        st = os.stat(path)
    except (OSError, TypeError):
        return tuple(detect_codes(page))
    return _cached(os.path.abspath(path), st.st_mtime_ns, st.st_size, page.number)


def payload_rects(page, payloads) -> list["fitz.Rect"]:
    """Rectangles of the codes on `page` whose payload is one of `payloads`."""
    wanted = set(payloads)
    return [h.box for h in codes_for_page(page) if h.payload in wanted]
