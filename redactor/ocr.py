"""OCR support for scanned PDFs (uses Tesseract through PyMuPDF)."""
import io
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pymupdf as fitz
from PIL import Image

from .detect import LANGUAGES

OCR_DPI = 300
UPSCALE_TARGET_DPI = 240   # Tesseract reads text well from about 200+ dpi
MAX_UPSCALE = 4.0
UPSCALE_BELOW_DPI = 170    # sharper images are left alone: enlarging a 200 dpi scan by 1.2x only adds noise
MIN_QUALITY = 80   # fewer word-like characters than this on a page: try other orientations
# PyMuPDF keeps ligatures by default: 'Griffin' is extracted as 'Griﬃn' and search_for('Griffin') finds nothing, so a
# name in a PDF that uses ligatures (LaTeX, Word and Office exports do) would be missed AND left unredacted.
# Every extraction and search goes through these helpers, which do not preserve them.
TEXT_FLAGS = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP
SEARCH_FLAGS = fitz.TEXT_DEHYPHENATE


def page_text(page) -> str:
    """The page's text layer with ligatures expanded ('ﬁ' -> 'fi')."""
    return page.get_text(flags=TEXT_FLAGS)


MIN_IMAGE_SHARE = 0.02      # an image smaller than this share of the page is a logo or an icon
MIN_IMAGE_PIXELS = (100, 30)  # ...or smaller than this many pixels: too small to hold readable text
MIN_OCR_CHARS = 40  # an OCR'd page with less text than this is probably unreadable
MIN_TEXT_CHARS = 20  # fewer characters than this on a page => treat it as scanned
PROJECT_TESSDATA = Path(__file__).resolve().parent.parent / "tessdata"


# ---- OCR engine --------------------------------------------------------------------------------------------
# "tesseract" (default: about 3 s a page), "paddle" (PaddleOCR: more accurate on photos, about 30 s a page, optional
# install) or "auto" (Tesseract first; PaddleOCR only for a page Tesseract reads badly).
BACKENDS = ("tesseract", "paddle", "auto")
BACKEND = os.environ.get("REDACTOR_OCR", "tesseract").lower()


def set_backend(name: str, paddle_models: str | None = None) -> None:
    """Choose the OCR engine. Raises if PaddleOCR is requested but not usable, so the user finds out now."""
    global BACKEND
    if name not in BACKENDS:
        raise ValueError(f"Unknown OCR engine {name!r}; choose one of {', '.join(BACKENDS)}.")
    if paddle_models:
        os.environ["REDACTOR_PADDLE_MODELS"] = str(paddle_models)
    if name != "tesseract":
        from . import paddle_ocr
        ok, why = paddle_ocr.available()
        if not ok:
            raise RuntimeError(f"Cannot use the {name!r} OCR engine: {why}.")
    BACKEND = name


def find_tessdata(lang: str = "en") -> str | None:
    """First tessdata folder that has every Tesseract language `lang` needs."""
    needed = LANGUAGES[lang][1].split("+")
    candidates = [str(PROJECT_TESSDATA), os.environ.get("TESSDATA_PREFIX")]
    exe = shutil.which("tesseract")
    if exe:
        candidates.append(str(Path(exe).parent / "tessdata"))
    candidates += [
        r"C:\Program Files\Tesseract-OCR\tessdata",
        r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tessdata",
        "/opt/homebrew/share/tessdata",
    ]
    for c in candidates:
        if c and all(Path(c, f"{n}.traineddata").exists() for n in needed):
            return c
    return None


def _effective_dpi(page) -> float | None:
    """Resolution of the page's main raster image (pixels per inch of page), or None
    if the page is not dominated by one image."""
    best = None
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"]) & page.rect
        if r.is_empty or r.get_area() < 0.5 * page.rect.get_area():
            continue
        dpi = info["width"] / (r.width / 72)
        best = dpi if best is None else max(best, dpi)
    return best


def _upscale_factor(page) -> float:
    """Low-resolution scans (phone photos, small PNGs) OCR badly: on one 579 px-wide letter
    Tesseract found 2/9 key items as-is and 9/9 after a 3x Lanczos upscale."""
    dpi = _effective_dpi(page)
    if dpi is None or dpi >= UPSCALE_BELOW_DPI:
        return 1.0
    return min(UPSCALE_TARGET_DPI / dpi, MAX_UPSCALE)


def _render_copy(page, factor: float, quarter_turns: int):
    """One-page PDF holding the page as the viewer shows it, enlarged by `factor` and turned
    clockwise by `quarter_turns` x 90 degrees. Returns (doc, page)."""
    dpi = round(_effective_dpi(page) or OCR_DPI)
    img = Image.open(io.BytesIO(page.get_pixmap(dpi=dpi).tobytes("png"))).convert("RGB")
    if factor != 1:
        img = img.resize((round(img.width * factor), round(img.height * factor)), Image.LANCZOS)
    if quarter_turns:
        img = img.rotate(-90 * quarter_turns, expand=True)  # PIL: negative angle = clockwise
    buf = io.BytesIO()
    img.save(buf, "PNG")
    w, h = page.rect.width, page.rect.height
    doc = fitz.open()
    new = doc.new_page(width=h if quarter_turns % 2 else w, height=w if quarter_turns % 2 else h)
    new.insert_image(new.rect, stream=buf.getvalue())
    return doc, new


def _back_matrix(page, quarter_turns: int) -> "fitz.Matrix":
    """Map points from a `_render_copy` page back to the viewer's page, then to PyMuPDF's
    unrotated page space (where search_for / redaction / drawing work)."""
    w, h = page.rect.width, page.rect.height
    undo = {0: fitz.Matrix(1, 0, 0, 1, 0, 0), 1: fitz.Matrix(0, -1, 1, 0, 0, h),
            2: fitz.Matrix(-1, 0, 0, -1, w, h), 3: fitz.Matrix(0, 1, -1, 0, w, 0)}[quarter_turns]
    return undo * page.derotation_matrix


_VOWELS = set("aeiouyàâäéèêëîïôöùûüœ")


def _word_like(word: str) -> bool:
    w = "".join(c for c in word if c.isalnum()).lower()
    if len(w) < 3 or not w.isalpha():
        return False
    return 0.2 <= sum(c in _VOWELS for c in w) / len(w) <= 0.65


def _quality(words) -> tuple[int, int]:
    """(characters in words of 3+ letters/digits, characters in word-like tokens). A sideways
    page gives scraps; an upside-down one gives mangled words; an upright page gives mostly
    word-like text. Digits and codes are legitimate, so this is a hint, never a verdict."""
    total = sum(len(w[4]) for w in words if sum(c.isalnum() for c in w[4]) >= 3)
    wordlike = sum(sum(c.isalnum() for c in w[4]) for w in words if _word_like(w[4]))
    return total, wordlike


def _run_ocr(page, lang: str, factor: float, quarter_turns: int, engine: str = "tesseract") -> "OcrPage":
    if engine == "paddle":
        from . import paddle_ocr
        text, words = paddle_ocr.run(page, factor, quarter_turns, _render_copy, _back_matrix)
        return OcrPage(text, words)
    tessdata = find_tessdata(lang)
    if tessdata is None:
        raise RuntimeError(
            f"Tesseract data for '{LANGUAGES[lang][1]}' not found. Put the .traineddata files in "
            f"{PROJECT_TESSDATA} or install Tesseract (https://github.com/UB-Mannheim/tesseract/wiki)."
        )
    if factor == 1 and quarter_turns == 0:
        # OCR the page itself: PyMuPDF already reports unrotated coordinates
        tp = page.get_textpage_ocr(language=LANGUAGES[lang][1], dpi=OCR_DPI, full=True, tessdata=tessdata)
        return OcrPage(tp.extractText(), tp.extractWORDS())
    doc, copy = _render_copy(page, factor, quarter_turns)
    tp = copy.get_textpage_ocr(language=LANGUAGES[lang][1], dpi=OCR_DPI, full=True, tessdata=tessdata)
    back = _back_matrix(page, quarter_turns)
    words = [(*fitz.Rect(w[:4]).transform(back).normalize(), *w[4:]) for w in tp.extractWORDS()]
    doc.close()
    return OcrPage(tp.extractText(), words)


@dataclass
class OcrPage:
    """OCR result detached from the document, so it can be cached and reused.
    words: (x0, y0, x1, y1, word, block_no, line_no, word_no) in page coordinates."""
    text: str
    words: list
    mixed: bool = False   # a text page whose embedded images were OCR'd and merged with its own text

    def search(self, term: str) -> list["fitz.Rect"]:
        """Rects of every occurrence of `term` (case- and whitespace-insensitive), one per
        line, always covering whole OCR words: OCR character boxes are too imprecise to
        trust ('Acacia|s' once left its last letter uncovered), words are not."""
        needle = "".join(term.split()).lower()
        if not needle or not self.words:
            return []
        owner, chars = [], []
        for i, w in enumerate(self.words):
            piece = "".join(w[4].split()).lower()
            chars.append(piece)
            owner += [i] * len(piece)
        hay = "".join(chars)
        rects, pos = [], hay.find(needle)
        while pos != -1:
            by_line: dict[tuple, fitz.Rect] = {}
            for wi in sorted(set(owner[pos:pos + len(needle)])):
                x0, y0, x1, y1, _, block, line, _ = self.words[wi]
                box = fitz.Rect(x0, y0, x1, y1)
                key = (block, line)
                by_line[key] = by_line[key] | box if key in by_line else box  # Rect | Rect returns a new Rect
            rects += list(by_line.values())
            pos = hay.find(needle, pos + len(needle))
        return rects


def _suspicious(words) -> bool:
    if not words:
        return False  # a blank page is not a sideways page
    total, wordlike = _quality(words)
    return wordlike < MIN_QUALITY or wordlike < 0.7 * total


def text_images(page) -> list:
    """Images on the page big enough to hold readable text: at least 2% of the page and 100 x 30 pixels.
    Logos, bullets and decorations are below that and are not worth an OCR pass."""
    page_area = page.rect.get_area() or 1.0
    found = []
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"]) & page.rect
        if (not r.is_empty and r.get_area() / page_area >= MIN_IMAGE_SHARE
                and info.get("width", 0) >= MIN_IMAGE_PIXELS[0] and info.get("height", 0) >= MIN_IMAGE_PIXELS[1]):
            found.append(r)
    return found


def _has_reliable_text(page) -> bool:
    return len(page_text(page).strip()) >= MIN_TEXT_CHARS and not _unreliable_text_layer(page.get_text())


def _run_mixed(page, lang: str, engine: str = "tesseract") -> OcrPage:
    """A page with its own text AND embedded pictures of text (a pasted screenshot, a scanned attachment):
    OCR only the images (`full=False`) and let PyMuPDF merge that with the text layer. The digital words keep
    their exact boxes; the picture's words come from OCR."""
    if engine == "paddle":
        from . import paddle_ocr
        text, words = paddle_ocr.run_images_only(page, text_images(page), TEXT_FLAGS)
        return OcrPage(text, words, mixed=True)
    tessdata = find_tessdata(lang)
    if tessdata is None:
        raise RuntimeError(f"Tesseract data for '{LANGUAGES[lang][1]}' not found (see {PROJECT_TESSDATA}).")
    tp = page.get_textpage_ocr(language=LANGUAGES[lang][1], dpi=OCR_DPI, full=False, tessdata=tessdata)
    return OcrPage(tp.extractText(), tp.extractWORDS(), mixed=True)


def _read_with(page, lang: str, engine: str) -> OcrPage:
    """One engine reads the page: images only on a text page, otherwise the whole page, trying the other
    orientations when the first reading looks like scraps."""
    if _has_reliable_text(page) and text_images(page):
        return _run_mixed(page, lang, engine)
    factor = _upscale_factor(page)
    best = _run_ocr(page, lang, factor, 0, engine)
    # Orientation retries are for Tesseract, which returns scraps for a sideways page. PaddleOCR is slow (~30 s a read) and
    # does not fail that way, so it is never asked four times.
    if engine == "tesseract" and _effective_dpi(page) is not None and _suspicious(best.words):
        # Possibly a sideways or upside-down scan without a rotate flag: read the other
        # orientations too and switch only if one is clearly better.
        for turns in (1, 3, 2):
            candidate = _run_ocr(page, lang, factor, turns, engine)
            total_c, wordlike_c = _quality(candidate.words)
            total_b, wordlike_b = _quality(best.words)
            # Clearly more word-like text AND more readable text overall. The second condition matters for
            # ID pages, which are mostly capitals, codes and '<<<' filler: a sideways page full of random
            # letter strings can look slightly more 'word-like' while yielding less text than the real one.
            if wordlike_c > max(1.25 * wordlike_b, MIN_QUALITY) and total_c > total_b:
                best = candidate
    return best


def _looks_blank(page) -> bool:
    """A page with nothing on it needs no second opinion."""
    pix = page.get_pixmap(dpi=30, colorspace=fitz.csGRAY)
    return float(np.frombuffer(pix.samples, dtype=np.uint8).std()) < 4.0


def _ocr_to_page(page, lang: str, backend: str | None = None) -> OcrPage:
    backend = backend or BACKEND
    result = _read_with(page, lang, "paddle" if backend == "paddle" else "tesseract")
    if backend == "auto" and (_suspicious(result.words) or (not result.words and not _looks_blank(page))):
        # Tesseract struggled: ask PaddleOCR (slow, so only now) and keep whichever read more.
        second = _read_with(page, lang, "paddle")
        if _quality(second.words)[0] >= _quality(result.words)[0]:
            result = second
    return result


@lru_cache(maxsize=128)
def _ocr_cached(path: str, mtime_ns: int, size: int, pno: int, lang: str, backend: str = "tesseract") -> OcrPage:
    with fitz.open(path) as doc:
        return _ocr_to_page(doc[pno], lang, backend)


def ocr_page(page, lang: str = "en") -> OcrPage:
    """OCR a page, reusing an earlier result for the same unchanged file. The UI scans,
    previews and redacts the same pages many times; OCR is by far the slowest step."""
    path = page.parent.name
    try:
        st = os.stat(path)
    except (OSError, TypeError):  # in-memory document: no stable identity to cache on
        return _ocr_to_page(page, lang)
    return _ocr_cached(os.path.abspath(path), st.st_mtime_ns, st.st_size, page.number, lang, BACKEND)


def _unreliable_text_layer(text: str) -> bool:
    """Arabic PDFs often store shaped glyphs (Unicode 'presentation forms') in
    visual order, which cannot be read or matched reliably -> OCR the page instead."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    forms = sum(1 for c in letters if "ﭐ" <= c <= "﷿" or "ﹰ" <= c <= "﻿")
    return forms / len(letters) > 0.3


def page_textpage(page, ocr: bool = True, lang: str = "en"):
    """Return (text, ocr_page). ocr_page is None for normal pages and an OcrPage for scanned
    ones: use ocr_page.search(term) there instead of page.search_for."""
    text = page_text(page)
    if not ocr:
        return text, None
    # Reliability is judged on the RAW text: expanding ligatures also turns Arabic shaped forms into plain letters,
    # which would hide exactly the signal ('this layer is in visual order') that sends such pages to OCR.
    reliable = len(text.strip()) >= MIN_TEXT_CHARS and not _unreliable_text_layer(page.get_text())
    if reliable and not text_images(page):
        return text, None          # an ordinary digital page: nothing to OCR
    result = ocr_page(page, lang)  # a scanned page, or a text page with pictures of text on it
    return result.text, result
