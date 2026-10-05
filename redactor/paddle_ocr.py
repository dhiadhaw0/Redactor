"""Optional OCR engine: PaddleOCR (PP-OCR models), used instead of, or to rescue, Tesseract.

Why it exists: on photos and awkward layouts PaddleOCR reads noticeably better, including accents. Why it is not the
default: on a CPU it takes about 30 s per page against about 3 s for Tesseract, and it needs ~1 GB of packages
(`pip install -r requirements-paddle.txt`) plus the PP-OCR models on disk. Nothing here is imported unless it is chosen.

Models: a folder holding `<name>_det` and `<name>_rec` sub-folders in the PaddleOCR 3.x format (inference.json,
inference.pdiparams, inference.yml), e.g. PP-OCRv6_medium_det / PP-OCRv6_medium_rec. Point to it with the environment
variable REDACTOR_PADDLE_MODELS, `--paddle-models` on the command line or the box in the app.

PaddleOCR returns text lines with word boxes; they are turned into the same `OcrPage` words the rest of the pipeline
uses, so detection, redaction, verification and grading do not know which engine read the page.
"""
import importlib.util
import os
import re
from pathlib import Path

import numpy as np
import pymupdf as fitz

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")   # never reach out to download models

OCR_DPI = 150          # PaddleOCR is no slower at 300 dpi nor better below ~110; its time is not driven by resolution
DET_SIDE = 1280        # the detector works on images scaled down to this longest side
_engine = None
_engine_key = None


def models_dir() -> Path | None:
    value = os.environ.get("REDACTOR_PADDLE_MODELS")
    return Path(value) if value else None


def _model_pair(folder: Path) -> tuple[Path, Path] | None:
    """(detection dir, recognition dir) inside `folder`, or None."""
    if not folder or not folder.is_dir():
        return None
    det = sorted(p for p in folder.iterdir() if p.is_dir() and p.name.lower().endswith("_det") and (p / "inference.yml").exists())
    rec = sorted(p for p in folder.iterdir() if p.is_dir() and p.name.lower().endswith("_rec") and (p / "inference.yml").exists())
    return (det[-1], rec[-1]) if det and rec else None


def _model_name(directory: Path) -> str:
    text = (directory / "inference.yml").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"model_name:\s*(\S+)", text)
    return m.group(1) if m else directory.name


def installed() -> bool:
    return importlib.util.find_spec("paddleocr") is not None


def available() -> tuple[bool, str]:
    """(usable, why not). Cheap: nothing heavy is imported."""
    if not installed():
        return False, "PaddleOCR is not installed (pip install -r requirements-paddle.txt)"
    folder = models_dir()
    if folder is None:
        return False, "No models folder set (REDACTOR_PADDLE_MODELS, --paddle-models, or the box in the app)"
    if _model_pair(folder) is None:
        return False, f"{folder} has no <name>_det and <name>_rec model folders"
    return True, ""


def engine():
    """The loaded PaddleOCR pipeline (loaded once; about 15 s)."""
    global _engine, _engine_key
    ok, why = available()
    if not ok:
        raise RuntimeError(why)
    det, rec = _model_pair(models_dir())
    key = (str(det), str(rec))
    if _engine is None or _engine_key != key:
        from paddleocr import PaddleOCR

        _engine = PaddleOCR(
            text_detection_model_name=_model_name(det), text_detection_model_dir=str(det),
            text_recognition_model_name=_model_name(rec), text_recognition_model_dir=str(rec),
            use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
            # oneDNN crashes some Paddle 3.x CPU builds on Windows ("ConvertPirAttribute2RuntimeAttribute")
            enable_mkldnn=False, device="cpu", cpu_threads=os.cpu_count() or 4,
            text_det_limit_type="max", text_det_limit_side_len=DET_SIDE,
        )
        _engine_key = key
    return _engine


# ------------------------------------------------------------------ reading
def read_image(rgb: np.ndarray) -> list[tuple[str, tuple, int, int]]:
    """[(word, (x0, y0, x1, y1) in pixels, line number, word number)] for an RGB image.

    PaddleOCR reports 'tokens' with their own boxes: whole words for plain ASCII, but one token PER CHARACTER for
    many accented words ('L', 'é', 'a'), with explicit ' ' tokens between real words. Characters between two space
    tokens are therefore merged into one word whose box is the union of theirs."""
    result = engine().predict(rgb, return_word_box=True)[0]
    out = []
    for line_no, (tokens, regions) in enumerate(zip(result["text_word"], result["text_word_region"])):
        word_no, chars, boxes = 0, [], []

        def flush():
            nonlocal word_no, chars, boxes
            if chars:
                out.append(("".join(chars), (min(b[0] for b in boxes), min(b[1] for b in boxes),
                                             max(b[2] for b in boxes), max(b[3] for b in boxes)), line_no, word_no))
                word_no += 1
            chars, boxes = [], []

        for token, poly in zip(tokens, regions):
            if not token.strip():
                flush()
                continue
            pts = np.asarray(poly, dtype=float)
            chars.append(token)
            boxes.append((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
        flush()
    return out


def _pixmap_rgb(pix) -> np.ndarray:
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return np.ascontiguousarray(arr[:, :, :3] if pix.n >= 3 else np.repeat(arr[:, :, :1], 3, axis=2))


def _to_words(read, matrix: "fitz.Matrix", scale: float, origin=(0.0, 0.0), block: int = 0) -> list:
    """Pixel boxes -> page-coordinate OCR words. `matrix` maps the rendering's frame to PyMuPDF's unrotated page space."""
    words = []
    for token, (x0, y0, x1, y1), line_no, word_no in read:
        box = fitz.Rect(origin[0] + x0 * scale, origin[1] + y0 * scale, origin[0] + x1 * scale, origin[1] + y1 * scale)
        box = fitz.Rect(box.transform(matrix)).normalize()
        words.append((box.x0, box.y0, box.x1, box.y1, token, block, line_no, word_no))
    return words


def _text_of(words: list) -> str:
    lines: dict[tuple, list] = {}
    for w in words:
        lines.setdefault((w[5], w[6]), []).append(w)
    ordered = sorted(lines.values(), key=lambda ws: (round(min(w[1] for w in ws)), min(w[0] for w in ws)))
    return "\n".join(" ".join(w[4] for w in sorted(ws, key=lambda w: w[0])) for ws in ordered)


def run(page, factor: float, quarter_turns: int, render_copy, back_matrix):
    """OCR a page (or an enlarged / turned copy of it, see ocr._render_copy). Returns (text, words)."""
    if factor == 1 and quarter_turns == 0:
        pix = page.get_pixmap(dpi=OCR_DPI)
        words = _to_words(read_image(_pixmap_rgb(pix)), page.derotation_matrix, 72.0 / OCR_DPI)
    else:
        doc, copy = render_copy(page, factor, quarter_turns)
        try:
            pix = copy.get_pixmap(dpi=OCR_DPI)
            words = _to_words(read_image(_pixmap_rgb(pix)), back_matrix(page, quarter_turns), 72.0 / OCR_DPI)
        finally:
            doc.close()
    return _text_of(words), words


def run_images_only(page, image_rects: list, text_flags: int):
    """A text page with pictures of text: the page's own words (exact boxes) plus the words OCR finds inside each
    picture. Returns (text, words)."""
    words = [tuple(w) for w in page.get_text("words", flags=text_flags)]
    digital = _text_of(words)
    pictures = []
    for i, rect in enumerate(image_rects):
        view = fitz.Rect(rect * page.rotation_matrix).normalize()          # unrotated space -> what get_pixmap(clip) uses
        pix = page.get_pixmap(dpi=OCR_DPI, clip=view)
        found = _to_words(read_image(_pixmap_rgb(pix)), page.derotation_matrix, 72.0 / OCR_DPI, origin=(view.x0, view.y0),
                          block=1000 + i)
        pictures.append(found)
        words += found
    text = "\n".join([digital] + [_text_of(w) for w in pictures if w])
    return text, words
