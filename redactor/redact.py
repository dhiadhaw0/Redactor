"""True redaction with PyMuPDF: removes the underlying text, not just overlays it."""
from dataclasses import dataclass, field, replace

import pymupdf as fitz

from . import codes as barcodes
from . import scoring as sc
from .detect import Finding, detect
from .ocr import MIN_OCR_CHARS, SEARCH_FLAGS, page_textpage
from .profiles import Profile, apply_allow, deny_findings
from .quality import Quality, assess_quality
from .risk import Risk, assess

OCR_PAD = 1.5  # points of margin around OCR word boxes
from .verify import Verification, describe_hidden, inspect_hidden, strip_hidden, verify_output


def scan_pdf(src: str, use_ner: bool = True, ocr: bool = True, lang: str = "en",
             warnings: list | None = None, codes: bool = True, profile: Profile | None = None,
             suppressed: list | None = None) -> list[Finding]:
    """Detect personal data without modifying anything (feeds the review UI).
    Pass a list as `warnings` to collect pages that need a human look. A `profile` adds its deny list (text the
    detector could not know) and drops what its allow list says never to flag; pass a list as `suppressed` to
    receive the findings it dropped, so they are never silently lost."""
    found: dict[tuple[str, str], Finding] = {}
    denied: dict[tuple[str, str], Finding] = {}
    with fitz.open(src) as doc:
        for page in doc:
            text, tp = page_textpage(page, ocr=ocr, lang=lang)
            if tp is not None and warnings is not None and len(text.strip()) < MIN_OCR_CHARS:
                warnings.append(f"Page {page.number + 1}: OCR found almost no text. The scan may be too "
                                "blurry or low-resolution to read - check the preview and draw boxes by hand.")
            for f in detect(text, use_ner=use_ner, lang=lang):
                if tp is not None and not tp.mixed:  # a scanned page read through OCR: slightly less certain
                    f = replace(f, score=sc.clamp(f.score + sc.OCR))
                old = found.get((f.kind, f.text))
                if old is None or f.score > old.score:
                    found[(f.kind, f.text)] = f
            for f in deny_findings(text, profile):   # "always redact": not subject to the allow list
                denied[(f.kind, f.text)] = f
            if codes:  # QR codes and barcodes are not text: found by decoding them, redacted by position
                for hit in barcodes.codes_for_page(page):
                    found.setdefault(("BARCODE", hit.payload), Finding("BARCODE", hit.payload, barcodes.payload_score(hit.payload)))
    kept, dropped = apply_allow(list(found.values()), profile)
    kept = [f for f in kept if (f.kind, f.text) not in denied] + list(denied.values())
    if suppressed is not None:
        suppressed.extend(sorted((f for f in dropped if (f.kind, f.text) not in denied), key=lambda f: (f.kind, f.text)))
    return sorted(kept, key=lambda f: (f.kind, f.text))


def _term_rects(page, terms, ocr):
    """(term, rect) for every occurrence. `ocr` is an OcrPage (scanned pages) or None."""
    for term in sorted(set(terms), key=len, reverse=True):
        if ocr is not None:
            for rect in ocr.search(term):
                yield term, fitz.Rect(rect.x0 - OCR_PAD, rect.y0 - OCR_PAD, rect.x1 + OCR_PAD, rect.y1 + OCR_PAD)
        else:
            for rect in page.search_for(term, flags=SEARCH_FLAGS):
                yield term, rect


# A manual box: (page_index, x0, y0, x1, y1) with coordinates as 0-1 fractions of the
# page, so it does not depend on the resolution the preview was drawn at.
Box = tuple[int, float, float, float, float]


def _box_rects(page, boxes):
    """Manual boxes are fractions of the page as the user SEES it (rotation applied);
    PyMuPDF draws and redacts in the unrotated page space, so convert."""
    w, h = page.rect.width, page.rect.height
    for pno, x0, y0, x1, y1 in boxes:
        if pno == page.number:
            view = fitz.Rect(min(x0, x1) * w, min(y0, y1) * h, max(x0, x1) * w, max(y0, y1) * h) & page.rect
            if not view.is_empty:
                yield fitz.Rect(view * page.derotation_matrix).normalize()


@dataclass
class RedactionResult:
    hits: int                     # areas blacked out
    hidden_removed: list[str]     # hidden content found in the source and stripped
    verification: Verification    # independent re-check of the saved file
    residual: list[Finding] = field(default_factory=list)  # what the detector still finds in the saved file
    pages: int = 0
    unreadable_pages: list[int] = field(default_factory=list)  # pages with no text and OCR could not read any
    ocr_pages: list[int] = field(default_factory=list)         # pages that had to be read through OCR
    image_text_pages: list[int] = field(default_factory=list)  # text pages whose embedded pictures were OCR'd
    hidden_remaining: list[str] = field(default_factory=list)  # hidden content still in the SAVED file (should be none)

    @property
    def quality(self) -> Quality:
        """A/B/C/F grade of the finished file with the reasons (see quality.py)."""
        return assess_quality(self)

    @property
    def risk_after(self) -> Risk:
        """Measured residual risk: the same scoring as before redaction, applied to what is still detectable."""
        return assess(self.residual)


def apply_redactions(src: str, dst: str, terms: list[str], ocr: bool = True, lang: str = "en",
                     boxes: list[Box] = (), verify: bool = True, rescan: bool = True,
                     use_ner: bool = True, codes: bool = True) -> RedactionResult:
    """Black out every occurrence of each term plus any manual boxes, delete the text AND
    image pixels underneath, strip hidden content (annotations, attachments, metadata, ...),
    save, then re-open the saved file, verify it and re-scan it for whatever is still detectable
    (`rescan`: that is the measured residual risk)."""
    hits = 0
    applied: list[tuple[int, fitz.Rect]] = []
    ocr_pages: set[int] = set()
    image_text: set[int] = set()
    code_payloads: set[str] = set()
    term_set = set(terms)
    unreadable: list[int] = []
    hidden_removed = describe_hidden(inspect_hidden(src))
    with fitz.open(src) as doc:
        n_pages = len(doc)
        for page in doc:
            rects = []
            text, scanned = page_textpage(page, ocr=ocr, lang=lang)  # cached: the scan already OCR'd this page
            if scanned is not None:
                ocr_pages.add(page.number)
                if scanned.mixed:
                    image_text.add(page.number)
            # an image page with (almost) no readable text: OCR failed, or OCR is off
            if len(text.strip()) < MIN_OCR_CHARS and page.get_images():
                unreadable.append(page.number)
            if terms:
                rects += [r for _, r in _term_rects(page, terms, scanned)]
            if terms and codes:  # a selected BARCODE finding: black out the code where it was decoded
                for hit in barcodes.codes_for_page(page):
                    if hit.payload in term_set:
                        rects.append(hit.box)
                        code_payloads.add(hit.payload)
            rects += list(_box_rects(page, boxes))
            for rect in rects:
                page.add_redact_annot(rect, fill=(0, 0, 0))
                applied.append((page.number, fitz.Rect(rect)))
                hits += 1
            # IMAGE_PIXELS blanks the pixels under each box (needed for scans)
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)
        strip_hidden(doc)
        doc.save(dst, garbage=4, deflate=True)
    verification = (verify_output(dst, terms, applied, ocr_pages if ocr else set(), lang, code_payloads)
                    if verify else Verification())
    residual = scan_pdf(dst, use_ner=use_ner, ocr=ocr, lang=lang, codes=codes) if rescan else []
    hidden_remaining = describe_hidden(inspect_hidden(dst))
    return RedactionResult(hits, hidden_removed, verification, residual, n_pages, unreadable,
                           sorted(ocr_pages), sorted(image_text), hidden_remaining)


def preview_pages(src: str, terms: list[str], ocr: bool = True, lang: str = "en", dpi: int = 110,
                  boxes: list[Box] = (), codes: bool = True) -> list[bytes]:
    """PNG of every page with the areas that WOULD be redacted shaded red.
    Nothing is written to disk and the source PDF is untouched."""
    pages = []
    with fitz.open(src) as doc:
        for page in doc:
            if terms:
                _, tp = page_textpage(page, ocr=ocr, lang=lang)
                shaded = [r for _, r in _term_rects(page, terms, tp)]
                if codes:
                    shaded += barcodes.payload_rects(page, terms)
                for rect in shaded:
                    page.draw_rect(rect, color=(0.85, 0, 0), fill=(0.85, 0, 0), fill_opacity=0.35, width=1)
            for rect in _box_rects(page, boxes):  # manual boxes in blue
                page.draw_rect(rect, color=(0, 0.3, 0.9), fill=(0, 0.3, 0.9), fill_opacity=0.35, width=1)
            pages.append(page.get_pixmap(dpi=dpi).tobytes("png"))
    return pages


def redact_pdf(src: str, dst: str, use_ner: bool = True, ocr: bool = True, lang: str = "en") -> list[Finding]:
    findings = scan_pdf(src, use_ner=use_ner, ocr=ocr, lang=lang)
    apply_redactions(src, dst, [f.text for f in findings], ocr=ocr, lang=lang)
    return findings
