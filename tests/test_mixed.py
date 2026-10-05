"""Text pages that carry pictures of text (a pasted screenshot, a scanned attachment)."""
import pymupdf as fitz
import pytest

from benchmark import corpus
from redactor.ocr import _ocr_cached, ocr_page, page_textpage, text_images
from redactor.redact import apply_redactions, scan_pdf

needs_tesseract = pytest.mark.skipif(__import__("redactor.ocr", fromlist=["x"]).find_tessdata() is None,
                                     reason="Tesseract not installed")


def doc_named(name):
    return next(d for d in corpus.build_corpus() if d.name == name)


@pytest.fixture
def mixed(tmp_path):
    d = doc_named("form_en")
    return d, corpus.render(d, "mixed", str(tmp_path))


def logo_page(path):
    """A page with ordinary text and a small logo-sized picture."""
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((60, 80), "Quarterly report with enough ordinary words to count as a text page.", fontsize=11)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 60, 20), False)
    pix.clear_with(200)
    page.insert_image(fitz.Rect(60, 120, 100, 135), pixmap=pix)
    pdf.save(path)
    return path


def test_a_small_logo_does_not_trigger_ocr(tmp_path):
    page = fitz.open(logo_page(str(tmp_path / "logo.pdf")))[0]
    assert text_images(page) == []
    text, scanned = page_textpage(page, ocr=True)
    assert scanned is None and "Quarterly report" in text


def test_a_picture_big_enough_to_hold_text_is_found(mixed):
    _, path = mixed
    [rect] = text_images(fitz.open(path)[0])
    assert rect.get_area() > 0.02 * 595 * 842


@needs_tesseract
def test_text_in_a_picture_is_read_and_merged_with_the_text_layer(mixed):
    d, path = mixed
    page = fitz.open(path)[0]
    text, ocr = page_textpage(page, ocr=True)
    assert ocr is not None and ocr.mixed
    head = [kind_text for kind_text in d.items if kind_text[1] in page.get_text()]      # items in the text layer
    tail = [kind_text for kind_text in d.items if kind_text[1] not in page.get_text()]  # items only in the picture
    assert head and tail, "the corpus must put items on both sides"
    squashed = "".join(text.split()).lower()
    for _, item in tail + head:
        assert "".join(item.split()).lower() in squashed, item


@needs_tesseract
def test_findings_include_the_items_that_only_exist_as_pixels(mixed):
    d, path = mixed
    page = fitz.open(path)[0]
    only_pixels = [t for _, t in d.items if t not in page.get_text()]
    found = "".join(" ".join(f.text for f in scan_pdf(path)).split()).lower()
    assert any("".join(t.split()).lower() in found for t in only_pixels)


@needs_tesseract
def test_mixed_page_findings_are_not_penalised_as_scans(mixed):
    _, path = mixed
    emails = [f for f in scan_pdf(path) if f.kind == "EMAIL"]
    assert emails and max(f.score for f in emails) >= 95    # no OCR penalty: the page has a real text layer


@needs_tesseract
def test_redaction_removes_text_from_the_picture_and_from_the_text_layer(mixed, tmp_path):
    d, path = mixed
    findings = scan_pdf(path)
    out = str(tmp_path / "out.pdf")
    res = apply_redactions(path, out, [f.text for f in findings if f.score >= 60])
    assert res.verification.ok, res.verification.problems
    assert res.image_text_pages == [0] and res.ocr_pages == [0]
    after_text = "".join(fitz.open(out)[0].get_text().split()).lower()
    after_pixels = "".join(ocr_page(fitz.open(out)[0]).text.split()).lower()      # OCR of what is left, pictures included
    for kind, item in d.items:
        if kind in ("EMAIL", "IBAN", "CREDIT_CARD", "ID", "PHONE"):
            squashed = "".join(item.split()).lower()
            assert squashed not in after_text and squashed not in after_pixels, item


@needs_tesseract
def test_quality_notes_say_pictures_were_read_through_ocr(mixed, tmp_path):
    _, path = mixed
    res = apply_redactions(path, str(tmp_path / "o.pdf"), [f.text for f in scan_pdf(path) if f.score >= 60])
    assert any("Pictures on page 1" in n for n in res.quality.notes)
    assert not any("was read through OCR" in n for n in res.quality.notes)         # it is not a scanned page


def test_ocr_is_skipped_when_ocr_is_off(mixed):
    _, path = mixed
    text, scanned = page_textpage(fitz.open(path)[0], ocr=False)
    assert scanned is None


@needs_tesseract
def test_a_pure_scan_is_not_flagged_as_mixed(tmp_path):
    d = doc_named("letter_en")
    path = corpus.render(d, "scan", str(tmp_path))
    _ocr_cached.cache_clear()
    assert page_textpage(fitz.open(path)[0], ocr=True)[1].mixed is False


def test_arabic_shaped_text_is_still_sent_to_ocr_after_ligature_expansion(tmp_path):
    """Expanding ligatures turns Arabic shaped forms into plain letters; the page must still be recognised as being
    in visual order (and read by OCR), or its text would be matched backwards."""
    d = doc_named("letter_ar")
    page = fitz.open(corpus.render(d, "digital", str(tmp_path)))[0]
    from redactor.ocr import _unreliable_text_layer, page_text
    assert _unreliable_text_layer(page.get_text()) and not _unreliable_text_layer(page_text(page))   # the trap
    text, scanned = page_textpage(page, ocr=True, lang="ar")
    assert scanned is not None and not scanned.mixed
