"""The optional PaddleOCR engine, tested with a fake engine (fast, deterministic) and, if asked, the real one."""
import os
from pathlib import Path

import numpy as np
import pymupdf as fitz
import pytest

from redactor import ocr, paddle_ocr
from redactor.cli import main as cli_main
from redactor.redact import apply_redactions, scan_pdf


class FakeEngine:
    """Stands in for PaddleOCR: returns canned lines of tokens with polygons (pixel coordinates)."""

    def __init__(self, lines):
        self.lines = lines          # [(tokens, [(x0, y0, x1, y1) per token])]
        self.calls = 0
        self.sizes = []

    def predict(self, rgb, return_word_box=True):
        self.calls += 1
        self.sizes.append(rgb.shape)
        poly = lambda b: [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
        return [{"text_word": [t for t, _ in self.lines], "text_word_region": [[poly(b) for b in boxes] for _, boxes in self.lines]}]


@pytest.fixture(autouse=True)
def restore_engine():
    saved_backend, saved_env = ocr.BACKEND, os.environ.get("REDACTOR_PADDLE_MODELS")
    yield
    ocr.BACKEND = saved_backend
    if saved_env is None:
        os.environ.pop("REDACTOR_PADDLE_MODELS", None)
    else:
        os.environ["REDACTOR_PADDLE_MODELS"] = saved_env
    ocr._ocr_cached.cache_clear()


@pytest.fixture
def fake(monkeypatch):
    """Install a fake engine and make PaddleOCR look available."""
    holder = {}

    def install(lines):
        holder["engine"] = FakeEngine(lines)
        monkeypatch.setattr(paddle_ocr, "engine", lambda: holder["engine"])
        monkeypatch.setattr(paddle_ocr, "available", lambda: (True, ""))
        return holder["engine"]

    return install


def blank_pdf(path, rotation=0, text=None):
    pdf = fitz.open()
    page = pdf.new_page()
    if text:
        page.insert_text((60, 80), text, fontsize=11)
    page.set_rotation(rotation)
    pdf.save(str(path))
    return str(path)


# ------------------------------------------------------------------ reading
def test_characters_between_spaces_are_merged_into_words(fake):
    """PaddleOCR reports one token per character for many accented words."""
    fake([(["L", "é", "a", " ", "CHEVALIER"], [(10, 10, 20, 30), (22, 10, 32, 30), (34, 10, 44, 30), (46, 10, 52, 30), (54, 10, 200, 30)])])
    words = paddle_ocr.read_image(np.zeros((50, 250, 3), dtype=np.uint8))
    assert [(w[0], w[1]) for w in words] == [("Léa", (10.0, 10.0, 44.0, 30.0)), ("CHEVALIER", (54.0, 10.0, 200.0, 30.0))]
    assert [(w[2], w[3]) for w in words] == [(0, 0), (0, 1)]                  # line 0, words 0 and 1


def test_whole_words_and_punctuation_tokens_are_handled(fake):
    fake([(["Vega", " ", "Telecom"], [(0, 0, 40, 10), (40, 0, 44, 10), (44, 0, 100, 10)]),
          (["+", "33", " ", "4"], [(0, 20, 8, 30), (8, 20, 30, 30), (30, 20, 34, 30), (34, 20, 44, 30)])])
    assert [w[0] for w in paddle_ocr.read_image(np.zeros((40, 120, 3), dtype=np.uint8))] == ["Vega", "Telecom", "+33", "4"]


def test_text_is_rebuilt_line_by_line_in_reading_order():
    words = [(100, 50, 140, 60, "world", 0, 1, 1), (60, 50, 95, 60, "hello", 0, 1, 0), (60, 10, 90, 20, "Top", 0, 0, 0)]
    assert paddle_ocr._text_of(words) == "Top\nhello world"


# ------------------------------------------------------------------ availability and settings
def test_availability_explains_what_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(paddle_ocr, "installed", lambda: False)
    assert paddle_ocr.available()[1].startswith("PaddleOCR is not installed")
    monkeypatch.setattr(paddle_ocr, "installed", lambda: True)
    monkeypatch.delenv("REDACTOR_PADDLE_MODELS", raising=False)
    assert "No models folder" in paddle_ocr.available()[1]
    monkeypatch.setenv("REDACTOR_PADDLE_MODELS", str(tmp_path))
    assert "no <name>_det and <name>_rec" in paddle_ocr.available()[1]
    for kind in ("det", "rec"):
        (tmp_path / f"PP-OCRv6_medium_{kind}").mkdir()
        (tmp_path / f"PP-OCRv6_medium_{kind}" / "inference.yml").write_text(f"Global:\n  model_name: PP-OCRv6_medium_{kind}\n")
    assert paddle_ocr.available() == (True, "")
    det, rec = paddle_ocr._model_pair(tmp_path)
    assert paddle_ocr._model_name(det) == "PP-OCRv6_medium_det" and paddle_ocr._model_name(rec) == "PP-OCRv6_medium_rec"


def test_set_backend_validates_and_explains(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="Unknown OCR engine"):
        ocr.set_backend("magic")
    monkeypatch.setattr(paddle_ocr, "available", lambda: (False, "no models"))
    with pytest.raises(RuntimeError, match="no models"):
        ocr.set_backend("paddle")
    assert ocr.BACKEND != "paddle"                                  # a failed switch changes nothing
    ocr.set_backend("tesseract")
    assert ocr.BACKEND == "tesseract"
    monkeypatch.setattr(paddle_ocr, "available", lambda: (True, ""))
    ocr.set_backend("auto", str(tmp_path))
    assert ocr.BACKEND == "auto" and os.environ["REDACTOR_PADDLE_MODELS"] == str(tmp_path)


# ------------------------------------------------------------------ coordinates
def test_word_boxes_land_where_the_pixels_were(fake, tmp_path):
    fake([(["Hello"], [(150, 300, 300, 330)])])                      # pixels at 150 dpi
    page = fitz.open(blank_pdf(tmp_path / "a.pdf"))[0]
    text, words = paddle_ocr.run(page, 1.0, 0, ocr._render_copy, ocr._back_matrix)
    x0, y0, x1, y1, word, *_ = words[0]
    assert word == "Hello" and (round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)) == (72.0, 144.0, 144.0, 158.4)
    assert text == "Hello"


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_boxes_follow_the_page_rotation_into_unrotated_space(fake, tmp_path, rotation):
    fake([(["Hello"], [(150, 300, 300, 330)])])
    page = fitz.open(blank_pdf(tmp_path / "a.pdf", rotation))[0]
    _, words = paddle_ocr.run(page, 1.0, 0, ocr._render_copy, ocr._back_matrix)
    view = fitz.Rect(72.0, 144.0, 144.0, 158.4)                       # where the word is, as the viewer shows the page
    expected = fitz.Rect(view.transform(page.derotation_matrix)).normalize()
    assert [round(v, 2) for v in words[0][:4]] == [round(v, 2) for v in expected]


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_boxes_from_a_turned_copy_map_back_inside_the_page(fake, tmp_path, turns):
    fake([(["Hello"], [(150, 300, 300, 330)])])
    page = fitz.open(blank_pdf(tmp_path / "a.pdf"))[0]
    _, words = paddle_ocr.run(page, 1.0, turns, ocr._render_copy, ocr._back_matrix)
    assert fitz.Rect(words[0][:4]) in page.rect


def test_an_enlarged_copy_keeps_the_same_page_coordinates(fake, tmp_path):
    fake([(["Hello"], [(150, 300, 300, 330)])])
    page = fitz.open(blank_pdf(tmp_path / "a.pdf"))[0]
    _, direct = paddle_ocr.run(page, 1.0, 0, ocr._render_copy, ocr._back_matrix)
    _, enlarged = paddle_ocr.run(page, 2.5, 0, ocr._render_copy, ocr._back_matrix)
    assert [round(v, 2) for v in direct[0][:4]] == [round(v, 2) for v in enlarged[0][:4]]


def test_a_text_page_with_a_picture_keeps_exact_digital_boxes_and_adds_the_picture_words(fake, tmp_path):
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((60, 80), "Quarterly report with ordinary words on the page itself.", fontsize=11)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 100), False)
    pix.clear_with(255)
    page.insert_image(fitz.Rect(100, 300, 400, 400), pixmap=pix)
    path = str(tmp_path / "m.pdf")
    pdf.save(path)
    engine = fake([(["IBAN", " ", "FR76"], [(0, 0, 100, 20), (100, 0, 110, 20), (110, 0, 300, 20)])])
    page = fitz.open(path)[0]
    text, words = paddle_ocr.run_images_only(page, ocr.text_images(page), ocr.TEXT_FLAGS)
    digital = [w for w in words if w[5] < 1000]
    picture = [w for w in words if w[5] >= 1000]
    assert digital and "Quarterly" in {w[4] for w in digital}
    assert [w[4] for w in picture] == ["IBAN", "FR76"] and all(fitz.Rect(w[:4]) in fitz.Rect(100, 300, 400, 400) for w in picture)
    assert "Quarterly report" in text and "IBAN FR76" in text
    assert engine.calls == 1 and engine.sizes[0][0] > 0            # only the picture was sent to the engine


# ------------------------------------------------------------------ using it in the pipeline
def test_the_chosen_engine_reads_the_page_and_its_result_is_cached_per_engine(fake, tmp_path):
    engine = fake([(["john.smith@example.com"], [(150, 300, 600, 330)])])
    path = blank_pdf(tmp_path / "scan.pdf")
    ocr.set_backend("paddle")
    page = fitz.open(path)[0]
    first = ocr.ocr_page(page)
    assert "john.smith@example.com" in first.text and engine.calls >= 1
    calls = engine.calls
    assert ocr.ocr_page(fitz.open(path)[0]) is first and engine.calls == calls            # cached
    ocr.BACKEND = "tesseract"                                                              # another engine: another cache entry
    assert ocr.ocr_page(fitz.open(path)[0]) is not first


def test_paddle_is_never_asked_to_retry_orientations(fake, tmp_path):
    engine = fake([(["xq"], [(10, 10, 40, 30)])])                                          # scraps: 'suspicious' for Tesseract
    path = tmp_path / "s.pdf"
    pdf = fitz.open()
    pg = pdf.new_page()
    pg.insert_image(pg.rect, pixmap=fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 400, 560), False))
    pdf.save(str(path))
    ocr.set_backend("paddle")
    ocr.ocr_page(fitz.open(str(path))[0])
    assert engine.calls == 1                                                                # one read, not four


def test_auto_keeps_tesseract_when_it_reads_well_and_rescues_when_it_does_not(fake, tmp_path, monkeypatch):
    engine = fake([(["rescued", "text", "from", "paddle"], [(0, 0, 50, 10), (60, 0, 100, 10), (110, 0, 150, 10), (160, 0, 220, 10)])])
    path = blank_pdf(tmp_path / "p.pdf")
    good = ocr.OcrPage("a good reading with many real words in it " * 3, [(0, 0, 9, 9, w, 0, 0, i) for i, w in
                                                                      enumerate(("good reading with many real words " * 6).split())])
    poor = ocr.OcrPage("xq vv", [(0, 0, 9, 9, "xq", 0, 0, 0), (0, 0, 9, 9, "vv", 0, 0, 1)])

    def tesseract_gives(result):
        monkeypatch.setattr(ocr, "_read_with", lambda page, lang, eng: result if eng == "tesseract" else ocr.OcrPage(
            "rescued text", [(0, 0, 9, 9, w, 0, 0, i) for i, w in enumerate(("rescued text from paddle plus words " * 8).split())]))

    ocr.BACKEND = "auto"
    tesseract_gives(good)
    assert ocr._ocr_to_page(fitz.open(path)[0], "en", "auto") is good                     # no second opinion needed
    tesseract_gives(poor)
    rescued = ocr._ocr_to_page(fitz.open(path)[0], "en", "auto")
    assert rescued is not poor and "rescued" in rescued.text                                # Paddle read more, so it wins
    assert engine.calls == 0                                                                # (the fake read_with stood in for it)


def test_auto_does_not_rescue_a_blank_page(fake, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ocr, "_read_with", lambda page, lang, eng: calls.append(eng) or ocr.OcrPage("", []))
    path = tmp_path / "blank.pdf"
    pdf = fitz.open()
    pg = pdf.new_page()
    white = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 200, 280), False)
    white.clear_with(255)
    pg.insert_image(pg.rect, pixmap=white)
    pdf.save(str(path))
    ocr._ocr_to_page(fitz.open(str(path))[0], "en", "auto")
    assert calls == ["tesseract"]                                                           # nothing to read: no 30 s detour


def test_a_redaction_made_with_the_paddle_engine_is_verified_like_any_other(fake, tmp_path):
    """Words come from the fake engine; redaction and verification run exactly as for Tesseract."""
    scan = fitz.open()
    page = scan.new_page()
    shot = fitz.open()
    sp = shot.new_page()
    sp.insert_text((60, 100), "Email : julien@example.org", fontsize=14)
    page.insert_image(page.rect, stream=sp.get_pixmap(dpi=150).tobytes("png"))
    src = str(tmp_path / "scan.pdf")
    scan.save(src)
    # the email sits at about x 130-290 pt, y 88-104 pt; at 150 dpi that is 270-600 px, 183-217 px
    fake([(["Email", ":", "julien@example.org"], [(125, 183, 215, 217), (218, 183, 232, 217), (240, 183, 600, 217)])])
    ocr.set_backend("paddle")
    findings = scan_pdf(src, use_ner=False, codes=False)
    assert any(f.kind == "EMAIL" and f.text == "julien@example.org" for f in findings)
    out = str(tmp_path / "out.pdf")
    res = apply_redactions(src, out, ["julien@example.org"], use_ner=False, codes=False, rescan=False, verify=False)
    assert res.hits == 1
    pixels = fitz.open(out)[0].get_pixmap(dpi=150)
    box = fitz.Rect(130, 88, 290, 104)
    crop = fitz.open(out)[0].get_pixmap(dpi=150, clip=box, colorspace=fitz.csGRAY)
    assert sum(1 for b in crop.samples if b < 40) / len(crop.samples) > 0.9            # the email's area is black


# ------------------------------------------------------------------ command line
def test_cli_explains_why_paddle_cannot_be_used(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("REDACTOR_PADDLE_MODELS", raising=False)
    monkeypatch.setattr(paddle_ocr, "installed", lambda: True)
    assert cli_main([blank_pdf(tmp_path / "a.pdf"), str(tmp_path / "o.pdf"), "--ocr", "paddle"]) == 2
    assert "Cannot use the 'paddle' OCR engine" in capsys.readouterr().out


def test_cli_records_the_engine_in_the_audit(tmp_path):
    import json
    audit = str(tmp_path / "a.json")
    cli_main([blank_pdf(tmp_path / "a.pdf", text="Plain text page with a few ordinary words on it for the test."),
              str(tmp_path / "o.pdf"), "--no-ner", "--no-ocr", "--ocr", "tesseract", "--audit", audit])
    assert json.loads(Path(audit).read_text(encoding="utf-8"))["files"][0]["ocr_engine"] == "tesseract"


# ------------------------------------------------------------------ the real engine (slow: off unless asked)
@pytest.mark.skipif(os.environ.get("RUN_PADDLE_TESTS") != "1" or not paddle_ocr.installed(),
                    reason="set RUN_PADDLE_TESTS=1 with PaddleOCR installed and REDACTOR_PADDLE_MODELS pointing at the models")
def test_the_real_engine_redacts_a_scanned_letter_and_tesseract_agrees(tmp_path):
    from benchmark import corpus
    ok, why = paddle_ocr.available()
    assert ok, why
    doc = next(d for d in corpus.build_corpus() if d.name == "letter_fr")
    path = corpus.render(doc, "scan", str(tmp_path))
    ocr.set_backend("paddle")
    findings = scan_pdf(path, lang="fr", codes=False)
    found = "".join("".join(f.text.split()) for f in findings if f.score >= 60).lower()
    assert all("".join(t.split()).lower() in found for _, t in doc.items)
    out = str(tmp_path / "o.pdf")
    apply_redactions(path, out, [f.text for f in findings if f.score >= 60], lang="fr", verify=False, rescan=False, codes=False)
    ocr.set_backend("tesseract")                                                            # an independent engine checks the result
    ocr._ocr_cached.cache_clear()
    left = "".join(ocr.ocr_page(fitz.open(out)[0], "fr").text.split()).lower()
    assert not [t for _, t in doc.items if "".join(t.split()).lower() in left]
