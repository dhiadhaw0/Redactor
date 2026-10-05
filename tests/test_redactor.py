import pymupdf as fitz
import pytest

from redactor.detect import detect_regex, detect_ner
from redactor.make_sample import make_sample
from redactor.redact import redact_pdf

SECRETS_REGEX = [
    "john.smith@example.com", "maria.garcia@acme.org", "+44 20 7946 0958",
    "GB82 WEST 1234 5698 7654 32", "4111 1111 1111 1111",
]
SECRETS_NER = ["John Smith", "Maria Garcia", "London", "Paris"]


def text_of(path):
    return "".join(p.get_text() for p in fitz.open(path))


@pytest.fixture
def sample(tmp_path):
    p = tmp_path / "sample.pdf"
    make_sample(str(p))
    return str(p)


def test_day1_secrets_exist_before(sample):
    t = text_of(sample)
    assert all(s in t for s in SECRETS_REGEX)


def test_day2_regex_finds_all(sample):
    kinds = {f.kind for f in detect_regex(text_of(sample))}
    assert {"EMAIL", "PHONE", "IBAN", "CREDIT_CARD"} <= kinds
    texts = {f.text for f in detect_regex(text_of(sample))}
    assert set(SECRETS_REGEX) <= texts


def test_day2_rejects_invalid_numbers():
    # 16 digits failing Luhn: not labelled a card, but still flagged as a long number (OCR may have misread a digit)
    assert [f.kind for f in detect_regex("card 1234 5678 9012 3456")] == ["NUMBER"]
    # bad checksum is still flagged as an IBAN on purpose: an OCR misread or a typo must not
    # leave a half-visible account number. Dates and amounts must NOT be taken for phones.
    assert [f.kind for f in detect_regex("GB00 WEST 1234 5698 7654 32")] == ["IBAN"]
    for not_a_phone in ["18/09/2026", "2026-09-18", "24 000 000", "1 240,00 €"]:
        assert detect_regex(not_a_phone) == [], not_a_phone


def test_day3_ner_finds_names():
    texts = {f.text for f in detect_ner("Employee: John Smith lives in London.")}
    assert "John Smith" in texts and "London" in texts


def test_day1_regex_only_redaction_removes_text(sample, tmp_path):
    out = str(tmp_path / "out.pdf")
    redact_pdf(sample, out, use_ner=False)
    t = text_of(out)
    for s in SECRETS_REGEX:
        assert s not in t
    assert "EMPLOYMENT CONTRACT" in t  # non-sensitive text survives


def test_day3_full_redaction_masks_names(sample, tmp_path):
    out = str(tmp_path / "out.pdf")
    redact_pdf(sample, out)
    t = text_of(out)
    for s in SECRETS_REGEX + SECRETS_NER:
        assert s not in t, s


def test_gap_fixes_title_and_address_caught(sample, tmp_path):
    out = str(tmp_path / "out.pdf")
    redact_pdf(sample, out)
    t = text_of(out)
    assert "Mr. Smith" not in t and "Baker Street" not in t


def test_gap_fixes_label_not_redacted(sample, tmp_path):
    out = str(tmp_path / "out.pdf")
    redact_pdf(sample, out)
    assert "Email:" in text_of(out)


def test_apply_respects_rejected_terms(sample, tmp_path):
    from redactor.redact import apply_redactions
    out = str(tmp_path / "out.pdf")
    apply_redactions(sample, out, ["john.smith@example.com"])
    t = text_of(out)
    assert "john.smith@example.com" not in t
    assert "maria.garcia@acme.org" in t  # not approved -> untouched


# ---- OCR (scanned PDFs) ----
from redactor.make_sample import make_scanned_sample  # noqa: E402
from redactor.ocr import find_tessdata  # noqa: E402

needs_tesseract = pytest.mark.skipif(find_tessdata() is None, reason="Tesseract not installed")


def ocr_text_of(path):
    from redactor.ocr import find_tessdata
    out = ""
    for page in fitz.open(path):
        tp = page.get_textpage_ocr(language="eng", dpi=300, full=True, tessdata=find_tessdata())
        out += page.get_text(textpage=tp)
    return out


@pytest.fixture
def scanned(tmp_path):
    p = tmp_path / "scan.pdf"
    make_scanned_sample(str(p))
    return str(p)


@needs_tesseract
def test_scan_has_no_text_layer(scanned):
    assert text_of(scanned).strip() == ""


@needs_tesseract
def test_ocr_scan_is_detected_and_pixels_removed(scanned, tmp_path):
    out = str(tmp_path / "out.pdf")
    findings = redact_pdf(scanned, out)
    kinds = {f.kind for f in findings}
    assert {"EMAIL", "PERSON"} <= kinds
    after = ocr_text_of(out)  # re-OCR the result: secrets must be gone from the pixels
    assert "john.smith" not in after and "maria.garcia" not in after
    assert "EMPLOYMENT" in after  # rest of the page survives


# ---- French / Arabic ----
from redactor.make_sample import SAMPLES  # noqa: E402
from redactor.redact import preview_pages  # noqa: E402


def make(tmp_path, lang, scanned=False):
    p = str(tmp_path / f"{lang}{'_scan' if scanned else ''}.pdf")
    (make_scanned_sample if scanned else make_sample)(p, lang)
    return p


def test_french_regex_and_titles():
    kinds = {(f.kind, f.text) for f in detect_regex("\n".join(SAMPLES["fr"]))}
    for expected in [
        ("EMAIL", "jean.dupont@exemple.fr"), ("IBAN", "FR14 2004 1010 0505 0001 3M02 606"),
        ("PHONE", "+33 1 42 68 53 00"), ("ADDRESS", "12 rue de la Paix"), ("PERSON", "M. Dupont"),
    ]:
        assert expected in kinds


def test_french_end_to_end(tmp_path):
    src, out = make(tmp_path, "fr"), str(tmp_path / "out.pdf")
    redact_pdf(src, out, lang="fr")
    t = text_of(out)
    for secret in ["jean.dupont@exemple.fr", "Jean Dupont", "Marie Curie", "rue de la Paix", "M. Dupont", "Lyon"]:
        assert secret not in t, secret
    assert "CONTRAT DE TRAVAIL" in t


def test_arabic_regex_and_titles():
    kinds = {(f.kind, f.text) for f in detect_regex("\n".join(SAMPLES["ar"]))}
    assert ("EMAIL", "mohamed.benali@example.com") in kinds
    assert ("PERSON", "السيد بن علي") in kinds
    assert any(k == "ADDRESS" and "شارع الحبيب بورقيبة" in t for k, t in kinds)


def test_arabic_indic_digits_phone():
    assert ("PHONE", "٢١٦ ٧١ ٢٣٤ ٥٦٧") in {(f.kind, f.text) for f in detect_regex("هاتف: ٢١٦ ٧١ ٢٣٤ ٥٦٧")}


@needs_tesseract
def test_arabic_end_to_end_via_ocr(tmp_path):
    src, out = make(tmp_path, "ar"), str(tmp_path / "out.pdf")
    redact_pdf(src, out, lang="ar")
    t = text_of(out)
    assert "example.com" not in t and "acme.org" not in t and "GB82" not in t


@needs_tesseract
def test_french_scan(tmp_path):
    src, out = make(tmp_path, "fr", scanned=True), str(tmp_path / "out.pdf")
    redact_pdf(src, out, lang="fr")
    from redactor.ocr import ocr_page
    page = fitz.open(out)[0]
    after = ocr_page(page, "fr").text
    assert "dupont" not in after.lower() and "marie.curie" not in after.lower()
    assert "CONTRAT" in after


def test_preview_does_not_modify_source(sample):
    before = open(sample, "rb").read()
    pages = preview_pages(sample, ["john.smith@example.com"])
    assert len(pages) == 1 and pages[0][:4] == b"\x89PNG"
    assert open(sample, "rb").read() == before


# ---- manual boxes + image input ----
from redactor.boxes import box_to_row, drag_to_box, rows_to_boxes  # noqa: E402
from redactor.loader import to_pdf  # noqa: E402
from redactor.redact import apply_redactions  # noqa: E402


def test_drag_to_box_normalises_and_orders():
    drag = {"x1": 320, "y1": 100, "x2": 64, "y2": 20, "width": 640, "height": 800}
    assert drag_to_box(drag, 0) == (0, 0.1, 0.025, 0.5, 0.125)


def test_drag_ignores_clicks_and_garbage():
    assert drag_to_box({"x": 5, "y": 5}, 0) is None
    assert drag_to_box({"x1": 10, "y1": 10, "x2": 11, "y2": 11, "width": 640, "height": 800}, 0) is None
    assert drag_to_box(None, 0) is None


def test_rows_roundtrip_and_skip_incomplete():
    box = (1, 0.1, 0.2, 0.3, 0.4)
    assert rows_to_boxes([box_to_row(box), {"page": 1}]) == [box]


def test_manual_box_removes_text_underneath(sample, tmp_path):
    out = str(tmp_path / "out.pdf")
    page = fitz.open(sample)[0]
    r = page.search_for("Acme Corp")[0]
    w, h = page.rect.width, page.rect.height
    box = (0, r.x0 / w, r.y0 / h, r.x1 / w, r.y1 / h)
    apply_redactions(sample, out, [], boxes=[box])
    t = text_of(out)
    assert "Acme Corp" not in t and "EMPLOYMENT CONTRACT" in t


def test_manual_box_preview_marks_area(sample):
    from redactor.redact import preview_pages
    plain = preview_pages(sample, [])[0]
    boxed = preview_pages(sample, [], boxes=[(0, 0.1, 0.1, 0.5, 0.2)])[0]
    assert plain != boxed


@needs_tesseract
def test_photo_input_is_wrapped_and_redacted(scanned, tmp_path):
    img = str(tmp_path / "photo.png")
    fitz.open(scanned)[0].get_pixmap(dpi=200).save(img)
    pdf = to_pdf(img, str(tmp_path / "photo.pdf"))
    assert len(fitz.open(pdf)) == 1
    out = str(tmp_path / "out.pdf")
    redact_pdf(pdf, out)
    assert "john.smith" not in ocr_text_of(out)


def test_unsupported_type_rejected(tmp_path):
    f = tmp_path / "x.docx"
    f.write_bytes(b"x")
    with pytest.raises(ValueError):
        to_pdf(str(f), str(tmp_path / "o.pdf"))


def test_card_with_ocr_misread_digit_still_flagged():
    kinds = [f.kind for f in detect_regex("Card: 4111 1111 1111 1112")]  # Luhn-invalid
    assert "CREDIT_CARD" not in kinds and "NUMBER" in kinds


def test_card_with_dropped_space_is_a_card():
    assert [(f.kind, f.text) for f in detect_regex("Card: 4111 111111111111")] == [("CREDIT_CARD", "4111 111111111111")]


@needs_tesseract
def test_unreadable_scan_triggers_warning(tmp_path):
    from redactor.redact import scan_pdf
    doc = fitz.open()
    page = doc.new_page()
    white = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 200, 280), False)
    white.clear_with(255)
    page.insert_image(page.rect, pixmap=white)
    blank = str(tmp_path / "blank.pdf")
    doc.save(blank)
    warns = []
    assert scan_pdf(blank, warnings=warns) == []
    assert len(warns) == 1 and "Page 1" in warns[0]


# ---- hidden content + verification ----
from redactor.verify import describe_hidden, inspect_hidden, verify_output  # noqa: E402


@pytest.fixture
def booby_trapped(tmp_path):
    """A PDF that hides personal data everywhere except in the visible text."""
    path = str(tmp_path / "trap.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Public report", fontsize=12)
    page.insert_text((72, 120), "hidden.person@leak.org", fontsize=12, render_mode=3)  # invisible text
    page.add_text_annot((300, 300), "call John Smith on 0612345678")                   # comment
    page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(72, 60, 150, 80), "uri": "https://leak.example/?u=jsmith"})
    doc.embfile_add("notes.txt", b"John Smith lives at 12 Baker Street")                # attachment
    doc.set_toc([[1, "Contract of John Smith", 1]])                                      # bookmark
    doc.set_metadata({"author": "John Smith", "title": "Contract"})
    doc.save(path)
    return path


def test_inspect_finds_all_hidden_content(booby_trapped):
    r = inspect_hidden(booby_trapped)
    assert r["annotations"] >= 1 and r["attachments"] == 1 and r["bookmarks"] == 1
    assert r["links"] == 1 and r["invisible_text_spans"] >= 1
    assert "author" in r["metadata_fields"]
    assert len(describe_hidden(r)) >= 5


def test_apply_strips_all_hidden_content(booby_trapped, tmp_path):
    out = str(tmp_path / "out.pdf")
    res = apply_redactions(booby_trapped, out, [])
    r = inspect_hidden(out)
    assert r["annotations"] == 0 and r["attachments"] == 0 and r["bookmarks"] == 0 and r["links"] == 0
    assert r["invisible_text_spans"] == 0 and r["metadata_fields"] == [] and not r["xmp_metadata"]
    assert "hidden.person@leak.org" not in text_of(out)
    assert "Public report" in text_of(out)
    assert res.hidden_removed and res.verification.ok


def test_verification_passes_on_a_good_redaction(sample, tmp_path):
    out = str(tmp_path / "out.pdf")
    res = apply_redactions(sample, out, ["john.smith@example.com", "John Smith"])
    assert res.hits == 2 and res.verification.ok, res.verification.problems


def test_verification_catches_a_term_that_is_still_readable(sample):
    v = verify_output(sample, ["john.smith@example.com"], [])  # the unredacted file
    assert not v.ok and "still readable" in v.problems[0]


def test_verification_catches_a_box_that_is_not_black(sample):
    page = fitz.open(sample)[0]
    v = verify_output(sample, [], [(0, page.search_for("Acme Corp")[0])])  # text visible, nothing blacked out
    assert not v.ok and "not fully black" in v.problems[0]


def test_verification_catches_a_partly_visible_number(sample):
    # only the first half of the IBAN is "redacted" away; its other digits are still readable
    v = verify_output(sample, ["GB82 WEST 1234 5698 7654 32 00"], [])
    assert not v.ok and "partly readable" in v.problems[0]


@needs_tesseract
def test_verification_reocrs_scanned_pages(scanned, tmp_path):
    out = str(tmp_path / "out.pdf")
    findings = redact_pdf(scanned, out)
    assert apply_redactions(scanned, out, [f.text for f in findings]).verification.ok
    leaked = verify_output(scanned, ["john.smith@example.com"], [], ocr_pages={0})  # original scan
    assert not leaked.ok and "OCR'd again" in leaked.problems[0]


def test_low_confidence_findings_are_flagged():
    from redactor.detect import detect_ner
    hits = {(f.text, f.low) for f in detect_ner("Employee: John Smith lives in London.")}
    assert ("London", True) in hits and ("John Smith", False) in hits


# ---- lessons from real documents: low-res scans, labelled tables, names, references ----
from redactor.detect import Finding, detect, detect_context, detect_name_shaped, _lower_words  # noqa: E402
from redactor.ocr import _effective_dpi, _upscale_factor  # noqa: E402

LETTER = [
    "Atlantis Assurances", "", "M. Julien MARCHAND", "14 rue des Acacias", "69003 LYON", "",
    "Dossier sinistre SIN-2026-0093178", "Contrat Multirisque Habitation Confort n° HAB-2026-0731245", "",
    "Nous vous adressons une indemnité de 2 037,00 EUR par virement sur le compte",
    "FR76 1234 5000 1000 0001 2345 636 sous cinq jours ouvrés.",
    "Pour toute question appelez le 01 99 00 12 34, dossier du 24/07/2026.", "",
    "Nathalie ROBIN", "Gestionnaire Sinistres",
]
LETTER_SECRETS = ["Julien MARCHAND", "14 rue des Acacias", "69003 LYON", "SIN-2026-0093178", "HAB-2026-0731245",
                  "FR76 1234 5000 1000 0001 2345 636", "01 99 00 12 34", "Nathalie ROBIN"]


def make_letter_scan(tmp_path, dpi):
    """A letter as a low-resolution PNG, like a small scan or a screenshot."""
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in LETTER:
        page.insert_text((72, y), line, fontsize=11)
        y += 20
    png = str(tmp_path / f"letter_{dpi}.png")
    page.get_pixmap(dpi=dpi).save(png)
    return png


def test_context_rules_ocr_table_line_and_digital_two_line_layout():
    ocr_style = "Nom et prénom MARCHAND Julien\nAdresse 14 rue des Acacias, 69003 LYON\nNuméro de client 20458712"
    got = {(f.kind, f.text) for f in detect_context(ocr_style)}
    assert {("PERSON", "MARCHAND Julien"), ("ADDRESS", "14 rue des Acacias, 69003 LYON"), ("ID", "20458712")} <= got
    digital_style = "Nom et prénom\nMARCHAND Julien\nNuméro de client\n20458712"  # value on the next line
    got = {(f.kind, f.text) for f in detect_context(digital_style)}
    assert {("PERSON", "MARCHAND Julien"), ("ID", "20458712")} <= got


def test_context_rules_ignore_empty_or_nonsense_values():
    assert detect_context("Nom\n\n\n") == []
    assert detect_context("Numéro de client : à renseigner") == []  # no digits -> not an ID


def test_name_shaped_only_on_short_lines():
    lower = _lower_words("")
    assert [f.text for f in detect_name_shaped("Nathalie ROBIN", lower)] == ["Nathalie ROBIN"]
    assert detect_name_shaped("Pour le suivi du dossier Nathalie ROBIN vous répondra", lower) == []
    assert detect_name_shaped("Atlantis ASSURANCES", lower) == []  # organisation


def test_names_propagate_across_spellings():
    kinds = {(f.kind, f.text) for f in detect("M. Julien MARCHAND\nplus loin\nNom et prénom MARCHAND Julien", lang="fr")}
    assert ("PERSON", "MARCHAND") in kinds and ("PERSON", "Julien") in kinds


def test_common_words_are_not_places():
    text = "Les frais réels sont couverts.\nFrais de dossier: voir contrat.\nle contrat est signé\nContrat n° 5"
    got = {f.text for f in detect(text, lang="fr")}
    assert "Frais" not in got and "Contrat" not in got


def test_emails_do_not_make_names_look_like_common_words():
    got = {f.text for f in detect("Contact: Maria Garcia\nmaria.garcia@acme.org", lang="en")}
    assert "Maria Garcia" in got


def test_reference_and_iban_shapes():
    kinds = {(f.kind, f.text) for f in detect_regex(
        "Dossier SIN-2026-0093178 contrat HAB-2026-0731245 compte FR76 1234 5000 1000 0001 2345 636")}
    assert {("REFERENCE", "SIN-2026-0093178"), ("REFERENCE", "HAB-2026-0731245"),
            ("IBAN", "FR76 1234 5000 1000 0001 2345 636")} <= kinds


def test_low_resolution_scan_is_upscaled_for_ocr(tmp_path):
    low = to_pdf(make_letter_scan(tmp_path, 70), str(tmp_path / "low.pdf"))
    high = to_pdf(make_letter_scan(tmp_path, 300), str(tmp_path / "high.pdf"))
    assert _effective_dpi(fitz.open(low)[0]) < 100 and _upscale_factor(fitz.open(low)[0]) > 2
    assert _upscale_factor(fitz.open(high)[0]) == 1  # already sharp enough


@needs_tesseract
@pytest.mark.parametrize("dpi", [70, 110])
def test_low_resolution_letter_end_to_end(tmp_path, dpi):
    pdf = to_pdf(make_letter_scan(tmp_path, dpi), str(tmp_path / "in.pdf"))
    out = str(tmp_path / "out.pdf")
    findings = scan_pdf_for_test(pdf)
    found_blob = "".join("".join(f.text.split()) for f in findings).lower()
    for secret in LETTER_SECRETS:
        assert "".join(secret.split()).lower() in found_blob, f"not found at {dpi} dpi: {secret}"
    res = apply_redactions(pdf, out, [f.text for f in findings], lang="fr")
    assert res.verification.ok, res.verification.problems
    # the secrets must be unreadable in the pixels of the result
    after = "".join(ocr_text_of(out).split()).lower()
    for secret in ("MARCHAND", "SIN-2026", "HAB-2026", "FR761234", "Acacias"):
        assert secret.lower() not in after, secret


def scan_pdf_for_test(pdf, lang="fr"):
    from redactor.redact import scan_pdf
    return scan_pdf(pdf, lang=lang)


# ---- OCR cache and word-based search ----
from redactor.ocr import OcrPage, _ocr_cached, ocr_page  # noqa: E402


def test_ocr_search_covers_whole_words_one_rect_per_line():
    words = [(10, 10, 40, 20, "14", 0, 0, 0), (45, 10, 70, 20, "rue", 0, 0, 1), (75, 10, 140, 20, "Acacias,", 0, 0, 2),
             (10, 30, 60, 40, "69003", 0, 1, 0), (65, 30, 100, 40, "LYON", 0, 1, 1)]
    page = OcrPage("14 rue Acacias,\n69003 LYON", words)
    rects = page.search("14 rue des Acacias".replace(" des", ""))  # 'Acacias' word matched whole, with its comma
    assert len(rects) == 1 and rects[0].x1 == 140
    two_lines = page.search("Acacias, 69003 LYON")
    assert len(two_lines) == 2 and two_lines[0].y0 == 10 and two_lines[1].y0 == 30
    assert page.search("nothing") == [] and page.search("") == []


def test_ocr_search_is_case_and_space_insensitive():
    page = OcrPage("Nathalie ROBIN", [(0, 0, 50, 10, "Nathalie", 0, 0, 0), (55, 0, 90, 10, "ROBIN", 0, 0, 1)])
    assert len(page.search("nathalie   robin")) == 1


@needs_tesseract
def test_ocr_results_are_cached_per_unchanged_file(scanned):
    _ocr_cached.cache_clear()
    page = fitz.open(scanned)[0]
    first, second = ocr_page(page), ocr_page(page)
    assert first is second and _ocr_cached.cache_info().hits == 1
    assert "john.smith@example.com" in first.text


# ---- rotated pages and EXIF-rotated photos ----
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_rotated_pdf_page_redaction_and_verification(sample, tmp_path, rotation):
    rotated = str(tmp_path / "rot.pdf")
    d = fitz.open(sample)
    d[0].set_rotation(rotation)
    d.save(rotated)
    out = str(tmp_path / "out.pdf")
    res = apply_redactions(rotated, out, ["john.smith@example.com"])
    assert res.hits == 1 and res.verification.ok, res.verification.problems
    assert "john.smith" not in text_of(out) and "EMPLOYMENT" in text_of(out)


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_manual_box_follows_what_the_user_sees_on_rotated_pages(sample, tmp_path, rotation):
    rotated = str(tmp_path / "rot.pdf")
    d = fitz.open(sample)
    d[0].set_rotation(rotation)
    d.save(rotated)
    page = fitz.open(rotated)[0]
    title = fitz.Rect(page.search_for("EMPLOYMENT CONTRACT")[0] * page.rotation_matrix).normalize()  # as displayed
    w, h = page.rect.width, page.rect.height
    box = (0, title.x0 / w, title.y0 / h, title.x1 / w, title.y1 / h)
    out = str(tmp_path / "out.pdf")
    res = apply_redactions(rotated, out, [], boxes=[box])
    t = text_of(out)
    assert res.verification.ok and "EMPLOYMENT" not in t and "Employee" in t  # only the title went


def make_rotated_scan(path, rotation):
    """Image stored sideways, /Rotate set so a viewer shows it upright (common scanner output)."""
    import io
    from PIL import Image
    from redactor.make_sample import _draw
    up = fitz.open()
    pg = up.new_page()
    _draw(pg, "en")
    img = Image.open(io.BytesIO(pg.get_pixmap(dpi=200).tobytes("png"))).convert("RGB")
    buf = io.BytesIO()
    img.rotate(rotation, expand=True).save(buf, "PNG")
    d = fitz.open()
    p = d.new_page(width=595 if rotation in (0, 180) else 842, height=842 if rotation in (0, 180) else 595)
    p.insert_image(p.rect, stream=buf.getvalue())
    p.set_rotation(rotation)
    d.save(path)


@needs_tesseract
@pytest.mark.parametrize("rotation", [90, 180])
def test_rotated_scan_is_read_and_redacted_in_place(tmp_path, rotation):
    src, out = str(tmp_path / "scan.pdf"), str(tmp_path / "out.pdf")
    make_rotated_scan(src, rotation)
    findings = scan_pdf_for_test(src, "en")
    assert any(f.kind == "EMAIL" for f in findings)
    res = apply_redactions(src, out, [f.text for f in findings])
    assert res.verification.ok, res.verification.problems
    assert "john.smith" not in ocr_text_of(out).lower().replace(" ", "")


def test_exif_rotated_photo_is_straightened(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (400, 200), "white")  # stored landscape...
    exif = Image.Exif()
    exif[0x0112] = 6                              # ...but flagged "rotate 90 degrees to view"
    path = str(tmp_path / "photo.jpg")
    img.save(path, exif=exif)
    pdf = to_pdf(path, str(tmp_path / "photo.pdf"))
    r = fitz.open(pdf)[0].rect
    assert r.height > r.width  # displayed portrait, not sideways


def test_photo_without_exif_flag_unchanged(tmp_path):
    from PIL import Image
    path = str(tmp_path / "plain.png")
    Image.new("RGB", (400, 200), "white").save(path)
    r = fitz.open(to_pdf(path, str(tmp_path / "plain.pdf")))[0].rect
    assert r.width > r.height


def make_sideways_scan(path, quarter_turns):
    """Pixels stored sideways/upside down and NO /Rotate flag: what some scanner apps produce."""
    import io
    from PIL import Image
    from redactor.make_sample import _draw
    up = fitz.open()
    pg = up.new_page()
    _draw(pg, "en")
    img = Image.open(io.BytesIO(pg.get_pixmap(dpi=200).tobytes("png"))).convert("RGB")
    buf = io.BytesIO()
    img.rotate(-90 * quarter_turns, expand=True).save(buf, "PNG")
    d = fitz.open()
    p = d.new_page(width=842 if quarter_turns % 2 else 595, height=595 if quarter_turns % 2 else 842)
    p.insert_image(p.rect, stream=buf.getvalue())
    d.save(path)


@needs_tesseract
@pytest.mark.parametrize("turns", [1, 2, 3])
def test_sideways_scan_is_auto_oriented_and_redacted_in_place(tmp_path, turns):
    src, out = str(tmp_path / "side.pdf"), str(tmp_path / "out.pdf")
    make_sideways_scan(src, turns)
    findings = scan_pdf_for_test(src, "en")
    assert any(f.kind == "EMAIL" and "john.smith" in f.text for f in findings), [f.text for f in findings]
    res = apply_redactions(src, out, [f.text for f in findings])
    assert res.verification.ok, res.verification.problems
    # the boxes must sit on the text: after redaction a fresh OCR of the output cannot find it
    from redactor.ocr import ocr_page
    assert "john.smith" not in "".join(ocr_page(fitz.open(out)[0]).text.split()).lower()


@needs_tesseract
def test_blank_scan_does_not_invent_orientation(tmp_path):
    white = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 300, 400), False)
    white.clear_with(255)
    d = fitz.open()
    pg = d.new_page()
    pg.insert_image(pg.rect, pixmap=white)
    path = str(tmp_path / "blank.pdf")
    d.save(path)
    from redactor.ocr import ocr_page
    assert ocr_page(fitz.open(path)[0]).words == []


# ---- command line ----
import json  # noqa: E402
from redactor.cli import main as cli_main  # noqa: E402


def test_cli_single_file(sample, tmp_path, capsys):
    out = str(tmp_path / "out.pdf")
    assert cli_main([sample, out, "--no-ocr"]) == 0
    assert "verification passed" in capsys.readouterr().out
    assert "john.smith@example.com" not in text_of(out)


def test_cli_batch_writes_audit_without_personal_text(sample, tmp_path):
    src_dir, out_dir = tmp_path / "in", tmp_path / "out"
    src_dir.mkdir()
    for name in ("a.pdf", "b.pdf"):
        make_sample(str(src_dir / name))
    (src_dir / "notes.bin").write_bytes(b"ignored")
    assert cli_main([str(src_dir), str(out_dir), "--no-ocr"]) == 0
    assert (out_dir / "a_redacted.pdf").exists() and (out_dir / "b_redacted.pdf").exists()
    raw = (out_dir / "audit.json").read_text(encoding="utf-8")
    audit = json.loads(raw)
    assert [f["file"] for f in audit["files"]] == ["a.pdf", "b.pdf"]
    assert all(f["verification_passed"] and f["items_by_type"]["EMAIL"] == 2 for f in audit["files"])
    assert len(audit["files"][0]["sha256_of_input"]) == 64
    for secret in ("john.smith", "Garcia", "4111", "GB82"):
        assert secret not in raw, secret  # the audit log must not leak what was redacted


def test_cli_skip_low_leaves_bare_places(sample, tmp_path):
    keep, skip = str(tmp_path / "keep.pdf"), str(tmp_path / "skip.pdf")
    cli_main([sample, keep, "--no-ocr"])
    cli_main([sample, skip, "--no-ocr", "--skip-low"])
    assert "Paris" not in text_of(keep) and "Paris" in text_of(skip)


def test_cli_exit_code_signals_failure(tmp_path, capsys):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not a pdf")
    assert cli_main([str(bad), str(tmp_path / "o.pdf")]) == 1
    assert "NOT verified" in capsys.readouterr().out


def test_cli_empty_folder(tmp_path):
    (tmp_path / "in").mkdir()
    assert cli_main([str(tmp_path / "in"), str(tmp_path / "out")]) == 2


# ---- lessons from a real CV: handles, contact block, profile links, headings ----
CV = """Dhia Eddine Toumi
Tunis, Tunisia | +216 29 045 129 | dhiaeddinetoumi7@gmail.com | linkedin.com/in/dhia-eddine-toumi-9759181ab | github.com/dhiadhaw0
PROFIL
Etudiant en troisième année à l'Ecole Supérieure d'Ingénierie, spécialité Informatique Financière.
EXPERIENCE PROFESSIONNELLE
Plateforme Recrutement – ArabSoft | Jan 2025 - 2025
Développement avec Spring Boot et Docker. Projet de Monte Carlo.
2023 - 2025 : Cycle ingénieur
"""


def cv_findings():
    return {(f.kind, f.text, f.low) for f in detect(CV, lang="fr")}


def test_cv_name_is_found_through_email_and_profile_handles():
    found = {(k, t) for k, t, _ in cv_findings()}
    assert ("PERSON", "Dhia Eddine Toumi") in found


def test_cv_contact_details_and_profile_links():
    found = {(k, t) for k, t, _ in cv_findings()}
    assert {("EMAIL", "dhiaeddinetoumi7@gmail.com"), ("PHONE", "+216 29 045 129"), ("ADDRESS", "Tunis, Tunisia"),
            ("URL", "linkedin.com/in/dhia-eddine-toumi-9759181ab"), ("URL", "github.com/dhiadhaw0")} <= found


def test_cv_year_ranges_are_not_phone_numbers():
    assert not any(k == "PHONE" and t != "+216 29 045 129" for k, t, _ in cv_findings())
    assert detect_regex("2023 - 2025") == [] and detect_regex("Jan 2025 - 2025") == []


def test_cv_name_pieces_are_redacted_wherever_they_occur():
    texts = {t for k, t, low in cv_findings() if k == "PERSON" and not low}
    assert {"Dhia", "Eddine", "Toumi"} <= texts  # even though they also sit inside the email and URL


def test_cv_headings_and_skills_are_not_pre_selected():
    selected = {t for k, t, low in cv_findings() if not low}
    for noise in ("Informatique Financière", "Spring Boot", "Docker", "Plateforme Recrutement", "Ingénierie"):
        assert noise not in selected, noise


def test_person_only_vouched_for_by_ner_is_still_found_in_prose():
    got = {(f.kind, f.text, f.low) for f in detect("Hier, Maria Garcia a signé le contrat avec Jean Dupont.", lang="fr")}
    assert ("PERSON", "Maria Garcia", False) in got or ("PERSON", "Jean Dupont", False) in got


def test_postcode_and_city_line_is_an_address():
    got = {(f.kind, f.text) for f in detect_regex("M. Julien MARCHAND\n14 rue des Acacias\n69003 LYON\nParis, le 18/09/2026")}
    assert ("ADDRESS", "14 rue des Acacias") in got and ("ADDRESS", "69003 LYON") in got
    assert detect_regex("Le total est de 12345 euros") == []  # a number followed by lower-case text is not an address


# ---- confidence scores ----
from redactor import scoring as sc  # noqa: E402


def scored(text, lang="en", **kw):
    return {(f.kind, f.text): f.score for f in detect(text, lang=lang, **kw)}


def test_scores_are_between_0_and_100_and_ordered_by_certainty():
    got = scored("Email: a.b@x.org\nIBAN: GB82 WEST 1234 5698 7654 32\nIBAN: GB00 WEST 1234 5698 7654 32\n"
                 "Phone: +44 20 7946 0958\nHistory: Spring Boot and Leadership")
    assert all(0 <= v <= 100 for v in got.values())
    assert got[("IBAN", "GB82 WEST 1234 5698 7654 32")] > got[("IBAN", "GB00 WEST 1234 5698 7654 32")]  # checksum
    assert got[("EMAIL", "a.b@x.org")] >= 95
    assert got[("PHONE", "+44 20 7946 0958")] > sc.PHONE


def test_ner_only_findings_score_below_rule_findings():
    got = scored("Hier, Maria Garcia a rencontré le client à Lyon.\nM. Dupont confirme.", lang="fr")
    assert got[("PERSON", "M. Dupont")] > got[("PERSON", "Maria Garcia")] >= sc.DEFAULT_THRESHOLD
    assert got.get(("LOCATION", "Lyon"), 0) < sc.DEFAULT_THRESHOLD  # a bare place: listed, not pre-selected


def test_name_pieces_score_below_their_parent():
    got = scored("M. Julien MARCHAND\nNom et prénom MARCHAND Julien", lang="fr")
    parent = got[("PERSON", "M. Julien MARCHAND")]
    assert got[("PERSON", "MARCHAND")] < parent and got[("PERSON", "Julien")] < parent
    # a piece never outscores the best name it was taken from, even when NER also saw the name
    got = scored("Contact: M. Julien Marchand")
    assert got[("PERSON", "Marchand")] < got[("PERSON", "M. Julien Marchand")]


def test_agreement_between_sources_raises_the_score():
    text = "Name: Maria Garcia"                    # a label rule AND the NER model both see this name
    rule_only = scored(text, use_ner=False)[("PERSON", "Maria Garcia")]
    both = scored(text)[("PERSON", "Maria Garcia")]
    assert rule_only == sc.LABELLED and both == sc.LABELLED + sc.CORROBORATED


def test_uncertain_text_that_repeats_scores_higher():
    once = scored("Il a rencontré Maria Garcia hier.", lang="fr")
    thrice = scored("Il a rencontré Maria Garcia hier. Maria Garcia a signé. Puis Maria Garcia est partie.", lang="fr")
    assert thrice[("PERSON", "Maria Garcia")] > once[("PERSON", "Maria Garcia")]


def test_ocr_reading_costs_a_few_points(sample, scanned):
    from redactor.redact import scan_pdf
    digital = {(f.kind, f.text): f.score for f in scan_pdf(sample, use_ner=False)}
    via_ocr = {(f.kind, f.text): f.score for f in scan_pdf(scanned, use_ner=False)}
    assert digital[("EMAIL", "john.smith@example.com")] == sc.EMAIL
    assert via_ocr[("EMAIL", "john.smith@example.com")] == sc.EMAIL + sc.OCR


def test_low_is_derived_from_the_score():
    assert Finding("EMAIL", "a@b.co", 98).low is False
    assert Finding("LOCATION", "Lyon", 30).low is True
    assert Finding("PERSON", "X Y", sc.DEFAULT_THRESHOLD).low is False


def test_score_bands():
    assert [sc.band(s) for s in (99, 90, 70, 30)] == ["very high", "high", "medium", "low"]


def test_cli_min_score_filters_and_is_audited(sample, tmp_path):
    out = str(tmp_path / "out.pdf")
    audit = str(tmp_path / "a.json")
    assert cli_main([sample, out, "--no-ocr", "--min-score", "95", "--audit", audit]) == 0
    rec = json.loads(open(audit, encoding="utf-8").read())["files"][0]
    assert rec["min_score"] == 95 and rec["score_range_of_redacted"][0] >= 95 and rec["skipped_below_min_score"] > 0
    t = text_of(out)
    assert "john.smith@example.com" not in t and "Acme Corp" in t and "4111" not in t  # emails/cards/IBAN gone
    assert "+44 20 7946 0958" in t  # phone scores 92 < 95: left alone


def test_ui_lists_findings_best_first_with_a_preselection_threshold(sample):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120)
    at.run()
    at.file_uploader[0].upload("s.pdf", open(sample, "rb").read(), "application/pdf")
    at.run()
    assert not at.exception
    findings = at.session_state["findings"]
    assert [f.score for f in findings] == sorted((f.score for f in findings), reverse=True)  # best first
    assert 0 < sum(f.score >= sc.DEFAULT_THRESHOLD for f in findings) <= len(findings)
    assert at.slider[0].value == sc.DEFAULT_THRESHOLD


# ---- empty labels, residual detection ----
def test_empty_field_followed_by_another_label_is_not_a_value():
    assert detect_context("Address: \nEmail: \nPhone: ") == []
    assert detect_context("Nom :\nPrénom :") == []
    assert [f.text for f in detect_context("Adresse\nNuméro de client\n20458712")] == ["20458712"]
    assert [f.text for f in detect_context("Employee: \nAddress: 12 Baker Street")] == ["12 Baker Street"]


def test_a_redacted_page_does_not_invent_findings_from_its_empty_labels(sample, tmp_path):
    from redactor.redact import scan_pdf
    out = str(tmp_path / "out.pdf")
    findings = scan_pdf(sample)
    res = apply_redactions(sample, out, [f.text for f in findings if f.score >= sc.DEFAULT_THRESHOLD])
    assert not [f for f in res.residual if f.kind in ("EMAIL", "IBAN", "CREDIT_CARD", "PHONE")]
    assert not any(f.kind == "ADDRESS" and "Email" in f.text for f in res.residual)


# ---- document risk score ----
from redactor import risk as rk  # noqa: E402


def F(kind, text, score=95):
    return Finding(kind, text, score)


def test_risk_of_nothing_is_zero_and_none():
    r = rk.assess([])
    assert (r.score, r.level) == (0, "none") and r.by_kind == [] and r.bonuses == []


def test_risk_grows_with_more_or_more_dangerous_findings():
    email = rk.assess([F("EMAIL", "a@b.co")])
    iban = rk.assess([F("IBAN", "GB82 WEST 1234 5698 7654 32", 99)])
    both = rk.assess([F("EMAIL", "a@b.co"), F("IBAN", "GB82 WEST 1234 5698 7654 32", 99)])
    assert 0 < email.score < iban.score < both.score  # an IBAN is worse than an email; adding never lowers it


def test_risk_scales_with_confidence():
    sure = rk.assess([F("PERSON", "Maria Garcia", 95)]).score
    unsure = rk.assess([F("PERSON", "Maria Garcia", 35)]).score
    assert 0 < unsure < sure


def test_repeats_count_for_less_and_are_bounded():
    one = rk.assess([F("EMAIL", "a1@b.co")])
    ten = rk.assess([F("EMAIL", f"a{i}@b.co") for i in range(10)])
    assert one.score < ten.score < 2 * rk.WEIGHTS["EMAIL"] + 1  # geometric decay: never more than ~2x the weight


def test_pieces_of_a_name_are_not_counted_as_extra_people():
    whole = rk.assess([F("PERSON", "Dhia Eddine Toumi")])
    with_pieces = rk.assess([F("PERSON", "Dhia Eddine Toumi"), F("PERSON", "Toumi", 85), F("PERSON", "Dhia", 85)])
    assert whole.score == with_pieces.score and with_pieces.by_kind[0][1] == 1


def test_identifiability_bonuses():
    name = F("PERSON", "Maria Garcia")
    assert rk.assess([name]).bonuses == []
    assert [b for _, b in rk.assess([name, F("EMAIL", "m@g.co")]).bonuses] == [rk.BONUS_NAME_WITH_DETAILS]
    two = rk.assess([name, F("EMAIL", "m@g.co"), F("PHONE", "+33 1 42 68 53 00")]).bonuses
    assert sum(b for _, b in two) == rk.BONUS_NAME_WITH_DETAILS + rk.BONUS_NAME_WITH_MANY_DETAILS
    money = rk.assess([name, F("IBAN", "GB82 WEST 1234 5698 7654 32", 99)])
    assert ("financial data tied to a named person", rk.BONUS_FINANCIAL_WITH_NAME) in money.bonuses
    assert rk.assess([F("IBAN", "GB82 WEST 1234 5698 7654 32", 99)]).bonuses == []  # no name, no bonus


def test_a_low_confidence_name_does_not_trigger_a_bonus():
    r = rk.assess([F("PERSON", "Spring Boot", 40), F("EMAIL", "info@x.co")])
    assert r.bonuses == []


def test_risk_is_capped_and_levels_match_scores():
    big = rk.assess([F(k, f"{k}{i}", 99) for k in rk.WEIGHTS for i in range(3)])
    assert big.score == 100 and big.level == "critical"
    assert [rk.level_for(s) for s in (0, 1, 29, 30, 54, 55, 84, 85, 100)] == \
        ["none", "low", "low", "moderate", "moderate", "high", "high", "critical", "critical"]


def test_explanation_never_contains_the_personal_text():
    r = rk.assess([F("PERSON", "Maria Garcia"), F("EMAIL", "maria@garcia.org")])
    assert "Maria" not in r.explain() and "garcia" not in str(r.as_dict()).lower()
    assert r.as_dict()["by_kind"]["EMAIL"]["count"] == 1


def test_remaining_after_ignores_what_is_selected_or_contained_in_it():
    fs = [F("PERSON", "Dhia Eddine Toumi"), F("PERSON", "Toumi", 85), F("EMAIL", "d@t.co"), F("LOCATION", "Tunis", 30)]
    left = rk.remaining_after(fs, ["Dhia Eddine Toumi", "d@t.co"])
    assert [f.text for f in left] == ["Tunis"]  # 'Toumi' is inside a selected term: covered


def test_redaction_lowers_measured_risk(sample, tmp_path):
    from redactor.redact import scan_pdf
    findings = scan_pdf(sample)
    before = rk.assess(findings)
    res = apply_redactions(sample, str(tmp_path / "o.pdf"),
                           [f.text for f in findings if f.score >= sc.DEFAULT_THRESHOLD])
    assert before.level == "critical"
    assert res.risk_after.score < before.score / 2 and res.risk_after.level in ("none", "low", "moderate")
    assert res.risk_after.score < 55  # at most moderate once the confident findings are gone


def test_redacting_nothing_leaves_the_risk_unchanged(sample, tmp_path):
    from redactor.redact import scan_pdf
    before = rk.assess(scan_pdf(sample)).score
    res = apply_redactions(sample, str(tmp_path / "o.pdf"), [])
    assert abs(res.risk_after.score - before) <= 3


def test_cli_prints_and_audits_risk_before_and_after(sample, tmp_path, capsys):
    out, audit = str(tmp_path / "o.pdf"), str(tmp_path / "a.json")
    assert cli_main([sample, out, "--no-ocr", "--min-score", "60", "--audit", audit]) == 0
    assert "privacy risk:" in capsys.readouterr().out
    rec = json.loads(open(audit, encoding="utf-8").read())["files"][0]
    assert rec["risk_before"]["score"] > rec["risk_after"]["score"] and rec["risk_before"]["level"] == "critical"
    assert "john" not in json.dumps(rec).lower()


def test_ui_shows_before_and_measured_after_risk(sample):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=180)
    at.run()
    at.file_uploader[0].upload("s.pdf", open(sample, "rb").read(), "application/pdf")
    at.run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert labels[0] == "Original document" and "estimate" in labels[1]
    assert at.metric[0].value.endswith("/100") and at.metric[0].delta.lower() == "critical"
    [b for b in at.button if b.label == "Apply redaction"][0].click()
    at.run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "measured" in labels[1]
    assert int(at.metric[1].value.split("/")[0]) < int(at.metric[0].value.split("/")[0])


# ---- the digit-run rule must not fire on coincidences ----
def test_digit_run_rule_ignores_coincidences_across_lines_and_zeros():
    from redactor.verify import _digits, _leaks, _squash
    iban = "FR76 1234 5000 1000 0001 2345 636"
    footer = "Societe anonyme au capital de 24 000 000 EUR - RCS Paris 512 345 679\nPage 1 / 1"
    assert _leaks(iban, _squash(footer), _digits(footer)) is None            # '000000' is not the IBAN
    split = "1234500\n0100000"                                               # IBAN digits, 7 per line: no run of 8
    assert _leaks(iban, _squash(split), _digits(split)) is None              # a run cannot span lines
    real = "compte FR76 1234 5000 1000"                                      # genuinely half visible
    assert _leaks(iban, _squash(real), _digits(real)) is not None


def test_an_ocr_garbled_field_name_is_not_taken_for_a_value():
    for garbled in ("Adress", "Telefone", "Emial", "Adresse", "Nom"):
        assert detect_context(f"Employee\n{garbled}") == [], garbled
    assert [f.text for f in detect_context("Employee\nDupont")] == ["Dupont"]   # a real stacked value still works
    assert [f.text for f in detect_context("Nom\nMARCHAND")] == ["MARCHAND"]
