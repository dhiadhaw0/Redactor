"""Word, Excel, PowerPoint, e-mail and text files: converted to PDF, with what is dropped reported."""
import io
import zipfile
from email.message import EmailMessage

import pymupdf as fitz
import pytest
from PIL import Image, ImageDraw

from redactor import office
from redactor.cli import main as cli_main
from redactor.loader import SUPPORTED_SUFFIXES, convert, is_supported
from redactor.ocr import find_tessdata
from redactor.redact import apply_redactions, scan_pdf

docx = pytest.importorskip("docx")
openpyxl = pytest.importorskip("openpyxl")
pptx = pytest.importorskip("pptx")

IBAN = "FR14 2004 1010 0505 0001 3M02 606"
needs_tesseract = pytest.mark.skipif(find_tessdata() is None, reason="Tesseract not installed")


def pdf_text(path):
    return " ".join(p.get_text(flags=fitz.TEXT_PRESERVE_WHITESPACE) for p in fitz.open(path))


def to_pdf_text(path, tmp_path):
    out = str(tmp_path / "converted.pdf")
    dropped = convert(str(path), out)
    return pdf_text(out), dropped, out


def rewrite_zip(path, edit):
    """Copy a zip, letting `edit(name, data) -> bytes | None` change or drop parts, and add new ones via edit(None, None)."""
    src = zipfile.ZipFile(path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out:
        for item in src.infolist():
            data = edit(item.filename, src.read(item.filename))
            if data is not None:
                out.writestr(item.filename, data)
        extra = edit(None, None)
        for name, data in (extra or {}).items():
            out.writestr(name, data)
    src.close()
    open(path, "wb").write(buf.getvalue())


# ------------------------------------------------------------------ Word
@pytest.fixture
def word_file(tmp_path):
    d = docx.Document()
    d.core_properties.author = "Julien Marchand"
    d.core_properties.last_modified_by = "RH Atlantis"
    d.sections[0].header.paragraphs[0].text = "Confidentiel - Atlantis Assurances"
    d.sections[0].footer.paragraphs[0].text = "Footer: contact@atlantis.example"
    d.add_heading("Contrat de travail", 1)
    d.add_paragraph("Nom : Julien Marchand, demeurant 14 rue des Acacias, 69003 LYON.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "IBAN", IBAN
    t.cell(1, 0).text, t.cell(1, 1).text = "Telephone", "+33 6 12 34 56 78"
    p = d.add_paragraph("Visible text. ")
    p.add_run("SECRET HIDDEN TEXT").font.hidden = True
    path = tmp_path / "contract.docx"
    d.save(path)
    return path


def test_word_text_tables_headers_and_footers_are_kept(word_file, tmp_path):
    text, _, _ = to_pdf_text(word_file, tmp_path)
    for kept in ("Contrat de travail", "Julien Marchand", "14 rue des Acacias", IBAN, "+33 6 12 34 56 78",
                 "[Header] Confidentiel", "[Footer] Footer: contact@atlantis.example", "Visible text."):
        assert kept in text, kept
    assert "IBAN | " + IBAN in text                 # a table row stays a row


def test_word_hidden_text_and_properties_are_dropped_and_reported(word_file, tmp_path):
    text, dropped, _ = to_pdf_text(word_file, tmp_path)
    assert "SECRET HIDDEN TEXT" not in text
    assert any("hidden text" in d for d in dropped)
    assert any("document properties" in d and "creator" in d and "lastModifiedBy" in d for d in dropped)


def test_word_tracked_deletions_and_comments_are_dropped(word_file, tmp_path):
    def edit(name, data):
        if name == "word/document.xml":
            xml = data.decode("utf-8")
            deleted = ('<w:del w:id="9" w:author="Julien"><w:r><w:delText>DELETED SECRET SALARY</w:delText></w:r></w:del>'
                       '<w:ins w:id="10" w:author="Julien"><w:r><w:t>inserted visible words</w:t></w:r></w:ins>')
            return xml.replace("</w:body>", f"<w:p>{deleted}</w:p></w:body>", 1).encode("utf-8")
        if name is None:
            return {"word/comments.xml": '<w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                    '<w:comment w:id="0" w:author="HR"><w:p><w:r><w:t>COMMENT SECRET</w:t></w:r></w:p></w:comment></w:comments>'}
        return data

    rewrite_zip(word_file, edit)
    text, dropped, _ = to_pdf_text(word_file, tmp_path)
    assert "DELETED SECRET" not in text and "COMMENT SECRET" not in text
    assert "inserted visible words" in text                      # a tracked insertion is visible text
    assert any("tracked deletions" in d for d in dropped) and any("1 comment" in d for d in dropped)


def test_word_picture_is_carried_over_as_a_picture(word_file, tmp_path):
    img = Image.new("RGB", (500, 120), "white")
    ImageDraw.Draw(img).text((10, 40), "pictures stay pictures", fill="black")
    png = tmp_path / "pic.png"
    img.save(png)
    d = docx.Document(word_file)
    d.add_paragraph("Below is a screenshot:")
    d.add_picture(str(png))
    d.save(word_file)
    out = str(tmp_path / "c.pdf")
    convert(str(word_file), out)
    assert sum(len(p.get_images()) for p in fitz.open(out)) == 1


def test_word_text_box_content_is_not_lost_or_doubled(tmp_path):
    d = docx.Document()
    d.add_paragraph("Anchor paragraph")
    path = tmp_path / "tb.docx"
    d.save(path)

    def edit(name, data):
        if name == "word/document.xml":
            box = ('<w:p xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"><w:r><mc:AlternateContent>'
                   '<mc:Choice><w:pict><w:txbxContent><w:p><w:r><w:t>TEXT BOX CONTENT</w:t></w:r></w:p></w:txbxContent></w:pict></mc:Choice>'
                   '<mc:Fallback><w:pict><w:txbxContent><w:p><w:r><w:t>TEXT BOX CONTENT</w:t></w:r></w:p></w:txbxContent></w:pict></mc:Fallback>'
                   '</mc:AlternateContent></w:r></w:p>')
            return data.decode("utf-8").replace("</w:body>", box + "</w:body>", 1).encode("utf-8")
        return data if name else None

    rewrite_zip(path, edit)
    assert pdf_text(to_pdf_text(path, tmp_path)[2]).count("TEXT BOX CONTENT") == 1


# ------------------------------------------------------------------ Excel
@pytest.fixture
def excel_file(tmp_path):
    from openpyxl.comments import Comment
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Clients"
    ws.append(["Nom", "Email", "IBAN"])
    ws.append(["Julien Marchand", "julien@example.org", IBAN])
    ws.append(["HIDDEN ROW PERSON", "x@y.z", "000"])
    ws.row_dimensions[3].hidden = True
    ws["A2"].comment = Comment("note about Julien", "HR")
    secret = wb.create_sheet("Secret")
    secret.append(["HIDDEN SHEET DATA"])
    secret.sheet_state = "hidden"
    wb.properties.creator = "Julien Marchand"
    path = tmp_path / "clients.xlsx"
    wb.save(path)
    return path


def test_excel_visible_cells_become_rows(excel_file, tmp_path):
    text, _, _ = to_pdf_text(excel_file, tmp_path)
    assert "Sheet: Clients" in text and "Nom | Email | IBAN" in text
    assert f"Julien Marchand | julien@example.org | {IBAN}" in text


def test_excel_hidden_sheets_rows_comments_and_properties_are_dropped_and_reported(excel_file, tmp_path):
    text, dropped, _ = to_pdf_text(excel_file, tmp_path)
    assert "HIDDEN SHEET DATA" not in text and "HIDDEN ROW PERSON" not in text
    assert any("1 hidden sheet" in d for d in dropped) and any("1 hidden row" in d for d in dropped)
    assert any("cell comments" in d for d in dropped) and any("document properties" in d for d in dropped)


def test_excel_numbers_booleans_and_shared_strings(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Total", 1240.5, True, "repeat", "repeat"])
    path = tmp_path / "n.xlsx"
    wb.save(path)
    assert "Total | 1240.5 | TRUE | repeat | repeat" in to_pdf_text(path, tmp_path)[0]


# ------------------------------------------------------------------ PowerPoint
@pytest.fixture
def slides_file(tmp_path):
    pr = pptx.Presentation()
    s = pr.slides.add_slide(pr.slide_layouts[1])
    s.shapes.title.text = "Dossier Marchand"
    s.placeholders[1].text = "Contact: julien@example.org"
    s.notes_slide.notes_text_frame.text = "SPEAKER NOTE SECRET"
    pr.core_properties.author = "Julien Marchand"
    path = tmp_path / "deck.pptx"
    pr.save(path)
    return path


def test_powerpoint_slide_text_is_kept_notes_are_dropped(slides_file, tmp_path):
    text, dropped, _ = to_pdf_text(slides_file, tmp_path)
    assert "Slide 1" in text and "Dossier Marchand" in text and "Contact: julien@example.org" in text
    assert "SPEAKER NOTE SECRET" not in text
    assert any("speaker notes on 1 slide" in d for d in dropped) and any("document properties" in d for d in dropped)


def test_powerpoint_hidden_slides_are_dropped(slides_file, tmp_path):
    pr = pptx.Presentation(slides_file)
    hidden = pr.slides.add_slide(pr.slide_layouts[1])
    hidden.shapes.title.text = "HIDDEN SLIDE TITLE"
    hidden._element.set("show", "0")
    pr.save(slides_file)
    text, dropped, _ = to_pdf_text(slides_file, tmp_path)
    assert "HIDDEN SLIDE TITLE" not in text and any("1 hidden slide" in d for d in dropped)


# ------------------------------------------------------------------ e-mail
def make_eml(tmp_path, html=False, attachment=True):
    m = EmailMessage()
    m["From"], m["To"], m["Cc"] = "Julien Marchand <julien@example.org>", "rh@atlantis.example", "boss@atlantis.example"
    m["Subject"], m["Date"] = "Dossier 2026", "Mon, 04 Oct 2026 10:00:00 +0000"
    m["Received"] = "from mail.example.org ([203.0.113.7]) by mx.atlantis.example"
    m["X-Originating-IP"] = "203.0.113.7"
    m["Message-ID"] = "<abc@example.org>"
    if html:
        m.set_content("fallback")
        m.add_alternative("<html><head><style>p{color:red}</style></head><body><p>Bonjour,</p><p>Mon IBAN : "
                          f"{IBAN}</p><table><tr><td>Tel</td><td>+33 6 12 34 56 78</td></tr></table></body></html>", subtype="html")
    else:
        m.set_content(f"Bonjour,\nMon IBAN : {IBAN}\nMerci.")
    if attachment:
        m.add_attachment(b"%PDF-1.4 fake", maintype="application", subtype="pdf", filename="releve_Marchand.pdf")
    path = tmp_path / "mail.eml"
    path.write_bytes(bytes(m))
    return path


def test_email_headers_and_body_are_kept(tmp_path):
    text, _, _ = to_pdf_text(make_eml(tmp_path), tmp_path)
    for kept in ("From: Julien Marchand <julien@example.org>", "To: rh@atlantis.example", "Subject: Dossier 2026",
                 "Bonjour,", f"Mon IBAN : {IBAN}"):
        assert kept in text, kept


def test_email_routing_headers_and_attachments_are_dropped_but_named(tmp_path):
    text, dropped, _ = to_pdf_text(make_eml(tmp_path), tmp_path)
    assert "203.0.113.7" not in text and "mx.atlantis.example" not in text
    assert any("routing" in d for d in dropped) and any("1 attachment" in d for d in dropped)
    assert "Attachment: releve_Marchand.pdf" in text        # the file name is personal data too: it stays so it can be redacted


def test_html_email_becomes_readable_text_without_styles_or_scripts(tmp_path):
    text, _, _ = to_pdf_text(make_eml(tmp_path, html=True), tmp_path)
    assert f"Mon IBAN : {IBAN}" in text and "Tel | +33 6 12 34 56 78" in text.replace("  ", " ") or "+33 6 12 34 56 78" in text
    assert "color:red" not in text and "<p>" not in text


def test_html_to_text():
    assert office.html_to_text("<div>a<br>b</div><script>evil()</script><style>x{}</style><p>c &amp; d</p>") == "a\nb\nc & d"


# ------------------------------------------------------------------ plain text, supported types, safety
def test_text_csv_and_markdown_files(tmp_path):
    for name, body in (("a.txt", "Nom : Julien Marchand\nTel : +33 6 12 34 56 78\n"), ("b.csv", "nom,email\nJulien,julien@example.org\n"),
                       ("c.md", "# Notes\nContact julien@example.org\n"), ("d.html", "<p>Contact julien@example.org</p>")):
        p = tmp_path / name
        p.write_text(body, encoding="utf-8")
        text, dropped, _ = to_pdf_text(p, tmp_path)
        assert ("julien@example.org" in text or "Julien Marchand" in text) and dropped == []


def test_text_files_in_old_encodings_are_read(tmp_path):
    p = tmp_path / "latin.txt"
    p.write_bytes("Nom : Hélène Brûlé, 12 rue de l'Église".encode("cp1252"))
    assert "Hélène Brûlé" in to_pdf_text(p, tmp_path)[0]


def test_accents_euro_sign_and_ligatures_survive_and_are_searchable(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("Griffin Fischer paid 12 € to œuvre at office.flow@example.org", encoding="utf-8")
    out = str(tmp_path / "o.pdf")
    convert(str(p), out)
    page = fitz.open(out)[0]
    assert "€" in page.get_text() and "Griffin Fischer" in pdf_text(out)
    findings = scan_pdf(out, use_ner=False)
    res = apply_redactions(out, str(tmp_path / "r.pdf"), [f.text for f in findings if f.score >= 60], use_ner=False)
    assert res.verification.ok and "office.flow" not in pdf_text(str(tmp_path / "r.pdf"))


def test_supported_types_and_unsupported_ones():
    assert {".docx", ".xlsx", ".pptx", ".eml", ".txt", ".pdf", ".png"} <= SUPPORTED_SUFFIXES
    assert is_supported("a.DOCX") and not is_supported("a.exe") and not is_supported("a.doc")


def test_an_unsupported_file_is_refused_clearly(tmp_path):
    p = tmp_path / "old.doc"
    p.write_bytes(b"\xd0\xcf\x11\xe0")
    with pytest.raises(ValueError, match="Unsupported"):
        convert(str(p), str(tmp_path / "o.pdf"))


def test_a_damaged_office_file_gives_a_clear_error(tmp_path):
    p = tmp_path / "broken.docx"
    p.write_bytes(b"not a zip at all")
    with pytest.raises(ValueError, match="damaged"):
        convert(str(p), str(tmp_path / "o.pdf"))


def test_a_zip_bomb_is_refused(tmp_path, monkeypatch):
    p = tmp_path / "bomb.docx"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", "<a/>" + " " * 50_000)
    monkeypatch.setattr(office, "MAX_UNCOMPRESSED", 10_000)
    with pytest.raises(ValueError, match="zip bomb"):
        convert(str(p), str(tmp_path / "o.pdf"))


def test_an_empty_document_still_produces_a_pdf(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    out = str(tmp_path / "o.pdf")
    convert(str(p), out)
    assert len(fitz.open(out)) == 1


def test_a_long_document_flows_onto_several_pages(tmp_path):
    p = tmp_path / "long.txt"
    p.write_text("\n".join(f"line {i} of a long document with some words" for i in range(300)), encoding="utf-8")
    out = str(tmp_path / "o.pdf")
    convert(str(p), out)
    assert len(fitz.open(out)) >= 3


# ------------------------------------------------------------------ the whole round trip
def test_a_word_file_is_redacted_verified_and_graded_like_any_pdf(word_file, tmp_path):
    pdf = str(tmp_path / "in.pdf")
    convert(str(word_file), pdf)
    findings = scan_pdf(pdf, use_ner=False)
    assert {"IBAN", "PHONE", "ADDRESS"} <= {f.kind for f in findings}
    out = str(tmp_path / "out.pdf")
    res = apply_redactions(pdf, out, [f.text for f in findings if f.score >= 60], use_ner=False)
    left = pdf_text(out).lower()
    assert res.verification.ok and res.quality.grade == "A"
    for secret in ("2004 1010", "+33 6 12", "acacias", "marchand"):
        assert secret not in left, secret


@needs_tesseract
def test_personal_data_inside_a_picture_in_a_word_file_is_redacted(tmp_path):
    img = Image.new("RGB", (900, 200), "white")
    ImageDraw.Draw(img).text((20, 40), "Customer: Julien Marchand", fill="black")
    ImageDraw.Draw(img).text((20, 100), "Email: julien.marchand@example.org", fill="black")
    png = tmp_path / "shot.png"
    img.resize((1800, 400)).save(png)
    d = docx.Document()
    d.add_paragraph("Report with a pasted screenshot of the customer record, for review by the department.")
    d.add_picture(str(png), width=docx.shared.Inches(6))
    path = tmp_path / "shot.docx"
    d.save(path)
    pdf = str(tmp_path / "in.pdf")
    convert(str(path), pdf)
    findings = scan_pdf(pdf, use_ner=False)
    assert any("julien.marchand@example.org" in f.text.lower() for f in findings), [f.text for f in findings]
    out = str(tmp_path / "out.pdf")
    res = apply_redactions(pdf, out, [f.text for f in findings if f.score >= 60], use_ner=False)
    assert res.verification.ok and res.image_text_pages == [0]


# ------------------------------------------------------------------ CLI
def test_cli_processes_a_folder_of_mixed_formats_and_audits_what_was_dropped(word_file, excel_file, slides_file, tmp_path):
    import json
    folder, out = tmp_path / "in", tmp_path / "out"
    folder.mkdir()
    for f in (word_file, excel_file, slides_file, make_eml(tmp_path)):
        (folder / f.name).write_bytes(f.read_bytes())
    assert cli_main([str(folder), str(out), "--no-ner", "--no-ocr", "--min-score", "60"]) == 0
    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    by_name = {r["file"]: r for r in audit["files"]}
    assert set(by_name) == {"contract.docx", "clients.xlsx", "deck.pptx", "mail.eml"}
    assert by_name["contract.docx"]["source_format"] == "docx" and by_name["contract.docx"]["verification_passed"]
    assert any("hidden text" in d for d in by_name["contract.docx"]["dropped_on_conversion"])
    assert any("speaker notes" in d for d in by_name["deck.pptx"]["dropped_on_conversion"])
    assert "julien" not in json.dumps(audit["files"][0]["quality"]).lower()
    for pdf in out.glob("*_redacted.pdf"):
        assert IBAN not in pdf_text(str(pdf))


def test_cli_case_mode_accepts_office_files_and_shares_across_formats(word_file, tmp_path):
    folder, out = tmp_path / "case", tmp_path / "case_out"
    folder.mkdir()
    (folder / "contract.docx").write_bytes(word_file.read_bytes())
    (folder / "memo.txt").write_text("Note interne\nDossier suivi par Marchand cette annee.\n", encoding="utf-8")
    assert cli_main([str(folder), str(out), "--case", "--lang", "fr", "--no-ner", "--no-ocr", "--min-score", "60"]) == 0
    assert "marchand" not in pdf_text(str(out / "memo_redacted.pdf")).lower()       # confirmed in the Word file, removed from the note


# ------------------------------------------------------------------ UI
def test_ui_accepts_a_word_file_and_shows_what_was_dropped(word_file):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=180)
    at.run()
    at.file_uploader[0].upload("contract.docx", word_file.read_bytes(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    at.run()
    assert not at.exception
    assert any("Converted to PDF" in i.value and "hidden text" in i.value for i in at.info)
    [b for b in at.button if b.label == "Apply redaction"][0].click()
    at.run()
    assert not at.exception and any("Quality of the saved file" in s.value for s in list(at.success) + list(at.warning))
