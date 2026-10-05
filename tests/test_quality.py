"""Redaction quality grade: the rules (on hand-made results) and the real thing (on redacted files)."""
import json

import pymupdf as fitz
import pytest

from benchmark.corpus import SEED  # noqa: F401  (keeps the benchmark package importable from the tests)
from redactor import quality as q
from redactor.cli import main as cli_main
from redactor.detect import Finding
from redactor.make_sample import make_sample
from redactor.redact import RedactionResult, apply_redactions, scan_pdf
from redactor.verify import Verification


def result(**kw) -> RedactionResult:
    base = dict(hits=5, hidden_removed=[], verification=Verification(), residual=[], pages=1,
                unreadable_pages=[], ocr_pages=[], hidden_remaining=[])
    base.update(kw)
    return RedactionResult(**base)


def severities(quality):
    return [s for s, _ in quality.issues]


# ------------------------------------------------------------------ the rules
def test_a_clean_result_is_an_A():
    r = q.assess_quality(result())
    assert (r.score, r.grade, r.issues) == (100, "A", [])
    assert "Nothing detectable remains in the saved file." in r.notes


def test_failed_verification_is_an_automatic_F_with_a_capped_score():
    r = q.assess_quality(result(verification=Verification(["Page 1: a redaction box is not fully black"])))
    assert r.grade == "F" and r.score <= 40 and severities(r)[0] == "critical" and "Do not share" in r.issues[0][1]


def test_hidden_content_surviving_the_strip_is_an_automatic_F():
    r = q.assess_quality(result(hidden_remaining=["1 comments / annotations"]))
    assert r.grade == "F" and "annotations" in r.issues[0][1]


def test_an_unreadable_page_caps_the_grade_at_C_and_costs_more_the_more_pages_are_affected():
    one_of_four = q.assess_quality(result(pages=4, unreadable_pages=[2]))
    all_four = q.assess_quality(result(pages=4, unreadable_pages=[0, 1, 2, 3]))
    assert one_of_four.grade == "C" and "Page 3" in one_of_four.issues[0][1]
    assert all_four.score < one_of_four.score <= 100 - 10 and all_four.grade in ("C", "F")


def test_confident_data_still_in_the_file_caps_the_grade_at_C():
    r = q.assess_quality(result(residual=[Finding("IBAN", "GB82 WEST 1234 5698 7654 32", 99)]))
    assert r.grade == "C" and r.issues[0][0] == "major" and "IBAN x1" in r.issues[0][1]


def test_plausible_leftovers_cost_points_but_not_the_grade_cap():
    r = q.assess_quality(result(residual=[Finding("PERSON", "Maria Garcia", 65), Finding("PERSON", "Jean Dupont", 70)]))
    assert r.grade in ("A", "B") and severities(r) == ["minor"] and r.score < 100


def test_low_confidence_leftovers_are_not_even_an_issue():
    r = q.assess_quality(result(residual=[Finding("LOCATION", "Paris", 30), Finding("LOCATION", "Lyon", 30)]))
    assert r.issues == [] and r.grade == "A"


def test_more_residual_risk_means_a_lower_score():
    few = q.assess_quality(result(residual=[Finding("EMAIL", "a@b.co", 98)]))
    many = q.assess_quality(result(residual=[Finding("EMAIL", "a@b.co", 98), Finding("PHONE", "+33 1 42 68 53 00", 92),
                                              Finding("PERSON", "Maria Garcia", 95), Finding("IBAN", "GB82 WEST 1234 5698 7654 32", 99)]))
    assert many.score < few.score < 100


def test_nothing_redacted_while_data_remains_is_flagged():
    r = q.assess_quality(result(hits=0, residual=[Finding("EMAIL", "a@b.co", 98)]))
    assert any("Nothing was redacted" in t for _, t in r.issues) and r.grade == "C"


def test_notes_report_ocr_pages_and_removed_hidden_content_without_penalty():
    r = q.assess_quality(result(pages=3, ocr_pages=[0, 2], hidden_removed=["3 hyperlinks"]))
    assert r.grade == "A" and r.score == 100
    assert any("Page 1, 3 were read through OCR" in n for n in r.notes) and any("3 hyperlinks" in n for n in r.notes)


def test_issues_are_sorted_worst_first_and_the_worst_grade_wins():
    r = q.assess_quality(result(pages=2, unreadable_pages=[1], verification=Verification(["x"]),
                                residual=[Finding("PERSON", "Maria Garcia", 65)]))
    assert severities(r) == sorted(severities(r), key={"critical": 0, "major": 1, "minor": 2}.get) and r.grade == "F"
    assert q.worse("A", "C") == "C" and q.worse("F", "B") == "F" and q.worse("B", "B") == "B"


def test_grade_boundaries():
    assert [q.grade_for(s) for s in (100, 90, 89, 75, 74, 50, 49, 0)] == ["A", "A", "B", "B", "C", "C", "F", "F"]


def test_the_explanation_never_contains_personal_text():
    r = q.assess_quality(result(residual=[Finding("PERSON", "Maria Garcia", 95), Finding("EMAIL", "maria@garcia.org", 98)]))
    blob = json.dumps(r.as_dict()).lower() + r.explain().lower()
    assert "maria" not in blob and "garcia" not in blob


# ------------------------------------------------------------------ real redactions
@pytest.fixture
def sample_pdf(tmp_path):
    path = str(tmp_path / "s.pdf")
    make_sample(path)
    return path


def redact(path, tmp_path, keep=lambda f: f.score >= 60, **kw):
    findings = scan_pdf(path)
    return apply_redactions(path, str(tmp_path / "out.pdf"), [f.text for f in findings if keep(f)], **kw)


def test_a_properly_redacted_text_pdf_is_an_A(sample_pdf, tmp_path):
    r = redact(sample_pdf, tmp_path).quality
    assert r.grade == "A" and r.score >= 95 and r.issues == []


def test_leaving_a_phone_number_unredacted_drops_the_grade_and_says_why(sample_pdf, tmp_path):
    r = redact(sample_pdf, tmp_path, keep=lambda f: f.kind != "PHONE").quality
    assert r.grade == "C" and any("PHONE" in t for _, t in r.issues)


def test_redacting_nothing_is_flagged(sample_pdf, tmp_path):
    r = redact(sample_pdf, tmp_path, keep=lambda f: False)
    assert r.hits == 0 and r.quality.grade == "C" and any("Nothing was redacted" in t for _, t in r.quality.issues)


def test_a_redacted_scan_is_graded_and_notes_that_ocr_was_used(tmp_path):
    from redactor.make_sample import make_scanned_sample
    path = str(tmp_path / "scan.pdf")
    make_scanned_sample(path)
    r = redact(path, tmp_path, keep=lambda f: f.score >= 60)
    assert r.verification.ok and r.quality.grade in ("A", "B") and r.ocr_pages == [0]
    assert any("OCR" in n for n in r.quality.notes)


def test_a_page_ocr_cannot_read_is_reported_as_unreadable(tmp_path):
    white = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 300, 400), False)
    white.clear_with(255)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(page.rect, pixmap=white)
    path = str(tmp_path / "blank.pdf")
    doc.save(path)
    r = apply_redactions(path, str(tmp_path / "o.pdf"), [])
    assert r.unreadable_pages == [0] and r.quality.grade == "C"
    assert "Page 1 could not be read" in r.quality.issues[0][1]


def test_a_scan_with_ocr_switched_off_counts_as_unreadable(tmp_path):
    from redactor.make_sample import make_scanned_sample
    path = str(tmp_path / "scan.pdf")
    make_scanned_sample(path)
    r = apply_redactions(path, str(tmp_path / "o.pdf"), [], ocr=False)
    assert r.unreadable_pages == [0] and r.quality.grade == "C"


def test_hidden_content_that_was_removed_is_a_note_not_a_penalty(tmp_path):
    path = str(tmp_path / "trap.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Public report with enough words in it to be readable text", fontsize=12)
    page.add_text_annot((300, 300), "call John Smith")
    doc.set_metadata({"author": "John Smith"})
    doc.save(path)
    r = apply_redactions(path, str(tmp_path / "o.pdf"), [])
    assert r.hidden_remaining == [] and r.quality.grade == "A"
    assert any("Removed hidden content" in n for n in r.quality.notes)


def test_if_stripping_ever_fails_the_file_is_an_F(tmp_path, monkeypatch):
    path = str(tmp_path / "trap.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Public report with enough words in it to be readable text", fontsize=12)
    page.add_text_annot((300, 300), "call John Smith")
    doc.save(path)
    monkeypatch.setattr("redactor.redact.strip_hidden", lambda doc: None)  # simulate the strip not working
    r = apply_redactions(path, str(tmp_path / "o.pdf"), [])
    assert r.hidden_remaining and r.quality.grade == "F" and "annotations" in r.quality.issues[0][1]


# ------------------------------------------------------------------ CLI and UI
def test_cli_prints_and_audits_the_grade(sample_pdf, tmp_path, capsys):
    out, audit = str(tmp_path / "o.pdf"), str(tmp_path / "a.json")
    assert cli_main([sample_pdf, out, "--no-ocr", "--min-score", "60", "--audit", audit]) == 0
    assert "quality of the saved file: A" in capsys.readouterr().out
    rec = json.loads(open(audit, encoding="utf-8").read())["files"][0]
    assert rec["quality"]["grade"] == "A" and "john" not in json.dumps(rec["quality"]).lower()


def test_cli_min_grade_fails_a_file_below_the_bar(sample_pdf, tmp_path, capsys):
    out = str(tmp_path / "o.pdf")
    # --min-score 99 leaves most data in the file: grade C, still "verified", but below a B bar
    assert cli_main([sample_pdf, out, "--no-ocr", "--min-score", "99"]) == 0
    assert cli_main([sample_pdf, out, "--no-ocr", "--min-score", "99", "--min-grade", "B"]) == 1
    assert "below the required grade" in capsys.readouterr().out


def test_ui_shows_the_quality_grade_after_applying(sample_pdf):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=180)
    at.run()
    at.file_uploader[0].upload("s.pdf", open(sample_pdf, "rb").read(), "application/pdf")
    at.run()
    [b for b in at.button if b.label == "Apply redaction"][0].click()
    at.run()
    assert not at.exception
    assert any("Quality of the saved file: A" in s.value for s in at.success)
