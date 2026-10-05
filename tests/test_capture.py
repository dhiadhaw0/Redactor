"""Saving a review as a labeled benchmark case, and scoring the detector on it."""
import json
from pathlib import Path

import pymupdf as fitz
import pytest

from benchmark import capture, run
from redactor.redact import scan_pdf

CV_LINES = ["Dhia Eddine Toumi", "Tunis, Tunisia | +216 29 045 129 | dhiaeddinetoumi7@gmail.com",
            "Skills: Spring Boot, Docker, Monte Carlo simulation, Big Data", "Experience 2023 - 2025 in software"]


@pytest.fixture
def cv_pdf(tmp_path):
    pdf = fitz.open()
    page = pdf.new_page()
    for i, line in enumerate(CV_LINES):
        page.insert_text((50, 80 + 16 * i), line, fontsize=10)
    path = tmp_path / "cv.pdf"
    pdf.save(path)
    return str(path)


@pytest.fixture
def local_dir(tmp_path, monkeypatch):
    d = tmp_path / "local"
    monkeypatch.setattr(capture, "LOCAL_DIR", d)
    monkeypatch.setattr(run, "LOCAL_DIR", d)
    return d


ACCEPTED = [("PERSON", "Dhia Eddine Toumi"), ("PERSON", "Toumi"), ("PERSON", "Dhia"), ("EMAIL", "dhiaeddinetoumi7@gmail.com"),
            ("PHONE", "+216 29 045 129"), ("ADDRESS", "Tunis, Tunisia")]


# ------------------------------------------------------------------ saving
def test_a_case_is_written_with_a_copy_of_the_document(cv_pdf, local_dir):
    label = capture.save_case(cv_pdf, "fr", ACCEPTED, rejected=[("PERSON", "Monte Carlo")], manual=["Zorblax Quux"],
                              boxes=[(0, 0.1, 0.1, 0.3, 0.2)], name="my cv", source="cv.pdf", directory=local_dir)
    spec = json.loads(label.read_text(encoding="utf-8"))
    assert label.parent == local_dir and label.name == "my_cv.labels.json"
    assert spec["file"] == "docs/my_cv.pdf" and spec["lang"] == "fr" and spec["source"] == "cv.pdf"
    assert (local_dir / "docs" / "my_cv.pdf").read_bytes() == Path(cv_pdf).read_bytes()
    assert spec["rejected"] == [{"kind": "PERSON", "text": "Monte Carlo"}] and spec["boxes"] == [[0, 0.1, 0.1, 0.3, 0.2]]
    assert spec["created"].endswith("+00:00")


def test_name_pieces_are_not_separate_items_and_manual_terms_are_missed_items(cv_pdf, local_dir):
    spec = json.loads(capture.save_case(cv_pdf, "fr", ACCEPTED, manual=["  Zorblax Quux ", ""], name="c", directory=local_dir)
                      .read_text(encoding="utf-8"))
    items = [(i["kind"], i["text"]) for i in spec["items"]]
    assert ("PERSON", "Toumi") not in items and ("PERSON", "Dhia") not in items          # inside the accepted full name
    assert ("PERSON", "Dhia Eddine Toumi") in items and ("MANUAL", "Zorblax Quux") in items
    assert len(items) == 5


def test_names_are_made_safe_and_never_overwrite_an_existing_case(cv_pdf, local_dir):
    first = capture.save_case(cv_pdf, "en", ACCEPTED, name="../../etc/passwd", directory=local_dir)
    second = capture.save_case(cv_pdf, "en", ACCEPTED, name="../../etc/passwd", directory=local_dir)
    assert first.parent == local_dir and second.parent == local_dir and first != second
    assert first.name == "passwd.labels.json" and second.name == "passwd_2.labels.json"
    assert capture._slug("???") == "case" and len(capture._slug("x" * 200)) == 60


def test_saved_cases_are_found_by_the_local_benchmark(cv_pdf, local_dir):
    capture.save_case(cv_pdf, "fr", ACCEPTED, name="cv", directory=local_dir)
    [doc] = run.load_local(local_dir)
    assert doc["name"] == "cv" and doc["path"] == str(local_dir / "docs" / "cv.pdf") and doc["lang"] == "fr"
    assert ("EMAIL", "dhiaeddinetoumi7@gmail.com") in doc["items"]


# ------------------------------------------------------------------ scoring
def test_a_case_that_matches_the_detector_scores_perfect_recall(cv_pdf, local_dir):
    findings = {(f.kind, f.text) for f in scan_pdf(cv_pdf, use_ner=True, lang="fr") if f.score >= 60}
    accepted = [(k, t) for k, t in findings if k != "PERSON" or t == "Dhia Eddine Toumi"]
    label = capture.save_case(cv_pdf, "fr", accepted, name="good", directory=local_dir)
    score = capture.evaluate_case(label)
    assert score.recall == 1.0 and score.missed == [] and score.items >= 4


def test_a_term_you_typed_in_is_reported_as_missed(cv_pdf, local_dir):
    label = capture.save_case(cv_pdf, "fr", ACCEPTED, manual=["Zorblax Quux"], name="m", directory=local_dir)
    score = capture.evaluate_case(label)
    assert [(k, t) for k, t, _ in score.missed] == [("MANUAL", "Zorblax Quux")]
    assert score.covered == score.items - 1 and 0 < score.recall < 1


def test_something_you_unticked_counts_against_precision(cv_pdf, local_dir):
    """'Monte Carlo' is flagged as a person by the model; the reviewer unticked it: that is a false positive."""
    flagged = [f for f in scan_pdf(cv_pdf, use_ner=True, lang="fr") if f.text == "Monte Carlo" and f.score >= 60]
    assert flagged, "the fixture relies on the model flagging 'Monte Carlo'"
    label = capture.save_case(cv_pdf, "fr", ACCEPTED, rejected=[("PERSON", "Monte Carlo")], name="fp", directory=local_dir)
    score = capture.evaluate_case(label)
    assert ("PERSON", "Monte Carlo", flagged[0].score) in score.false_positives
    assert score.precision < 1.0


def test_the_benchmark_cli_scores_saved_cases(cv_pdf, local_dir, capsys):
    capture.save_case(cv_pdf, "fr", ACCEPTED, manual=["Zorblax Quux"], name="viaCli", directory=local_dir)
    assert run.main(["--local"]) == 0
    printed = capsys.readouterr().out
    assert "viaCli" in printed and "kind:MANUAL" in printed
    assert "insurance_letter" not in printed        # only the redirected folder was used, not the real local cases


# ------------------------------------------------------------------ the UI
def test_ui_saves_a_single_file_review_and_scores_it(cv_pdf, local_dir):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=240)
    at.run()
    at.file_uploader[0].upload("my_cv.pdf", open(cv_pdf, "rb").read(), "application/pdf")
    at.selectbox[0].set_value("fr").run()
    at.text_area[0].set_value("Zorblax Quux").run()
    [b for b in at.button if b.label == "Save as benchmark case"][0].click()
    at.run()
    assert not at.exception
    files = sorted(p.name for p in local_dir.rglob("*") if p.is_file())
    assert files == ["my_cv.labels.json", "my_cv.pdf"]
    spec = json.loads((local_dir / "my_cv.labels.json").read_text(encoding="utf-8"))
    assert ("MANUAL", "Zorblax Quux") in {(i["kind"], i["text"]) for i in spec["items"]}
    assert any(m.label.startswith("Recall") for m in at.metric)
    assert any("Zorblax Quux" in w.value for w in at.markdown if "Missed" in w.value)


def test_ui_saves_every_file_of_a_case(tmp_path, local_dir):
    from streamlit.testing.v1 import AppTest
    paths = []
    for name, text in (("a.pdf", "Nom : Julien Marchand\nCourriel : julien.marchand@example.org"),
                       ("b.pdf", "Dossier suivi par Marchand pendant l annee 2026 avec le plus grand soin.")):
        pdf = fitz.open()
        page = pdf.new_page()
        for i, line in enumerate(text.split("\n")):
            page.insert_text((50, 80 + 16 * i), line, fontsize=10)
        p = tmp_path / name
        pdf.save(p)
        paths.append(p)
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=240)
    at.run()
    for p in paths:
        at.file_uploader[0].upload(p.name, p.read_bytes(), "application/pdf")
    at.selectbox[0].set_value("fr").run()
    [b for b in at.button if b.label == "Save every file as a benchmark case"][0].click()
    at.run()
    assert not at.exception
    assert sorted(p.name for p in local_dir.glob("*.labels.json")) == ["a.pdf.labels.json".replace(".pdf", ""), "b.labels.json"] or \
        len(list(local_dir.glob("*.labels.json"))) == 2
    assert len(at.session_state["case_capture_rows"]) == 2
