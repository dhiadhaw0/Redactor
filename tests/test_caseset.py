"""Several documents treated as one case: shared entities, per-file verification, set-level risk and audit."""
import json
import zipfile
from io import BytesIO

import pymupdf as fitz
import pytest

from redactor import caseset as cs
from redactor.detect import Finding
from redactor.risk import assess, assess_set

IBAN = "FR14 2004 1010 0505 0001 3M02 606"


def make_pdf(path, lines):
    doc = fitz.open()
    page = doc.new_page()
    for i, line in enumerate(lines):
        page.insert_text((60, 70 + 16 * i), line, fontsize=10)
    doc.save(str(path))
    return str(path)


def text_of(path):
    return " ".join(p.get_text() for p in fitz.open(path))


@pytest.fixture
def case(tmp_path):
    """Three documents about one person. Only the first identifies him in a way every rule trusts."""
    a = make_pdf(tmp_path / "a.pdf", ["Nom : Julien Marchand", "Courriel : julien.marchand@example.org",
                                       "Telephone : +33 6 12 34 56 78"])
    b = make_pdf(tmp_path / "b.pdf", ["Notes de dossier", "Le dossier a ete suivi par Marchand pendant toute l'annee 2026."])
    c = make_pdf(tmp_path / "c.pdf", ["Virement", f"Compte : {IBAN}"])
    scans = cs.scan_files([("a.pdf", a), ("b.pdf", b), ("c.pdf", c)], use_ner=False, lang="fr")
    return scans, cs.build_registry(scans), tmp_path


def entity(registry, text):
    return next(e for e in registry if e.text.casefold() == text.casefold())


# ------------------------------------------------------------------ registry
def test_registry_merges_the_same_text_across_files_and_keeps_the_best_score(case):
    scans, registry, _ = case
    assert len({e.key for e in registry}) == len(registry)                       # one entry per (kind, text)
    assert [e.score for e in registry] == sorted((e.score for e in registry), reverse=True)
    assert entity(registry, IBAN).files == ["c.pdf"]


def test_registry_merge_is_case_insensitive_and_keeps_the_most_certain_spelling():
    scans = [cs.FileScan("x", "x", [Finding("PERSON", "MARCHAND", 70)]),
             cs.FileScan("y", "y", [Finding("PERSON", "Marchand", 90)])]
    [e] = cs.build_registry(scans)
    assert (e.text, e.score, e.files) == ("Marchand", 90, ["x", "y"])


# ------------------------------------------------------------------ what each file is searched for
def test_a_file_is_searched_for_its_own_selected_entities(case):
    scans, registry, _ = case
    selected = cs.preselected(registry, 60)
    terms_c = cs.terms_for_file(scans[2], registry, selected, share=False)
    assert IBAN in terms_c and not any("marchand" in t.lower() for t in terms_c)


def test_confident_entities_are_shared_to_files_that_did_not_report_them(case):
    scans, registry, _ = case
    selected = cs.preselected(registry, 60)
    shared = cs.terms_for_file(scans[2], registry, selected, share=True)
    assert any(t.lower() == "marchand" for t in shared)           # confirmed in a.pdf, searched for in c.pdf
    assert IBAN in shared


def test_low_confidence_entities_are_never_shared():
    low = cs.Entity("LOCATION", "Paris", 40, ["a"])
    scan_b = cs.FileScan("b", "b", [])
    assert cs.terms_for_file(scan_b, [low], {low.key}, share=True) == []          # selected, but too uncertain to spread
    sure = cs.Entity("PERSON", "Marchand", 85, ["a"])
    assert cs.terms_for_file(scan_b, [sure], {sure.key}, share=True) == ["Marchand"]  # exactly at the threshold


def test_unselected_entities_are_never_applied_anywhere(case):
    scans, registry, _ = case
    assert cs.terms_for_file(scans[1], registry, set(), share=True) == []


def test_sharing_can_be_switched_off(case):
    scans, registry, _ = case
    assert cs.terms_for_file(scans[1], registry, cs.preselected(registry, 60), share=False) == []


def test_the_threshold_decides_how_much_is_shared(case):
    scans, registry, _ = case
    selected = cs.preselected(registry, 0)
    strict = cs.terms_for_file(scans[1], registry, selected, share=True, threshold=99)
    loose = cs.terms_for_file(scans[1], registry, selected, share=True, threshold=60)
    assert len(strict) <= len(loose) and set(strict) <= set(loose)


# ------------------------------------------------------------------ processing the set
def test_a_surname_confirmed_in_one_file_is_removed_from_the_others(case):
    scans, registry, tmp = case
    assert not any(f.text.lower() == "marchand" and f.score >= cs.SHARE_THRESHOLD for f in scans[1].findings)  # b.pdf alone misses it
    result = cs.process_set(scans, str(tmp / "out"), cs.preselected(registry, 60), registry, share=True, lang="fr", use_ner=False)
    assert "marchand" not in text_of(result.outcomes[1].output).lower()
    assert "Marchand" not in text_of(result.outcomes[0].output)
    assert IBAN not in text_of(result.outcomes[2].output)
    assert result.ok and result.shared_terms > 0


def test_without_sharing_the_other_file_keeps_the_surname(case):
    scans, registry, tmp = case
    result = cs.process_set(scans, str(tmp / "out2"), cs.preselected(registry, 60), registry, share=False, lang="fr", use_ner=False)
    assert "marchand" in text_of(result.outcomes[1].output).lower() and result.shared_terms == 0


def test_every_file_is_verified_and_graded_on_its_own(case):
    scans, registry, tmp = case
    result = cs.process_set(scans, str(tmp / "out3"), cs.preselected(registry, 60), registry, lang="fr", use_ner=False)
    assert [o.name for o in result.outcomes] == ["a.pdf", "b.pdf", "c.pdf"]
    assert all(o.result.verification.ok for o in result.outcomes)
    assert all(o.result.quality.grade in "AB" for o in result.outcomes) and result.worst_grade in ("A", "B")


def test_the_worst_file_sets_the_grade_of_the_set(case, tmp_path):
    scans, registry, tmp = case
    white = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 300, 400), False)
    white.clear_with(255)
    doc = fitz.open()
    doc.new_page().insert_image(fitz.Rect(0, 0, 595, 842), pixmap=white)
    blank = str(tmp / "blank.pdf")
    doc.save(blank)
    scans = scans + [cs.FileScan("blank.pdf", blank)]
    result = cs.process_set(scans, str(tmp / "out4"), cs.preselected(registry, 60), registry, lang="fr", use_ner=False)
    assert result.outcomes[-1].result.quality.grade == "C" and result.worst_grade == "C"
    assert result.ok                                    # C is below A but still verified: the set is usable


def test_progress_is_reported_for_every_file(case):
    scans, registry, tmp = case
    seen = []
    cs.process_set(scans, str(tmp / "out5"), set(), registry, lang="fr", use_ner=False, progress=lambda i, n, name: seen.append((i, n, name)))
    assert seen == [(0, 3, "a.pdf"), (1, 3, "b.pdf"), (2, 3, "c.pdf")]


def test_output_names_stay_unique_when_stems_collide():
    assert cs.output_names(["a.pdf", "a.png", "b.pdf", "a.jpg"]) == ["a_redacted.pdf", "a_redacted_2.pdf", "b_redacted.pdf", "a_redacted_3.pdf"]


# ------------------------------------------------------------------ set-level risk
def test_a_set_is_riskier_than_its_files_because_documents_link_up(case):
    scans, registry, _ = case
    per_file = {s.name: s.findings for s in scans}
    alone = max(assess(s.findings).score for s in scans)
    together = assess_set(per_file)
    assert together.score > alone
    assert any("financial data tied to a named person" in r for r, _ in together.bonuses)   # name in a.pdf + IBAN in c.pdf


def test_the_same_person_in_several_documents_adds_a_small_bonus():
    f = lambda: [Finding("PERSON", "Maria Garcia", 95)]
    one, two = assess_set({"a": f()}), assess_set({"a": f(), "b": f()})
    assert ("the same person is identifiable across several documents", 5) in two.bonuses and one.bonuses == []
    assert 4 <= two.score - one.score <= 5          # +5 points before rounding


def test_a_repeated_identity_is_counted_once():
    f = lambda: [Finding("EMAIL", "a@b.co", 98)]
    assert assess_set({"a": f(), "b": f(), "c": f()}).score == assess(f()).score


def test_redaction_lowers_the_measured_risk_of_the_set(case):
    scans, registry, tmp = case
    result = cs.process_set(scans, str(tmp / "out6"), cs.preselected(registry, 60), registry, lang="fr", use_ner=False)
    assert result.risk_after.score < result.risk_before.score / 2


# ------------------------------------------------------------------ audit and zip
def test_the_set_audit_holds_no_personal_text(case):
    scans, registry, tmp = case
    result = cs.process_set(scans, str(tmp / "out7"), cs.preselected(registry, 60), registry, lang="fr", use_ner=False)
    audit = cs.set_audit(result, registry, True, cs.SHARE_THRESHOLD)
    blob = json.dumps(audit).lower()
    for secret in ("marchand", "julien", "example.org", "6 12 34", "2004 1010"):
        assert secret not in blob, secret
    assert audit["set"]["files"] == 3 and audit["set"]["sharing"]["terms_shared_to_other_files"] > 0
    assert audit["set"]["distinct_entities_by_type"]["EMAIL"] >= 1 and len(audit["files"]) == 3


def test_the_zip_holds_every_redacted_file_and_the_audit(case):
    scans, registry, tmp = case
    result = cs.process_set(scans, str(tmp / "out8"), cs.preselected(registry, 60), registry, lang="fr", use_ner=False)
    data = cs.build_zip(result, cs.set_audit(result, registry, True, cs.SHARE_THRESHOLD))
    with zipfile.ZipFile(BytesIO(data)) as z:
        assert sorted(z.namelist()) == ["a_redacted.pdf", "audit.json", "b_redacted.pdf", "c_redacted.pdf"]
        assert json.loads(z.read("audit.json"))["set"]["worst_grade"] in "AB"
        redacted = z.read("b_redacted.pdf")
    assert b"%PDF" in redacted[:8]


# ------------------------------------------------------------------ CLI --case
def test_cli_case_mode_shares_across_files_and_writes_a_set_audit(case, capsys):
    from redactor.cli import main
    scans, registry, tmp = case
    folder, out = tmp / "in", tmp / "cli_out"
    folder.mkdir()
    for s in scans:
        (folder / s.name).write_bytes(open(s.pdf, "rb").read())
    code = main([str(folder), str(out), "--case", "--lang", "fr", "--no-ner", "--min-score", "60"])
    printed = capsys.readouterr().out
    assert code == 0 and "Case of 3 files" in printed and "Terms shared to files that did not report them:" in printed
    assert "marchand" not in text_of(out / "b_redacted.pdf").lower()
    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert audit["set"]["files"] == 3 and audit["set"]["sharing"]["enabled"] and "marchand" not in json.dumps(audit).lower()


def test_cli_case_mode_without_sharing_threshold_effect(case):
    from redactor.cli import main
    scans, registry, tmp = case
    folder, out = tmp / "in2", tmp / "cli_out2"
    folder.mkdir()
    for s in scans:
        (folder / s.name).write_bytes(open(s.pdf, "rb").read())
    assert main([str(folder), str(out), "--case", "--lang", "fr", "--no-ner", "--min-score", "60", "--share-threshold", "100"]) == 0
    assert "marchand" in text_of(out / "b_redacted.pdf").lower()      # threshold 100: nothing is confident enough to share


def test_cli_case_min_grade_fails_the_set(case):
    from redactor.cli import main
    scans, registry, tmp = case
    folder, out = tmp / "in3", tmp / "cli_out3"
    folder.mkdir()
    for s in scans:
        (folder / s.name).write_bytes(open(s.pdf, "rb").read())
    # --min-score 99 leaves data in the files: graded C, so a B bar fails the case
    assert main([str(folder), str(out), "--case", "--lang", "fr", "--no-ner", "--min-score", "99", "--min-grade", "B"]) == 1


# ------------------------------------------------------------------ the case UI
def _case_app(case):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    scans, _, _ = case
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=240)
    at.run()
    for s in scans:
        at.file_uploader[0].upload(s.name, open(s.pdf, "rb").read(), "application/pdf")
    at.selectbox[0].set_value("fr").run()
    return at


def _apply(at):
    [b for b in at.button if b.label == "Apply to all files"][0].click()
    at.run()
    assert not at.exception
    return at.session_state["case_result"]


def test_ui_several_uploads_open_case_mode_and_share_confident_findings(case):
    at = _case_app(case)
    assert not at.exception and any("Case of 3 files" in i.value for i in at.info)
    assert at.metric[0].label == "Original documents, together" and "estimate" in at.metric[1].label
    result = _apply(at)
    assert "marchand" not in text_of(result.outcomes[1].output).lower() and result.ok
    assert "measured" in at.metric[1].label and int(at.metric[1].value.split("/")[0]) < int(at.metric[0].value.split("/")[0])
    assert any("Set grade: A" in s.value for s in at.success)


def test_ui_sharing_can_be_switched_off(case):
    at = _case_app(case)
    at.toggle(key="case_share").set_value(False).run()
    result = _apply(at)
    assert "marchand" in text_of(result.outcomes[1].output).lower() and result.shared_terms == 0


def test_ui_the_zip_and_audit_are_ready_after_applying(case):
    at = _case_app(case)
    _apply(at)
    with zipfile.ZipFile(BytesIO(at.session_state["case_zip"])) as z:
        assert "audit.json" in z.namelist() and sum(n.endswith("_redacted.pdf") for n in z.namelist()) == 3
    assert at.session_state["case_audit"]["set"]["files"] == 3


def test_ui_one_upload_still_uses_the_single_file_flow(case):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    scans, _, _ = case
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120)
    at.run()
    at.file_uploader[0].upload("a.pdf", open(scans[0].pdf, "rb").read(), "application/pdf")
    at.run()
    assert not at.exception and "findings" in at.session_state and not any("Case of" in i.value for i in at.info)
