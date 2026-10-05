"""Profiles: allow list (never flag), deny list (always redact) and defaults."""
import json
import re
import time
from pathlib import Path

import pymupdf as fitz
import pytest

from redactor import caseset, profiles
from redactor.cli import main as cli_main
from redactor.detect import Finding
from redactor.profiles import Profile, ProfileError, Rule, apply_allow, deny_findings, list_profiles, load_profile
from redactor.redact import apply_redactions, scan_pdf


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    shipped, local = tmp_path / "profiles", tmp_path / "profiles" / "local"
    shipped.mkdir()
    monkeypatch.setattr(profiles, "PROFILES_DIR", shipped)
    monkeypatch.setattr(profiles, "LOCAL_DIR", local)
    return shipped, local


def make_pdf(path, lines):
    pdf = fitz.open()
    page = pdf.new_page()
    for i, line in enumerate(lines):
        page.insert_text((60, 80 + 16 * i), line, fontsize=10)
    pdf.save(str(path))
    return str(path)


def text_of(path):
    return " ".join(p.get_text() for p in fitz.open(path))


# ------------------------------------------------------------------ rules are validated
def test_good_rules_pass():
    for rule in (Rule("Atlantis"), Rule(r"EMP-\d{5}", regex=True), Rule("x", score=0), Rule("x", kind="ID", score=100)):
        rule.validate()


@pytest.mark.parametrize("rule, message", [
    (Rule("  "), "empty"),
    (Rule("x" * 201), "longer"),
    (Rule("(a+)+$", regex=True), "nested quantifiers"),
    (Rule("(.*)*", regex=True), "nested quantifiers"),
    (Rule("(a|b*)+", regex=True), "nested quantifiers"),
    (Rule("[unclosed", regex=True), "not a valid regular expression"),
    (Rule("a*", regex=True), "empty string"),
    (Rule("x", score=101), "between 0 and 100"),
])
def test_dangerous_or_broken_rules_are_refused(rule, message):
    with pytest.raises(ProfileError, match=message):
        rule.validate()


def test_a_legitimate_pattern_is_fast_even_on_hostile_looking_text():
    rule = Rule(r"\bEMP[-\s]?\d{4,6}\b", regex=True)
    rule.validate()
    start = time.perf_counter()
    assert not rule.compiled().search("EMP" * 20000 + "a" * 50000)
    assert time.perf_counter() - start < 2


def test_profile_validation_checks_the_whole_profile():
    with pytest.raises(ProfileError, match="min_score"):
        Profile(min_score=150).validate()
    with pytest.raises(ProfileError, match="language"):
        Profile(language="de").validate()
    with pytest.raises(ProfileError, match="nested"):
        Profile(deny=[Rule("(a+)+", regex=True)]).validate()


# ------------------------------------------------------------------ files
def test_a_profile_survives_a_round_trip_through_json(dirs):
    p = Profile("hr", "employment", "fr", 55, 90, [Rule("Atlantis"), Rule("^Paris$", regex=True, kind="LOCATION")],
                [Rule("EMP-\\d{5}", regex=True, kind="ID", score=99, note="employee")])
    path = p.save()
    assert path.parent == dirs[1] and path.name == "hr.json"
    again = load_profile("hr")
    assert again == p and again.deny[0].score == 99 and again.allow[1].kind == "LOCATION"


def test_local_profiles_hide_shipped_ones_of_the_same_name(dirs):
    (dirs[0] / "team.json").write_text(json.dumps({"name": "team", "description": "shipped"}), encoding="utf-8")
    assert load_profile("team").description == "shipped"
    Profile("team", "mine").save()
    assert load_profile("team").description == "mine" and set(list_profiles()) == {"team"}


def test_loading_by_path_and_by_missing_name(dirs, tmp_path):
    path = tmp_path / "any.json"
    path.write_text(json.dumps({"name": "x", "allow": ["Atlantis"]}), encoding="utf-8")
    assert load_profile(path).allow == [Rule("Atlantis")]            # a bare string is a literal rule
    assert load_profile(None) is None and load_profile("") is None
    with pytest.raises(ProfileError, match="No profile called 'nope'"):
        load_profile("nope")


def test_broken_files_give_clear_errors(dirs, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProfileError, match="not valid JSON"):
        load_profile(bad)
    bad.write_text(json.dumps({"allow": 5}), encoding="utf-8")
    with pytest.raises(ProfileError):
        load_profile(bad)


def test_the_shipped_example_profiles_are_valid():
    shipped = Path(profiles.__file__).resolve().parent.parent / "profiles"
    found = sorted(shipped.glob("example_*.json"))
    assert len(found) >= 3
    for path in found:
        profile = load_profile(path)                      # validates every rule, including the regexes
        assert profile.deny and profile.allow


def test_the_hr_example_deny_pattern_finds_employee_numbers():
    shipped = Path(profiles.__file__).resolve().parent.parent / "profiles"
    hits = deny_findings("Fiche de EMP-04821 et emp 1234, matricule : AB-12", load_profile(shipped / "example_hr.json"))
    assert {f.text for f in hits} >= {"EMP-04821", "emp 1234"} and all(f.kind == "ID" for f in hits)


def test_adding_choices_returns_a_copy_without_duplicates():
    base = Profile("p", allow=[Rule("Atlantis")])
    more = base.with_allowed(["atlantis", "Helios", " ", "Helios"]).with_denied(["Project Falcon", "project falcon"])
    assert [r.text for r in more.allow] == ["Atlantis", "Helios"] and [r.text for r in more.deny] == ["Project Falcon"]
    assert [r.text for r in base.allow] == ["Atlantis"] and base.deny == []             # the original is untouched


# ------------------------------------------------------------------ allow semantics
def F(kind, text, score=90):
    return Finding(kind, text, score)


def kept_texts(findings, profile):
    return [f.text for f in apply_allow(findings, profile)[0]]


def test_allow_matches_exact_text_ignoring_case():
    p = Profile(allow=[Rule("Atlantis Assurances")])
    assert kept_texts([F("PERSON", "ATLANTIS assurances"), F("PERSON", "Julien Marchand")], p) == ["Julien Marchand"]


def test_allow_also_covers_pieces_of_an_allowed_phrase_but_not_longer_text():
    p = Profile(allow=[Rule("Atlantis Assurances")])
    assert kept_texts([F("PERSON", "Atlantis"), F("PERSON", "Assurances")], p) == []
    assert kept_texts([F("PERSON", "Atlantis Assurances Group")], p) == ["Atlantis Assurances Group"]      # more than allowed


def test_allow_regex_must_match_the_whole_finding():
    p = Profile(allow=[Rule(r"INV-\d{4}-\d+", regex=True)])
    assert kept_texts([F("REFERENCE", "INV-2026-004211"), F("REFERENCE", "SIN-INV-2026-004211")], p) == ["SIN-INV-2026-004211"]


def test_allow_can_be_limited_to_one_kind():
    p = Profile(allow=[Rule("Paris", kind="LOCATION")])
    assert kept_texts([F("LOCATION", "Paris"), F("PERSON", "Paris")], p) == ["Paris"]            # the PERSON one stays
    assert apply_allow([F("LOCATION", "Paris")], p)[1] == [F("LOCATION", "Paris")]


def test_no_profile_changes_nothing():
    findings = [F("EMAIL", "a@b.co")]
    assert apply_allow(findings, None) == (findings, []) and apply_allow(findings, Profile()) == (findings, [])


# ------------------------------------------------------------------ deny semantics
def test_deny_finds_every_occurrence_of_a_literal_ignoring_case():
    p = Profile(deny=[Rule("Project Falcon")])
    found = deny_findings("Project Falcon starts. Later, project falcon ends.", p)
    assert [f.text for f in found] == ["Project Falcon", "project falcon"]
    assert {(f.kind, f.score) for f in found} == {("CUSTOM", 95)}


def test_deny_regex_uses_its_own_kind_and_score():
    p = Profile(deny=[Rule(r"EMP-\d{5}", regex=True, kind="ID", score=99)])
    assert [(f.kind, f.text, f.score) for f in deny_findings("see EMP-04821 and EMP-1", p)] == [("ID", "EMP-04821", 99)]


def test_deny_is_capped_per_page():
    p = Profile(deny=[Rule("x")])
    assert len(deny_findings("x " * 5000, p)) == profiles.MAX_MATCHES_PER_PAGE


# ------------------------------------------------------------------ in the scanner
LINES = ["Rapport interne du Projet Falcon.", "Client : Julien Marchand, courriel julien.marchand@example.org",
         "Edité par Atlantis Assurances, support@atlantis.example"]


def test_allow_suppresses_and_returns_what_it_hid(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", LINES)
    plain = {f.text for f in scan_pdf(pdf, use_ner=False)}
    hidden = []
    kept = {f.text for f in scan_pdf(pdf, use_ner=False, profile=Profile(allow=[Rule("support@atlantis.example")]), suppressed=hidden)}
    assert "support@atlantis.example" in plain and "support@atlantis.example" not in kept
    assert [f.text for f in hidden] == ["support@atlantis.example"] and "julien.marchand@example.org" in kept


def test_deny_adds_what_the_detector_could_not_know_and_redaction_removes_it(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", LINES)
    assert "Projet Falcon" not in {f.text for f in scan_pdf(pdf, use_ner=False)}
    profile = Profile(deny=[Rule("Projet Falcon")])
    findings = scan_pdf(pdf, use_ner=False, profile=profile)
    assert ("CUSTOM", "Projet Falcon", 95) in {(f.kind, f.text, f.score) for f in findings}
    out = str(tmp_path / "o.pdf")
    res = apply_redactions(pdf, out, [f.text for f in findings if f.score >= 60], use_ner=False)
    assert res.verification.ok and "Falcon" not in text_of(out)


def test_always_redact_beats_never_flag(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", LINES)
    both = Profile(allow=[Rule("Projet Falcon")], deny=[Rule("Projet Falcon", kind="ID")])
    hidden = []
    findings = scan_pdf(pdf, use_ner=False, profile=both, suppressed=hidden)
    assert ("ID", "Projet Falcon") in {(f.kind, f.text) for f in findings} and not any(f.text == "Projet Falcon" for f in hidden)


def test_an_allowed_barcode_payload_is_not_flagged(tmp_path):
    import io
    import numpy as np
    import zxingcpp
    from PIL import Image
    arr = np.asarray(zxingcpp.create_barcode("TICKET-0042-ABCDEF", zxingcpp.BarcodeFormat.QRCode).to_image()) \
        if hasattr(zxingcpp, "create_barcode") else np.asarray(zxingcpp.write_barcode(zxingcpp.BarcodeFormat.QRCode, "TICKET-0042-ABCDEF", width=300, height=300))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((60, 80), "Entry ticket for the museum visit, valid on the printed date only.", fontsize=10)
    page.insert_image(fitz.Rect(380, 560, 520, 700), stream=buf.getvalue())
    path = str(tmp_path / "t.pdf")
    pdf.save(path)
    assert any(f.kind == "BARCODE" for f in scan_pdf(path, use_ner=False))
    assert not any(f.kind == "BARCODE" for f in scan_pdf(path, use_ner=False, profile=Profile(allow=[Rule("TICKET-0042-ABCDEF")])))


def test_profiles_apply_in_case_mode(tmp_path):
    a = make_pdf(tmp_path / "a.pdf", LINES)
    b = make_pdf(tmp_path / "b.pdf", ["Encore le Projet Falcon, sans autre donnée personnelle ici."])
    profile = Profile(allow=[Rule("support@atlantis.example")], deny=[Rule("Projet Falcon")])
    scans = caseset.scan_files([("a.pdf", a), ("b.pdf", b)], use_ner=False, lang="fr", profile=profile)
    assert [f.text for f in scans[0].suppressed] == ["support@atlantis.example"]
    registry = caseset.build_registry(scans)
    falcon = next(e for e in registry if e.text == "Projet Falcon")
    assert sorted(falcon.files) == ["a.pdf", "b.pdf"] and falcon.score == 95


# ------------------------------------------------------------------ command line
def test_cli_uses_the_profile_for_defaults_rules_and_the_audit(tmp_path, dirs):
    pdf = make_pdf(tmp_path / "a.pdf", LINES)
    prof = tmp_path / "p.json"
    prof.write_text(json.dumps({"name": "team", "language": "fr", "min_score": 90,
                                "allow": [{"text": "support@atlantis.example"}], "deny": [{"text": "Projet Falcon"}]}), encoding="utf-8")
    out, audit = str(tmp_path / "o.pdf"), str(tmp_path / "audit.json")
    assert cli_main([pdf, out, "--profile", str(prof), "--no-ner", "--no-ocr", "--audit", audit]) == 0
    record = json.loads(Path(audit).read_text(encoding="utf-8"))["files"][0]
    assert record["profile"] == "team" and record["language"] == "fr" and record["min_score"] == 90
    assert record["suppressed_by_allow_list"] == {"EMAIL": 1} and "atlantis" not in json.dumps(record)
    left = text_of(out)
    assert "Falcon" not in left and "support@atlantis.example" in left            # denied: gone; allowed: left alone


def test_explicit_cli_flags_beat_the_profile(tmp_path, dirs):
    pdf = make_pdf(tmp_path / "a.pdf", LINES)
    prof = tmp_path / "p.json"
    prof.write_text(json.dumps({"name": "team", "language": "fr", "min_score": 99}), encoding="utf-8")
    audit = str(tmp_path / "a.json")
    assert cli_main([pdf, str(tmp_path / "o.pdf"), "--profile", str(prof), "--lang", "en", "--min-score", "10", "--no-ner",
                     "--no-ocr", "--audit", audit]) == 0
    record = json.loads(Path(audit).read_text(encoding="utf-8"))["files"][0]
    assert record["language"] == "en" and record["min_score"] == 10


def test_cli_rejects_an_unknown_or_broken_profile(tmp_path, dirs, capsys):
    pdf = make_pdf(tmp_path / "a.pdf", LINES)
    assert cli_main([pdf, str(tmp_path / "o.pdf"), "--profile", "nope"]) == 2
    assert "No profile called" in capsys.readouterr().out
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"deny": [{"text": "(a+)+", "regex": True}]}), encoding="utf-8")
    assert cli_main([pdf, str(tmp_path / "o.pdf"), "--profile", str(bad)]) == 2
    assert "nested quantifiers" in capsys.readouterr().out


def test_cli_case_mode_uses_the_profile(tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    make_pdf(folder / "a.pdf", LINES)
    make_pdf(folder / "b.pdf", ["Autre note sur le Projet Falcon."])
    prof = tmp_path / "p.json"
    prof.write_text(json.dumps({"name": "team", "deny": [{"text": "Projet Falcon"}]}), encoding="utf-8")
    out = tmp_path / "out"
    assert cli_main([str(folder), str(out), "--case", "--profile", str(prof), "--no-ner", "--no-ocr", "--lang", "fr", "--min-score", "60"]) == 0
    assert "Falcon" not in text_of(str(out / "a_redacted.pdf")) and "Falcon" not in text_of(str(out / "b_redacted.pdf"))


# ------------------------------------------------------------------ the app
def test_ui_remembers_choices_in_a_profile_and_selects_it(tmp_path, dirs):
    from streamlit.testing.v1 import AppTest
    pdf = make_pdf(tmp_path / "a.pdf", LINES)
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=240)
    at.run()
    at.file_uploader[0].upload("a.pdf", open(pdf, "rb").read(), "application/pdf")
    at.selectbox[0].set_value("fr").run()
    at.text_area[0].set_value("Projet Falcon").run()
    at.text_input(key="profile_save_name").set_value("mine").run()
    [b for b in at.button if b.label == "Save to profile"][0].click()
    at.run()
    assert not at.exception
    saved = load_profile("mine")
    assert [r.text for r in saved.deny] == ["Projet Falcon"]
    assert at.session_state["profile_select"] == "mine"
    assert any("Projet Falcon" == f.text for f in at.session_state["findings"])           # the deny rule is applied straight away


def test_ui_hides_allowed_findings_and_says_so(tmp_path, dirs):
    from streamlit.testing.v1 import AppTest
    Profile("quiet", allow=[Rule("support@atlantis.example")]).save()
    pdf = make_pdf(tmp_path / "a.pdf", LINES)
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=240)
    at.run()
    at.selectbox(key="profile_select").set_value("quiet").run()
    at.file_uploader[0].upload("a.pdf", open(pdf, "rb").read(), "application/pdf")
    at.run()
    assert not at.exception
    assert [f.text for f in at.session_state["hidden"]] == ["support@atlantis.example"]
    assert not any(f.text == "support@atlantis.example" for f in at.session_state["findings"])
    assert any("hidden by the allow list" in e.label for e in at.expander)
