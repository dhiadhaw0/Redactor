"""Tests of the benchmark itself: its corpus, its metrics and its regression gate."""
import json
import random

import pymupdf as fitz
import pytest

from benchmark import corpus, metrics
from benchmark.identities import make_card, make_iban, make_person
from benchmark.run import RECALL_TOLERANCE, check, load_local, run_corpus
from redactor.detect import Finding, _iban_valid, _luhn_valid


# ------------------------------------------------------------------ corpus
def test_generated_ibans_and_cards_are_valid():
    rng = random.Random(1)
    assert all(_iban_valid(make_iban(rng, c)) for c in ("FR", "GB", "TN") for _ in range(30))
    assert all(_luhn_valid(make_card(rng)) for _ in range(100))


def test_corpus_is_reproducible_and_seed_dependent():
    a, b = corpus.build_corpus(7), corpus.build_corpus(7)
    assert [d.lines for d in a] == [d.lines for d in b]
    assert [d.lines for d in corpus.build_corpus(8)] != [d.lines for d in a]


def test_persons_are_consistent():
    p = make_person(random.Random(3), "fr")
    assert p.postcode_city.endswith(p.city) and p.iban.startswith("FR") and "@" in p.email


@pytest.mark.parametrize("doc", corpus.build_corpus(), ids=lambda d: d.name)
def test_every_labeled_item_is_really_in_the_rendered_page(doc, tmp_path):
    path = corpus.render(doc, "digital", str(tmp_path))
    page = fitz.open(path)[0]
    if doc.lang == "ar":  # the Arabic text layer is shaped glyphs, so it cannot be compared as text
        return
    text = " ".join(page.get_text().split())
    for kind, item in doc.items:
        if kind == "BARCODE":  # not printed text: it must decode from the page to exactly the labeled payload
            from redactor.codes import detect_codes
            assert item in {h.payload for h in detect_codes(page)}, item
            continue
        assert " ".join(item.split()) in text, item
    for line in doc.lines:  # nothing may run off the page, or a label would be silently unreadable
        assert not line or fitz.get_text_length(line, fontsize=corpus.FONT_SIZE) < page.rect.width - 70, line


def test_corpus_has_traps_for_precision():
    docs = {d.name: d for d in corpus.build_corpus()}
    text = " ".join(" ".join(d.lines) for d in docs.values())
    assert "1 240,00 EUR" in text and "12/01/2026" in text and "Spring Boot" in text and "2023 - 2024" in text
    assert docs["letter_en"].ignore and docs["invoice_fr"].ignore  # company details: allowed, not required


# ------------------------------------------------------------------ metrics
def test_coverage_counts_characters_not_all_or_nothing():
    iban = "GB82 WEST 1234 5698 7654 32"
    assert metrics.coverage(iban, [iban]) == 1.0
    assert metrics.coverage(iban, ["GB82 WEST 1234"]) == pytest.approx(12 / 22, abs=0.01)  # half an IBAN
    assert not metrics.ItemResult("IBAN", iban, metrics.coverage(iban, ["GB82 WEST 1234"])).covered
    assert metrics.coverage(iban, ["x " + iban + " y"]) == 1.0                              # bigger finding covers it


def test_pieces_together_cover_a_name():
    assert metrics.coverage("Dhia Toumi", ["Dhia", "Toumi"]) == 1.0
    assert metrics.coverage("Dhia Toumi", ["Dhia"]) < metrics.COVERED_AT


def test_fuzzy_matching_forgives_an_ocr_misread_only_for_scans():
    assert metrics.coverage("john.smith@example.com", ["jonn.smith@example.com"], fuzzy=False) < metrics.COVERED_AT
    assert metrics.coverage("john.smith@example.com", ["jonn.smith@example.com"], fuzzy=True) >= metrics.COVERED_AT


def test_right_to_left_words_compare_as_a_set():
    assert metrics.coverage("+216 62 987 559", ["559 987 62 216"]) < metrics.COVERED_AT
    assert metrics.coverage("+216 62 987 559", ["559 987 62 216"], rtl=True) == 1.0


def test_touches_credits_findings_that_are_mostly_an_item():
    assert metrics.touches("Mme MARCHAND", "MARCHAND Julien")            # blacks out the surname
    assert metrics.touches("Dhia", "Dhia Eddine Toumi")
    assert not metrics.touches("Invoice INV", "INV-2026-004211")         # only 3 shared characters
    assert not metrics.touches("Spring Boot", "Dhia Toumi")


def test_precision_ignores_acceptable_findings_but_not_traps():
    items = [("EMAIL", "a@b.co")]
    findings = [Finding("EMAIL", "a@b.co", 98), Finding("ADDRESS", "58 Elm Road", 88), Finding("PERSON", "Spring Boot", 70)]
    res = metrics.evaluate("d", "en", "digital", items, ["58 Elm Road"], findings, fuzzy=False)
    assert [p.useful for p in res.preds] == [True, True, False]
    c = metrics.summarize([res], 60)["overall"]
    assert (c.items, c.covered, c.preds, c.useful) == (1, 1, 3, 2)
    assert c.recall == 1.0 and c.precision == pytest.approx(2 / 3) and 0 < c.f1 < 1


def test_threshold_trades_recall_for_precision():
    items = [("EMAIL", "a@b.co"), ("PERSON", "Maria Garcia")]
    findings = [Finding("EMAIL", "a@b.co", 98), Finding("PERSON", "Maria Garcia", 65), Finding("PERSON", "Big Data", 40)]
    res = metrics.evaluate("d", "en", "digital", items, [], findings, fuzzy=False)
    everything, only_sure = metrics.summarize([res], 0)["overall"], metrics.summarize([res], 90)["overall"]
    assert everything.recall == 1.0 and everything.precision < 1.0   # nothing missed, one false positive
    assert only_sure.recall == 0.5 and only_sure.precision == 1.0    # nothing wrong, one missed


def test_empty_groups_do_not_divide_by_zero():
    c = metrics.Counts()
    assert c.recall == 1.0 and c.precision == 1.0 and c.f1 == 1.0


def test_summary_is_split_by_variant_language_and_kind():
    res = metrics.evaluate("d", "fr", "scan", [("EMAIL", "a@b.co")], [], [Finding("EMAIL", "a@b.co", 98)], fuzzy=True)
    assert {"overall", "variant:scan", "lang:fr", "kind:EMAIL"} <= set(metrics.summarize([res], 60))


# ------------------------------------------------------------------ regression gate
def snap(items=100, covered=96, predictions=100, useful=99):
    return {"recall": covered / items, "precision": useful / predictions, "items": items,
            "covered": covered, "predictions": predictions, "useful": useful}


def test_gate_passes_when_nothing_changed_or_things_improved():
    base = {"metrics": {"overall": snap(), "lang:fr": snap()}}
    assert check(base["metrics"], base) == ([], [])
    failures, better = check({"overall": snap(covered=99), "lang:fr": snap()}, base)
    assert failures == [] and len(better) == 1


def test_gate_fails_when_recall_drops_more_than_the_tolerance():
    base = {"metrics": {"overall": snap(), "kind:IBAN": snap(items=8, covered=8)}}
    failures, _ = check({"overall": snap(covered=93), "kind:IBAN": snap(items=8, covered=6)}, base)
    assert len(failures) == 2 and "overall" in failures[0] and "kind:IBAN" in failures[1]


def test_gate_allows_one_item_of_noise_in_a_small_group():
    base = {"metrics": {"kind:URL": snap(items=12, covered=11)}}
    assert check({"kind:URL": snap(items=12, covered=10)}, base)[0] == []   # one item: tolerated
    assert check({"kind:URL": snap(items=12, covered=9)}, base)[0] != []    # two items: not


def test_gate_fails_when_precision_drops():
    failures, _ = check({"overall": snap(useful=80)}, {"metrics": {"overall": snap()}})
    assert failures and "precision" in failures[0]


def test_gate_missing_group_fails_only_for_a_complete_run():
    base = {"metrics": {"variant:photo": snap()}}
    assert check({}, base, complete=True)[0] != [] and check({}, base, complete=False)[0] == []


def test_overall_recall_has_the_tightest_tolerance():
    assert RECALL_TOLERANCE["overall"] <= 0.01


# ------------------------------------------------------------------ local documents + a real mini run
def test_local_labels_are_loaded_relative_to_the_label_file(tmp_path):
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "a.labels.json").write_text(json.dumps({
        "file": "doc.pdf", "lang": "fr", "items": [{"kind": "PERSON", "text": "Jean Dupont"}], "ignore": ["Acme"]}),
        encoding="utf-8")
    [d] = load_local(tmp_path)
    assert d["name"] == "a" and d["lang"] == "fr" and d["path"] == str(tmp_path / "doc.pdf")
    assert d["items"] == [("PERSON", "Jean Dupont")] and d["ignore"] == ["Acme"]


def test_a_real_digital_run_meets_the_floor():
    """Not the full gate (CI runs that against the baseline): a quick floor on the English digital documents."""
    results = run_corpus({"digital"}, {"en"}, corpus.SEED, e2e=False, threshold=60)
    overall = metrics.summarize(results, 60)["overall"]
    assert overall.recall >= 0.95 and overall.precision >= 0.95
