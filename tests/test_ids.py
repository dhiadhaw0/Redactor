"""Machine-readable zones, national ID numbers and QR codes / barcodes."""
import io

import numpy as np
import pymupdf as fitz
import pytest
import zxingcpp
from PIL import Image

from redactor import codes, mrz
from redactor import scoring as sc
from redactor.detect import detect, detect_regex
from redactor.redact import apply_redactions, preview_pages, scan_pdf
from redactor.verify import verify_output

# The ICAO 9303 specimen passport (publicly documented example)
ICAO = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]


def kinds(findings):
    return {(f.kind, f.text) for f in findings}


# ------------------------------------------------------------------ MRZ
def test_check_digits_match_the_icao_specimen():
    assert [mrz.check_digit(f) for f in ("L898902C3", "740812", "120415", "ZE184226B<<<<<")] == ["6", "2", "9", "1"]
    assert mrz.valid("740812", "2") and not mrz.valid("740812", "3")


def test_the_specimen_passport_is_found_with_full_confidence():
    got = mrz.detect_mrz("\n".join(ICAO))
    assert [(k, s) for k, _, s in got if k == "MRZ"] == [("MRZ", sc.MRZ_VALID)] * 2
    assert {("PERSON", "Eriksson"), ("PERSON", "Anna"), ("PERSON", "Maria"), ("PERSON", "Anna Maria Eriksson"),
            ("ID", "L898902C3")} <= {(k, t) for k, t, _ in got}


def test_built_passports_and_id_cards_pass_their_own_check_digits():
    td3 = mrz.build_td3("Garnier", "Hugo Pierre", "FRA", "12AB34567", "FRA", "850312", "M", "300101")
    td1 = mrz.build_td1("Garnier", "Hugo", "FRA", "123456789", "FRA", "850312", "M", "300101")
    assert [len(l) for l in td3] == [44, 44] and [len(l) for l in td1] == [30, 30, 30]
    assert {s for k, _, s in mrz.detect_mrz("\n".join(td3)) if k == "MRZ"} == {sc.MRZ_VALID}
    assert {s for k, _, s in mrz.detect_mrz("\n".join(td1)) if k == "MRZ"} == {sc.MRZ_VALID}


def test_a_wrong_check_digit_lowers_the_score_but_the_line_is_still_found():
    bad = [ICAO[0], ICAO[1][:9] + "7" + ICAO[1][10:]]          # document-number check digit corrupted
    scores = {s for k, _, s in mrz.detect_mrz("\n".join(bad)) if k == "MRZ"}
    assert scores == {sc.MRZ_VALID} or scores == {sc.MRZ_PARTIAL}          # other digits still verify
    nothing_valid = [ICAO[0], "L898902C3" + "7" + ICAO[1][10:28] + "0" + ICAO[1][29:42] + "0" + "0"]
    assert all(k != "MRZ" or s < sc.MRZ_VALID for k, _, s in mrz.detect_mrz("\n".join(nothing_valid)))


def test_ocr_damage_spaces_and_lookalike_filler_is_forgiven():
    noisy = ICAO[0].replace("<", "«", 3).replace("ERIKSSON", "ERIK SSON"), ICAO[1]
    assert any(k == "MRZ" and s == sc.MRZ_VALID for k, _, s in mrz.detect_mrz("\n".join(noisy)))


def test_a_line_with_extra_ocr_characters_is_still_an_mrz_line():
    first = ICAO[0][:-2] + "e<e<<"                                                # 46 characters
    got = mrz.detect_mrz(first + "\n" + ICAO[1])
    assert any(k == "MRZ" and t == first and s == sc.MRZ_SHAPE for k, t, s in got) and ("PERSON", "Eriksson", 90) in got


def test_a_lone_second_line_is_found():
    assert any(k == "MRZ" for k, _, _ in mrz.detect_mrz("Date of birth : 12/08/1974\n" + ICAO[1]))


def test_ocr_junk_is_never_taken_for_a_name():
    junk = "P<GBRCARTER<<LUCAS<<<<<<<<<<<<<<< £6 6<< £66666"
    assert {t for k, t, _ in mrz.detect_mrz(junk) if k == "PERSON"} == {"Carter", "Lucas", "Lucas Carter"}


def test_ordinary_capital_text_is_not_an_mrz():
    for text in ("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG TODAY ONLY", "TERMS AND CONDITIONS OF SALE AND DELIVERY 2026",
                 "A" * 44, "SURNAME CARTER GIVEN NAMES LUCAS DOCUMENT NUMBER HF8266682 ISSUED"):
        assert mrz.detect_mrz(text) == [], text


def test_the_name_and_number_in_an_mrz_are_redacted_everywhere_else_on_the_page():
    """The printed fields repeat what the MRZ holds: they must go too (findings inside an MRZ line are not 'just pieces')."""
    text = "Surname : ERIKSSON\nGiven names : Anna Maria\nPassport No : L898902C3\n\n" + "\n".join(ICAO)
    found = kinds(detect(text, use_ner=False))
    assert {("PERSON", "Eriksson"), ("ID", "L898902C3"), ("PERSON", "Anna")} <= found


def test_a_passport_page_is_redacted_completely(tmp_path):
    path = str(tmp_path / "passport.pdf")
    pdf = fitz.open()
    page = pdf.new_page()
    for i, line in enumerate(["PASSPORT", "Surname : ERIKSSON", "Given names : Anna Maria", "Passport No : L898902C3", ""]):
        page.insert_text((60, 80 + 15 * i), line, fontsize=10)
    for i, line in enumerate(ICAO):
        page.insert_text((60, 170 + 15 * i), line, fontsize=10, fontname="cour")
    pdf.save(path)
    out = str(tmp_path / "out.pdf")
    findings = scan_pdf(path, use_ner=False)
    res = apply_redactions(path, out, [f.text for f in findings if f.score >= 60], use_ner=False)
    left = fitz.open(out)[0].get_text().lower()
    assert res.verification.ok and res.quality.grade == "A"
    for secret in ("eriksson", "anna", "l898902c3", "utoeriksson", "<<"):
        assert secret not in left, secret


# ------------------------------------------------------------------ national ID numbers
def make_nir(base13="1850369123456"):
    return f"{base13[0]} {base13[1:3]} {base13[3:5]} {base13[5:7]} {base13[7:10]} {base13[10:13]} {97 - int(base13) % 97:02d}"


def test_a_french_nir_is_recognised_only_with_a_valid_key():
    good = make_nir()
    assert [(f.kind, f.score) for f in detect_regex("NIR " + good)] == [("ID", sc.NATIONAL_ID_VALID)]
    wrong_key = good[:-2] + f"{(int(good[-2:]) + 1) % 97:02d}"
    assert all(f.kind != "ID" for f in detect_regex("NIR " + wrong_key))       # 15 digits without a verifying key
    assert [f.kind for f in detect_regex(wrong_key)] == ["NUMBER"]              # still caught as a long number


def test_corsican_nir_with_2a_is_handled():
    base = "1850" + "32" + "A" + "123456"       # department 2A -> 19 for the key
    digits = f"1850319123456"
    key = 97 - int(digits) % 97
    assert mrz is not None and [f.kind for f in detect_regex(f"1 85 03 2A 123 456 {key:02d}")] == ["ID"]


def test_uk_national_insurance_us_ssn_and_tunisian_tax_id():
    got = {f.text: f.kind for f in detect_regex("NI AB 12 34 56 C ; SSN 123-45-6789 ; MF 1234567 A/M/P/000")}
    assert got == {"AB 12 34 56 C": "ID", "123-45-6789": "ID", "1234567 A/M/P/000": "ID"}


def test_invalid_national_ids_are_not_flagged():
    assert detect_regex("SSN 000-12-3456 and 666-12-3456 and 900-12-3456") == []          # never issued
    assert all(f.kind != "ID" for f in detect_regex("NI QQ 12 34 56 C"))                    # Q is not a valid NI prefix


# ------------------------------------------------------------------ QR codes and barcodes
PAYLOAD = "PAYER:Maria Garcia|IBAN:FR14 2004 1010 0505 0001 3M02 606|REF:SIN-2026-0012345"


def qr_png(payload, size=300):
    arr = np.asarray(zxingcpp.write_barcode(zxingcpp.BarcodeFormat.QRCode, payload, width=size, height=size))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def page_with_qr(path, payloads=(PAYLOAD,), rotation=0, text="Quarterly statement with enough ordinary words to be a text page."):
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((60, 80), text, fontsize=11)
    for i, payload in enumerate(payloads):
        page.insert_image(fitz.Rect(380, 560 + 130 * i, 520, 700 + 130 * i), stream=qr_png(payload))
    page.set_rotation(rotation)
    pdf.save(path)
    return str(path)


def test_zxing_is_available():
    assert codes.available()


def test_a_qr_code_is_decoded_and_located(tmp_path):
    [hit] = codes.detect_codes(fitz.open(page_with_qr(tmp_path / "q.pdf"))[0])
    assert hit.payload == PAYLOAD and "QR" in hit.format
    r = hit.box
    assert r.x0 <= 380 and r.x1 >= 520 and r.y0 <= 560 and r.y1 >= 700        # covers the image, with a margin around it
    assert r.width < 200 and r.height < 200                                     # ...but is not the whole page


def test_two_codes_are_both_found(tmp_path):
    hits = codes.detect_codes(fitz.open(page_with_qr(tmp_path / "q.pdf", ("ONE-1234567890", "TWO-0987654321")))[0])
    assert sorted(h.payload for h in hits) == ["ONE-1234567890", "TWO-0987654321"]


def test_a_code_becomes_a_barcode_finding_scored_by_what_it_holds(tmp_path):
    findings = {f.text: f for f in scan_pdf(page_with_qr(tmp_path / "q.pdf", (PAYLOAD, "5901234123457")), use_ner=False)}
    assert findings[PAYLOAD].kind == "BARCODE" and findings[PAYLOAD].score == 97          # holds an IBAN
    assert findings["5901234123457"].score == sc.BARCODE                                  # an opaque product-style code


def test_a_selected_code_is_blacked_out_and_can_no_longer_be_decoded(tmp_path):
    src = page_with_qr(tmp_path / "q.pdf")
    out = str(tmp_path / "out.pdf")
    res = apply_redactions(src, out, [PAYLOAD], use_ner=False)
    assert res.hits == 1 and res.verification.ok, res.verification.problems
    assert codes.detect_codes(fitz.open(out)[0]) == []
    assert res.quality.grade == "A" and "Quarterly statement" in fitz.open(out)[0].get_text()


def test_an_unselected_code_is_left_alone(tmp_path):
    src = page_with_qr(tmp_path / "q.pdf")
    res = apply_redactions(src, str(tmp_path / "out.pdf"), ["something else"], use_ner=False)
    assert res.hits == 0 and len(codes.detect_codes(fitz.open(str(tmp_path / "out.pdf"))[0])) == 1


def test_codes_can_be_switched_off(tmp_path):
    src = page_with_qr(tmp_path / "q.pdf")
    assert [f for f in scan_pdf(src, use_ner=False, codes=False) if f.kind == "BARCODE"] == []
    assert apply_redactions(src, str(tmp_path / "o.pdf"), [PAYLOAD], use_ner=False, codes=False).hits == 0


def test_verification_notices_a_code_that_is_still_readable(tmp_path):
    src = page_with_qr(tmp_path / "q.pdf")
    v = verify_output(src, [PAYLOAD], [], code_payloads={PAYLOAD})          # the unredacted file
    assert not v.ok and "can still be decoded" in v.problems[0]


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_a_code_on_a_rotated_page_is_redacted_where_it_is(tmp_path, rotation):
    src = page_with_qr(tmp_path / "q.pdf", rotation=rotation)
    out = str(tmp_path / "out.pdf")
    res = apply_redactions(src, out, [PAYLOAD], use_ner=False)
    assert res.hits == 1 and res.verification.ok, res.verification.problems
    assert codes.detect_codes(fitz.open(out)[0]) == []


def test_a_code_on_a_scanned_page_is_found_and_redacted(tmp_path):
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((60, 80), "Statement", fontsize=14)
    page.insert_image(fitz.Rect(380, 560, 520, 700), stream=qr_png(PAYLOAD))
    scan = fitz.open()
    sp = scan.new_page()
    sp.insert_image(sp.rect, stream=page.get_pixmap(dpi=200).tobytes("png"))
    src = str(tmp_path / "scan.pdf")
    scan.save(src)
    out = str(tmp_path / "out.pdf")
    res = apply_redactions(src, out, [PAYLOAD], use_ner=False, ocr=False)
    assert res.hits == 1 and res.verification.ok and codes.detect_codes(fitz.open(out)[0]) == []


def test_the_preview_shades_a_selected_code(tmp_path):
    src = page_with_qr(tmp_path / "q.pdf")
    assert preview_pages(src, [PAYLOAD])[0] != preview_pages(src, ["nothing"])[0]


def test_barcode_text_is_never_treated_as_a_piece_of_something_else():
    """A name inside a QR payload and the same name printed elsewhere are both findings."""
    from redactor.detect import CONTAINERS
    assert {"MRZ", "BARCODE", "EMAIL", "URL"} == CONTAINERS


def test_risk_counts_mrz_and_barcodes():
    from redactor.detect import Finding
    from redactor.risk import WEIGHTS, assess
    assert WEIGHTS["MRZ"] >= 25 and WEIGHTS["BARCODE"] > WEIGHTS["LOCATION"]
    assert assess([Finding("MRZ", "P<UTO", 99), Finding("PERSON", "Anna Eriksson", 90)]).bonuses  # name + identity data
