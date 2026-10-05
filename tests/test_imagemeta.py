"""Photo metadata (EXIF, GPS, XMP, comments) must not survive into the redacted file."""
import io
import json

import pymupdf as fitz
import pytest
from PIL import Image, ImageDraw, PngImagePlugin

from redactor import imagemeta
from redactor.cli import main as cli_main
from redactor.loader import convert, to_pdf
from redactor.redact import apply_redactions
from redactor.verify import describe_hidden, inspect_hidden

CAMERA = b"SecretCameraMaker"
GPS_TUNIS = {1: "N", 2: (36.0, 48.0, 25.0), 3: "E", 4: (10.0, 10.0, 30.0)}


def picture(text="Customer: Julien Marchand", size=(800, 400)):
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).text((20, 40), text, fill="black")
    return img


def jpeg_bytes(img=None, gps=True, camera=True, comment=None, quality=92, icc=None):
    exif = Image.Exif()
    if camera:
        exif[0x010F], exif[0x0110], exif[0x0132] = CAMERA.decode(), "ModelXYZ-9000", "2026:09:18 10:15:00"
    if gps:
        exif[0x8825] = GPS_TUNIS
    buf = io.BytesIO()
    kw = {"exif": exif, "quality": quality}
    if comment:
        kw["comment"] = comment
    if icc:
        kw["icc_profile"] = icc
    (img or picture()).save(buf, "JPEG", **kw)
    return buf.getvalue()


def add_segment(data: bytes, marker: int, payload: bytes) -> bytes:
    """Insert an APPn / COM segment right after the SOI marker."""
    return data[:2] + bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload + data[2:]


def pixels(jpeg: bytes):
    return Image.open(io.BytesIO(jpeg)).convert("RGB").tobytes()


# ------------------------------------------------------------------ detection
def test_exif_and_gps_are_detected():
    assert imagemeta.metadata_in_jpeg(jpeg_bytes()) == ["EXIF", "GPS location"]


def test_exif_without_gps_does_not_claim_a_location():
    assert imagemeta.metadata_in_jpeg(jpeg_bytes(gps=False)) == ["EXIF"]


def test_xmp_iptc_and_comments_are_detected():
    data = jpeg_bytes(gps=False, camera=False)
    data = add_segment(data, 0xE1, b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta>owner: Julien</x:xmpmeta>")
    data = add_segment(data, 0xED, b"Photoshop 3.0\x008BIM")
    data = add_segment(data, 0xFE, b"taken by Julien Marchand")
    kinds = imagemeta.metadata_in_jpeg(data)
    assert {"XMP", "IPTC / Photoshop", "comment"} <= set(kinds)


def test_a_clean_jpeg_has_no_metadata_and_other_bytes_are_ignored():
    clean = imagemeta.strip_jpeg(jpeg_bytes())
    assert imagemeta.metadata_in_jpeg(clean) == []
    assert imagemeta.metadata_in_jpeg(b"not a jpeg") == [] and imagemeta.strip_jpeg(b"not a jpeg") == b"not a jpeg"


# ------------------------------------------------------------------ lossless stripping
def test_stripping_is_lossless_and_removes_everything_identifying():
    src = jpeg_bytes(comment=b"private note")
    clean = imagemeta.strip_jpeg(src)
    assert pixels(clean) == pixels(src)                                      # the picture is bit-for-bit the same
    for secret in (b"Exif", CAMERA, b"ModelXYZ", b"private note"):
        assert secret not in clean
    assert len(clean) < len(src)
    assert imagemeta.strip_jpeg(clean) == clean                              # idempotent


def test_the_colour_profile_and_jfif_header_are_kept():
    icc = b"\x00" * 128 + b"fake icc profile body"
    clean = imagemeta.strip_jpeg(jpeg_bytes(icc=icc))
    markers = [m for m, _, _ in imagemeta._segments(clean) if m is not None]
    assert 0xE2 in markers                                                   # APP2 (ICC) kept
    assert 0xE1 not in markers and 0xFE not in markers


def test_a_truncated_or_odd_jpeg_does_not_crash():
    data = jpeg_bytes()
    for cut in (2, 3, 10, 40, len(data) // 2):
        imagemeta.strip_jpeg(data[:cut])
        imagemeta.metadata_in_jpeg(data[:cut])


# ------------------------------------------------------------------ photos turned into PDFs
def test_a_photo_loses_its_exif_when_it_becomes_a_pdf(tmp_path):
    src = tmp_path / "photo.jpg"
    src.write_bytes(jpeg_bytes())
    out = str(tmp_path / "photo.pdf")
    convert(str(src), out)
    data = open(out, "rb").read()
    assert b"Exif" not in data and CAMERA not in data and b"ModelXYZ" not in data
    page = fitz.open(out)[0]
    assert len(page.get_images()) == 1 and round(page.rect.width) == 600          # 800 px at 96 dpi
    assert inspect_hidden(out)["photo_metadata"] == {}


def test_a_sideways_photo_is_straightened_and_still_has_no_exif(tmp_path):
    exif = Image.Exif()
    exif[0x0112], exif[0x0110] = 6, "ModelXYZ-9000"
    exif[0x8825] = GPS_TUNIS
    src = tmp_path / "sideways.jpg"
    picture(size=(400, 200)).save(src, exif=exif)
    out = str(tmp_path / "s.pdf")
    convert(str(src), out)
    r = fitz.open(out)[0].rect
    assert r.height > r.width and b"Exif" not in open(out, "rb").read() and b"ModelXYZ" not in open(out, "rb").read()


def test_png_text_chunks_do_not_reach_the_pdf(tmp_path):
    info = PngImagePlugin.PngInfo()
    info.add_text("Author", "Julien Marchand")
    info.add_text("Location", "36.8N 10.1E")
    src = tmp_path / "p.png"
    picture().save(src, pnginfo=info)
    out = str(tmp_path / "p.pdf")
    convert(str(src), out)
    data = open(out, "rb").read()
    assert b"Julien Marchand" not in data and b"36.8N" not in data and b"tEXt" not in data


def test_a_multi_page_tiff_keeps_all_its_pages(tmp_path):
    frames = [picture(f"page {i}") for i in range(3)]
    src = tmp_path / "scan.tiff"
    frames[0].save(src, save_all=True, append_images=frames[1:])
    out = str(tmp_path / "scan.pdf")
    convert(str(src), out)
    assert len(fitz.open(out)) == 3


def test_transparent_palette_and_grayscale_images_still_convert(tmp_path):
    cases = {"gray.png": picture().convert("L"), "pal.png": picture().convert("P"),
             "rgba.png": Image.new("RGBA", (300, 200), (255, 0, 0, 80)), "bmp.bmp": picture(), "w.webp": picture()}
    for name, img in cases.items():
        src = tmp_path / name
        img.save(src)
        out = str(tmp_path / (name + ".pdf"))
        convert(str(src), out)
        assert len(fitz.open(out)) == 1, name


# ------------------------------------------------------------------ a PDF that already holds such a photo
@pytest.fixture
def pdf_with_exif_photo(tmp_path):
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((60, 80), "Inspection report with a photograph attached below, taken on site by the engineer.", fontsize=10)
    page.insert_image(fitz.Rect(60, 120, 360, 270), stream=jpeg_bytes())            # embedded as-is: EXIF inside
    path = tmp_path / "report.pdf"
    pdf.save(path)
    return str(path)


def test_hidden_content_inspection_sees_photo_metadata_inside_a_pdf(pdf_with_exif_photo):
    report = inspect_hidden(pdf_with_exif_photo)
    assert report["photo_metadata"] == {"EXIF": 1, "GPS location": 1}
    assert any("photo metadata in 1 embedded image(s) (EXIF, GPS location)" in line for line in describe_hidden(report))


def test_redaction_strips_it_even_when_no_box_touches_the_photo(pdf_with_exif_photo, tmp_path):
    out = str(tmp_path / "out.pdf")
    before = fitz.open(pdf_with_exif_photo)[0].get_pixmap(dpi=72).samples
    res = apply_redactions(pdf_with_exif_photo, out, [], use_ner=False, ocr=False)
    data = open(out, "rb").read()
    assert b"Exif" not in data and CAMERA not in data
    assert any("photo metadata" in h for h in res.hidden_removed) and res.hidden_remaining == []
    assert res.quality.grade == "A"
    assert fitz.open(out)[0].get_pixmap(dpi=72).samples == before                    # the picture itself is unchanged


def test_if_stripping_failed_the_file_would_be_an_F(pdf_with_exif_photo, tmp_path, monkeypatch):
    monkeypatch.setattr("redactor.verify.strip_pdf_images", lambda doc: 0)
    res = apply_redactions(pdf_with_exif_photo, str(tmp_path / "o.pdf"), [], use_ner=False, ocr=False)
    assert any("photo metadata" in h for h in res.hidden_remaining) and res.quality.grade == "F"


def test_a_photo_inside_a_word_file_arrives_without_exif(tmp_path):
    docx = pytest.importorskip("docx")
    photo = tmp_path / "p.jpg"
    photo.write_bytes(jpeg_bytes())
    d = docx.Document()
    d.add_paragraph("Site report with a photo for the file.")
    d.add_picture(str(photo))
    path = tmp_path / "r.docx"
    d.save(path)
    out = str(tmp_path / "r.pdf")
    convert(str(path), out)
    assert b"Exif" not in open(out, "rb").read() and inspect_hidden(out)["photo_metadata"] == {}


def test_cli_reports_photo_metadata_in_the_audit(tmp_path):
    src = tmp_path / "photo.jpg"
    src.write_bytes(add_segment(jpeg_bytes(), 0xFE, b"comment " * 4))
    out, audit = str(tmp_path / "o.pdf"), str(tmp_path / "a.json")
    assert cli_main([str(src), out, "--no-ner", "--no-ocr", "--audit", audit]) in (0, 1)   # no OCR: the page is "unreadable" (C)
    record = json.loads(open(audit, encoding="utf-8").read())["files"][0]
    assert b"Exif" not in open(out, "rb").read() and CAMERA not in open(out, "rb").read()
    assert record["source_format"] == "jpg"


def test_resolution_is_kept_so_a_scan_stays_the_size_of_its_paper(tmp_path):
    """A 300 dpi scan of an A4 page must give an A4 page, whatever format it arrives in."""
    for name, kwargs in (("a.png", {}), ("a.tif", {}), ("a.jpg", {"quality": 90})):
        src = tmp_path / name
        picture(size=(2480, 3508)).save(src, dpi=(300, 300), **kwargs)
        out = str(tmp_path / (name + ".pdf"))
        convert(str(src), out)
        r = fitz.open(out)[0].rect
        assert abs(r.width - 595) < 3 and abs(r.height - 842) < 3, (name, r)
