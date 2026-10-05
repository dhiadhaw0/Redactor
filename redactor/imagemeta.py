"""Photo metadata: EXIF (camera, date, GPS position, a thumbnail of the original), XMP, IPTC and comments.

It lives inside the JPEG bytes. PyMuPDF embeds a JPEG as it is, and an image that no redaction box touches is
never re-encoded, so without this the "redacted" PDF would still tell a reader where and when the photo was taken.

JPEGs are cleaned losslessly: the metadata segments are cut out of the byte stream and the compressed picture is
not touched. Segments that affect how the picture looks are kept (JFIF, ICC colour profile, Adobe colour flag).
"""
import io
import struct

SOI, EOI, SOS = 0xD8, 0xD9, 0xDA
KEEP_APP = {0xE0, 0xE2, 0xEE}            # APP0 JFIF, APP2 ICC profile, APP14 Adobe colour transform
STANDALONE = set(range(0xD0, 0xD8)) | {0x01}


def _segments(data: bytes):
    """Yield (marker, start, end) for every segment before the compressed scan; then (None, pos, len)."""
    if data[:2] != bytes([0xFF, SOI]):
        return
    pos = 2
    n = len(data)
    while pos + 4 <= n:
        if data[pos] != 0xFF:
            break
        marker = data[pos + 1]
        if marker == 0xFF:                      # fill byte
            pos += 1
            continue
        if marker in STANDALONE:
            yield marker, pos, pos + 2
            pos += 2
            continue
        if marker == EOI:
            break
        length = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        end = pos + 2 + length
        yield marker, pos, end
        if marker == SOS:
            pos = end
            break
        pos = end
    yield None, pos, n                          # the entropy-coded scan, copied verbatim


def _kind(marker: int, payload: bytes) -> str | None:
    """What a segment holds, if it is metadata; None for the segments a picture needs."""
    if marker == 0xE1:
        if payload.startswith(b"Exif"):
            return "EXIF"
        if payload.startswith(b"http://ns.adobe.com/xap"):
            return "XMP"
        return "APP1 data"
    if marker == 0xED:
        return "IPTC / Photoshop"
    if marker == 0xFE:
        return "comment"
    if 0xE0 <= marker <= 0xEF and marker not in KEEP_APP:
        return f"APP{marker - 0xE0} data"
    return None


def metadata_in_jpeg(data: bytes) -> list[str]:
    """Names of the metadata kinds a JPEG carries, e.g. ['EXIF', 'GPS location', 'XMP']."""
    found = []
    for marker, start, end in _segments(data):
        if marker is None or marker in (SOS,):
            continue
        kind = _kind(marker, data[start + 4:end])
        if kind and kind not in found:
            found.append(kind)
    if "EXIF" in found and has_gps(data):
        found.insert(found.index("EXIF") + 1, "GPS location")
    return found


def has_gps(data: bytes) -> bool:
    try:
        from PIL import Image
        gps = Image.open(io.BytesIO(data)).getexif().get_ifd(0x8825)
        return bool(gps)
    except Exception:
        return False


def strip_jpeg(data: bytes) -> bytes:
    """The same JPEG without EXIF / XMP / IPTC / comments. Lossless."""
    if data[:2] != bytes([0xFF, SOI]):
        return data
    out = bytearray(b"\xff\xd8")
    for marker, start, end in _segments(data):
        if marker is None:
            out += data[start:end]
        elif _kind(marker, data[start + 4:end]) is None:
            out += data[start:end]
    return bytes(out)


# ------------------------------------------------------------------ inside a PDF
def _jpeg_xrefs(doc) -> list[int]:
    seen, out = set(), []
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen:
                continue
            seen.add(xref)
            if "DCTDecode" in (doc.xref_get_key(xref, "Filter")[1] or ""):
                out.append(xref)
    return out


def pdf_image_metadata(doc) -> dict[str, int]:
    """{kind: number of embedded JPEG images carrying it} for a PDF."""
    counts: dict[str, int] = {}
    for xref in _jpeg_xrefs(doc):
        for kind in metadata_in_jpeg(doc.xref_stream_raw(xref)):
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def strip_pdf_images(doc) -> int:
    """Cut the metadata out of every embedded JPEG. Returns how many images changed.

    update_stream() writes the new bytes but drops the image's /Filter, which would corrupt the picture, so the
    filter is put back. Only PyMuPDF's own errors are tolerated: anything else is a bug and must be seen."""
    changed = 0
    for xref in _jpeg_xrefs(doc):
        raw = doc.xref_stream_raw(xref)
        clean = strip_jpeg(raw)
        if clean == raw:
            continue
        filt = doc.xref_get_key(xref, "Filter")[1]
        doc.update_stream(xref, clean, compress=False)
        doc.xref_set_key(xref, "Filter", filt)
        changed += 1
    return changed
