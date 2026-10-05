"""Accept PDFs or images (photos/scans of documents) and normalise to PDF."""
import io
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageOps, ImageSequence

from . import office
from .imagemeta import strip_jpeg
from .office import OFFICE_SUFFIXES

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
EXIF_ORIENTATION = 0x0112


def _png(img: Image.Image, dpi=None) -> bytes:
    """One picture as PNG: pixels only. Ancillary chunks (text, EXIF, ICC, timestamps) do not survive. The
    resolution is kept: PyMuPDF sizes the PDF page from it, and a 300 dpi scan must stay A4, not become a poster."""
    if img.mode not in ("RGB", "L", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() or img.mode == "P" and "transparency" in img.info else "RGB")
    buf = io.BytesIO()
    img.save(buf, "PNG", **({"dpi": dpi} if dpi else {}))
    return buf.getvalue()


def _image_to_pdf_bytes(src: str) -> bytes:
    """Wrap an image (or every frame of a multi-page TIFF) in a PDF, without its metadata.

    * Phone photos are often stored sideways with an EXIF 'rotate me' flag that PDF conversion ignores, which
      would hand OCR a sideways page: the rotation is baked in first.
    * EXIF carries the camera, the date and often the GPS position. PyMuPDF would embed the JPEG as it is, and
      a picture no redaction box touches is never re-encoded, so the metadata must be cut out here (losslessly
      for JPEG), and other formats are re-encoded from their pixels.
    """
    pdf = fitz.open()
    with Image.open(src) as img:
        is_jpeg = img.format == "JPEG"
        flagged = img.getexif().get(EXIF_ORIENTATION, 1) not in (0, 1)
        if is_jpeg and not flagged:
            pages = [("jpeg", strip_jpeg(Path(src).read_bytes()))]
        else:
            pages = [("png", _png(ImageOps.exif_transpose(frame.copy()), frame.info.get("dpi")))
                     for frame in ImageSequence.Iterator(img)]
    for kind, data in pages:
        with fitz.open(stream=data, filetype=kind) as pic:
            pdf.insert_pdf(fitz.open("pdf", pic.convert_to_pdf()))
    return pdf.tobytes()


SUPPORTED_SUFFIXES = {".pdf"} | IMAGE_SUFFIXES | OFFICE_SUFFIXES


def is_supported(path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_SUFFIXES


def convert(src: str, dst: str) -> list[str]:
    """Write `src` as a PDF at `dst` and return what the original carried that the PDF does not (comments, speaker
    notes, hidden sheets, document properties, ...). PDFs and images lose nothing worth reporting here: an
    image's pixels are carried over whole and a PDF is copied."""
    suffix = Path(src).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        Path(dst).write_bytes(_image_to_pdf_bytes(src))
    elif suffix == ".pdf":
        Path(dst).write_bytes(Path(src).read_bytes())
    elif suffix in OFFICE_SUFFIXES:
        return office.convert(src, dst)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return []


def to_pdf(src: str, dst: str) -> str:
    """Copy a PDF, or convert an image / Office file / e-mail / text file to a PDF, at `dst`."""
    convert(src, dst)
    return dst
