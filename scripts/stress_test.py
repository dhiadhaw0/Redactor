"""How well does detection survive bad scans? Degrades the fake sample page
(skew, noise, blur, JPEG compression) at increasing levels and reports which
secrets are still detected.

    python scripts/stress_test.py [--lang en]
"""
import argparse
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pymupdf as fitz
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from redactor.loader import to_pdf  # noqa: E402
from redactor.make_sample import make_sample  # noqa: E402
from redactor.redact import scan_pdf  # noqa: E402

# (name, skew degrees, noise std, blur radius, jpeg quality, dpi)
LEVELS = [
    ("clean scan", 0.0, 0, 0.0, 95, 200),
    ("slight skew", 1.5, 0, 0.0, 90, 200),
    ("noisy office scan", 1.0, 12, 0.6, 60, 150),
    ("phone photo", 3.0, 18, 1.0, 45, 120),
    ("poor photo", 4.0, 28, 1.5, 30, 100),
]
SECRETS = {
    "en": ["john.smith@example.com", "maria.garcia@acme.org", "+44 20 7946 0958",
           "GB82 WEST 1234 5698 7654 32", "4111 1111 1111 1111", "John Smith", "Maria Garcia", "12 Baker Street"],
    "fr": ["jean.dupont@exemple.fr", "marie.curie@acme.fr", "+33 1 42 68 53 00",
           "FR14 2004 1010 0505 0001 3M02 606", "4111 1111 1111 1111", "Jean Dupont", "Marie Curie", "12 rue de la Paix"],
}


def degrade(page_png: bytes, skew, noise, blur, quality) -> bytes:
    img = Image.open(io.BytesIO(page_png)).convert("L")
    if skew:
        img = img.rotate(skew, expand=True, fillcolor=255, resample=Image.BICUBIC)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if noise:
        arr = np.asarray(img, dtype=np.float32)
        arr += np.random.default_rng(0).normal(0, noise, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "fr"], default="en")
    lang = ap.parse_args().lang
    secrets = SECRETS[lang]
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "base.pdf")
        make_sample(base, lang)
        for name, skew, noise, blur, q, dpi in LEVELS:
            png = fitz.open(base)[0].get_pixmap(dpi=dpi).tobytes("png")
            img_path = Path(tmp) / "scan.jpg"
            img_path.write_bytes(degrade(png, skew, noise, blur, q))
            pdf = to_pdf(str(img_path), str(Path(tmp) / "scan.pdf"))
            found = {f.text for f in scan_pdf(pdf, lang=lang)}
            # OCR often drops or moves spaces, so compare ignoring whitespace
            squash = lambda t: "".join(t.split())
            blob = squash(" | ".join(found))
            hit = [s for s in secrets if squash(s) in blob]
            miss = [s for s in secrets if squash(s) not in blob]
            print(f"{name:20s} {len(hit)}/{len(secrets)} found" + (f"   missed: {miss}" if miss else ""))


if __name__ == "__main__":
    main()
