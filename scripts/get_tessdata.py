"""Download the Tesseract language files used for OCR into ./tessdata."""
import urllib.request
from pathlib import Path

BASE = "https://github.com/tesseract-ocr/{repo}/raw/main/{lang}.traineddata"
# Arabic uses the larger 'best' model: the 'fast' one misreads digits.
FILES = {"eng": "tessdata_fast", "fra": "tessdata_fast", "ara": "tessdata_best"}

dest = Path(__file__).resolve().parent.parent / "tessdata"
dest.mkdir(exist_ok=True)
for lang, repo in FILES.items():
    target = dest / f"{lang}.traineddata"
    if target.exists():
        print(f"{lang}: already present")
        continue
    print(f"{lang}: downloading from {repo}...")
    urllib.request.urlretrieve(BASE.format(repo=repo, lang=lang), target)
print("Done.")
