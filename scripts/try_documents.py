"""Run the detector over a folder of real documents (PDFs or photos) and write,
for each one, a preview PNG per page plus the redacted PDF, so you can check by eye
what was found and what was missed.

    python scripts/try_documents.py samples/real other.pdf --lang fr

Output goes to ./out/ (git-ignored). Nothing leaves your machine.
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from redactor.loader import is_supported, to_pdf  # noqa: E402
from redactor.redact import apply_redactions, preview_pages, scan_pdf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["samples/real"], help="files and/or folders")
    ap.add_argument("--lang", choices=["en", "fr", "ar"], default="en")
    ap.add_argument("--no-ner", action="store_true")
    args = ap.parse_args()

    candidates = []
    for arg in args.paths:
        path = Path(arg)
        candidates += sorted(path.glob("*")) if path.is_dir() else [path]
    files = [p for p in candidates if is_supported(p)]
    if not files:
        sys.exit(f"No PDFs or images found in {args.paths}")
    out = Path("out")
    out.mkdir(exist_ok=True)

    for f in files:
        print(f"\n=== {f.name}")
        with tempfile.TemporaryDirectory() as tmp:
            pdf = to_pdf(str(f), str(Path(tmp) / "in.pdf"))
            warns = []
            findings = scan_pdf(pdf, use_ner=not args.no_ner, lang=args.lang, warnings=warns)
            for w in warns:
                print(f"  WARNING: {w}")
            for x in findings:
                print(f"  [{x.kind}] {x.text}")
            print(f"  -> {len(findings)} detections")
            terms = [x.text for x in findings]
            for i, png in enumerate(preview_pages(pdf, terms, lang=args.lang), 1):
                (out / f"{f.stem}_preview_p{i}.png").write_bytes(png)
            res = apply_redactions(pdf, str(out / f"{f.stem}_redacted.pdf"), terms, lang=args.lang)
            if res.hidden_removed:
                print("  hidden content removed: " + "; ".join(res.hidden_removed))
            print("  verification: " + ("PASSED" if res.verification.ok else "FAILED"))
            for problem in res.verification.problems:
                print("    - " + problem)
    print(f"\nPreviews and redacted PDFs written to {out.resolve()}")
    print("Check the previews: anything personal that is NOT red was missed.")


if __name__ == "__main__":
    main()
