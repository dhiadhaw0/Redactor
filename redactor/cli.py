"""Command line: redact one file, or every PDF/image in a folder.

    python -m redactor.cli in.pdf out.pdf --lang fr
    python -m redactor.cli scans/ redacted/ --lang fr          # batch, writes redacted/audit.json

Exit code 0 only if every file was redacted AND passed verification.
"""
import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .loader import convert, is_supported
from .profiles import ProfileError, load_profile
from . import ocr as ocr_module
from .caseset import SHARE_THRESHOLD, build_registry, process_set, scan_files, set_audit
from .quality import GRADE_ORDER
from .redact import apply_redactions, scan_pdf
from .risk import assess
from .scoring import DEFAULT_THRESHOLD


def process(src: Path, dst: Path, args) -> dict:
    """Redact one file. Returns an audit record: what kinds of data were removed and whether
    the result verified. It deliberately holds no personal text, only types and counts."""
    with tempfile.TemporaryDirectory() as tmp:
        pdf = str(Path(tmp) / "input.pdf")
        dropped = convert(str(src), pdf)
        if dropped:
            print(f"  converted from {src.suffix}; not carried into the PDF: " + "; ".join(dropped))
        warnings: list[str] = []
        hidden: list = []   # findings the profile's allow list dropped
        findings = scan_pdf(pdf, use_ner=not args.no_ner, ocr=not args.no_ocr, lang=args.lang, warnings=warnings,
                            profile=args.profile_obj, suppressed=hidden)
        if hidden:
            print(f"  {len(hidden)} finding(s) hidden by the allow list of profile {args.profile_obj.name!r}: "
                  + ", ".join(f"{k} x{n}" for k, n in sorted(Counter(f.kind for f in hidden).items())))
        chosen = [f for f in findings if f.score >= args.min_score]
        for f in sorted(findings, key=lambda f: -f.score):
            skipped = f" (below --min-score {args.min_score}, skipped)" if f.score < args.min_score else ""
            print(f"  {f.score:3d} [{f.kind}] {f.text}{skipped}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        res = apply_redactions(pdf, str(dst), [f.text for f in chosen], ocr=not args.no_ocr, lang=args.lang,
                               use_ner=not args.no_ner)
    risk_before, risk_after, quality = assess(findings), res.risk_after, res.quality
    for w in warnings:
        print(f"  WARNING: {w}")
    print(f"  redacted {res.hits} areas for {len(chosen)} items -> {dst}")
    print(f"  privacy risk: {risk_before.explain()}  ->  {risk_after.explain()}")
    print(f"  quality of the saved file: {quality.explain()}")
    for severity, text in quality.issues:
        print(f"    [{severity}] {text}")
    for note in quality.notes:
        print(f"    note: {note}")
    if res.hidden_removed:
        print("  removed hidden content: " + "; ".join(res.hidden_removed))
    if res.verification.ok:
        print("  verification passed: boxes are black and no redacted text is readable")
    else:
        print("  VERIFICATION FAILED - do not share this file:")
        for problem in res.verification.problems:
            print("    - " + problem)
    return {
        "file": src.name,
        "sha256_of_input": hashlib.sha256(src.read_bytes()).hexdigest(),
        "output": dst.name,
        "language": args.lang,
        "profile": args.profile_obj.name if args.profile_obj else None,
        "suppressed_by_allow_list": dict(Counter(f.kind for f in hidden)),   # kinds and counts only
        "ocr_engine": ocr_module.BACKEND,
        "items_by_type": dict(Counter(f.kind for f in chosen)),
        "areas_redacted": res.hits,
        "skipped_below_min_score": sum(1 for f in findings if f.score < args.min_score),
        "min_score": args.min_score,
        "score_range_of_redacted": [min((f.score for f in chosen), default=None),
                                    max((f.score for f in chosen), default=None)],
        "risk_before": risk_before.as_dict(),   # kinds, counts and points only, never the text
        "risk_after": risk_after.as_dict(),
        "quality": quality.as_dict(),          # grade, score, reasons: never the text
        "hidden_content_removed": res.hidden_removed,
        "source_format": src.suffix.lower().lstrip("."),
        "dropped_on_conversion": dropped,       # comments, notes, hidden sheets, properties... of an Office file or e-mail
        "verification_passed": res.verification.ok,
        "verification_problems": res.verification.problems,
        "warnings": warnings,
    }


def run_case(files: list[Path], out: Path, args) -> int:
    """A folder as ONE case: scan everything, merge what was found, redact every file with the merged
    decisions (confident findings are searched for in all files), verify and grade each file."""
    with tempfile.TemporaryDirectory() as tmp:
        pdfs = []
        for i, f in enumerate(files):
            pdf = str(Path(tmp) / f"{i}.pdf")
            dropped = convert(str(f), pdf)
            if dropped:
                print(f"  {f.name}: not carried into the PDF: " + "; ".join(dropped))
            pdfs.append((f.name, pdf))
        print(f"Case of {len(files)} files: scanning...")
        scans = scan_files(pdfs, use_ner=not args.no_ner, ocr=not args.no_ocr, lang=args.lang,
                           progress=lambda i, n, name: print(f"  [{i + 1}/{n}] {name}"), profile=args.profile_obj)
        hidden_total = sum(len(sc_.suppressed) for sc_ in scans)
        if hidden_total:
            print(f"{hidden_total} finding(s) hidden by the allow list of profile {args.profile_obj.name!r}")
        for scan in scans:
            for w in scan.warnings:
                print(f"  WARNING {scan.name}: {w}")
        registry = build_registry(scans)
        selected = {e.key for e in registry if e.score >= args.min_score}
        print(f"{len(registry)} distinct entities, {len(selected)} selected (score >= {args.min_score}); "
              f"confident ones (>= {args.share_threshold}) are shared across files")
        for e in registry:
            mark = "x" if e.key in selected else " "
            print(f"  [{mark}] {e.score:3d} {e.kind:12s} {e.text}   ({len(e.files)} file{'s' if len(e.files) > 1 else ''})")
        print("Redacting...")
        result = process_set(scans, str(out), selected, registry, share=True, threshold=args.share_threshold,
                             ocr=not args.no_ocr, lang=args.lang, use_ner=not args.no_ner,
                             progress=lambda i, n, name: print(f"  [{i + 1}/{n}] {name}"))
    for o in result.outcomes:
        q = o.result.quality
        print(f"  {o.name}: {o.result.hits} areas, verification {'passed' if o.result.verification.ok else 'FAILED'}, "
              f"quality {q.explain()}")
        for severity, text in q.issues:
            print(f"      [{severity}] {text}")
        for problem in o.result.verification.problems:
            print(f"      - {problem}")
    print(f"Set privacy risk: {result.risk_before.explain()}  ->  {result.risk_after.explain()}")
    print(f"Terms shared to files that did not report them: {result.shared_terms}. Worst grade: {result.worst_grade}")
    audit_path = Path(args.audit) if args.audit else out / "audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(set_audit(result, registry, True, args.share_threshold), indent=2, ensure_ascii=False),
                          encoding="utf-8")
    print(f"Audit log: {audit_path}")
    too_low = bool(args.min_grade) and GRADE_ORDER.index(result.worst_grade) > GRADE_ORDER.index(args.min_grade)
    if not result.ok or too_low:
        print("Case NOT accepted: a file failed verification or is below the required grade.")
        return 1
    print("Case redacted and verified.")
    return 0


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    p = argparse.ArgumentParser(description="Locally redact personal data from PDFs and document photos.")
    p.add_argument("input", help="a PDF, image, Word/Excel/PowerPoint file, e-mail or text file, or a folder of them")
    p.add_argument("output", help="output PDF, or an output folder when input is a folder")
    p.add_argument("--no-ner", action="store_true", help="regex and label rules only (no names/places model)")
    p.add_argument("--no-ocr", action="store_true", help="skip OCR of scanned pages")
    p.add_argument("--lang", choices=["en", "fr", "ar"], default=None, help="document language (default: the profile's, else en)")
    p.add_argument("--profile", metavar="NAME|FILE.json", help="a profile: your allow list (never flag), deny list (always "
                   "redact) and defaults. NAME is looked up in profiles/ and profiles/local/")
    p.add_argument("--ocr", choices=["tesseract", "paddle", "auto"], default=None,
                   help="OCR engine: tesseract (default, ~3 s/page), paddle (more accurate, ~30 s/page, optional install) "
                        "or auto (Paddle only for pages Tesseract reads badly)")
    p.add_argument("--paddle-models", metavar="FOLDER", help="folder with the PaddleOCR <name>_det and <name>_rec models")
    p.add_argument("--min-score", type=int, default=None, metavar="0-100",
                   help="only redact detections scoring at least this (default 0: everything; "
                        f"{DEFAULT_THRESHOLD} is what the UI pre-selects). See redactor/scoring.py")
    p.add_argument("--skip-low", action="store_true", help=f"shorthand for --min-score {DEFAULT_THRESHOLD}")
    p.add_argument("--min-grade", choices=["A", "B", "C"], help="exit 1 if a file's quality grade is worse than this "
                   "(grades are A best, then B, C, F; F always fails)")
    p.add_argument("--case", action="store_true",
                   help="treat a folder as ONE case: merge what is found in all files and redact confident findings "
                        "(--share-threshold) in every file, not just where each was found")
    p.add_argument("--share-threshold", type=int, default=None, metavar="0-100",
                   help=f"with --case: only findings scoring at least this are searched for in files that did not "
                        f"report them (default {SHARE_THRESHOLD})")
    p.add_argument("--audit", metavar="FILE.json", help="write an audit log (types and counts, never the text)")
    args = p.parse_args(argv)
    try:
        args.profile_obj = load_profile(args.profile)
        if args.ocr or args.paddle_models:
            ocr_module.set_backend(args.ocr or ocr_module.BACKEND, args.paddle_models)
    except (ProfileError, RuntimeError, ValueError) as e:
        print(f"Error: {e}")
        return 2
    prof = args.profile_obj
    args.lang = args.lang or (prof.language if prof and prof.language else "en")
    if args.min_score is None:
        args.min_score = prof.min_score if prof and prof.min_score is not None else 0
    if args.share_threshold is None:
        args.share_threshold = prof.share_threshold if prof and prof.share_threshold is not None else SHARE_THRESHOLD
    if args.skip_low:
        args.min_score = max(args.min_score, DEFAULT_THRESHOLD)

    src, out = Path(args.input), Path(args.output)
    if src.is_dir():
        files = sorted(f for f in src.iterdir() if is_supported(f))
        if not files:
            print(f"No PDFs or images in {src}")
            return 2
        if args.case:
            return run_case(files, out, args)
        jobs = [(f, out / f"{f.stem}_redacted.pdf") for f in files]
        audit_path = Path(args.audit) if args.audit else out / "audit.json"
    else:
        jobs = [(src, out)]
        audit_path = Path(args.audit) if args.audit else None

    records = []
    for f, dst in jobs:
        print(f"== {f.name}")
        try:
            records.append(process(f, dst, args))
        except Exception as e:  # keep going in batch mode, but never report success for this file
            print(f"  ERROR: {e}")
            records.append({"file": f.name, "verification_passed": False, "verification_problems": [f"error: {e}"]})

    if audit_path:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps({
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": records,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Audit log: {audit_path}")

    def too_low(r) -> bool:
        grade = (r.get("quality") or {}).get("grade")
        if grade == "F":  # never acceptable, whatever --min-grade says
            return True
        return bool(args.min_grade and grade and GRADE_ORDER.index(grade) > GRADE_ORDER.index(args.min_grade))

    failed = [r["file"] for r in records if not r.get("verification_passed") or too_low(r)]
    if failed:
        print(f"\n{len(failed)} of {len(records)} file(s) NOT verified or below the required grade: {', '.join(failed)}")
        return 1
    print(f"\nAll {len(records)} file(s) redacted and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
