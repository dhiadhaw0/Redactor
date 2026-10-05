"""Run the benchmark.

    python -m benchmark.run                      # synthetic corpus, prints precision / recall / F1
    python -m benchmark.run --details            # also list every missed item and every false positive
    python -m benchmark.run --sweep              # precision/recall at several score thresholds
    python -m benchmark.run --e2e                # also redact and check that no item is still readable
    python -m benchmark.run --check              # compare with benchmark/baseline.json; exit 1 on a regression
    python -m benchmark.run --update-baseline    # accept the current numbers as the new baseline
    python -m benchmark.run --local              # your own documents: benchmark/local/*.labels.json

The synthetic corpus is generated from a seed (see corpus.py), so results are reproducible. Local documents
stay on your machine: benchmark/local/ is git-ignored.
"""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import pymupdf as fitz

from redactor.ocr import ocr_page, page_text
from redactor.redact import apply_redactions, scan_pdf
from redactor.scoring import DEFAULT_THRESHOLD
from redactor.verify import _digits, _leaks, _squash

from . import corpus
from .metrics import DocResult, Counts, at_threshold, evaluate, summarize

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "baseline.json"
LOCAL_DIR = HERE / "local"
RESULTS = HERE / "results" / "latest.json"
# A group may lose this share of its items versus the baseline (but always at least one item) before the
# check fails: OCR is deterministic, yet a library update can move a single item.
RECALL_TOLERANCE = {"overall": 0.01, "variant:digital": 0.01}
DEFAULT_TOLERANCE = 0.03
PRECISION_TOLERANCE = 0.05
SWEEP = (0, 30, 40, 60, 75, 90)


def load_local(directory: Path | None = None) -> list[dict]:
    """Labels for real documents: {"file": path (absolute, or relative to the labels file), "lang": "fr",
    "items": [{"kind": "PERSON", "text": "..."}], "ignore": ["text that may be redacted but need not be"]}"""
    directory = Path(directory) if directory else LOCAL_DIR     # looked up now, not when the module was imported
    docs = []
    for label_file in sorted(directory.glob("*.labels.json")):
        spec = json.loads(label_file.read_text(encoding="utf-8"))
        path = Path(spec["file"])
        path = path if path.is_absolute() else label_file.parent / path
        docs.append({"name": label_file.name.removesuffix(".labels.json"), "lang": spec.get("lang", "en"), "path": str(path),
                     "items": [(i["kind"], i["text"]) for i in spec["items"]], "ignore": spec.get("ignore", [])})
    return docs


def leaked_items(path: str, items, lang: str, scanned: bool) -> list[str]:
    """Items still readable in a redacted file: text layer, plus OCR for scanned pages."""
    with fitz.open(path) as doc:
        parts = [page_text(p) for p in doc]
        if scanned:
            parts += [ocr_page(p, lang).text for p in doc]
    blob = "\n".join(parts)
    squashed, digits = _squash(blob), _digits(blob)
    return [text for _, text in items if _leaks(text, squashed, digits)]


def run_one(name, lang, variant, path, items, ignore, e2e: bool, threshold: int, workdir: str) -> DocResult:
    scanned = variant != "digital"
    findings = scan_pdf(path, lang=lang)
    result = evaluate(name, lang, variant, items, ignore, findings, fuzzy=scanned, rtl=lang == "ar")
    if e2e:
        out = str(Path(workdir) / f"out_{name}_{variant}.pdf")
        apply_redactions(path, out, [f.text for f in findings if f.score >= threshold], lang=lang,
                         verify=False, rescan=False)
        result.leaked = leaked_items(out, items, lang, scanned)
    return result


def run_corpus(variants, langs, seed, e2e, threshold) -> list[DocResult]:
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for doc in corpus.build_corpus(seed):
            if doc.lang not in langs:
                continue
            for variant in corpus.variants_for(doc):
                if variant not in variants:
                    continue
                path = corpus.render(doc, variant, tmp)
                print(f"  {doc.name:12s} {variant:8s}", flush=True)
                results.append(run_one(doc.name, doc.lang, variant, path, doc.items, doc.ignore, e2e, threshold, tmp))
    return results


def run_local(e2e, threshold) -> list[DocResult]:
    docs = load_local()
    if not docs:
        print(f"No *.labels.json files in {LOCAL_DIR}")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for d in docs:
            from redactor.loader import to_pdf  # photos/PNGs are wrapped like the app does
            pdf = to_pdf(d["path"], str(Path(tmp) / "in.pdf"))
            print(f"  {d['name']}", flush=True)
            # real PDFs may be digital or scanned; OCR'd pages are matched fuzzily either way
            results.append(run_one(d["name"], d["lang"], "real", pdf, d["items"], d["ignore"], e2e, threshold, tmp))
    return results


# ------------------------------------------------------------------ reporting
def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def print_summary(summary: dict[str, Counts], threshold: int) -> None:
    print(f"\nScores at threshold {threshold}  (recall = items covered; precision = findings that are useful)")
    print(f"{'group':18s} {'recall':>8s} {'precision':>10s} {'F1':>7s}   items          findings")
    order = ["overall"] + sorted(k for k in summary if k.startswith("variant:")) \
        + sorted(k for k in summary if k.startswith("lang:")) + sorted(k for k in summary if k.startswith("kind:"))
    for key in order:
        c = summary[key]
        print(f"{key:18s} {pct(c.recall):>8s} {pct(c.precision):>10s} {c.f1:7.3f}   "
              f"{c.covered:3d}/{c.items:<3d}        {c.useful:3d}/{c.preds:<3d}")


def print_sweep(results: list[DocResult]) -> None:
    print("\nThreshold sweep (overall): a higher threshold trades recall for precision")
    print(f"{'threshold':>10s} {'recall':>8s} {'precision':>10s} {'F1':>7s}")
    for t in SWEEP:
        c = summarize(results, t)["overall"]
        print(f"{t:10d} {pct(c.recall):>8s} {pct(c.precision):>10s} {c.f1:7.3f}")


def print_details(results: list[DocResult], threshold: int) -> None:
    print("\nMISSED (not fully covered):")
    n = 0
    for r in results:
        items, _ = at_threshold(r, threshold)
        for i in items:
            if not i.covered:
                n += 1
                print(f"  {r.name}/{r.variant}  [{i.kind}] {i.text}   (covered {i.coverage:.0%})")
    print(f"  {n} item(s)")
    print("\nFALSE POSITIVES (findings on text that is not personal):")
    n = 0
    for r in results:
        _, kept = at_threshold(r, threshold)
        for p in kept:
            if not p.useful:
                n += 1
                print(f"  {r.name}/{r.variant}  {p.score:3d} [{p.kind}] {p.text}")
    print(f"  {n} finding(s)")


def print_e2e(results: list[DocResult]) -> None:
    done = [r for r in results if r.leaked is not None]
    if not done:
        return
    total = sum(len(r.items) for r in done)
    leaked = sum(len(r.leaked) for r in done)
    print(f"\nEnd to end (redact, then look for the items in the saved file): {leaked} of {total} items still "
          f"readable = {pct(1 - leaked / total) if total else 'n/a'} protected")
    for r in done:
        for text in r.leaked:
            print(f"  LEAK {r.name}/{r.variant}: {text}")


# ------------------------------------------------------------------ regression gate
def snapshot(summary: dict[str, Counts]) -> dict:
    return {k: c.as_dict() for k, c in summary.items()}


def check(current: dict, baseline: dict, complete: bool = True) -> tuple[list[str], list[str]]:
    """(failures, improvements). Groups are compared when present in both. A group missing from a
    `complete` run (same variants and languages as the baseline) is a failure; a partial run just skips it."""
    failures, better = [], []
    for key, base in baseline["metrics"].items():
        now = current.get(key)
        if now is None:
            if complete:
                failures.append(f"{key}: missing from this run (was in the baseline)")
            continue
        tol = RECALL_TOLERANCE.get(key, DEFAULT_TOLERANCE)
        slack = max(1, math.ceil(tol * base["items"])) if base["items"] else 0
        if now["items"] and now["covered"] < base["covered"] - slack and now["items"] == base["items"]:
            failures.append(f"{key}: recall fell from {base['recall']:.1%} ({base['covered']}/{base['items']}) to "
                            f"{now['recall']:.1%} ({now['covered']}/{now['items']})")
        elif now["covered"] > base["covered"] and now["items"] == base["items"]:
            better.append(f"{key}: recall rose from {base['recall']:.1%} to {now['recall']:.1%}")
        if now["predictions"] and base["predictions"] and now["precision"] < base["precision"] - PRECISION_TOLERANCE:
            failures.append(f"{key}: precision fell from {base['precision']:.1%} to {now['precision']:.1%}")
    return failures, better


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="digital,scan,photo,mixed", help="comma separated")
    ap.add_argument("--langs", default="en,fr,ar", help="comma separated")
    ap.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help="count findings scoring at least this")
    ap.add_argument("--seed", type=int, default=corpus.SEED)
    ap.add_argument("--details", action="store_true", help="list misses and false positives")
    ap.add_argument("--sweep", action="store_true", help="precision/recall at several thresholds")
    ap.add_argument("--e2e", action="store_true", help="redact and verify that no item is still readable")
    ap.add_argument("--local", action="store_true", help="benchmark your own labeled documents instead")
    ap.add_argument("--check", action="store_true", help="fail (exit 1) if worse than benchmark/baseline.json")
    ap.add_argument("--update-baseline", action="store_true", help="write the current numbers to the baseline")
    ap.add_argument("--json", metavar="FILE", help=f"write full results (default {RESULTS})")
    args = ap.parse_args(argv)

    print("Benchmark: " + ("local documents" if args.local else f"synthetic corpus, seed {args.seed}"))
    if args.local:
        results = run_local(args.e2e, args.threshold)
    else:
        results = run_corpus(set(args.variants.split(",")), set(args.langs.split(",")), args.seed, args.e2e, args.threshold)
    if not results:
        return 2

    summary = summarize(results, args.threshold)
    print_summary(summary, args.threshold)
    if args.sweep:
        print_sweep(results)
    if args.details:
        print_details(results, args.threshold)
    print_e2e(results)

    out = Path(args.json) if args.json else RESULTS
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"threshold": args.threshold, "metrics": snapshot(summary)}, indent=2), encoding="utf-8")

    current = snapshot(summary)
    if args.update_baseline and not args.local:
        BASELINE.write_text(json.dumps({"seed": args.seed, "threshold": args.threshold, "variants": args.variants,
                                        "langs": args.langs, "metrics": current}, indent=2), encoding="utf-8")
        print(f"\nBaseline written to {BASELINE}")
    if args.check and not args.local:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        complete = args.variants == baseline.get("variants") and args.langs == baseline.get("langs")
        if not complete:
            print("\n(partial run: only the groups measured here are compared with the baseline)")
        failures, better = check(current, baseline, complete)
        for line in better:
            print("  improved:", line)
        if failures:
            print("\nREGRESSION:")
            for line in failures:
                print("  -", line)
            return 1
        print("\nNo regression against the baseline." + ("  (Numbers improved: run --update-baseline to keep them.)" if better else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
