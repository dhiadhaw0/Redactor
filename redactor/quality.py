"""Redaction quality grade (A / B / C / F) for a finished file.

The risk score says how sensitive a document is. This says how much you can trust the *output*: it is built
only from checks the pipeline already makes, and every deduction comes with a reason a person can act on.

Gate (any of these is an automatic F: do not share the file)
  - verification found a problem: a box that is not black, a redacted term still readable in the text layer,
    or readable again when the redacted scan is OCR'd;
  - hidden content (annotations, attachments, metadata, invisible text...) survived the strip.

Major (the grade can be at most C)
  - a page could not be read by OCR, so what was or was not redacted there is unknown;
  - confident personal data (a finding scoring 85+) is still detectable in the saved file;
  - nothing was redacted although personal data is still detectable.

Deductions from 100 (the rest of the score)
  - measured residual risk: half of its points, at most 40;
  - each unreadable page share: up to 40;
  - plausible (60-84) findings still detectable: 2 each, at most 10.

Grades: A 90+ · B 75+ · C 50+ · below 50 F. Notes (information, no deduction): pages read through OCR,
hidden content that was removed.
"""
from dataclasses import dataclass, field

CRITICAL, MAJOR, MINOR = "critical", "major", "minor"
CONFIDENT = 85            # a residual finding at least this sure is "confident personal data"
PLAUSIBLE = 60
MAX_RISK_DEDUCTION = 40
MAX_UNREADABLE_DEDUCTION = 40
MAX_PLAUSIBLE_DEDUCTION = 10
GRADE_ORDER = "ABCF"      # best to worst


def grade_for(score: int) -> str:
    return "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 50 else "F"


def worse(a: str, b: str) -> str:
    """The worse of two grades."""
    return max(a, b, key=GRADE_ORDER.index)


@dataclass
class Quality:
    score: int
    grade: str
    issues: list[tuple[str, str]] = field(default_factory=list)   # (severity, text), worst first
    notes: list[str] = field(default_factory=list)

    def explain(self) -> str:
        return f"{self.grade} ({self.score}/100)" + (f", {len(self.issues)} issue(s)" if self.issues else "")

    def as_dict(self) -> dict:
        return {"grade": self.grade, "score": self.score,
                "issues": [{"severity": s, "text": t} for s, t in self.issues], "notes": self.notes}


def _pages(numbers) -> str:
    return ", ".join(str(n + 1) for n in numbers)


def assess_quality(result) -> Quality:
    """Grade a RedactionResult. No personal text is ever put in an issue or a note."""
    issues: list[tuple[str, str]] = []
    notes: list[str] = []
    score = 100.0
    cap = "A"  # best grade still possible

    # ---- gate ----
    if result.verification.problems:
        n = len(result.verification.problems)
        issues.append((CRITICAL, f"Verification failed ({n} problem{'s' if n > 1 else ''}): the saved file still shows "
                                 "something that should be gone. Do not share it."))
        cap = "F"
    if result.hidden_remaining:
        issues.append((CRITICAL, "Hidden content is still in the file after stripping: " + "; ".join(result.hidden_remaining) + "."))
        cap = "F"

    # ---- unreadable pages ----
    pages = max(result.pages, 1)
    if result.unreadable_pages:
        share = len(result.unreadable_pages) / pages
        score -= MAX_UNREADABLE_DEDUCTION * share
        issues.append((MAJOR, f"Page {_pages(result.unreadable_pages)} could not be read (OCR found almost no text), so "
                              "what was redacted there is unknown. Check it by eye or draw boxes by hand."))
        cap = worse(cap, "C")

    # ---- what is still detectable ----
    from .risk import assess  # local import: risk imports detect, which imports nothing from here

    residual = result.residual
    risk = assess(residual)
    score -= min(MAX_RISK_DEDUCTION, risk.score * 0.5)
    confident = [f for f in residual if f.score >= CONFIDENT]
    plausible = [f for f in residual if PLAUSIBLE <= f.score < CONFIDENT]
    if confident:
        kinds = ", ".join(f"{k} x{n}" for k, n in _count_kinds(confident))
        issues.append((MAJOR, f"Confident personal data is still detectable in the saved file ({kinds}). "
                              "If that was not intended, redact it and run again."))
        cap = worse(cap, "C")
    if plausible:
        kinds = ", ".join(f"{k} x{n}" for k, n in _count_kinds(plausible))
        score -= min(MAX_PLAUSIBLE_DEDUCTION, 2 * len(plausible))
        issues.append((MINOR, f"Plausible personal data remains detectable ({kinds})."))
    if result.hits == 0 and (confident or plausible):
        issues.append((MAJOR, "Nothing was redacted although personal data is still detectable."))
        cap = worse(cap, "C")

    # ---- information ----
    scanned = [n for n in result.ocr_pages if n not in result.image_text_pages]
    if scanned:
        notes.append(f"Page {_pages(scanned)} {'was' if len(scanned) == 1 else 'were'} read through OCR: "
                     "detection there is less certain, so review the preview.")
    if result.image_text_pages:
        notes.append(f"Pictures on page {_pages(result.image_text_pages)} were read through OCR (the rest of the page has a "
                     "text layer): check that the preview covers the text inside the pictures.")
    if result.hidden_removed:
        notes.append("Removed hidden content: " + "; ".join(result.hidden_removed) + ".")
    if not residual:
        notes.append("Nothing detectable remains in the saved file.")

    score_int = max(0, min(100, round(score)))
    if cap == "F":
        score_int = min(score_int, 40)
    grade = worse(grade_for(score_int), cap)
    order = {CRITICAL: 0, MAJOR: 1, MINOR: 2}
    issues.sort(key=lambda i: order[i[0]])
    return Quality(score_int, grade, issues, notes)


def _count_kinds(findings) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.kind] = counts.get(f.kind, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])
