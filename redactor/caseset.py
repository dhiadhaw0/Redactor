"""Treat several documents as one case.

Scan every file first, merge what was found into one registry of entities, let the user decide once, then
redact every file with the merged decisions. Each file is still verified and graded on its own, so one bad
file cannot hide behind good ones.

Why a shared list: the language model is less sure of a bare surname in a table than of the same name in a
sentence. Once `MARCHAND` is confirmed in one document, it should go from all of them.

Safety valve: sharing is limited to confident findings (SHARE_THRESHOLD). A common word flagged at a low score
in one file ('Monte', 'Paris') must never be spread to every other file.
"""
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from .detect import Finding
from .quality import worse
from .redact import RedactionResult, apply_redactions, scan_pdf
from .risk import Risk, assess_set

SHARE_THRESHOLD = 85   # only findings at least this sure are searched for in files that did not report them


@dataclass
class FileScan:
    name: str                       # what the user calls it (the uploaded file name)
    pdf: str                        # path of the PDF to read (images are wrapped beforehand)
    findings: list[Finding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suppressed: list[Finding] = field(default_factory=list)   # dropped by the profile's allow list (kept to show and count)


@dataclass
class Entity:
    kind: str
    text: str
    score: int                      # the best score any file gave it
    files: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return (self.kind, self.text.casefold())


def scan_files(files: list[tuple[str, str]], use_ner: bool = True, ocr: bool = True, lang: str = "en",
               progress=None, profile=None) -> list[FileScan]:
    """files: (display name, pdf path). `progress(i, n, name)` is called before each file."""
    scans = []
    for i, (name, pdf) in enumerate(files):
        if progress:
            progress(i, len(files), name)
        warnings: list[str] = []
        hidden: list[Finding] = []
        findings = scan_pdf(pdf, use_ner=use_ner, ocr=ocr, lang=lang, warnings=warnings, profile=profile, suppressed=hidden)
        scans.append(FileScan(name, pdf, findings, warnings, suppressed=hidden))
    return scans


def build_registry(scans: list[FileScan]) -> list[Entity]:
    """One entity per (kind, text), case-insensitive, with the best score and the files that reported it."""
    merged: dict[tuple[str, str], Entity] = {}
    for scan in scans:
        for f in scan.findings:
            key = (f.kind, f.text.casefold())
            e = merged.get(key)
            if e is None:
                merged[key] = Entity(f.kind, f.text, f.score, [scan.name])
                continue
            if f.score > e.score:
                e.score, e.text = f.score, f.text      # keep the spelling of the most certain finding
            if scan.name not in e.files:
                e.files.append(scan.name)
    return sorted(merged.values(), key=lambda e: (-e.score, e.kind, e.text))


def terms_for_file(scan: FileScan, registry: list[Entity], selected: set[tuple[str, str]],
                   share: bool = True, threshold: int = SHARE_THRESHOLD) -> list[str]:
    """What to search for in one file: the selected entities it reported itself, plus (when sharing) the
    selected entities from other files that are confident enough to trust there."""
    own = {f.text.casefold(): f.text for f in scan.findings}
    terms = []
    for e in registry:
        if e.key not in selected:
            continue
        if scan.name in e.files:
            terms.append(own.get(e.text.casefold(), e.text))
        elif share and e.score >= threshold:
            terms.append(e.text)
    return terms


def preselected(registry: list[Entity], min_score: int) -> set[tuple[str, str]]:
    return {e.key for e in registry if e.score >= min_score}


@dataclass
class FileOutcome:
    name: str
    output: str
    terms: list[str]
    result: RedactionResult


@dataclass
class SetResult:
    outcomes: list[FileOutcome]
    risk_before: Risk
    risk_after: Risk
    shared_terms: int               # terms searched for in a file that did not report them

    @property
    def worst_grade(self) -> str:
        grade = "A"
        for o in self.outcomes:
            grade = worse(grade, o.result.quality.grade)
        return grade

    @property
    def ok(self) -> bool:
        """Every file verified and none graded F."""
        return all(o.result.verification.ok and o.result.quality.grade != "F" for o in self.outcomes)


def output_names(names: list[str]) -> list[str]:
    """`<stem>_redacted.pdf`, made unique if two files share a stem (a.pdf and a.png)."""
    used: dict[str, int] = {}
    out = []
    for name in names:
        base = f"{Path(name).stem}_redacted"
        n = used.get(base, 0)
        used[base] = n + 1
        out.append(f"{base}.pdf" if n == 0 else f"{base}_{n + 1}.pdf")
    return out


def process_set(scans: list[FileScan], out_dir: str, selected: set[tuple[str, str]], registry: list[Entity] | None = None,
                share: bool = True, threshold: int = SHARE_THRESHOLD, ocr: bool = True, lang: str = "en",
                use_ner: bool = True, progress=None) -> SetResult:
    """Redact every file with the merged decisions, verify and grade each, assess the set before and after."""
    registry = registry if registry is not None else build_registry(scans)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    outcomes: list[FileOutcome] = []
    shared = 0
    for i, (scan, filename) in enumerate(zip(scans, output_names([s.name for s in scans]))):
        if progress:
            progress(i, len(scans), scan.name)
        terms = terms_for_file(scan, registry, selected, share, threshold)
        shared += sum(1 for e in registry if e.key in selected and scan.name not in e.files
                      and share and e.score >= threshold)
        dst = str(Path(out_dir) / filename)
        result = apply_redactions(scan.pdf, dst, terms, ocr=ocr, lang=lang, use_ner=use_ner)
        outcomes.append(FileOutcome(scan.name, dst, terms, result))
    return SetResult(
        outcomes,
        risk_before=assess_set({s.name: s.findings for s in scans}),
        risk_after=assess_set({o.name: o.result.residual for o in outcomes}),
        shared_terms=shared,
    )


def set_audit(result: SetResult, registry: list[Entity], share: bool, threshold: int) -> dict:
    """The audit record of a set: kinds, counts, grades and risk, never the personal text."""
    kinds: dict[str, int] = {}
    for e in registry:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    return {
        "files": [{
            "file": o.name, "output": Path(o.output).name, "terms_applied": len(o.terms),
            "areas_redacted": o.result.hits,
            "verification_passed": o.result.verification.ok,
            "quality": o.result.quality.as_dict(),
            "risk_after": o.result.risk_after.as_dict(),
            "hidden_content_removed": o.result.hidden_removed,
        } for o in result.outcomes],
        "set": {
            "files": len(result.outcomes), "worst_grade": result.worst_grade, "all_verified": result.ok,
            "sharing": {"enabled": share, "threshold": threshold, "terms_shared_to_other_files": result.shared_terms},
            "distinct_entities_by_type": kinds,
            "risk_before": result.risk_before.as_dict(), "risk_after": result.risk_after.as_dict(),
        },
    }


def build_zip(result: SetResult, audit: dict | None = None) -> bytes:
    """The redacted files (and the audit log, if given) in one zip, in memory."""
    import json
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for o in result.outcomes:
            z.write(o.output, Path(o.output).name)
        if audit is not None:
            z.writestr("audit.json", json.dumps(audit, indent=2, ensure_ascii=False))
    return buf.getvalue()
