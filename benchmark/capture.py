"""Turn a review you did in the app into a labeled benchmark case.

What you decided while reviewing a document is exactly the ground truth the benchmark needs:

  ticked findings          -> items that MUST be redacted (recall: did the detector find them?)
  unticked findings        -> text that must NOT be flagged (precision: a finding there is a false positive)
  terms you typed in       -> items the detector MISSED, kind MANUAL (recall for them is 0 until it learns them)
  boxes you drew           -> kept in the file for reference; text metrics cannot score a region

The case is saved under benchmark/local/ (git-ignored) with a copy of the document, because the copy is the
unredacted original: it holds the personal data the labels refer to. Nothing here ever leaves the machine.
"""
import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import metrics
from .run import LOCAL_DIR, load_local, run_one

DOCS = "docs"


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._-")
    return slug[:60] or "case"


def _without_pieces(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """'Toumi' inside the accepted 'Dhia Eddine Toumi' is not a second item: it is part of the first."""
    texts = {(k, t.casefold()) for k, t in items}
    kept, seen = [], set()
    for kind, text in items:
        key = (kind, text.casefold())
        if key in seen or any(k == kind and text.casefold() != t and text.casefold() in t for k, t in texts):
            continue
        seen.add(key)
        kept.append((kind, text))
    return kept


def unique_name(directory: Path, name: str) -> str:
    base, n = _slug(name), 1
    candidate = base
    while (directory / f"{candidate}.labels.json").exists():
        n += 1
        candidate = f"{base}_{n}"
    return candidate


def save_case(pdf_path: str, lang: str, accepted: list[tuple[str, str]], rejected: list[tuple[str, str]] = (),
              manual: list[str] = (), boxes: list = (), name: str | None = None, source: str | None = None,
              directory: Path | None = None) -> Path:
    """Write `<name>.labels.json` and a copy of the document. Returns the label file."""
    directory = Path(directory) if directory else LOCAL_DIR
    (directory / DOCS).mkdir(parents=True, exist_ok=True)
    name = unique_name(directory, name or source or Path(pdf_path).stem)
    shutil.copyfile(pdf_path, directory / DOCS / f"{name}.pdf")

    items = _without_pieces(list(accepted)) + [("MANUAL", t.strip()) for t in manual if t.strip()]
    spec = {
        "file": f"{DOCS}/{name}.pdf",
        "lang": lang,
        "source": source or name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "items": [{"kind": k, "text": t} for k, t in items],
        "ignore": [],
        "rejected": [{"kind": k, "text": t} for k, t in rejected],     # documentation: they are traps automatically
        "boxes": [list(b) for b in boxes],                              # not scored: text metrics cannot score a region
    }
    path = directory / f"{name}.labels.json"
    path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


@dataclass
class CaseScore:
    items: int
    covered: int
    findings: int
    useful: int
    missed: list[tuple[str, str, float]] = field(default_factory=list)      # (kind, text, share covered)
    false_positives: list[tuple[str, str, int]] = field(default_factory=list)  # (kind, text, score)

    @property
    def recall(self) -> float:
        return self.covered / self.items if self.items else 1.0

    @property
    def precision(self) -> float:
        return self.useful / self.findings if self.findings else 1.0


def evaluate_case(label_file: Path, threshold: int = 60) -> CaseScore:
    """Score the detector on one saved case (the same metrics as `python -m benchmark.run --local`)."""
    [doc] = [d for d in load_local(label_file.parent) if d["name"] == label_file.name.removesuffix(".labels.json")]
    with tempfile.TemporaryDirectory() as tmp:
        result = run_one(doc["name"], doc["lang"], "real", doc["path"], doc["items"], doc["ignore"], False, threshold, tmp)
    items, kept = metrics.at_threshold(result, threshold)
    return CaseScore(
        items=len(items), covered=sum(i.covered for i in items), findings=len(kept), useful=sum(p.useful for p in kept),
        missed=[(i.kind, i.text, i.coverage) for i in items if not i.covered],
        false_positives=[(p.kind, p.text, p.score) for p in kept if not p.useful],
    )
