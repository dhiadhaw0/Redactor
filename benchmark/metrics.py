"""Scoring detections against ground truth.

Redaction works on text, so the question is "would redacting what was found cover the item?":

recall     an item is *covered* when at least 95% of its characters lie inside findings. Characters, not
           all-or-nothing, so a half-redacted IBAN is correctly not covered, while `Dhia` + `Toumi` together
           do cover `Dhia Toumi`. Recall is the number that matters for a redaction tool: a miss is a leak.
precision  a finding is *useful* when it overlaps an item that must be redacted, or text that is acceptable to
           redact (a company's own address). Anything else (a date, an amount, a skill) is a false positive.

For scans and photos the text went through OCR, so matching is fuzzy: a misread `jonn.smith` still counts for
`john.smith`, because redacting the OCR-read text blacks out the same spot on the page.
"""
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

COVERED_AT = 0.95
FUZZY = 0.8


def squash(s: str) -> str:
    return "".join(s.split()).lower()


def _mark(covered: list[bool], gt: str, pred: str, fuzzy: bool) -> None:
    """Mark the characters of `gt` that `pred` covers."""
    if not pred:
        return
    if pred in gt:                                     # the finding is a piece (or all) of the item
        start = gt.find(pred)
        while start != -1:
            covered[start:start + len(pred)] = [True] * len(pred)
            start = gt.find(pred, start + 1)
    elif gt in pred:                                   # the finding is bigger than the item
        covered[:] = [True] * len(gt)
    elif fuzzy and len(pred) >= 4:
        sm = SequenceMatcher(None, gt, pred, autojunk=False)
        if sm.ratio() >= FUZZY:
            for block in sm.get_matching_blocks():
                covered[block.a:block.a + block.size] = [True] * block.size


def words(s: str) -> list[str]:
    return re.findall(r"[^\W_]+", s.lower())


def coverage(gt: str, preds: list[str], fuzzy: bool = False, rtl: bool = False) -> float:
    g = squash(gt)
    if not g:
        return 1.0
    covered = [False] * len(g)
    for p in preds:
        _mark(covered, g, squash(p), fuzzy)
    chars = sum(covered) / len(g)
    if rtl:
        # Right-to-left lines come out of OCR in visual order ('+216 62 987 559' reads '559 987 62 216'):
        # the same spot on the page is redacted, so compare as a set of words instead of a sequence.
        seen = {w for p in preds for w in words(p)}
        gw = words(gt)
        total = sum(len(w) for w in gw)
        by_words = sum(len(w) for w in gw if w in seen) / total if total else 0.0
        return max(chars, by_words)
    return chars


MIN_OVERLAP = 4        # characters
MIN_OVERLAP_SHARE = 0.5  # of the finding


def touches(pred: str, text: str, fuzzy: bool = False, rtl: bool = False) -> bool:
    """Is the finding (mostly) made of `text`? 'Mme MARCHAND' is mostly the item 'MARCHAND Julien', so it
    counts as useful: it blacks out the surname. 'Invoice INV' against 'INV-2026-004211' does not."""
    p, t = squash(pred), squash(text)
    if not p or not t:
        return False
    if p in t or t in p:
        return True
    if rtl and words(pred) and set(words(pred)) <= set(words(text)):
        return True
    sm = SequenceMatcher(None, p, t, autojunk=False)
    longest = sm.find_longest_match(0, len(p), 0, len(t)).size
    if longest >= max(MIN_OVERLAP, MIN_OVERLAP_SHARE * len(p)):
        return True
    return fuzzy and len(p) >= 4 and sm.ratio() >= FUZZY


@dataclass
class ItemResult:
    kind: str
    text: str
    coverage: float

    @property
    def covered(self) -> bool:
        return self.coverage >= COVERED_AT


@dataclass
class PredResult:
    kind: str
    text: str
    score: int
    useful: bool


@dataclass
class DocResult:
    name: str
    lang: str
    variant: str
    items: list[ItemResult] = field(default_factory=list)
    preds: list[PredResult] = field(default_factory=list)
    fuzzy: bool = False                  # OCR'd variants are matched fuzzily
    rtl: bool = False                    # right-to-left: compare as sets of words
    leaked: list[str] | None = None      # end-to-end: items still readable in the redacted file (None = not run)


def evaluate(name: str, lang: str, variant: str, items: list[tuple[str, str]], ignore: list[str],
             findings, fuzzy: bool, rtl: bool = False) -> DocResult:
    """Compare detections (any score) with the ground truth of one document."""
    result = DocResult(name, lang, variant, fuzzy=fuzzy, rtl=rtl)
    for kind, text in items:
        result.items.append(ItemResult(kind, text, 0.0))
    result.preds = [
        PredResult(f.kind, f.text, f.score,
                   useful=any(touches(f.text, t, fuzzy, rtl) for _, t in items) or any(touches(f.text, t, fuzzy, rtl) for t in ignore))
        for f in findings
    ]
    return result


def at_threshold(result: DocResult, threshold: int):
    """(items with coverage computed from findings scoring >= threshold, those findings)."""
    kept = [p for p in result.preds if p.score >= threshold]
    texts = [p.text for p in kept]
    items = [ItemResult(i.kind, i.text, coverage(i.text, texts, result.fuzzy, result.rtl)) for i in result.items]
    return items, kept


@dataclass
class Counts:
    items: int = 0
    covered: int = 0
    preds: int = 0
    useful: int = 0

    @property
    def recall(self) -> float:
        return self.covered / self.items if self.items else 1.0

    @property
    def precision(self) -> float:
        return self.useful / self.preds if self.preds else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict:
        return {"recall": round(self.recall, 4), "precision": round(self.precision, 4), "f1": round(self.f1, 4),
                "items": self.items, "covered": self.covered, "predictions": self.preds, "useful": self.useful}


def summarize(results: list[DocResult], threshold: int) -> dict[str, Counts]:
    """Micro-averaged counts for: overall, each variant, each language, each variant x language cell
    (what a partial run can be compared on) and each item kind."""
    out: dict[str, Counts] = {}

    def bucket(key: str) -> Counts:
        return out.setdefault(key, Counts())

    for r in results:
        items, kept = at_threshold(r, threshold)
        for key in ("overall", f"variant:{r.variant}", f"lang:{r.lang}", f"cell:{r.variant}/{r.lang}"):
            c = bucket(key)
            c.items += len(items)
            c.covered += sum(i.covered for i in items)
            c.preds += len(kept)
            c.useful += sum(p.useful for p in kept)
        for i in items:  # recall by what the item is
            c = bucket(f"kind:{i.kind}")
            c.items += 1
            c.covered += i.covered
        for p in kept:   # precision by what the finding says it is
            c = bucket(f"kind:{p.kind}")
            c.preds += 1
            c.useful += p.useful
    return out
