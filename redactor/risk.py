"""Document privacy risk score (0-100): how much harm sharing this document as-is could do.

It is a transparent heuristic, not a probability. It answers "how identifying and how sensitive is
what is in this document?", so it can be shown before redaction ("87, critical") and again for what
is still detectable afterwards ("4, low").

How it is computed
  1. Each finding contributes  weight(kind) x confidence/100.  Weights reflect harm: an IBAN, a card or
     an ID number can be abused directly (20-25), a name or a street address identifies a person (10-12),
     an email or phone lets you reach them (8-10), a bare city says very little (2).
  2. Several findings of one kind count for geometrically less (each next one x0.5): ten emails are not
     ten times worse than one.
  3. Pieces are ignored: 'Toumi' inside 'Dhia Eddine Toumi' is one identity, not two.
  4. Identifiability bonuses: a name together with contact/identity details is far riskier than either
     alone (+15, +10 more for two or more kinds), and financial data tied to a named person (+10).
  Capped at 100. Levels: 0 none · 1-29 low · 30-54 moderate · 55-84 high · 85+ critical.
"""
from dataclasses import dataclass, field

from .detect import Finding

WEIGHTS = {
    "IBAN": 25, "CREDIT_CARD": 25, "MRZ": 25, "ID": 20, "BARCODE": 15, "DATE_OF_BIRTH": 15, "NUMBER": 12, "ADDRESS": 12,
    "PERSON": 10, "PHONE": 10, "EMAIL": 8, "REFERENCE": 8, "URL": 6, "LOCATION": 2,
}
DEFAULT_WEIGHT = 5          # kinds not listed above
DECAY = 0.5                 # each further finding of the same kind counts half as much
IDENTITY_MIN_SCORE = 60     # a finding must be at least this confident to count towards identifiability
CONTACT_KINDS = {"EMAIL", "PHONE", "ADDRESS", "URL", "ID", "DATE_OF_BIRTH", "MRZ"}
FINANCIAL_KINDS = {"IBAN", "CREDIT_CARD"}
BONUS_NAME_WITH_DETAILS = 15
BONUS_NAME_WITH_MANY_DETAILS = 10   # two or more kinds of details
BONUS_FINANCIAL_WITH_NAME = 10
LEVELS = [(85, "critical"), (55, "high"), (30, "moderate"), (1, "low"), (0, "none")]


def level_for(score: int) -> str:
    return next(name for floor, name in LEVELS if score >= floor)


@dataclass
class Risk:
    score: int
    level: str
    by_kind: list[tuple[str, int, float]] = field(default_factory=list)   # (kind, findings counted, points)
    bonuses: list[tuple[str, int]] = field(default_factory=list)          # (reason, points)

    def explain(self) -> str:
        """One line without any personal text: kinds and counts only."""
        parts = [f"{kind} x{n}" for kind, n, _ in self.by_kind]
        return f"{self.score}/100 ({self.level})" + (": " + ", ".join(parts) if parts else "")

    def as_dict(self) -> dict:
        return {"score": self.score, "level": self.level,
                "by_kind": {k: {"count": n, "points": round(p, 1)} for k, n, p in self.by_kind},
                "bonuses": [{"reason": r, "points": p} for r, p in self.bonuses]}


def _without_pieces(findings: list[Finding]) -> list[Finding]:
    """Drop findings that are just a piece of a longer finding of the same kind."""
    texts = {(f.kind, f.text.lower()) for f in findings}
    return [f for f in findings
            if not any(k == f.kind and f.text.lower() != t and f.text.lower() in t for k, t in texts)]


def assess(findings: list[Finding], extra_bonuses: list[tuple[str, int]] = ()) -> Risk:
    """Risk of a set of findings (those present in a document, or those still detectable after redaction)."""
    findings = _without_pieces(findings)
    by_kind: dict[str, list[float]] = {}
    for f in findings:
        by_kind.setdefault(f.kind, []).append(WEIGHTS.get(f.kind, DEFAULT_WEIGHT) * f.score / 100)

    rows, total = [], 0.0
    for kind, contributions in by_kind.items():
        points = sum(c * DECAY ** i for i, c in enumerate(sorted(contributions, reverse=True)))
        rows.append((kind, len(contributions), points))
        total += points
    rows.sort(key=lambda r: -r[2])

    confident = {f.kind for f in findings if f.score >= IDENTITY_MIN_SCORE}
    bonuses = []
    details = confident & CONTACT_KINDS
    if "PERSON" in confident and details:
        bonuses.append(("a name together with contact or identity details", BONUS_NAME_WITH_DETAILS))
        if len(details) >= 2:
            bonuses.append(("several kinds of contact or identity details", BONUS_NAME_WITH_MANY_DETAILS))
    if "PERSON" in confident and confident & FINANCIAL_KINDS:
        bonuses.append(("financial data tied to a named person", BONUS_FINANCIAL_WITH_NAME))
    bonuses += list(extra_bonuses)
    total += sum(p for _, p in bonuses)

    score = min(100, round(total))
    return Risk(score, level_for(score), rows, bonuses)


def remaining_after(findings: list[Finding], redacted_terms: list[str]) -> list[Finding]:
    """Estimate what stays detectable if exactly `redacted_terms` are redacted: findings that are
    not selected and not contained in a selected term. An estimate only: the measured residual
    risk comes from re-scanning the saved file."""
    selected = {t.lower() for t in redacted_terms}
    return [f for f in findings
            if f.text.lower() not in selected and not any(f.text.lower() in t for t in selected)]


BONUS_SAME_PERSON_ACROSS_DOCUMENTS = 5


def assess_set(per_file: dict[str, list[Finding]]) -> Risk:
    """Risk of a whole bundle of documents. Each distinct identity counts once (a name repeated in five
    files is one person), but findings from different files are pooled, so a name in one file and an IBAN in
    another do trigger the 'financial data tied to a named person' bonus: linking the documents is exactly
    how a set identifies someone that no single file does. A person who appears in several files adds a little."""
    pooled: dict[tuple[str, str], Finding] = {}
    seen_in: dict[tuple[str, str], int] = {}
    for findings in per_file.values():
        for key in {(f.kind, f.text.casefold()) for f in findings}:
            seen_in[key] = seen_in.get(key, 0) + 1
        for f in findings:
            key = (f.kind, f.text.casefold())
            if key not in pooled or f.score > pooled[key].score:
                pooled[key] = f
    extra = []
    if any(kind == "PERSON" and n >= 2 and pooled[(kind, t)].score >= IDENTITY_MIN_SCORE
           for (kind, t), n in seen_in.items()):
        extra.append(("the same person is identifiable across several documents", BONUS_SAME_PERSON_ACROSS_DOCUMENTS))
    return assess(list(pooled.values()), extra)
