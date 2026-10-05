"""Profiles: your own allow list ("never flag this"), deny list ("always redact this") and defaults.

A profile is a small JSON file, for example `profiles/hr.json`:

    {
      "name": "hr",
      "description": "Employment documents",
      "language": "fr",
      "min_score": 60,
      "share_threshold": 85,
      "allow": [ {"text": "Atlantis Assurances"}, {"text": "support@atlantis.example"}, {"text": "^Paris$", "regex": true} ],
      "deny":  [ {"text": "EMP-\\d{5}", "regex": true, "kind": "ID"}, {"text": "Project Falcon", "kind": "CUSTOM"} ]
    }

allow  suppresses findings: the finding's text equals the entry, is part of it ('Atlantis' inside the allowed
       'Atlantis Assurances'), or matches its regex. An entry may be limited to one kind. Suppressed findings are
       not lost: they are returned separately, shown in the app and counted in the audit log (kinds and counts only).
deny   adds findings the detector could not know: a literal or a regex, with a kind (default CUSTOM) and a score
       (default 95). They are redacted like any other finding.

Profiles ship in `profiles/` (examples); the ones you save from the app go to `profiles/local/` (git-ignored).
Regular expressions are validated: a pattern that could make the tool hang (nested quantifiers) is refused.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .detect import Finding

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
LOCAL_DIR = PROFILES_DIR / "local"
DENY_KIND, DENY_SCORE = "CUSTOM", 95
MAX_PATTERN = 200
MAX_MATCHES_PER_PAGE = 500
_NESTED_QUANTIFIER = re.compile(r"\((?:[^()\\]|\\.)*[+*](?:[^()\\]|\\.)*\)[+*{]")   # (a+)+  (.*)*  (a|b*)+ ...


class ProfileError(ValueError):
    pass


@dataclass
class Rule:
    text: str
    regex: bool = False
    kind: str | None = None        # allow: only suppress findings of this kind. deny: the kind of what it finds.
    score: int | None = None       # deny only
    note: str = ""

    def compiled(self) -> "re.Pattern":
        pattern = self.text if self.regex else re.escape(self.text)
        return re.compile(pattern, re.IGNORECASE | re.MULTILINE)

    def validate(self) -> None:
        if not self.text.strip():
            raise ProfileError("A rule cannot be empty.")
        if len(self.text) > MAX_PATTERN:
            raise ProfileError(f"A rule is longer than {MAX_PATTERN} characters.")
        if self.regex:
            if _NESTED_QUANTIFIER.search(self.text):
                raise ProfileError(f"The pattern {self.text!r} has nested quantifiers and could hang the tool.")
            try:
                if self.compiled().match(""):
                    raise ProfileError(f"The pattern {self.text!r} matches the empty string.")
            except re.error as e:
                raise ProfileError(f"The pattern {self.text!r} is not a valid regular expression: {e}") from e
        if self.score is not None and not 0 <= self.score <= 100:
            raise ProfileError("A score must be between 0 and 100.")

    def as_dict(self) -> dict:
        d = {"text": self.text}
        if self.regex:
            d["regex"] = True
        if self.kind:
            d["kind"] = self.kind
        if self.score is not None:
            d["score"] = self.score
        if self.note:
            d["note"] = self.note
        return d

    @classmethod
    def from_dict(cls, d) -> "Rule":
        if isinstance(d, str):
            d = {"text": d}
        return cls(str(d.get("text", "")), bool(d.get("regex", False)), d.get("kind"), d.get("score"), str(d.get("note", "")))


@dataclass
class Profile:
    name: str = "default"
    description: str = ""
    language: str | None = None
    min_score: int | None = None
    share_threshold: int | None = None
    allow: list[Rule] = field(default_factory=list)
    deny: list[Rule] = field(default_factory=list)

    def validate(self) -> "Profile":
        for rule in self.allow + self.deny:
            rule.validate()
        for label, value in (("min_score", self.min_score), ("share_threshold", self.share_threshold)):
            if value is not None and not 0 <= value <= 100:
                raise ProfileError(f"{label} must be between 0 and 100.")
        if self.language not in (None, "en", "fr", "ar"):
            raise ProfileError("language must be en, fr or ar.")
        return self

    def as_dict(self) -> dict:
        d = {"name": self.name, "description": self.description}
        for key in ("language", "min_score", "share_threshold"):
            if getattr(self, key) is not None:
                d[key] = getattr(self, key)
        d["allow"] = [r.as_dict() for r in self.allow]
        d["deny"] = [r.as_dict() for r in self.deny]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        try:
            return cls(
                name=str(d.get("name", "default")), description=str(d.get("description", "")), language=d.get("language"),
                min_score=d.get("min_score"), share_threshold=d.get("share_threshold"),
                allow=[Rule.from_dict(r) for r in d.get("allow", [])], deny=[Rule.from_dict(r) for r in d.get("deny", [])],
            ).validate()
        except (AttributeError, TypeError) as e:
            raise ProfileError(f"The profile is malformed: {e}") from e

    def save(self, directory: Path | None = None) -> Path:
        self.validate()
        directory = Path(directory) if directory else LOCAL_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_slug(self.name)}.json"
        path.write_text(json.dumps(self.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def with_allowed(self, texts) -> "Profile":
        """A copy that also never flags `texts`."""
        extra = _new_literals(texts, self.allow)
        return Profile(self.name, self.description, self.language, self.min_score, self.share_threshold,
                       self.allow + extra, list(self.deny))

    def with_denied(self, texts, kind: str = DENY_KIND) -> "Profile":
        """A copy that also always redacts `texts`."""
        extra = [Rule(r.text, kind=kind) for r in _new_literals(texts, self.deny)]
        return Profile(self.name, self.description, self.language, self.min_score, self.share_threshold,
                       list(self.allow), self.deny + extra)


def _new_literals(texts, existing: list[Rule]) -> list[Rule]:
    """Literal rules for `texts` that the list does not hold yet (case-insensitive), each at most once."""
    seen = {r.text.casefold() for r in existing if not r.regex}
    out = []
    for t in texts:
        t = t.strip()
        if t and t.casefold() not in seen:
            seen.add(t.casefold())
            out.append(Rule(t))
    return out


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")[:60] or "profile"


# ------------------------------------------------------------------ files
def list_profiles() -> dict[str, Path]:
    """name -> path, for the shipped examples and the local ones (a local profile hides a shipped one of the same name)."""
    found: dict[str, Path] = {}
    for directory in (PROFILES_DIR, LOCAL_DIR):
        if directory.is_dir():
            for path in sorted(directory.glob("*.json")):
                found[path.stem] = path
    return found


def load_profile(ref: str | Path | None) -> Profile | None:
    """A profile by name (from profiles/ or profiles/local/) or by path. None -> no profile."""
    if not ref:
        return None
    path = Path(ref)
    if not path.suffix:
        known = list_profiles()
        if str(ref) not in known:
            raise ProfileError(f"No profile called {str(ref)!r}. Available: {', '.join(sorted(known)) or 'none'}.")
        path = known[str(ref)]
    try:
        return Profile.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as e:
        raise ProfileError(f"{path.name} is not valid JSON: {e}") from e
    except OSError as e:
        raise ProfileError(f"Cannot read {path}: {e}") from e


# ------------------------------------------------------------------ applying a profile
def _allowed(finding: Finding, rules: list[Rule]) -> bool:
    cf = finding.text.casefold().strip()
    for rule in rules:
        if rule.kind and rule.kind != finding.kind:
            continue
        if rule.regex:
            if rule.compiled().fullmatch(finding.text.strip()):
                return True
        else:
            entry = rule.text.casefold().strip()
            if cf == entry or (cf and cf in entry):
                return True
    return False


def apply_allow(findings: list[Finding], profile: Profile | None) -> tuple[list[Finding], list[Finding]]:
    """(kept, suppressed). Applied to what the detector found; the deny list's own findings are added afterwards
    (see scan_pdf), so always-redact beats never-flag."""
    if profile is None or not profile.allow:
        return list(findings), []
    kept, suppressed = [], []
    for f in findings:
        (suppressed if _allowed(f, profile.allow) else kept).append(f)
    return kept, suppressed


def deny_findings(text: str, profile: Profile | None) -> list[Finding]:
    """Findings for every place the deny list matches in `text`."""
    if profile is None:
        return []
    out = []
    for rule in profile.deny:
        kind, score = rule.kind or DENY_KIND, rule.score if rule.score is not None else DENY_SCORE
        for n, m in enumerate(rule.compiled().finditer(text)):
            if n >= MAX_MATCHES_PER_PAGE:
                break
            if m.group().strip():
                out.append(Finding(kind, m.group().strip(), score))
    return out
