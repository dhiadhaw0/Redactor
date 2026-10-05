"""People, places and profile links: the heuristics that need more than a regular expression.

* spaCy NER, with confidence tiers: the models tag headings and skills ('Spring Boot',
  'Informatique Financière') as people and places, so a bare place or a name that is a whole
  standalone line is only "low confidence" (listed, but not pre-selected);
* names confirmed by an email address or profile URL (dhiaeddinetoumi7@... confirms
  'Dhia Eddine Toumi');
* the contact block at the top of a CV or letter (a short name line next to an email or phone);
* social/profile links, which identify a person just like an email does.
"""
import re

from . import scoring as sc
from .detect import LANGUAGES, NOT_NAMES, Finding, _is_org, _lower_words, _plausible_entity

SOCIAL_URL = re.compile(
    r"(?:https?://)?(?:www\.)?(?:linkedin\.com/(?:in|pub)/|github\.com/|gitlab\.com/|twitter\.com/|x\.com/"
    r"|facebook\.com/|instagram\.com/|t\.me/|wa\.me/|behance\.net/|medium\.com/@)[^\s,;)\]>]+",
    re.I,
)
_TOKEN = re.compile(r"[^\W\d_][\w'’-]*")
# 2-4 capitalised words and nothing else: a name on a line of its own
_NAME_LINE = re.compile(r"(?:[A-ZÀ-Ý][a-zà-ÿ'’-]+|[A-ZÀ-Ý]{2,})(?: (?:[A-ZÀ-Ý][a-zà-ÿ'’-]+|[A-ZÀ-Ý]{2,})){1,3}")
# 'Tunis, Tunisia' / 'Lyon, France': city and country on a line of their own
_PLACE_LINE = re.compile(r"[A-ZÀ-Ý][A-Za-zà-ÿ'’ -]+(?:, [A-ZÀ-Ý][A-Za-zà-ÿ'’ -]+)+")

PERSON_LABELS = {"PERSON", "PER"}
PLACE_LABELS = {"GPE", "LOC", "LOCATION"}


def detect_social(text: str) -> list[Finding]:
    return [Finding("URL", m.group().rstrip(".,;:"), sc.URL_PROFILE) for m in SOCIAL_URL.finditer(text)]


def _letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def names_from_handles(text: str, findings: list[Finding]) -> list[Finding]:
    """A run of capitalised words whose letters spell an email local part or profile handle
    ('Dhia Eddine Toumi' ~ dhiaeddinetoumi7@gmail.com) is that person's name."""
    handles = []
    for f in findings:
        if f.kind == "EMAIL":
            handles.append(_letters(f.text.split("@")[0]))
        elif f.kind == "URL":
            handles.append(_letters(f.text.rstrip("/").rsplit("/", 1)[-1]))
    handles = [h for h in handles if len(h) >= 6]
    if not handles:
        return []
    out = []
    for line in text.splitlines():
        toks = [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(line)]
        i = 0
        while i < len(toks):
            for j in range(min(i + 4, len(toks)), i + 1, -1):  # longest window first
                win = toks[i:j]
                single_spaced = all(line[win[k][2]:win[k + 1][1]] == " " for k in range(len(win) - 1))
                if single_spaced and all(t[0][0].isupper() for t in win) \
                        and any(_letters("".join(t[0] for t in win)) in h for h in handles) \
                        and len(_letters("".join(t[0] for t in win))) >= 6:
                    out.append(Finding("PERSON", line[win[0][1]:win[-1][2]], sc.NAME_FROM_HANDLE))
                    i = j - 1
                    break
            i += 1
    return out


def contact_block(text: str, findings: list[Finding], lower: set[str]) -> list[Finding]:
    """Lines within two of an email, phone or profile link that are just a name or just
    'City, Country' belong to the person whose contact details those are."""
    lines = text.splitlines()
    anchors = [i for i, l in enumerate(lines)
               if any(f.kind in ("EMAIL", "PHONE", "URL") and f.text in l for f in findings)]
    near = sorted({k for i in anchors for k in range(i - 2, i + 3) if 0 <= k < len(lines)})
    out = []
    for k in near:
        # headers are often 'Tunis, Tunisia | +216 29 045 129 | me@mail.com | ...' on one line
        for segment in re.split(r"\s*[|•·]\s*", lines[k]):
            line = segment.strip()
            if not line or re.search(r"[\d@/:]", line):
                continue
            tokens = line.split()
            if _NAME_LINE.fullmatch(line):
                if (not any(t.upper() in NOT_NAMES or t.lower() in lower for t in tokens)) and not _is_org(line):
                    out.append(Finding("PERSON", line, sc.NAME_CONTACT_BLOCK))
            elif _PLACE_LINE.fullmatch(line):
                out.append(Finding("ADDRESS", line, sc.ADDRESS_PLACE_LINE))
    return out


# ---------------------------------------------------------------- NER

_nlps: dict[str, object] = {}


def _get_nlp(lang: str):
    if lang not in _nlps:
        import spacy

        _nlps[lang] = spacy.load(LANGUAGES[lang][0], exclude=["lemmatizer", "parser", "senter"])
    return _nlps[lang]


def _bare(line: str) -> str:
    """The line without bullets and surrounding punctuation, for 'is this the whole line?'."""
    return line.strip(" 	•·*-–—|:;,.")


def ner(text: str, lang: str = "en", lower: set[str] | None = None) -> list[Finding]:
    """Names and places from spaCy. Lines are analysed one by one so an entity never spans a
    line break (and can be found again on the page)."""
    nlp = _get_nlp(lang)
    lower = _lower_words(text) if lower is None else lower
    lines = [l for l in text.splitlines() if l.strip()]
    out = []
    for doc in nlp.pipe(lines):
        for ent in doc.ents:
            kind = "PERSON" if ent.label_ in PERSON_LABELS else "LOCATION" if ent.label_ in PLACE_LABELS else None
            if kind is None:
                continue
            hit = ent.text.strip()
            tokens = [t for t in ent if not t.is_punct and not t.is_space]
            # a lone capitalised word is usually a form label ("Email"), not a person
            if kind == "PERSON" and len(tokens) < 2:
                continue
            # "Label :" - the model sometimes tags the field name itself
            if doc.text[ent.end_char:].lstrip("  ").startswith(":"):
                continue
            if lang != "ar":
                if not _plausible_entity(kind, hit, lower):
                    continue
            # Part-of-speech tags cannot help here: on an isolated capitalised phrase the French
            # model tags everything PROPN ('Informatique Financière') and real names as NOUN/ADJ.
            standalone = _bare(doc.text) == _bare(hit) and len(tokens) <= 4
            if kind == "PERSON":
                score = sc.NER_PERSON_STANDALONE if standalone else sc.NER_PERSON
            else:
                score = sc.NER_LOCATION_SINGLE if len(tokens) == 1 else sc.NER_LOCATION
            out.append(Finding(kind, hit, score))
    return out
