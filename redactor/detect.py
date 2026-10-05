"""Detect personal data: regex rules, label/context rules and spaCy NER (en / fr / ar)."""
import difflib
import re
from dataclasses import dataclass

from . import scoring as sc


@dataclass(frozen=True)
class Finding:
    kind: str
    text: str
    score: int = sc.LABELLED  # 0-100 confidence that this should be redacted (see scoring.py)

    @property
    def low(self) -> bool:
        """Below the default threshold: listed for review, not pre-selected."""
        return self.score < sc.DEFAULT_THRESHOLD


# language code -> (spaCy model, Tesseract language string)
LANGUAGES = {
    "en": ("en_core_web_sm", "eng"),
    "fr": ("fr_core_news_sm", "fra+eng"),
    "ar": ("xx_ent_wiki_sm", "ara+eng"),  # multilingual model; no Arabic-specific small model
}

# Arabic-Indic and Persian digits -> ASCII, so regexes work on them too
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")
IBAN_COUNTRIES = set(
    "AD AE AL AT AZ BA BE BG BH BR BY CH CR CY CZ DE DK DO EE EG ES FI FO FR GB GE GI GL GR GT HR HU IE IL IQ "
    "IS IT JO KW KZ LB LC LI LT LU LV LY MC MD ME MK MR MT MU NL NO PK PL PS PT QA RO RS SA SC SE SI SK SM ST "
    "SV TL TN TR UA VA VG XK".split()
)
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
# National identification numbers
NIR = re.compile(r"\b[12](?:[ .]?\d{2}){2}[ .]?(?:\d{2}|2[AB])[ .]?\d{3}[ .]?\d{3}[ .]?\d{2}\b")   # France, 15 digits + key
UK_NINO = re.compile(r"\b[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z](?:[ ]?\d{2}){3}[ ]?[A-D]\b")        # UK National Insurance
US_SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")                      # US Social Security
TN_TAX_ID = re.compile(r"\b\d{7}[ ]?[A-Z]/[A-Z]/[A-Z]/\d{3}\b")                                   # Tunisia, matricule fiscal


def _nir_valid(s: str) -> bool:
    """The last two digits of a French NIR are 97 - (the first 13 as a number mod 97); Corsica uses 2A/2B."""
    d = re.sub(r"[ .]", "", s).upper().replace("2A", "19").replace("2B", "18")
    return len(d) == 15 and d.isdigit() and 97 - int(d[:13]) % 97 == int(d[13:])


POSTCODE_LINE = re.compile(r"(?m)^[ \t]*\d{5}[ \t]+[A-ZÀ-Ý][A-Za-zÀ-ÿ'’ -]{2,}[ \t]*$")
# '59000 LILLE' inside a sentence (an all-caps city of 4+ letters; '15000 EUR' has only 3)
POSTCODE_CITY_CAPS = re.compile(r"\b\d{5}[ \t]+[A-ZÀ-Ý]{4,}(?:[ -][A-ZÀ-Ý]{3,})*\b")
# UK: 'London SW1A 1AA' / 'SW1A 1AA London' on a line of its own, or a bare postcode
_UK_POST = r"[A-Z]{1,2}\d[A-Z\d]?[ \t]?\d[A-Z]{2}"
UK_POSTCODE_LINE = re.compile(
    rf"(?m)^[ \t]*(?:[A-Z][a-z]+(?:[ -][A-Z][a-z]+)*[ \t,]+)?{_UK_POST}(?:[ \t]+[A-Z][a-z]+(?:[ -][A-Z][a-z]+)*)?[ \t]*$")
UK_POSTCODE = re.compile(rf"\b(?:[A-Z][a-z]+[ \t]+)?{_UK_POST}\b")  # 'Bristol BS1 5TR' / bare 'BS1 5TR'

# A name that follows a phrase which introduces a person: 'in the presence of Daniel Ellis'
_CAP = r"[A-ZÀ-Ý][\wà-ÿ'’-]+"
NAME_AFTER_CUE = re.compile(
    r"(?i:in the presence of|witnessed by|signed by|represented by|on behalf of|prepared by|approved by|issued to"
    r"|en pr[ée]sence de|sign[ée] par|repr[ée]sent[ée] par|au nom de|[àa] l.attention de)[ \t]+"
    rf"({_CAP}(?:[ \t]+(?:{_CAP}|[A-ZÀ-Ý]{{2,}})){{0,2}})"
)
PHONE = re.compile(r"(?<!\w)\+?\d[\d ()./-]{7,}\d(?!\w)")
# Contract / claim / customer references such as SIN-2026-0093178 or HAB-2026-0731245
REFERENCE = re.compile(r"\b[A-Z]{2,6}[-/]\d{2,4}(?:[-/]\d{3,}){1,2}\b|\b[A-Z]{2,6}[-/]\d{5,}\b")

_DATE = re.compile(r"^(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}[/.-]\d{1,2}[/.-]\d{1,2}|\d{2,4}[/.-]\d{4})$")
_YEAR_RANGE = re.compile(r"^(?:(?:19|20)\d{2}\s*[-–/]\s*)+(?:19|20)\d{2}$")  # 2023 - 2025
_THOUSANDS = re.compile(r"^\d{1,3}(?: \d{3}){2,}$")  # 24 000 000

# Titles followed by a name: English, French, Arabic
TITLE_NAME = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Mme|Mlle|M)\.?[ \t]+[A-ZÀ-Ý][a-zà-ÿ'’-]+(?:[ \t]+[A-ZÀ-Ý][A-Za-zà-ÿ'’-]+)?"
    r"|(?:السيد|السيدة|الآنسة|الدكتور|الدكتورة|الأستاذ|الأستاذة)\s+[؀-ۿ]+(?:\s+(?:بن|ابن|بنت)\s+[؀-ۿ]+|\s+[؀-ۿ]+)?"
)
# same-line only: a term that spans a line break is fragile to find again on the page
_POSTCODE_CITY = r"(?:,?[ \t]*\d{5}[ \t]+[A-ZÀ-Ý][A-Za-zÀ-ÿ'’-]+(?:[ -][A-ZÀ-Ý][A-Za-zÀ-ÿ'’-]+)*)?"
ADDRESS = re.compile(
    # English: 12 Baker Street
    r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Boulevard|Blvd|Drive|Dr|Way|Square)\b"
    # French: 12 bis rue de la Paix, 75002 PARIS
    r"|\b\d{1,4}(?:\s?(?:bis|ter))?,?\s+(?:rue|avenue|av\.|boulevard|bd|place|chemin|allée|impasse|quai|cours)"
    r"(?:\s+(?:de|du|des|d'|de la|l')?\s?[A-ZÀ-Ý][\wà-ÿ'’-]+){1,4}" + _POSTCODE_CITY +
    # Arabic: شارع الحبيب بورقيبة 12 / 12 شارع ...
    r"|(?:\d{1,4}[ \t]+)?(?:شارع|زنقة|نهج|حي|طريق)[ \t]+[؀-ۿ]+(?:[ \t]+[؀-ۿ]+){0,3}(?:[ \t]+\d{1,4})?"
)

# Words that mean "this is an organisation / a field of study / a job word, not a person or a place"
ORG_WORDS = {
    "assurance", "assurances", "société", "societe", "sarl", "sas", "sasu", "banque", "bank", "ltd", "inc",
    "corp", "gmbh", "group", "groupe", "company", "mutuelle", "cabinet", "agence", "association", "ministère",
    "université", "university", "hôpital", "clinique", "شركة", "بنك",
    # CVs and reports are full of capitalised study, job and tech words that NER tags as people
    "institut", "institute", "entreprise", "enterprise", "département", "department", "filière", "faculté",
    "école", "ecole", "lycée", "collège", "club", "projet", "project", "système", "systeme", "system",
    "informatique", "mathématiques", "mathematiques", "physique", "ingénierie", "ingénieur", "engineering",
    "master", "licence", "diplôme", "stage", "management", "marketing", "développement", "developpement",
    "développeur", "developer", "engineer", "software", "data", "cloud", "web", "mobile", "backend",
    "frontend", "fullstack", "framework", "frameworks", "financière", "financiere", "junior", "senior",
    "plateforme", "platform", "service", "services", "solutions", "technologies", "consulting",
    "spring", "boot", "angular", "react", "django", "flask", "docker", "kubernetes", "jenkins", "linux",
    "machine", "learning", "deep",
}
NOT_NAMES = {  # all-caps abbreviations and field words that look like surnames
    "TTC", "HT", "TVA", "IBAN", "BIC", "RIB", "SIRET", "SIREN", "RCS", "PDF", "EUR", "USD", "SARL", "SAS",
    "NOM", "DATE", "TOTAL", "PAGE",
}
PARTICLES = {"de", "du", "des", "la", "le", "van", "von", "bin", "ben", "ibn", "al", "el", "d", "di", "da"}

# ---- label / context rules: the value that follows a label such as "Nom et prénom" ----
_SEP = r"[\s:|·\-–—]*"
LABELS = [
    ("ID", r"iban|rib|compte\s+bancaire|num[ée]ro\s+de\s+(?:client|contrat|dossier|sinistre|police|compte|membre|s[ée]curit[ée]\s+sociale)"
           r"|n[°o]\s*(?:de\s+)?(?:client|contrat|dossier|police|compte|sécu)|client\s+n[°o]"
           r"|(?:customer|account|policy|member|case|file)\s+(?:id|number|no\.?)|id\s+(?:number|no\.?)"
           r"|(?:national|identity)\s+id(?:\s+(?:number|no\.?))?|id\s+card(?:\s+(?:number|no\.?))?"
           r"|num[ée]ro\s+de\s+(?:cin|passeport|passport|carte(?:\s+d.identit[ée])?)"
           r"|passport(?:\s+(?:number|no\.?))?|passeport|c\.?i\.?n\.?|matricule|ssn|nir|insee|رقم\s+[؀-ۿ]+"),
    ("DATE_OF_BIRTH", r"date\s+de\s+naissance|né(?:e)?\s+le|date\s+of\s+birth|dob|born(?:\s+on)?|تاريخ\s+الميلاد"),
    ("ADDRESS", r"adresse(?:\s+(?:du\s+risque|postale|de\s+facturation|de\s+livraison))?|address|domicile"
               r"|résidence|العنوان"),
    # accents are optional everywhere: OCR and plain-text exports often drop them
    ("PERSON", r"nom(?:\s+et\s+pr[ée]nom|\s+complet)?|pr[ée]nom|full\s+name|first\s+name|last\s+name|surname|name"
               r"|client|souscripteur|assur[ée](?:e)?|titulaire|b[ée]n[ée]ficiaire|signataire|patient|employee"
               r"|employ[ée](?:e)?|salari[ée](?:e)?|applicant|الاسم|اسم|الموظف|المستفيد"
               r"|emergency\s+contact|contact\s+person|person\s+to\s+contact|next\s+of\s+kin"
               r"|personne\s+[àa]\s+contacter|contact\s+d.urgence"),
]
# For a person's name, the value ends where the name does: 'Emily Ellis, +44 73 5711' -> 'Emily Ellis'
_NAME_END = re.compile(r"[,;(]|\s-\s|\s\d")
_LABEL_RES = [(k, re.compile(rf"^\W*(?:{p})\b{_SEP}(.*)$", re.I)) for k, p in LABELS]


def _iban_valid(s: str) -> bool:
    s = s.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    rearranged = s[4:] + s[:4]
    digits = "".join(str(int(c, 36)) for c in rearranged)
    return int(digits) % 97 == 1


def _luhn_valid(s: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", s)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _phone_like(s: str) -> bool:
    """A digit run that is a plausible phone number and not a date or an amount."""
    if not 8 <= len(re.sub(r"\D", "", s)) <= 15:  # E.164 max is 15 digits
        return False
    s = s.strip()
    if re.search(r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}", s) or re.search(r" {2,}", s):
        return False  # a date inside it, or table columns: '12/01/2026        1 240,00'
    return not (_DATE.match(s) or _THOUSANDS.match(s) or _YEAR_RANGE.match(s))


def detect_regex(text: str) -> list[Finding]:
    # translate() is 1:1 per character so offsets still line up with `text`
    norm = text.translate(_DIGITS)
    found: list[Finding] = []
    taken: list[tuple[int, int]] = []

    def add(kind, m, score):
        if any(m.start() < e and m.end() > s for s, e in taken):
            return
        taken.append((m.start(), m.end()))
        found.append(Finding(kind, text[m.start():m.end()].strip(), score))

    for m in EMAIL.finditer(norm):
        add("EMAIL", m, sc.EMAIL)
    for m in IBAN.finditer(norm):
        # Shape + a real IBAN country is enough: a failed checksum usually means an OCR
        # misread or a made-up number, and a partly-redacted IBAN is the worst outcome.
        if m.group()[:2] in IBAN_COUNTRIES:
            add("IBAN", m, sc.IBAN_VALID if _iban_valid(m.group()) else sc.IBAN_SHAPE)
    # National identification numbers come before the card rule: a 15-digit NIR must not become a 'NUMBER'.
    for m in NIR.finditer(norm):
        if _nir_valid(m.group()):
            add("ID", m, sc.NATIONAL_ID_VALID)
    for m in UK_NINO.finditer(norm):
        add("ID", m, sc.NATIONAL_ID_PATTERN)
    for m in US_SSN.finditer(norm):
        add("ID", m, sc.NATIONAL_ID_PATTERN)
    for m in TN_TAX_ID.finditer(norm):
        add("ID", m, sc.NATIONAL_ID_PATTERN)
    for m in CARD.finditer(norm):
        # Luhn-valid -> card. A long digit run that fails Luhn is most likely a card or
        # account number with one OCR misread digit, so still flag it (over-redact, not leak).
        if not _DATE.match(m.group().strip()):
            if _luhn_valid(m.group()):
                add("CREDIT_CARD", m, sc.CREDIT_CARD)
            else:
                add("NUMBER", m, sc.NUMBER)
    for m in REFERENCE.finditer(norm):
        add("REFERENCE", m, sc.REFERENCE)
    for m in ADDRESS.finditer(norm):
        add("ADDRESS", m, sc.ADDRESS_STREET)
    for m in POSTCODE_LINE.finditer(norm):  # '69003 LYON' as the second line of an address block
        add("ADDRESS", m, sc.ADDRESS_POSTCODE_LINE)
    for m in UK_POSTCODE_LINE.finditer(norm):  # 'London SW1A 1AA'
        add("ADDRESS", m, sc.ADDRESS_POSTCODE_LINE)
    for m in POSTCODE_CITY_CAPS.finditer(norm):  # '59000 LILLE (le Salarie) et'
        add("ADDRESS", m, sc.ADDRESS_POSTCODE_LINE)
    for m in UK_POSTCODE.finditer(norm):  # a bare UK postcode inside a sentence
        add("ADDRESS", m, sc.ADDRESS_POSTCODE_LINE)
    for m in TITLE_NAME.finditer(norm):
        add("PERSON", m, sc.NAME_TITLE)
    for m in NAME_AFTER_CUE.finditer(norm):
        a, b = m.span(1)
        found.append(Finding("PERSON", text[a:b].strip(), sc.NAME_AFTER_CUE))
    for m in PHONE.finditer(norm):
        if _phone_like(m.group()):
            add("PHONE", m, sc.PHONE_INTERNATIONAL if m.group().lstrip().startswith("+") else sc.PHONE)
    return found


# ---------------------------------------------------------------- label / context rules

def _value_ok(kind: str, v: str) -> bool:
    if not 2 <= len(v) <= 80 or not re.search(r"\w", v):
        return False
    if kind == "PERSON":  # not lower-case: Arabic has no case at all
        return v[0].isalpha() and not v[0].islower() and not re.search(r"\d", v)
    if kind in ("ID", "DATE_OF_BIRTH"):
        return bool(re.search(r"\d", v))
    return bool(re.search(r"\d", v)) or v[0].isupper()  # ADDRESS


FIELD_WORDS = [
    "address", "adresse", "email", "mail", "courriel", "phone", "telephone", "tel", "mobile", "name", "nom", "prenom",
    "date", "birth", "naissance", "passport", "passeport", "iban", "card", "carte", "customer", "client", "number",
    "numero", "contact", "employee", "salarie", "city", "ville", "country", "pays", "postcode", "signature", "reference",
]


def _is_field_word(value: str) -> bool:
    """A single word that is a (possibly OCR-garbled) form-field name: 'Adress', 'Telefone', 'Emial'."""
    v = re.sub(r"[^a-zà-ÿ]", "", value.lower())
    if " " in value.strip() or len(v) < 3:
        return False
    plain = v.translate(str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc"))
    return bool(difflib.get_close_matches(plain, FIELD_WORDS, n=1, cutoff=0.8))


def _is_label_line(line: str) -> bool:
    """A line that is itself a field label (ends with a colon, or is just a known label)."""
    line = line.strip()
    if line.endswith(":"):
        return True
    for _, rx in _LABEL_RES:  # a bare known label such as "Adresse" or "Nom et prénom": nothing follows it
        m = rx.match(line)
        if m and not m.group(1).strip(" :|·-–—"):
            return True
    return False


def detect_context(text: str) -> list[Finding]:
    """'Nom et prénom : MARCHAND Julien' -> PERSON. Also handles digital PDFs where
    the label and its value sit on consecutive lines."""
    lines = [l.strip() for l in text.translate(_DIGITS).splitlines()]
    out = []
    for i, line in enumerate(lines):
        for kind, rx in _LABEL_RES:
            m = rx.match(line)
            if not m:
                continue
            value = m.group(1).strip(" :|·-–—\t")
            stacked = not value
            if stacked:  # value is on the next non-empty line
                # OCR often puts the colon on its own line or in front of the value: skip lone-colon lines
                candidate = next((l for l in lines[i + 1:i + 4] if l.strip(" :|·-–—\t")), "")
                value = candidate.strip(" :|·-–—\t")
                # 'Address:' followed by 'Email:' or by 'Address: 12 Baker Street': an empty field, not a value.
                # Judged on the raw candidate: its trailing colon is what says it is a label.
                if (_is_label_line(candidate) or _is_field_word(value)
                        or any(rx.match(candidate) for _, rx in _LABEL_RES)):
                    value = ""
            # 'CLIENT FORM', 'Customer Service': a label word followed by one word and no colon is a heading
            if kind == "PERSON":
                value = _NAME_END.split(value, maxsplit=1)[0].strip()
                if not stacked and ":" not in line and len(value.split()) == 1:
                    value = ""
            if _value_ok(kind, value):
                out.append(Finding(kind, text_span(text, value), sc.LABELLED))
            break  # first matching label wins for this line
    return out


def text_span(original: str, value: str) -> str:
    """`value` was taken from digit-normalised text; return the same characters as in
    the original (lengths are identical because the translation is 1:1)."""
    idx = original.translate(_DIGITS).find(value)
    return original[idx:idx + len(value)] if idx >= 0 else value


# ---------------------------------------------------------------- name heuristics

_NAME_SHAPED = re.compile(
    r"\b[A-ZÀ-Ý][a-zà-ÿ'’-]{2,}[ \t]+[A-ZÀ-Ý]{3,}\b(?![-/])"   # Julien MARCHAND  (not 'Invoice INV-2026')
    r"|\b[A-ZÀ-Ý]{3,}[ \t]+[A-ZÀ-Ý][a-zà-ÿ'’-]{2,}\b"  # MARCHAND Julien
)
MAX_NAME_LINE_TOKENS = 3  # only lines that are essentially just the name (signature, addressee)


_NON_PROSE = re.compile(
    r"\S*[@/]\S*"                                                      # emails, URLs, paths
    r"|\S+\.(?:com|org|net|io|fr|tn|dev|co|app|me|example)\b\S*"       # bare domains
)


def _lower_words(text: str) -> set[str]:
    """Lower-case words that occur in running text. Emails and URLs are excluded:
    'maria.garcia@acme.org' must not make 'Maria' look like an ordinary word."""
    return set(re.findall(r"\b[a-zà-ÿ]{3,}\b", _NON_PROSE.sub(" ", text)))


def _is_org(hit: str) -> bool:
    return any(t.lower().strip(".,") in ORG_WORDS for t in hit.split())


def _plausible_entity(kind: str, hit: str, lower: set[str]) -> bool:
    """Filter NER output: drop OCR garbage, organisations and ordinary capitalised words."""
    tokens = hit.split()
    if not tokens or re.search(r"\d", hit) or not hit[0].isupper():
        return False
    if not re.fullmatch(r"[^\W\d_]+(?:[ '’.-]+[^\W\d_]+)*\.?", hit):
        return False
    core = [t for t in tokens if t.lower().strip(".'’") not in PARTICLES]
    if any(len(t.strip(".-'’")) < 3 for t in core):
        return False
    if _is_org(hit):
        return False
    # "Contrat", "CONDITIONS", "Frais": the same word also appears in lower case in the text,
    # so it is a common word, not a name or a place
    if all(t.lower().strip(".'’") in lower for t in core):
        return False
    return True


def detect_name_shaped(text: str, lower: set[str]) -> list[Finding]:
    out = []
    for line in text.splitlines():
        if len(line.split()) > MAX_NAME_LINE_TOKENS or re.search(r"\d", line):  # a name's line has no digits
            continue
        for m in _NAME_SHAPED.finditer(line):
            a, b = m.group().split()
            if (a.upper() in NOT_NAMES or b.upper() in NOT_NAMES or _is_org(m.group())
                    or a.lower() in lower or b.lower() in lower):
                continue
            out.append(Finding("PERSON", m.group(), sc.NAME_SHAPED))
    return out


def propagate_names(findings: list[Finding], lower: set[str], penalty: int = sc.PIECE_OF_RULE_NAME) -> list[Finding]:
    """If 'Julien MARCHAND' is a person, 'MARCHAND Julien' and a lone 'MARCHAND' are too.
    Searching is case-insensitive, so one token finding covers every spelling. A piece is a
    guess derived from its parent, so it scores below the parent by `penalty` (bigger when only
    the NER model vouched for the parent)."""
    extra = []
    for f in findings:
        if f.kind != "PERSON":
            continue
        for t in re.findall(r"[^\W\d_][\w'’-]*", f.text):
            if (len(t) >= 4 and t[0].isupper() and t.lower() not in lower and t.lower() not in PARTICLES
                    and t.upper() not in NOT_NAMES and t.rstrip(".").lower() not in {"madame", "monsieur", "mademoiselle"}):
                extra.append(Finding("PERSON", t, sc.clamp(f.score + penalty)))
    return extra


# ---------------------------------------------------------------- NER

def detect_ner(text: str, lang: str = "en", min_score: float = 0.4,
               lower: set[str] | None = None) -> list[Finding]:
    """Names and places from spaCy (see people.ner). `min_score` is kept for compatibility."""
    from . import people
    return people.ner(text, lang, lower)


# ---------------------------------------------------------------- everything together

# Findings whose text merely CONTAINS other personal data (a name in an email, in a profile URL, in an MRZ line):
# the data inside must still be redacted wherever else it is printed, so it is never dropped as "just a piece".
CONTAINERS = {"EMAIL", "URL", "MRZ", "BARCODE"}


def _best(findings: list[Finding]) -> list[Finding]:
    """One finding per (kind, text): the highest-scoring one. A regex finds the same email on
    every line it appears; that is one opinion, not several."""
    best: dict[tuple[str, str], Finding] = {}
    for f in findings:
        old = best.get((f.kind, f.text))
        if old is None or f.score > old.score:
            best[(f.kind, f.text)] = f
    return list(best.values())


def detect(text: str, use_ner: bool = True, lang: str = "en") -> list[Finding]:
    """All findings with their confidence scores. Each independent source (patterns, labels,
    name heuristics, handles, contact block, NER, name pieces) gives its own opinion; when several
    agree on the same text the score rises (corroboration)."""
    from . import people  # imports this module, so load it lazily

    lower = _lower_words(text)
    # Rules (patterns, labels, titles, handles, contact block) are trusted; the NER model is not:
    # it tags headings and skills as people, so it scores lower and its name pieces lower still.
    from . import mrz  # passport / ID-card machine-readable zones, verified by their check digits

    groups = [detect_regex(text), detect_context(text), people.detect_social(text),
              [Finding(k, t, s) for k, t, s in mrz.detect_mrz(text)]]
    if lang != "ar":
        groups.append(detect_name_shaped(text, lower))
    rules = [f for g in groups for f in g]
    handles = people.names_from_handles(text, rules)
    rules += handles
    contact = people.contact_block(text, rules, lower)
    rules += contact
    ner = people.ner(text, lang, lower) if use_ner else []
    # Name pieces derived from rule names and from NER names are ONE opinion (both are guesses
    # from the same parents), so they must not corroborate each other.
    pieces = propagate_names(rules, lower, sc.PIECE_OF_RULE_NAME) + propagate_names(ner, lower, sc.PIECE_OF_NER_NAME)
    groups += [handles, contact, ner, pieces]

    opinions: dict[tuple[str, str], list[int]] = {}
    for group in groups:
        for f in _best(group):
            opinions.setdefault((f.kind, f.text), []).append(f.score)
    lowered = text.lower()
    unique = []
    for (kind, t), scores in opinions.items():
        score = max(scores) + sc.CORROBORATED * (len(scores) - 1)
        if score < sc.UNCERTAIN_BELOW and lowered.count(t.lower()) >= 2:
            score += sc.REPEATED  # uncertain, but it keeps coming back
        unique.append(Finding(kind, t, sc.clamp(score)))

    # Drop a finding that is just a piece of a longer one of a different kind (the 'LYON' inside
    # an ADDRESS); same-kind pieces stay (name tokens). A person's name inside their own email or
    # profile URL is not "just a piece": the name must still be redacted wherever else it occurs.
    kept = [f for f in unique
            if not any(g.kind != f.kind and f.text.lower() in g.text.lower() and len(g.text) > len(f.text)
                       and g.kind not in CONTAINERS
                       for g in unique)]
    # longest first so partial strings don't pre-empt full ones
    return sorted(kept, key=lambda f: -len(f.text))
