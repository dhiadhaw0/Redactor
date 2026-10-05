"""Machine-readable zones (MRZ) of passports and ID cards, ICAO 9303.

An MRZ is the two or three lines of capital letters, digits and '<' filler printed at the bottom of a passport
or identity card. It repeats the holder's name, document number and date of birth, and it carries check
digits, which makes it far more reliable to detect than free text: a line that has the right length, the right
alphabet AND passes its check digits is an MRZ line.

  TD3  passport            2 lines x 44 characters
  TD2  some ID documents   2 lines x 36
  TD1  ID card             3 lines x 30

OCR reads the OCR-B font badly: '<' comes out as '«', '(', 'K' or is dropped, spaces appear. Lines are
normalised first, and a line with the right length but failing check digits is still reported, at a lower score,
because a misread digit must not leave the line readable.
"""
import re

from . import scoring as sc

ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
LENGTHS = {44: 2, 36: 2, 30: 3}          # characters per line -> lines in the zone
_OCR_FILLER = str.maketrans({"«": "<", "‹": "<", "〈": "<", "(": "<", "[": "<", "{": "<", "‹": "<", "⟨": "<"})


def char_value(c: str) -> int:
    if c.isdigit():
        return int(c)
    if c == "<":
        return 0
    return ord(c) - ord("A") + 10


def check_digit(field: str) -> str:
    """ICAO 9303 check digit: weights 7, 3, 1 repeating, sum modulo 10."""
    weights = (7, 3, 1)
    return str(sum(char_value(c) * weights[i % 3] for i, c in enumerate(field)) % 10)


def valid(field: str, digit: str) -> bool:
    return digit.isdigit() and check_digit(field) == digit


def normalize(raw: str) -> str:
    """Undo the usual OCR damage: spaces, look-alike filler characters, lower case."""
    return re.sub(r"\s+", "", raw.upper().translate(_OCR_FILLER))


def looks_like_mrz_line(norm: str) -> bool:
    return len(norm) in LENGTHS and set(norm) <= ALPHABET and norm.count("<") >= 3


# ------------------------------------------------------------------ field extraction
def _td3(lines: list[str]) -> dict:
    l1, l2 = lines
    checks = [valid(l2[0:9], l2[9]), valid(l2[13:19], l2[19]), valid(l2[21:27], l2[27]), valid(l2[28:42], l2[42])]
    composite = valid(l2[0:10] + l2[13:20] + l2[21:43], l2[43])
    return {"names": l1[5:], "number": l2[0:9], "dob": l2[13:19], "valid": sum(checks) + composite, "checks": 5}


def _td2(lines: list[str]) -> dict:
    l1, l2 = lines
    checks = [valid(l2[0:9], l2[9]), valid(l2[13:19], l2[19]), valid(l2[21:27], l2[27])]
    composite = valid(l2[0:10] + l2[13:20] + l2[21:35], l2[35])
    return {"names": l1[5:], "number": l2[0:9], "dob": l2[13:19], "valid": sum(checks) + composite, "checks": 4}


def _td1(lines: list[str]) -> dict:
    l1, l2, l3 = lines
    checks = [valid(l1[5:14], l1[14]), valid(l2[0:6], l2[6]), valid(l2[8:14], l2[14])]
    composite = valid(l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29], l2[29])
    return {"names": l3, "number": l1[5:14], "dob": l2[0:6], "valid": sum(checks) + composite, "checks": 4}


_PARSERS = {44: _td3, 36: _td2, 30: _td1}


def _names(field: str) -> tuple[list[str], list[str]]:
    """'ERIKSSON<<ANNA<MARIA<<<<' -> (['ERIKSSON'], ['ANNA', 'MARIA'])."""
    surname, _, given = field.partition("<<")
    split = lambda s: [w for w in re.split(r"<+", s) if re.fullmatch(r"[A-Z]{2,}", w)]   # OCR junk like '£66' is no name
    return split(surname), split(given)


# Tolerant patterns for lines OCR damaged (extra or garbled characters, so the length is off and no check digit
# can be trusted). The first line of a passport is 'P<' + a 3-letter country + SURNAME<<GIVEN<NAMES; the second
# holds '...<country><birth date><check><sex><expiry>...'. Both shapes are distinctive.
_LOOSE_FIRST = re.compile(r"^[PIAC][A-Z<][A-Z]{3}[A-Z]{2,}<<[A-Z<]{2,}")
_LOOSE_SECOND = re.compile(r"[A-Z0-9<]{9}[0-9<][A-Z]{3}[0-9]{6}[0-9<][MF<][0-9]{6}")


def _loose(raw: str, norm: str) -> list:
    """Findings for a line that has the look of an MRZ line but not its exact length."""
    out = []
    if 36 <= len(norm) <= 52 and set(norm) <= ALPHABET | set("0123456789") or len(norm) >= 36:
        if norm.count("<") >= 8 and _LOOSE_FIRST.match(norm):
            out.append(("MRZ", raw.strip(), sc.MRZ_SHAPE))
            surname, given = _names(norm[5:])
            for w in surname + given:
                out.append(("PERSON", w.title(), sc.NAME_FROM_MRZ))
            if surname and given:
                out.append(("PERSON", " ".join(given + surname).title(), sc.NAME_FROM_MRZ))
        else:
            m = _LOOSE_SECOND.search(norm)
            if m and norm.count("<") >= 3:
                out.append(("MRZ", raw.strip(), sc.MRZ_PARTIAL))
                number = norm[m.start():m.start() + 9].strip("<")
                if len(number) >= 5:
                    out.append(("ID", number, sc.ID_FROM_MRZ))
    return out


def detect_mrz(text: str):
    """Findings for every MRZ in `text`: each line, the holder's name and the document number.
    Returns (kind, text, score) triples; the caller wraps them in Finding objects."""
    raw_lines = [l for l in text.splitlines()]
    out = []
    i = 0
    while i < len(raw_lines):
        norm = normalize(raw_lines[i])
        n_lines = LENGTHS.get(len(norm)) if looks_like_mrz_line(norm) else None
        if n_lines is None:
            out += _loose(raw_lines[i], norm)
            i += 1
            continue
        group = [(raw_lines[i + k], normalize(raw_lines[i + k])) for k in range(n_lines) if i + k < len(raw_lines)]
        if len(group) < n_lines or any(len(n) != len(norm) or not looks_like_mrz_line(n) for _, n in group):
            out += _loose(raw_lines[i], norm)  # a lone line of the right length but no partner beside it
            i += 1
            continue
        fields = _PARSERS[len(norm)]([n for _, n in group])
        if fields["valid"] >= 3:
            score = sc.MRZ_VALID
        elif fields["valid"] >= 1:
            score = sc.MRZ_PARTIAL
        else:
            score = sc.MRZ_SHAPE
        for raw, _ in group:
            out.append(("MRZ", raw.strip(), score))
        surname, given = _names(fields["names"])
        for w in surname + given:
            out.append(("PERSON", w.title(), sc.NAME_FROM_MRZ))
        if surname and given:
            out.append(("PERSON", " ".join(given + surname).title(), sc.NAME_FROM_MRZ))
        number = fields["number"].strip("<")
        if len(number) >= 5:
            out.append(("ID", number, sc.ID_FROM_MRZ))
        i += n_lines
    return out


# ------------------------------------------------------------------ building (tests and the benchmark corpus)
def _pad(s: str, n: int) -> str:
    return (s + "<" * n)[:n]


def build_td3(surname: str, given: str, issuer: str, number: str, nationality: str, dob: str, sex: str,
              expiry: str, personal: str = "") -> list[str]:
    """A passport MRZ with correct check digits. dob / expiry are YYMMDD."""
    names = _pad(surname.upper().replace(" ", "<") + "<<" + given.upper().replace(" ", "<"), 39)
    l1 = "P<" + issuer + names
    number9, personal14 = _pad(number.upper(), 9), _pad(personal.upper(), 14)
    body = (number9 + check_digit(number9) + nationality + dob + check_digit(dob) + sex + expiry + check_digit(expiry)
            + personal14 + check_digit(personal14))
    composite = check_digit(body[0:10] + body[13:20] + body[21:43])
    return [l1, body + composite]


def build_td1(surname: str, given: str, issuer: str, number: str, nationality: str, dob: str, sex: str,
              expiry: str, optional: str = "") -> list[str]:
    """An ID-card MRZ (3 lines of 30) with correct check digits."""
    number9 = _pad(number.upper(), 9)
    opt1 = _pad(optional.upper(), 15)
    l1 = "I<" + issuer + number9 + check_digit(number9) + opt1
    opt2 = _pad("", 11)
    l2_head = dob + check_digit(dob) + sex + expiry + check_digit(expiry) + nationality + opt2
    composite = check_digit(l1[5:30] + dob + check_digit(dob) + expiry + check_digit(expiry) + opt2)
    l2 = l2_head + composite
    l3 = _pad(surname.upper().replace(" ", "<") + "<<" + given.upper().replace(" ", "<"), 30)
    return [l1, l2, l3]
