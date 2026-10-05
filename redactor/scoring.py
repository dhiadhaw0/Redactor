"""Confidence scores (0-100) for detections, in one place so they are easy to read and tune.

A score answers "how sure are we that this is personal data that should be redacted?":

  95-99  checked or self-identifying: a valid IBAN, an email, a profile link, a name spelled by an email handle
  85-94  a strong rule: a label ("Nom et prénom : ..."), a title ("M. Dupont"), a street address
  60-84  a plausible rule: a name-shaped signature line, a postcode line, a long digit string
  below 60 only a statistical model vouched for it (spaCy NER), or it is a bare place name: listed for review,
           not pre-selected

Adjustments: the same text found by independent sources is corroborated (+); uncertain text that
repeats in the document is more likely real (+); text read through OCR is slightly less certain (-);
a name piece inherits its parent's score minus a penalty (it is a guess derived from the parent).
"""

DEFAULT_THRESHOLD = 60   # detections scoring at least this are pre-selected in the UI / kept by the CLI

# ---- base scores by source ------------------------------------------------------------------
EMAIL = 98
IBAN_VALID = 99          # real IBAN: country, shape and mod-97 checksum
IBAN_SHAPE = 90          # right shape and country but the checksum fails (OCR misread, typo, sample number)
CREDIT_CARD = 97         # Luhn-valid
NUMBER = 70              # 13-19 digits failing Luhn: probably a card/account with a misread digit
REFERENCE = 85           # SIN-2026-0093178
URL_PROFILE = 95         # linkedin.com/in/..., github.com/...
PHONE = 85
PHONE_INTERNATIONAL = 92  # starts with +
ADDRESS_STREET = 88
ADDRESS_POSTCODE_LINE = 80
ADDRESS_PLACE_LINE = 85   # 'City, Country' in the contact block
NAME_TITLE = 92           # M. / Mr / Mme / السيد followed by a name
LABELLED = 90             # the value after "Nom et prénom", "Adresse", "Numéro de client"...
NAME_FROM_HANDLE = 95     # capitalised words that spell an email or profile handle
NAME_CONTACT_BLOCK = 85   # a name line next to an email/phone
NAME_AFTER_CUE = 85       # 'in the presence of Daniel Ellis', 'signé par ...'
MRZ_VALID = 99            # a passport/ID machine-readable zone whose check digits pass
MRZ_PARTIAL = 92          # some check digits fail (OCR misread): still an MRZ line
MRZ_SHAPE = 80            # right length and alphabet, no check digit verifies
NAME_FROM_MRZ = 90        # the holder's name, read from the MRZ
ID_FROM_MRZ = 96          # the document number, read from the MRZ
NATIONAL_ID_VALID = 97    # a national number with a verifying key (France NIR)
NATIONAL_ID_PATTERN = 90  # a distinctive national-ID format without a checksum (UK NI, US SSN, TN tax id)
BARCODE = 85              # a decoded QR code / barcode: these usually carry the person's data
NAME_SHAPED = 75          # 'Nathalie ROBIN' alone on a short line
NER_PERSON = 65           # above the default threshold even after the OCR penalty (-5)
NER_PERSON_STANDALONE = 35  # the whole line is the "name": a heading or a skill, not a person
NER_LOCATION = 40
NER_LOCATION_SINGLE = 30    # a bare place name rarely identifies anyone

# ---- adjustments ----------------------------------------------------------------------------
PIECE_OF_RULE_NAME = -5   # 'MARCHAND' taken from a name a rule found
PIECE_OF_NER_NAME = -15   # ...from a name only NER vouched for
CORROBORATED = 5          # per extra independent source that found the same text
REPEATED = 5              # uncertain text that occurs again in the document
OCR = -5                  # read through OCR
FLOOR, CEIL = 5, 99
UNCERTAIN_BELOW = 80      # only uncertain findings gain from repetition


def clamp(score: float) -> int:
    return int(max(FLOOR, min(CEIL, round(score))))


def band(score: int) -> str:
    """Human label for a score, used in the UI."""
    return "very high" if score >= 95 else "high" if score >= 85 else "medium" if score >= 60 else "low"
