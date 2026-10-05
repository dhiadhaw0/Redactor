"""Seeded generator of fake people and their data (names, addresses, valid IBANs, Luhn cards...).

Everything is invented: no real person, address or account. IBANs and cards pass their checksums so
they are detected exactly like real ones.
"""
import random
import unicodedata
from dataclasses import dataclass

FIRST = {
    "en": ["James", "Emily", "Oliver", "Sophia", "Daniel", "Hannah", "Ethan", "Grace", "Lucas", "Charlotte",
           "Henry", "Amelia", "Samuel", "Isabella", "Thomas", "Victoria"],
    "fr": ["Julien", "Camille", "Mathieu", "Léa", "Nicolas", "Chloé", "Antoine", "Manon", "Hugo", "Élodie",
           "Maxime", "Sarah", "Alexandre", "Pauline", "Romain", "Océane"],
    "ar": ["Mohamed Amine", "Yasmine", "Karim", "Ines", "Sofiene", "Rania", "Walid", "Mariem", "Dhia Eddine",
           "Sana", "Hatem", "Salma", "Anis", "Nour", "Fares", "Aya"],
}
LAST = {
    "en": ["Walker", "Reed", "Hughes", "Bennett", "Foster", "Mitchell", "Carter", "Brooks", "Turner", "Hayes",
           "Ellis", "Graham", "Morgan", "Parker", "Sutton", "Whitaker"],
    "fr": ["Marchand", "Lefèvre", "Bertrand", "Dubois", "Moreau", "Girard", "Faure", "Fontaine", "Rousseau",
           "Mercier", "Blanchard", "Chevalier", "Perrin", "Garnier", "Colin", "Renard"],
    "ar": ["Ben Salah", "Trabelsi", "Jebali", "Gharbi", "Toumi", "Khelifi", "Mansouri", "Bouazizi", "Hammami",
           "Ben Ammar", "Cherif", "Dridi", "Sassi", "Zouari", "Belhaj", "Chebbi"],
}
FEMALE = {"Emily", "Sophia", "Hannah", "Grace", "Charlotte", "Amelia", "Isabella", "Victoria", "Camille", "Léa", "Chloé",
          "Manon", "Élodie", "Sarah", "Pauline", "Océane", "Yasmine", "Ines", "Rania", "Mariem", "Sana", "Salma", "Nour", "Aya"}
ARABIC_NAMES = ["محمد بن علي", "فاطمة الزهراء", "أحمد الطرابلسي", "ليلى بن صالح", "يوسف المنصوري", "سلمى الغربي"]
STREETS = {
    "en": ["Baker Street", "Elm Road", "Kingsway Avenue", "Orchard Lane", "Maple Drive", "Station Road", "Queens Way"],
    "fr": ["rue des Acacias", "rue de la Paix", "avenue Victor Hugo", "boulevard Voltaire", "rue du Moulin",
           "avenue des Champs", "rue Pasteur"],
    "ar": ["شارع الحبيب بورقيبة", "شارع الجمهورية", "نهج الحرية", "شارع فلسطين"],
}
CITIES = {
    "en": [("London", "SW1A 1AA"), ("Leeds", "LS1 4AP"), ("Bristol", "BS1 5TR"), ("York", "YO1 7HH")],
    "fr": [("LYON", "69003"), ("PARIS", "75011"), ("LILLE", "59000"), ("NANTES", "44000"), ("TOULOUSE", "31000")],
    "ar": [("تونس", ""), ("صفاقس", ""), ("سوسة", ""), ("بنزرت", "")],
}
COMPANIES = ["Atlantis Assurances", "Northwind Bank", "Helios Energie", "Orion Logistics", "Vega Telecom"]
SKILLS = ["Python", "Spring Boot", "Docker", "Machine Learning", "Big Data", "Kubernetes", "React", "Monte Carlo",
          "Leadership", "Data Analysis", "Project Management"]
EMAIL_DOMAINS = ["gmail.com", "outlook.com", "orange.fr", "example.org", "mail.example"]


def ascii_lower(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return "".join(c for c in s.lower() if c.isalnum())


def ascii_upper(s: str) -> str:
    """'Lefèvre' -> 'LEFEVRE': what an MRZ can hold."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper()


def _mod97_check(country: str, bban: str) -> str:
    digits = "".join(str(int(c, 36)) for c in bban + country + "00")
    return f"{98 - int(digits) % 97:02d}"


def make_iban(rng: random.Random, country: str) -> str:
    if country == "GB":
        bban = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(4)) + "".join(rng.choice("0123456789") for _ in range(14))
    elif country == "TN":
        bban = "".join(rng.choice("0123456789") for _ in range(20))
    else:  # FR
        bban = "".join(rng.choice("0123456789") for _ in range(23))
    iban = country + _mod97_check(country, bban) + bban
    return " ".join(iban[i:i + 4] for i in range(0, len(iban), 4))


def make_card(rng: random.Random) -> str:
    digits = [4] + [rng.randint(0, 9) for _ in range(14)]
    total = 0
    for i, d in enumerate(reversed(digits)):  # the check digit is appended, so these are positions 1.. from the right
        d = d * 2 if i % 2 == 0 else d
        total += d - 9 if d > 9 else d
    digits.append((10 - total % 10) % 10)
    s = "".join(map(str, digits))
    return " ".join(s[i:i + 4] for i in range(0, 16, 4))


@dataclass
class Person:
    first: str
    last: str
    lang: str
    street: str
    number: int
    city: str
    postcode: str
    phone: str
    email: str
    iban: str
    card: str
    customer_no: str
    passport: str
    cin: str
    dob: str
    linkedin: str
    github: str
    reference: str
    arabic_name: str

    @property
    def full(self) -> str:
        return f"{self.first} {self.last}"

    @property
    def surname_first(self) -> str:
        return f"{self.last.upper()} {self.first}"

    @property
    def address_line(self) -> str:
        if self.lang == "fr":
            return f"{self.number} {self.street}"
        if self.lang == "ar":
            return f"{self.number} {self.street}"
        return f"{self.number} {self.street}"

    @property
    def postcode_city(self) -> str:
        if self.lang == "en":  # UK order: city, then postcode
            return f"{self.city} {self.postcode}".strip()
        return f"{self.postcode} {self.city}".strip()


def make_person(rng: random.Random, lang: str) -> Person:
    first, last = rng.choice(FIRST[lang]), rng.choice(LAST[lang])
    street, (city, postcode) = rng.choice(STREETS[lang]), rng.choice(CITIES[lang])
    handle = ascii_lower(first) + rng.choice([".", "", "_"]) + ascii_lower(last)
    if lang == "fr":
        phone = rng.choice([f"+33 {rng.randint(1, 9)} {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)}",
                            f"0{rng.randint(6, 7)} {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)}"])
        country = "FR"
    elif lang == "ar":
        phone = f"+216 {rng.randint(20, 99)} {rng.randint(100, 999)} {rng.randint(100, 999)}"
        country = "TN"
    else:
        phone = f"+44 {rng.randint(20, 79)} {rng.randint(1000, 9999)} {rng.randint(1000, 9999)}"
        country = "GB"
    return Person(
        first=first, last=last, lang=lang, street=street, number=rng.randint(2, 98), city=city, postcode=postcode,
        phone=phone, email=f"{handle}{rng.choice(['', str(rng.randint(1, 99))])}@{rng.choice(EMAIL_DOMAINS)}",
        iban=make_iban(rng, country), card=make_card(rng),
        customer_no=str(rng.randint(10_000_000, 99_999_999)),
        passport=rng.choice("ABCDEFGHJK") + rng.choice("ABCDEFGHJK") + str(rng.randint(1_000_000, 9_999_999)),
        cin=str(rng.randint(10_000_000, 99_999_999)),
        dob=f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1960, 2002)}",
        linkedin=f"linkedin.com/in/{handle.replace('.', '-').replace('_', '-')}-{rng.randint(100000, 999999)}",
        github=f"github.com/{ascii_lower(first)}{ascii_lower(last)[:5]}{rng.randint(0, 99)}",
        reference=f"SIN-{rng.randint(2024, 2026)}-{rng.randint(1_000_000, 9_999_999):07d}",
        arabic_name=rng.choice(ARABIC_NAMES),
    )
