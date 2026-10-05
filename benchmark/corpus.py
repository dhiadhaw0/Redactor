"""A labeled corpus of fake documents, generated from a seed.

Every document knows exactly which of its text is personal data that must be redacted (`items`) and
which text is acceptable to redact but not required, such as a company's own address (`ignore`).
Everything else (dates, amounts, headings, skills, company names...) is a trap for precision: a
detection that lands there is a false positive.

Each document renders in four variants that mirror real inputs:
  digital  a PDF with a text layer
  scan     the same page as an image-only PDF (200 dpi)
  photo    a degraded 120 dpi capture: skew, blur, noise, JPEG compression
  mixed    a text page whose lower half is an embedded picture of text (pasted screenshot, attachment)
"""
import io
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf as fitz
import zxingcpp
from PIL import Image, ImageFilter

from redactor import mrz

from .identities import COMPANIES, FEMALE, SKILLS, Person, ascii_lower, ascii_upper, make_iban, make_person

SEED = 7
LINE_HEIGHT = 15
FONT_SIZE = 10


@dataclass
class Doc:
    name: str
    lang: str
    lines: list[str] = field(default_factory=list)
    items: list[tuple[str, str]] = field(default_factory=list)   # (kind, text) that MUST be redacted
    ignore: list[str] = field(default_factory=list)              # text that may be redacted, not required
    codes: list[str] = field(default_factory=list)               # payloads of QR codes drawn on the page

    def add(self, *lines: str) -> None:
        self.lines += lines

    def item(self, kind: str, text: str) -> str:
        """Register `text` as personal data of `kind` and return it for use inside a line."""
        self.items.append((kind, text))
        return text

    def code(self, payload: str) -> str:
        """Draw a QR code holding `payload` on the page; its payload must be redacted (kind BARCODE)."""
        self.codes.append(payload)
        self.items.append(("BARCODE", payload))
        return payload

    def ign(self, text: str) -> str:
        self.ignore.append(text)
        return text


@dataclass
class Company:
    name: str
    street: str
    postcode_city: str
    phone: str
    email: str
    iban: str


def make_company(rng: random.Random, lang: str) -> Company:
    p = make_person(rng, lang)
    name = rng.choice(COMPANIES)
    return Company(name, p.address_line, p.postcode_city, p.phone,
                   f"contact@{ascii_lower(name)}.example", make_iban(rng, "GB" if lang == "en" else "FR"))


# ----------------------------------------------------------------------------- templates
T = {  # the little bit of language the templates need
    "en": dict(date="18 September 2026", tel="Tel", dear="Dear Sir or Madam,", subject="Subject: claim ref.",
               pay="We are transferring the settlement of EUR 2,037.00 to the account", questions="For any question call",
               regards="Yours faithfully,", job="Claims Manager", titles=("Mr", "Ms"), invoice="Invoice", customer="Customer",
               address="Address", custno="Customer number", email="Email", phone="Phone", card="Paid by card",
               item1="Annual subscription", item2="Support package", total="Total due", dob="Date of birth",
               fullname="Full name", passport="Passport number", cin="National ID number", emergency="Emergency contact",
               contract="EMPLOYMENT AGREEMENT", between="This agreement is made on 18 September 2026 between",
               residing="residing at", employee="(the Employee) and", employer="(the Employer).",
               witness="Signed in the presence of", duties="1. Duties and place of work",
               salary="2. Salary: EUR 2,400 per month, paid on the 28th.", profile="PROFILE", exp="PROFESSIONAL EXPERIENCE",
               skills="SKILLS", edu="EDUCATION", langs="LANGUAGES: English, French, Arabic"),
    "fr": dict(date="Lyon, le 18/09/2026", tel="Tel", dear="Madame, Monsieur,", subject="Objet : sinistre ref.",
               pay="Nous vous adressons une indemnite de 2 037,00 EUR par virement sur le compte",
               questions="Pour toute question appelez le", regards="Cordialement,", job="Gestionnaire Sinistres",
               titles=("M.", "Mme"), invoice="Facture", customer="Client", address="Adresse", custno="Numero de client",
               email="Courriel", phone="Telephone", card="Paiement par carte", item1="Abonnement annuel",
               item2="Forfait assistance", total="Total a payer", dob="Date de naissance", fullname="Nom et prenom",
               passport="Numero de passeport", cin="Numero de CIN", emergency="Personne a contacter",
               contract="CONTRAT DE TRAVAIL", between="Le present contrat est conclu le 18/09/2026 entre",
               residing="demeurant au", employee="(le Salarie) et", employer="(l'Employeur).",
               witness="Signe en presence de", duties="1. Fonctions et lieu de travail",
               salary="2. Remuneration : 2 400 EUR par mois, versee le 28 de chaque mois.", profile="PROFIL",
               exp="EXPERIENCE PROFESSIONNELLE", skills="COMPETENCES", edu="FORMATION",
               langs="LANGUES : Francais, Anglais, Arabe"),
}


def _header(d: Doc, co: Company, t: dict) -> None:
    d.add(d.ign(co.name), d.ign(co.street), d.ign(co.postcode_city), f"{t['tel']} : {d.ign(co.phone)}", d.ign(co.email), "")


def _address(d: Doc, p: Person) -> list[str]:
    """Two lines: street, then postcode and city (as on a letter)."""
    return [d.item("ADDRESS", p.address_line), d.item("ADDRESS", p.postcode_city)]


def letter(rng, lang) -> Doc:
    t, d = T[lang], Doc(f"letter_{lang}", lang)
    p, agent, co = make_person(rng, lang), make_person(rng, lang), make_company(rng, lang)
    _header(d, co, t)
    title = t["titles"][1 if p.first in FEMALE else 0]
    name = p.surname_first if lang == "fr" else p.full
    d.add(f"{title} {d.item('PERSON', name)}", *_address(d, p), "")
    d.add(t["date"], "")
    d.add(f"{t['subject']} {d.item('REFERENCE', p.reference)}", "", t["dear"], "")
    d.add(t["pay"], d.item("IBAN", p.iban), "")
    d.add(f"{t['questions']} {d.ign(co.phone)}.", "", t["regards"], "")
    d.add(d.item("PERSON", f"{agent.first} {agent.last.upper()}"), t["job"], d.ign(co.name))
    return d


def invoice(rng, lang) -> Doc:
    t, d = T[lang], Doc(f"invoice_{lang}", lang)
    p, co = make_person(rng, lang), make_company(rng, lang)
    _header(d, co, t)
    d.add(f"{t['invoice']} {d.ign('INV-2026-004211')}      18/09/2026", "")
    d.add(f"{t['customer']} : {d.item('PERSON', p.full)}")
    d.add(f"{t['address']} : {d.item('ADDRESS', p.address_line)}", d.item("ADDRESS", p.postcode_city))
    d.add(f"{t['custno']} : {d.item('ID', p.customer_no)}")
    d.add(f"{t['email']} : {d.item('EMAIL', p.email)}", f"{t['phone']} : {d.item('PHONE', p.phone)}", "")
    d.add(f"{t['item1']}        12/01/2026        1 240,00 EUR", f"{t['item2']}        12/01/2026          380,00 EUR",
          f"{t['total']}                                1 620,00 EUR", "")
    d.add(f"{t['card']} : {d.item('CREDIT_CARD', p.card)}", f"IBAN : {d.ign(co.iban)}")
    return d


def cv(rng, lang) -> Doc:
    t, d = T[lang], Doc(f"cv_{lang}", lang)
    p = make_person(rng, lang)
    country = "United Kingdom" if lang == "en" else "France"
    d.add(d.item("PERSON", p.full),
          f"{d.item('ADDRESS', f'{p.city.title()}, {country}')} | {d.item('PHONE', p.phone)} | {d.item('EMAIL', p.email)}",
          f"{d.item('URL', p.linkedin)} | {d.item('URL', p.github)}", "")
    d.add(t["profile"], "Third year engineering student looking for an end of studies internship in software and finance."
          if lang == "en" else "Etudiant en troisieme annee du cycle ingenieur, a la recherche d'un stage de fin d'etudes.", "")
    d.add(t["exp"], "Full Stack Developer Intern - Orion Logistics        Jan 2025 - 2025",
          "Built REST services with Spring Boot and deployed them with Docker.", "Data Analyst - Vega Telecom        2023 - 2024", "")
    d.add(t["skills"], ", ".join(rng.sample(SKILLS, 6)), "")
    d.add(t["edu"], "Higher School of Engineering and Technology (ESPRIT)        2021 - 2026", "Preparatory Institute        2019 - 2021", "")
    d.add(t["langs"])
    return d


def form(rng, lang) -> Doc:
    t, d = T[lang], Doc(f"form_{lang}", lang)
    p, other = make_person(rng, lang), make_person(rng, lang)
    d.add(t["customer"].upper() + " FORM", "")
    d.add(f"{t['fullname']} : {d.item('PERSON', p.full)}")
    d.add(t["dob"], d.item("DATE_OF_BIRTH", p.dob))  # label and value on separate lines
    d.add(f"{t['address']} : {d.item('ADDRESS', p.address_line)}", d.item("ADDRESS", p.postcode_city))
    d.add(f"{t['passport']} : {d.item('ID', p.passport)}", f"{t['cin']} : {d.item('ID', p.cin)}")
    d.add(f"{t['phone']} : {d.item('PHONE', p.phone)}", f"{t['email']} : {d.item('EMAIL', p.email)}", "")
    d.add(f"{t['emergency']} : {d.item('PERSON', other.full)}, {d.item('PHONE', other.phone)}")
    return d


def contract(rng, lang) -> Doc:
    t, d = T[lang], Doc(f"contract_{lang}", lang)
    p, witness, co = make_person(rng, lang), make_person(rng, lang), make_company(rng, lang)
    d.add(t["contract"], "")
    d.add(t["between"], f"{d.item('PERSON', p.full)}, {t['residing']} {d.item('ADDRESS', p.address_line)},",
          f"{d.item('ADDRESS', p.postcode_city)} {t['employee']}", f"{d.ign(co.name)} {t['employer']}", "")
    d.add(t["duties"], "The Employee shall perform the duties assigned by the Employer." if lang == "en"
          else "Le Salarie exerce les fonctions confiees par l'Employeur.", t["salary"], "")
    d.add(f"{t['titles'][0]} {d.item('PERSON', p.last)} {'agrees to the above terms.' if lang == 'en' else 'accepte les conditions ci-dessus.'}", "")
    d.add(f"{t['witness']} {d.item('PERSON', witness.full)}.")
    return d


def passport(rng, lang) -> Doc:
    """The printed data page of a passport, with its machine-readable zone."""
    t, d = T[lang], Doc(f"passport_{lang}", lang)
    p = make_person(rng, lang)
    fr = lang == "fr"
    country = "FRA" if fr else "GBR"
    yy_mm_dd = p.dob[8:10] + p.dob[3:5] + p.dob[0:2]
    sex = "F" if p.first in FEMALE else "M"
    zone = mrz.build_td3(ascii_upper(p.last), ascii_upper(p.first), country, p.passport, country, yy_mm_dd, sex, "300101")
    d.add("PASSEPORT" if fr else "PASSPORT", "")
    d.add(f"{'Nom' if fr else 'Surname'} : {d.item('PERSON', ascii_upper(p.last))}",
          f"{'Prenoms' if fr else 'Given names'} : {d.item('PERSON', p.first)}",
          f"{'Numero du document' if fr else 'Document No'} : {d.item('ID', p.passport)}",
          f"{t['dob']} : {d.item('DATE_OF_BIRTH', p.dob)}", "")
    d.add(d.item("MRZ", zone[0]), d.item("MRZ", zone[1]))
    return d


def invoice_qr(rng, lang) -> Doc:
    """An invoice whose payment QR code carries the payer's name and IBAN: nothing about them is printed."""
    t, d = T[lang], Doc(f"invoice_qr_{lang}", lang)
    p, co = make_person(rng, lang), make_company(rng, lang)
    _header(d, co, t)
    d.add(f"{t['invoice']} {d.ign('INV-2026-004211')}      18/09/2026", "")
    d.add(f"{t['customer']} : {d.item('PERSON', p.full)}", f"{t['address']} : {d.item('ADDRESS', p.address_line)}",
          d.item("ADDRESS", p.postcode_city), "")
    d.add(f"{t['item1']}        12/01/2026        1 240,00 EUR", f"{t['total']}                                1 240,00 EUR", "")
    d.code(f"PAYER:{p.full}|IBAN:{p.iban}|REF:{p.reference}")
    return d


# ----------------------------------------------------------------------------- Arabic (beta)
def letter_ar(rng, lang="ar") -> Doc:
    d, p, co = Doc("letter_ar", "ar"), make_person(rng, "ar"), make_company(rng, "ar")
    d.add(d.ign(co.name), f"الهاتف: {d.ign(co.phone)}", "")
    d.add(f"السيد {d.item('PERSON', p.arabic_name)}", f"العنوان: {d.item('ADDRESS', p.address_line)}", "")
    d.add(f"البريد الإلكتروني: {d.item('EMAIL', p.email)}", f"IBAN: {d.item('IBAN', p.iban)}", f"الهاتف: {d.item('PHONE', p.phone)}")
    return d


def form_ar(rng, lang="ar") -> Doc:
    d, p = Doc("form_ar", "ar"), make_person(rng, "ar")
    d.add("استمارة", "", f"الاسم: {d.item('PERSON', p.arabic_name)}", f"العنوان: {d.item('ADDRESS', p.address_line)}",
          f"رقم الهوية: {d.item('ID', p.cin)}", f"البريد: {d.item('EMAIL', p.email)}")
    return d


LATIN_TEMPLATES = [letter, invoice, cv, form, contract, passport, invoice_qr]
ARABIC_TEMPLATES = [letter_ar, form_ar]


def build_corpus(seed: int = SEED) -> list[Doc]:
    docs = []
    for lang in ("en", "fr"):
        for tpl in LATIN_TEMPLATES:
            docs.append(tpl(random.Random(f"{seed}-{lang}-{tpl.__name__}"), lang))
    for tpl in ARABIC_TEMPLATES:
        docs.append(tpl(random.Random(f"{seed}-ar-{tpl.__name__}")))
    return docs


# ----------------------------------------------------------------------------- rendering
def _is_mrz_line(line: str) -> bool:
    return len(line) >= 28 and set(line) <= mrz.ALPHABET and "<" in line


def _qr_png(payload: str) -> bytes:
    arr = np.asarray(zxingcpp.write_barcode(zxingcpp.BarcodeFormat.QRCode, payload, width=300, height=300))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def _draw(page, doc: Doc) -> None:
    y = 60
    for line in doc.lines:
        if doc.lang == "ar":
            rect = fitz.Rect(60, y - 11, page.rect.width - 60, y + 11)
            page.insert_htmlbox(rect, f'<div dir="rtl" style="font-size:10pt;text-align:right">{line}</div>')
        elif _is_mrz_line(line):
            page.insert_text((60, y), line, fontsize=FONT_SIZE, fontname="cour")  # OCR-B stand-in: monospaced
        elif line:
            page.insert_text((60, y), line, fontsize=FONT_SIZE)
        y += LINE_HEIGHT
    for i, payload in enumerate(doc.codes):
        top = 560 + 90 * i
        page.insert_image(fitz.Rect(400, top, 400 + 130, top + 130), stream=_qr_png(payload))


def render_digital(doc: Doc, path: str) -> None:
    pdf = fitz.open()
    _draw(pdf.new_page(), doc)
    pdf.save(path)


def _page_png(doc: Doc, dpi: int) -> bytes:
    pdf = fitz.open()
    page = pdf.new_page()
    _draw(page, doc)
    return page.get_pixmap(dpi=dpi).tobytes("png")


def _image_pdf(img_bytes: bytes, path: str, fmt: str) -> None:
    out = fitz.open()
    page = out.new_page()
    page.insert_image(page.rect, stream=img_bytes)
    out.save(path)


def render_scan(doc: Doc, path: str) -> None:
    _image_pdf(_page_png(doc, 200), path, "png")


def degrade(png: bytes, skew: float = 1.5, noise: float = 14, blur: float = 0.7, quality: int = 55, seed: int = 0) -> bytes:
    img = Image.open(io.BytesIO(png)).convert("L")
    img = img.rotate(skew, expand=True, fillcolor=255, resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(img, dtype=np.float32) + np.random.default_rng(seed).normal(0, noise, (img.height, img.width))
    buf = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def render_photo(doc: Doc, path: str) -> None:
    _image_pdf(degrade(_page_png(doc, 120)), path, "jpeg")


def render_mixed(doc: Doc, path: str) -> None:
    """A text page whose lower half is an embedded picture of text, like a pasted screenshot or a scanned
    attachment inside a digital document: the top lines have a text layer, the rest exist only as pixels."""
    split = len(doc.lines) // 2
    head, tail = doc.lines[:split], doc.lines[split:]
    pdf = fitz.open()
    page = pdf.new_page()
    _draw(page, Doc(doc.name, doc.lang, head, codes=doc.codes))
    shot = fitz.open()
    sp = shot.new_page()
    _draw(sp, Doc(doc.name, doc.lang, tail))
    height = LINE_HEIGHT * len(tail) + 20
    clip = fitz.Rect(40, 40, sp.rect.width - 40, 40 + height)
    png = sp.get_pixmap(dpi=200, clip=clip).tobytes("png")
    top = 60 + LINE_HEIGHT * len(head) + 10
    page.insert_image(fitz.Rect(40, top, 40 + clip.width, top + clip.height), stream=png)
    pdf.save(path)


RENDERERS = {"digital": render_digital, "scan": render_scan, "photo": render_photo, "mixed": render_mixed}


def variants_for(doc: Doc) -> list[str]:
    """Arabic is only measured digitally and as a clean scan: the fast OCR model cannot read photos."""
    return ["digital", "scan"] if doc.lang == "ar" else ["digital", "scan", "photo", "mixed"]


def render(doc: Doc, variant: str, directory: str) -> str:
    path = str(Path(directory) / f"{doc.name}_{variant}.pdf")
    RENDERERS[variant](doc, path)
    return path
