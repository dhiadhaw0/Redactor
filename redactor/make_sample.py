"""Generate fake PDFs containing personal data (all values are made up)."""
import sys

import pymupdf as fitz

SAMPLES = {
    "en": [
        "EMPLOYMENT CONTRACT (SAMPLE)",
        "",
        "Employee: John Smith",
        "Address: 12 Baker Street, London",
        "Email: john.smith@example.com",
        "Phone: +44 20 7946 0958",
        "IBAN: GB82 WEST 1234 5698 7654 32",
        "Card: 4111 1111 1111 1111",
        "",
        "Mr. Smith agrees to work for Acme Corp in Paris.",
        "Contact HR: Maria Garcia, maria.garcia@acme.org",
    ],
    "fr": [
        "CONTRAT DE TRAVAIL (EXEMPLE)",
        "",
        "Salarié : Jean Dupont",
        "Adresse : 12 rue de la Paix, Paris",
        "Courriel : jean.dupont@exemple.fr",
        "Téléphone : +33 1 42 68 53 00",
        "IBAN : FR14 2004 1010 0505 0001 3M02 606",
        "Carte : 4111 1111 1111 1111",
        "",
        "M. Dupont travaille pour la société Acme à Lyon.",
        "Contact RH : Marie Curie, marie.curie@acme.fr",
    ],
    "ar": [
        "عقد عمل (نموذج)",
        "",
        "الموظف: محمد بن علي",
        "العنوان: 12 شارع الحبيب بورقيبة، تونس",
        "البريد: mohamed.benali@example.com",
        "الهاتف: ٢١٦ ٧١ ٢٣٤ ٥٦٧+",
        "IBAN: GB82 WEST 1234 5698 7654 32",
        "السيد بن علي يعمل لدى شركة أكمي في صفاقس.",
        "جهة الاتصال: فاطمة الزهراء، fatima@acme.org",
    ],
}


def _draw(page, lang: str) -> None:
    y = 72
    for line in SAMPLES[lang]:
        if lang == "ar":
            # insert_htmlbox shapes Arabic and lays it out right-to-left
            rect = fitz.Rect(72, y - 14, page.rect.width - 72, y + 14)
            page.insert_htmlbox(
                rect, f'<div dir="rtl" style="font-size:12pt;text-align:right">{line}</div>'
            )
        else:
            page.insert_text((72, y), line, fontsize=12)
        y += 22


def make_sample(path: str, lang: str = "en") -> None:
    doc = fitz.open()
    _draw(doc.new_page(), lang)
    doc.save(path)


def make_scanned_sample(path: str, lang: str = "en", dpi: int = 200) -> None:
    """Image-only PDF (no text layer), like a scanner would produce."""
    tmp = fitz.open()
    page = tmp.new_page()
    _draw(page, lang)
    pix = page.get_pixmap(dpi=dpi)
    scan = fitz.open()
    sp = scan.new_page(width=page.rect.width, height=page.rect.height)
    sp.insert_image(sp.rect, pixmap=pix)
    scan.save(path)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "samples"
    for lang in SAMPLES:
        suffix = "" if lang == "en" else f"_{lang}"
        make_sample(f"{out_dir}/sample{suffix}.pdf", lang)
        make_scanned_sample(f"{out_dir}/sample{suffix}_scanned.pdf", lang)
        print(f"Wrote {lang} samples")
