# Local PDF Redactor

Detects personal data in PDFs, photos of documents, Word / Excel / PowerPoint files, e-mails and text files, and removes it **for real** (text and image pixels
deleted, metadata stripped, not just covered by a black box). Runs fully offline.
Languages: English, French, Arabic (beta).

## Run it
**Windows:** double-click `run.bat`. The first run creates the environment and downloads the models
(a few minutes, internet needed once); after that it just starts the app in your browser. Tested from a
clean folder with nothing installed.

**Manual setup / other platforms:**
```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt            # only to run the tests (python-docx, openpyxl, python-pptx)
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -m spacy download xx_ent_wiki_sm        # used for Arabic
python scripts/get_tessdata.py                 # OCR language files into tessdata/

streamlit run app.py                                        # review UI
python -m redactor.cli in.pdf out.pdf --lang fr             # one file
python -m redactor.cli scans/ redacted/ --lang fr           # a whole folder, writes redacted/audit.json
pytest                                                      # ~350 tests, about 1-3 minutes
```
CLI options: `--case` (a folder as one case, see below), `--min-grade A|B|C` (exit 1 if a file's quality grade is worse; an F always fails), `--min-score N` (only redact detections scoring at least N; default 0 = everything, `--skip-low` is
shorthand for 60), `--profile NAME|FILE.json`, `--ocr tesseract|paddle|auto`, `--paddle-models FOLDER`, `--no-ner`, `--no-ocr`,
`--audit FILE.json`.
The exit code is 0 only if every file was redacted **and** verified. The audit log records file hashes, types, counts,
risk before/after and score ranges of what was removed, never the personal text itself.

## The review UI
1. Upload a PDF or a photo/scan (PNG, JPG, TIFF...). Pick the language.
2. Every detection has a **confidence score** (0-100). Use the slider to choose what is pre-selected (default 60),
   untick false detections, add extra terms by hand. The **privacy risk** panel at the top shows the risk of the
   original document and, after you apply, the measured risk of what is still detectable in the saved file.
3. **Drag on the page preview to draw a box** for things no detector can find (signatures, photos, stamps).
   Boxes appear in a table you can edit. Red = detections, blue = manual boxes.
4. Apply. The tool then re-opens the saved file and **verifies it** (see below). Download only if it passed.

## What you can feed it
- **PDFs** and **photos / scans** (PNG, JPG, TIFF, WebP, BMP). Sideways, upside-down, rotated and low-resolution scans are
  handled, and so are phone photos with an EXIF orientation flag.
- **Word, Excel, PowerPoint, e-mail (.eml), text, CSV, Markdown and HTML.** They are converted to a PDF and then redacted
  like any other, so verification and grading apply too. A PDF holds only what a reader sees, which leaves far fewer
  places for personal data to hide. What the original carried and the PDF does not is **dropped and reported**: comments,
  tracked deletions, hidden text, speaker notes, hidden slides / sheets / rows, document properties (author, last
  modified by, company), macros, embedded objects, and an e-mail's routing headers and attachments. Embedded pictures
  come along as pictures, so text inside them is read by OCR. Attachments of an e-mail are named (the file name is
  personal data too) but not included: process them separately, or as one case.
- **Pictures of text inside a text page** (a pasted screenshot, a scanned attachment): the page's own text keeps its
  exact boxes and only the pictures are OCR'd.

## What it removes besides text
- **Machine-readable zones** of passports and ID cards (the `P<FRA...<<<` lines, ICAO 9303 TD1/TD2/TD3), verified by
  their check digits, tolerant of OCR damage. The holder's name and document number are read out of the zone and
  redacted wherever else they are printed on the page.
- **QR codes and barcodes**: decoded, listed as BARCODE findings (scored higher when the payload contains an IBAN, an
  email...), blacked out where they sit, and verified: the code must no longer decode in the saved file.
- **National ID numbers** with a checksum or a distinctive format: France NIR (validated key), UK National Insurance,
  US Social Security, Tunisian tax id.
- **Photo metadata**: EXIF (camera, date, **GPS position**), XMP, IPTC and comments inside photos are cut out losslessly,
  also from photos already inside a PDF. Without this, a photo that no redaction box touches keeps all of it.
- **Hidden content** in PDFs (see below).

## Safety: hidden content and verification
Black boxes are not enough, so every redaction also:
- **strips hidden content** that can carry personal data: comments/annotations, form fields, hyperlinks, embedded
  attachments, bookmarks, document metadata, XMP metadata and invisible text (e.g. a hidden OCR layer). The UI and
  CLI list what was found and removed.
- **verifies the saved file independently of the code that made it**:
  1. every redaction box renders (almost) fully black;
  2. no redacted term can be read back from the text layer, metadata, annotations or bookmarks;
  3. scanned pages are **OCR'd again**: no redacted term, and no run of its digits, may still be readable.

  On failure the UI shows a red error with the reasons and the CLI exits with code 1.

## Try it on your own documents
```powershell
mkdir samples\real            # put anonymised scans/PDFs here (git-ignored)
python scripts/try_documents.py samples/real some.pdf --lang fr
```
Writes a preview PNG per page and the redacted PDF to `out/`, and prints the verification result. Anything
personal that is **not** red in the preview was missed: that is the list to improve detection against.

## What it detects
- **Patterns:** emails, phones (dates, amounts and year ranges are excluded), IBANs (any real country code and
  shape, even with a bad checksum, because OCR misreads must not leave half an account number visible), credit
  cards (Luhn) and long digit strings that fail Luhn, contract/claim references (`SIN-2026-0093178`), street
  addresses with postcode and city, profile links (LinkedIn, GitHub, X, Instagram...), titled names
  (Mr / M. / السيد).
- **Context rules:** the value after a label: `Nom et prénom`, `Adresse`, `Numéro de client`, `Date de naissance`,
  `Passport`, `الاسم`... also when a digital PDF puts the label and value on consecutive lines.
- **Names:**
  - a name whose letters spell an email or profile handle (`Dhia Eddine Toumi` ~ `dhiaeddinetoumi7@gmail.com`);
  - the contact block of a CV or letter (a name or `City, Country` next to an email/phone, also in a
    `|`-separated header line);
  - short name-shaped lines (`Nathalie ROBIN`);
  - spaCy NER for names in running text.
  A name found once is searched everywhere in every spelling (`MARCHAND Julien`, `Julien MARCHAND`, `MARCHAND`).
- **False-positive filters:** organisations, study/job/tech words (`Institut`, `Marketing`, `Docker`...), OCR
  garbage, and capitalised ordinary words (a word that also appears in lower case in the document, like
  "Contrat"/"contrat", is not a name or a place).

### Scans and photos
- **Low-resolution scans** (small PNGs, screenshots, phone photos below ~170 dpi) are enlarged 2-4x before OCR.
  On one real 579 px-wide letter this took OCR from 2/9 to 9/9 key items.
- **Sideways or upside-down scans** (no rotate flag): if the first read looks like scraps, the other orientations
  are tried and one is kept only if clearly better. Boxes are mapped back to the page correctly.
- **Rotated PDF pages** and **EXIF-rotated phone photos** are handled; manual boxes follow what you see on screen.
- **OCR is cached** per unchanged file, so ticking boxes in the UI does not re-run it.
- Boxes always cover whole OCR words (OCR character boxes are too imprecise: `Acacia|s` once left a letter).

## Confidence scores
Each detection carries a score from 0 to 100 answering "how sure are we that this is personal data to redact?".
All numbers live in `redactor/scoring.py`.

| Score | What vouches for it | Examples |
|---|---|---|
| 95-99 | checked or self-identifying | valid IBAN (checksum), Luhn card, email, profile link, a name that spells an email handle |
| 85-94 | a strong rule | the value after a label (`Nom et prénom : ...`), a title (`M. Dupont`), a street address, `City, Country` in a contact block |
| 60-84 | a plausible rule | `Nathalie ROBIN` alone on a line, a postcode line, a long digit string that fails Luhn |
| below 60 | only the statistical model, or a bare place | NER-only names in running text (60), headings and skills tagged as names (35), bare city names (30) |

Adjustments: the same text found by independent sources **adds 5 per extra source** (a label rule and the model
agreeing on a name: 90 -> 95); uncertain text that **repeats** in the document gains 5; text read through **OCR**
loses 5; a **name piece** (`MARCHAND` from `Julien MARCHAND`) inherits its parent's score minus 5, or minus 15 when
only the model vouched for the parent, so a piece never outscores its parent.

Why scores instead of a yes/no flag: spaCy models tag headings and skills ("Spring Boot", "Informatique Financière")
as people and places, and part-of-speech tags cannot tell them apart (on an isolated phrase the French model tags
everything proper noun and real names as nouns). A threshold lets you choose between "never miss" and "never
bother me".

## Privacy risk score
A document-level score from 0 to 100 (`redactor/risk.py`): how much harm sharing the document as-is could do. It is a
transparent heuristic, not a probability.
- each finding adds `weight x confidence`: an IBAN, card or ID number weighs 20-25, a name or street address 10-12,
  an email or phone 8-10, a bare city 2;
- several findings of one kind count for geometrically less (each next one half as much);
- pieces of a name count once, not as extra people;
- **bonuses** for identifiability: a name together with contact or identity details (+15, +10 more for two or more
  kinds), and financial data tied to a named person (+10);
- levels: 0 none, 1-29 low, 30-54 moderate, 55-84 high, 85+ critical.

It is shown **before** redaction, as a live **estimate** for your current ticks, and then **measured** on the saved
file by re-scanning it with the same detector. On the real documents tested (redacting everything scoring 60+):

| Document | Before | After |
|---|---|---|
| letter with an IBAN, name, address, references | 100 critical | 1 low |
| insurance policy page (client number, name, address) | 82 high | 1 low |
| two-page CV (name, email, phone, city, profile links) | 76 high | 5 low |

The "after" number is not zero by design: it counts what you chose to leave (bare city names the model is unsure of)
and anything the detector can still read in the output.

## Several documents as one case
Upload several files (or point the CLI at a folder with `--case`) and they are treated as **one case**:

1. every file is scanned first;
2. everything found is merged into **one list of entities** (the same text in several files is one entity, with the
   best score any file gave it and the files that reported it);
3. you decide **once** (one table, one threshold slider);
4. every file is redacted with those decisions, then **verified and graded on its own**: the set is graded by its
   worst file, so one bad file cannot hide behind good ones;
5. you download one zip with the redacted files and an audit log.

**Why it improves recall.** The language model is less sure of a bare surname in a table than of the same name in a
sentence. Once `MARCHAND` is confirmed in one document it is searched for, and removed, in all of them. Each file
alone would have missed it.

**The safety valve.** Sharing is limited to findings scoring at least 85 (`--share-threshold`, or the slider). A common
word flagged at a low score in one file (a bare city, "Monte Carlo") is never spread to the other files. Sharing can
be switched off entirely for unrelated documents. Unselected entities are never applied anywhere.

**Set-level risk.** Each distinct identity counts once, but findings from different files are pooled, so a name in one
file and an IBAN in another still trigger the "financial data tied to a named person" bonus, and a person who appears
in several documents adds a little. A bundle can therefore score higher than any file in it. The measured residual risk
after redaction uses the same pooling.

```powershell
python -m redactor.cli scans/ redacted/ --case --lang fr --min-score 60
python -m redactor.cli scans/ redacted/ --case --share-threshold 90 --min-grade B
```
`--case` writes `redacted/audit.json` with per-file grades and the set's risk before/after (kinds and counts, never the
text). The exit code is 0 only if every file verified and the worst grade meets `--min-grade`. Manual boxes and the
page-by-page drawing tool stay in the single-file view; the case view shows a preview of any file's detections.

Deliberately not done: guessing that "J. Marchand" and "Julien MARCHAND" are the same person (error-prone: the
case-insensitive search of name pieces already covers most spellings), and consistent placeholders such as
`[PERSON_1]` across files (they keep every mention of one person linkable, so it would have to be an opt-in).

## Quality grade of the finished file
After you apply, the saved file gets a grade, **A / B / C / F**, with the reasons (`redactor/quality.py`). The risk score
says how sensitive the document is; this says how far you can trust the *output*. It is built from checks the pipeline
already makes, and every deduction names something you can act on (never the personal text itself).

| | Condition | Effect |
|---|---|---|
| **Gate** | verification found a problem (a box not black, a term still readable, readable again after re-OCR) | automatic **F**, score at most 40 |
| | hidden content (annotations, attachments, metadata, invisible text...) survived the strip | automatic **F** |
| **Major** | a page could not be read by OCR, so what was redacted there is unknown | grade at most **C**, up to -40 |
| | confident personal data (score 85+) is still detectable in the saved file | grade at most **C** |
| | nothing was redacted although data is still detectable | grade at most **C** |
| **Minor** | plausible findings (60-84) remain | -2 each, at most -10 |
| **Score** | residual privacy risk, measured by re-scanning the saved file | -half its points, at most -40 |
| **Notes** | pages read through OCR; hidden content that was removed | information only |

Grades: A 90+, B 75+, C 50+, below 50 F. The CLI prints it and records it in the audit log; `--min-grade B` makes a
batch fail on anything worse. On the real documents tested: the CV is A (98), both insurance documents are A (100).

## OCR engines
Tesseract is the default (about 3 s a page). **PaddleOCR is available as an optional engine** for pages Tesseract reads
badly (photos, awkward layouts): more accurate, but about 30 s a page on a CPU and it needs ~1 GB of packages.

```powershell
pip install -r requirements-paddle.txt
python -m redactor.cli in.pdf out.pdf --ocr paddle --paddle-models C:/path/to/paddleocr_models
python -m redactor.cli in.pdf out.pdf --ocr auto  --paddle-models C:/path/to/paddleocr_models   # Paddle only where Tesseract struggles
```
The models folder holds `<name>_det` and `<name>_rec` sub-folders in the PaddleOCR 3.x format (for example
`PP-OCRv6_medium_det` and `PP-OCRv6_medium_rec`). Nothing Paddle-related is imported unless it is chosen. The app has the
same choice under *Settings*. Both engines feed the same pipeline, so detection, redaction, verification and grading do
not know which one read the page. Measured on the degraded-photo benchmark: see *Benchmark* below.

## Profiles: allow list, deny list, defaults
A profile is a small JSON file (`profiles/<name>.json`; the ones you save from the app go to `profiles/local/`, git-ignored):
```json
{ "name": "hr", "language": "fr", "min_score": 60, "share_threshold": 85,
  "allow": [ {"text": "Atlantis Assurances"}, {"text": "^Paris$", "regex": true, "kind": "LOCATION"} ],
  "deny":  [ {"text": "EMP[-\\s]?\\d{4,6}", "regex": true, "kind": "ID"}, {"text": "Project Falcon"} ] }
```
- **allow** = never flag: the finding equals the entry, is part of it (`Atlantis` inside the allowed `Atlantis Assurances`)
  or matches its regex; an entry can be limited to one kind. Suppressed findings are not lost: the app lists them, and
  the audit log counts them by kind (never the text).
- **deny** = always redact: literals or regexes for what the detector cannot know (internal project names, employee-number
  formats). They get a kind (default CUSTOM) and a score (default 95). **Always-redact beats never-flag.**
- Defaults for language, `min_score` and `share_threshold`; explicit command-line flags win.
- Regexes are validated: nested quantifiers (`(a+)+`) that could hang the tool, invalid or empty-matching patterns are refused.
- `profiles/example_hr.json`, `example_legal.json` and `example_medical.json` show the shape.
- In the app, *Remember my choices in a profile* turns what you unticked into allow rules and your typed-in terms into deny rules.

## Benchmark
`python -m benchmark.run` measures the detector on a **labeled corpus of fake documents** (`benchmark/`): letters,
invoices, CVs, forms, contracts, passport pages (with a real MRZ) and invoices with a payment QR code, in English, French
and Arabic, each rendered four ways: a digital PDF, a clean scan, a degraded phone-style photo, and a **mixed** page
(a text page whose lower half is an embedded picture of text). Every document knows which text is personal data (`items`), which text is acceptable to
redact (a company's own address: `ignore`) and everything else is a **trap for precision**: dates, amounts, headings,
skills and company names. The corpus is generated from a seed, so results are reproducible; identities, IBANs and
card numbers are invented (and pass their checksums).

- **Recall** = items *covered*: at least 95% of an item's characters lie inside findings. Characters, not
  all-or-nothing, so a half-redacted IBAN does not count, while `Dhia` + `Toumi` together cover `Dhia Toumi`.
  Recall is the number that matters: a miss is a leak.
- **Precision** = findings that overlap an item or an ignorable text. A date or a skill tagged as personal is a false
  positive.
- OCR variants are matched fuzzily (a misread `jonn.smith` still covers `john.smith`: the same spot is redacted);
  right-to-left languages are compared as sets of words (OCR returns `+216 62 987 559` as `559 987 62 216`).
- Reported overall and per variant, per language and per kind (name, address, IBAN...).

```powershell
python -m benchmark.run                    # ~1 minute: precision / recall / F1
python -m benchmark.run --details          # every missed item and every false positive
python -m benchmark.run --sweep            # P/R at several score thresholds (60 gives the best F1)
python -m benchmark.run --e2e              # also redact, then search the saved file for the items
python -m benchmark.run --check            # compare with benchmark/baseline.json: exit 1 on a regression
python -m benchmark.run --update-baseline  # accept the current numbers (after a deliberate change)
python -m benchmark.run --local            # YOUR documents: benchmark/local/*.labels.json (git-ignored)
$env:REDACTOR_OCR = "paddle"               # compare OCR engines (also set REDACTOR_PADDLE_MODELS), then run any of the above
```
**Regression gate.** `--check` fails when a group loses more than its tolerance of items versus the committed
baseline (1% overall, 3% elsewhere, always at least one item of slack) or precision falls by 5 points. CI runs it
after the tests. It was verified to fail when email detection is deliberately broken.

**Teach it with your own reviews.** In the app, *Teach the benchmark with this review* saves the document and what you
decided (ticked = must be found, unticked = must not be flagged, typed-in terms = missed items, kind `MANUAL`) as a labeled
case under `benchmark/local/`, then scores the detector on it at once. In case mode it saves every file. The saved copy is
the original document, so the folder is git-ignored and never leaves your machine.

Local documents stay on your machine: write `benchmark/local/<name>.labels.json` with
`{"file": "path.pdf", "lang": "fr", "items": [{"kind": "PERSON", "text": "..."}], "ignore": ["company text"]}`.

Current numbers (threshold 60, synthetic corpus, 370 items, 634 findings, four variants):

| Group | Recall | Precision | F1 |
|---|---|---|---|
| **overall** | **97.3%** | **99.2%** | **0.982** |
| digital PDF | 96.9% | 99.4% | 0.981 |
| mixed page (text + picture of text) | 100% | 99.4% | 0.997 |
| clean scan | 99.0% | 98.8% | 0.989 |
| degraded photo | 93.2% | 99.3% | 0.962 |
| English / French / Arabic (beta) | 98.3% / 98.3% / 77.8% | 98.4% / 100% / 100% | 0.983 / 0.991 / 0.875 |

By kind: barcodes, cards, dates of birth, emails and references 100%; MRZ 93.8%; phones 97%; IBAN 90%; IDs 91%; URLs 87.5%.
End to end (redact, then search the saved file for each item): **98.9% protected**. The 4 leaks are all on degraded
photos: three ID numbers OCR misread and one name. Tesseract's photo misses are exactly where PaddleOCR helps (below).

**What the benchmark found and fixed in this round.** Mixed pages scored 57% before OCR of embedded pictures; MRZ
names and numbers were dropped as "pieces of a longer finding"; a *lone* second MRZ line was skipped; sideways-page
detection picked garbage on ID pages; OCR'd `Client` / `: Maxime Garnier` on separate lines was rejected; a redacted page's
empty `Given names :` label was read as a value; **PDF ligatures** (`ﬁ`, `ﬃ`) broke both extraction and search, so a name
such as `Griffin` was missed and left unredacted (fixed for all PDFs); expanding ligatures also hid Arabic text in
visual order, which the benchmark caught as a regression and which is now guarded by a test; and `load_local` captured
its folder at import time, so a test silently used real cases.

**PaddleOCR vs Tesseract on degraded photos** (same pipeline, only the OCR engine changes). On the 6 photo pages timed with
Paddle (letter, invoice, CV, form in English; form and contract in French) it covered **44 of 44** items; Tesseract missed 3
of the same pages' items (a LinkedIn URL and two ID numbers). Cost: 26-72 s a page against 2-7 s. Not measured on every
page (a full Paddle run did not fit in the time I had), so treat "better on photos, much slower" as the finding, not a
benchmark score. `--ocr auto` uses Paddle only for pages Tesseract reads badly.

## Accuracy
Tested on real documents, all processed locally:
- two French insurance documents (a letter and a policy page, as small PNGs): addressee and signature names,
  both addresses with postcode, client number, contract and dossier references, the whole IBAN and the phone
  number are all covered and verification passes;
- a two-page CV: name, city, phone, email and both profile links are covered; verification passes; hyperlinks and
  author metadata were stripped. The only high-confidence false positive was a tech term ("Monte Carlo").

Before these fixes the IBAN was only partly redacted, `MARCHAND Julien` and the client number were missed, the CV
owner's own name was missed, and dates, year ranges and ordinary words were flagged.

`python scripts/stress_test.py [--lang en|fr]` degrades a fake page (skew, noise, blur, JPEG) and reports what
is still found (8 secrets per page; OCR misreads such as `jonn.smith` count as misses there although the
text is still found and redacted as the OCR read it):

| Scan quality | English | French |
|---|---|---|
| clean 200 dpi | 8/8 | 8/8 |
| slight skew | 7/8 | 7/8 |
| noisy office scan | 8/8 | 8/8 |
| phone photo (120 dpi, blur, noise) | 6/8 | 7/8 |
| very poor photo (100 dpi, heavy blur) | 0/8 | 0/8 |

Cleaning the image before OCR (contrast, denoise, thresholding) was tried and gave no consistent gain; plain
upscaling of low-resolution images is what helped. The UI warns when OCR finds almost no text on a page.

## Known limits
- OCR misses text sometimes; a missed item is not redacted. **Always check the preview on scans and photos.**
  Verification proves that what was redacted is gone, not that everything sensitive was found.
- Standalone company details (a letterhead address or phone) are redacted like personal ones; untick them.
- Arabic NER is weak: titles, emails, IBANs, addresses and labelled fields are reliable, bare names are not.
  Arabic PDFs with shaped-glyph text layers are OCR'd instead of read directly.
- Handwriting and faces/signatures are not detected (use manual boxes).
- Re-OCR verification doubles OCR time on scanned pages (the first OCR of a file is cached, the check is not).
- Bilingual or code-heavy documents (CVs) produce more low-confidence rows; review them in the UI.
- OCR needs `tessdata/` (`scripts/get_tessdata.py`; Arabic uses the larger `tessdata_best` model because the
  `fast` one misreads digits). Without it, scanned pages raise a clear error; text PDFs still work.
- **Office files and e-mails become a text PDF**: the original layout, fonts and styles are not kept, and the output is a
  PDF, not a redacted `.docx`. Old binary formats (`.doc`, `.xls`, `.ppt`, `.msg`) are not supported: save them as the
  modern format first. Pictures in formats that cannot be carried over (EMF, WMF) are dropped and reported, not guessed at.
  E-mail attachments are not included (their names are): process them separately, or as one case.
- **Profiles can hide real personal data**: an allow rule that is too broad stops a finding from ever being flagged. The app
  lists everything an allow list hid and the audit log counts it by kind, so check that list when you build a profile.
- **A decoded barcode's rectangle is padded** to cover its quiet zone; a damaged code that cannot be decoded cannot be found.
  Use a manual box for it.
- **PaddleOCR** is optional, takes about 30 s a page on a CPU, and its models are the ones you supply. oneDNN is switched
  off because it crashes some Paddle 3.x CPU builds on Windows.
- MRZ lines are caught when OCR keeps their shape; a badly mangled line may only be found through the name and number
  printed elsewhere on the page.

## Project layout
`redactor/` detection (`detect.py`, `people.py`), scoring (`scoring.py`, `risk.py`, `quality.py`), OCR (`ocr.py`),
redaction (`redact.py`), several documents as one case (`caseset.py`), hidden content and verification (`verify.py`), input handling (`loader.py`), manual boxes (`boxes.py`), CLI (`cli.py`);
`benchmark/` the labeled corpus, metrics, regression gate and *save my review as a case*; `app.py` and `case_ui.py` the review UI (single file and case mode); `scripts/` helpers (`try_documents.py`, `stress_test.py`, `get_tessdata.py`);
`.github/workflows/ci.yml` runs the tests on Ubuntu and Windows (written but not yet run on GitHub).
