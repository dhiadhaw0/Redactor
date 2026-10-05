"""Review UI: upload a PDF or photo, approve/reject detections, draw extra boxes,
see everything on the page, download the redacted PDF. Upload several files to treat them as one case.

Run:  streamlit run app.py   (or double-click run.bat) - everything stays on your machine
"""
import io
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from benchmark.capture import evaluate_case, save_case
from case_ui import render_case
from redactor.boxes import box_to_row, drag_to_box, rows_to_boxes
from redactor import ocr as ocr_module
from redactor.loader import SUPPORTED_SUFFIXES, convert
from redactor.profiles import Profile, ProfileError, list_profiles, load_profile
from redactor.redact import apply_redactions, preview_pages, scan_pdf
from redactor.risk import assess, remaining_after
from redactor.scoring import DEFAULT_THRESHOLD

LANG_NAMES = {"en": "English", "fr": "Français", "ar": "العربية (beta)"}
BOX_COLUMNS = ["page", "left %", "top %", "right %", "bottom %"]
FILE_TYPES = sorted(s.lstrip(".") for s in SUPPORTED_SUFFIXES)

st.set_page_config(page_title="Local PDF Redactor", page_icon="🔒", layout="wide")
st.title("🔒 Local PDF Redactor")
st.caption("Files are processed on this machine only. Nothing is uploaded anywhere.")


@st.cache_data(show_spinner="Rendering preview…")
def cached_preview(src, terms, ocr, lang, boxes):
    return preview_pages(src, list(terms), ocr=ocr, lang=lang, boxes=list(boxes))


top = st.columns([3, 1, 1, 1])
uploads = top[0].file_uploader("PDF, photo/scan, Word, Excel, PowerPoint, e-mail or text (several files = one case)", type=FILE_TYPES,
                               accept_multiple_files=True)
upload = uploads[0] if len(uploads) == 1 else None
lang = top[1].selectbox("Language", list(LANG_NAMES), format_func=LANG_NAMES.get)
use_ner = top[2].toggle("Names & places", value=True, help="spaCy NER (slower)")
use_ocr = top[3].toggle("OCR scans", value=True, help="Read scanned pages and pictures of text (slower)")

# a profile saved from a previous run becomes the selected one (a widget's state can only be set before it is drawn)
if "_pending_profile" in st.session_state:
    st.session_state["profile_select"] = st.session_state.pop("_pending_profile")

with st.expander("Settings: OCR engine and profile"):
    c1, c2, c3 = st.columns([1, 2, 2])
    engine = c1.selectbox("OCR engine", ocr_module.BACKENDS, index=ocr_module.BACKENDS.index(ocr_module.BACKEND), key="ocr_engine",
                          help="tesseract: fast (about 3 s a page). paddle: more accurate on photos but about 30 s a page and an "
                               "optional install. auto: Tesseract first, PaddleOCR only for a page Tesseract reads badly.")
    models = c2.text_input("PaddleOCR models folder", value=os.environ.get("REDACTOR_PADDLE_MODELS", ""), key="paddle_models",
                           help="A folder holding <name>_det and <name>_rec model folders. Only used with paddle or auto.",
                           disabled=engine == "tesseract")
    try:
        ocr_module.set_backend(engine, models or None)
    except RuntimeError as e:
        c2.error(str(e))
        ocr_module.set_backend("tesseract")
    names = ["(none)"] + sorted(list_profiles())
    chosen = c3.selectbox("Profile", names, key="profile_select",
                          help="Your allow list (never flag), deny list (always redact) and defaults. Save one from the review "
                               "table below, or write profiles/<name>.json (see README).")
    try:
        profile = load_profile(None if chosen == "(none)" else chosen)
    except ProfileError as e:
        c3.error(str(e))
        profile = None
    if profile:
        c3.caption(f"{profile.description or profile.name}: {len(profile.allow)} allow, {len(profile.deny)} deny rule(s)")
engine_in_use = ocr_module.BACKEND
profile_key = (profile.name, str(profile.as_dict())) if profile else None

if len(uploads) > 1:
    render_case(uploads, lang, use_ner, use_ocr, profile)
elif upload:
    key = (upload.name, upload.size, use_ner, use_ocr, lang, engine_in_use, profile_key)
    if st.session_state.get("key") != key:
        tmp = tempfile.mkdtemp()
        raw = os.path.join(tmp, "upload" + Path(upload.name).suffix.lower())
        with open(raw, "wb") as fh:
            fh.write(upload.getvalue())
        src = os.path.join(tmp, "input.pdf")
        conversion_notes = convert(raw, src)
        scan_warnings = []
        hidden_by_profile = []
        with st.spinner("Scanning…" + (" (PaddleOCR is slow: about 30 s a page)" if engine_in_use != "tesseract" else "")):
            findings = scan_pdf(src, use_ner=use_ner, ocr=use_ocr, lang=lang, warnings=scan_warnings, profile=profile,
                                suppressed=hidden_by_profile)
        st.session_state.update(
            key=key, src=src, tmp=tmp, last_drag=None, boxes=[], warnings=scan_warnings, conversion=conversion_notes,
            hidden=hidden_by_profile,
            findings=sorted(findings, key=lambda f: (-f.score, f.kind, f.text)),
        )
        st.session_state.pop("result", None)

    if st.session_state.get("conversion"):
        st.info("Converted to PDF. Not carried over from the original (and therefore not in the output): "
                + "; ".join(st.session_state["conversion"]) + ".", icon="📄")
    for w in st.session_state["warnings"]:
        st.warning(w, icon="⚠️")
    if st.session_state.get("hidden"):
        with st.expander(f"{len(st.session_state['hidden'])} finding(s) hidden by the allow list of profile '{profile.name}'"):
            st.caption("These were detected but your profile says never to flag them. Remove the rule from the profile to see them again.")
            st.dataframe([{"type": f.kind, "text": f.text, "score": f.score} for f in st.session_state["hidden"]],
                         hide_index=True, width="stretch")

    left, right = st.columns([2, 3])

    with left:
        st.subheader("Privacy risk")
        risk_box = st.container()  # filled below, once we know what is selected
        st.subheader("1. Review detections")
        threshold = st.slider(
            "Pre-select detections with a score of at least", 0, 100,
            profile.min_score if profile and profile.min_score is not None else DEFAULT_THRESHOLD,
            help="Score = how sure we are that it is personal data. 95+ checked or self-identifying (valid IBAN, "
                 "email, profile link) · 85+ strong rule (label, title, street address) · 60+ plausible · "
                 "below 60 only the language model vouched for it, or a bare place name. "
                 "Moving the slider resets your ticks.")
        # the editor key includes the threshold so moving the slider re-initialises the ticks
        rows = st.data_editor(
            [{"redact": f.score >= threshold, "score": f.score, "type": f.kind, "text": f.text}
             for f in st.session_state["findings"]],
            column_config={
                "redact": st.column_config.CheckboxColumn("Redact?"),
                "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                "type": st.column_config.TextColumn("Type", disabled=True),
                "text": st.column_config.TextColumn("Text", disabled=True),
            },
            hide_index=True, width="stretch", key=f"editor_{threshold}_{st.session_state['key']}",
        )
        extra = st.text_area("Extra terms to redact (one per line)")
        terms = [r["text"] for r in rows if r["redact"]]
        terms += [t.strip() for t in extra.splitlines() if t.strip()]

        # ---- risk panel (above): before vs after ----
        before = assess(st.session_state["findings"])
        measured = st.session_state.get("result", (None, None))[1]
        if measured is not None:
            after, after_label = measured.risk_after, "After redaction (measured on the saved file)"
        else:
            after = assess(remaining_after(st.session_state["findings"], terms))
            after_label = "After redaction (estimate for your current ticks)"
        with risk_box:
            c1, c2 = st.columns(2)
            for col, label, risk in ((c1, "Original document", before), (c2, after_label, after)):
                col.metric(label, f"{risk.score}/100", risk.level.upper(), delta_color="off", delta_arrow="off")
                col.progress(risk.score / 100)
            with st.expander("Why this score?"):
                st.caption("A transparent heuristic, not a probability: each finding adds its weight x confidence, repeats "
                           "count half as much, and a name together with contact details or financial data adds a bonus.")
                st.write("**Original document**")
                st.table([{"type": k, "findings": str(n), "points": round(p, 1)} for k, n, p in before.by_kind]
                         + [{"type": "bonus: " + r, "findings": "", "points": b} for r, b in before.bonuses])

        with st.expander("Remember my choices in a profile"):
            st.caption("What you decided here can become permanent. Findings you pick below are never flagged again "
                       "(allow list); the terms you typed under 'Extra terms' are always redacted from now on (deny list). "
                       "Saved to profiles/local/ on this computer.")
            unticked = sorted({r["text"] for r in rows if not r["redact"]})
            never = st.multiselect("Never flag these again", unticked, key="never_flag")
            # no key on purpose: its identity changes with value/disabled, so it starts ticked as soon as extra terms exist
            always = st.checkbox("Always redact my extra terms", value=bool(extra.strip()), disabled=not extra.strip())
            profile_name = st.text_input("Profile name", value=profile.name if profile else "mine", key="profile_save_name")
            if st.button("Save to profile", key="profile_save"):
                base = profile or Profile(name=profile_name)
                updated = base.with_allowed(never).with_denied([t for t in extra.splitlines() if t.strip()] if always else [])
                updated.name = profile_name
                try:
                    path = updated.save()
                    st.session_state["_pending_profile"] = path.stem
                    st.session_state["profile_saved"] = f"Saved {path.name}: {len(updated.allow)} allow, {len(updated.deny)} deny rule(s)."
                    st.rerun()
                except ProfileError as e:
                    st.error(str(e))
            if st.session_state.get("profile_saved"):
                st.success(st.session_state.pop("profile_saved"))

        st.subheader("2. Manual boxes")
        st.caption("Drag on the page preview to draw a box (signatures, photos, stamps). "
                   "Edit the numbers or delete rows below.")
        box_rows = st.data_editor(
            pd.DataFrame(st.session_state["boxes"], columns=BOX_COLUMNS).astype("float64"),  # numeric even when empty
            num_rows="dynamic", hide_index=True,
            width="stretch", key=f"boxes_editor_{len(st.session_state['boxes'])}",
            column_config={"page": st.column_config.NumberColumn(min_value=1, step=1)},
        )
        box_rows = box_rows.to_dict("records")
        boxes = rows_to_boxes(box_rows)

        st.subheader("3. Redact")
        if st.button("Apply redaction", type="primary"):
            out = os.path.join(st.session_state["tmp"], "redacted.pdf")
            with st.spinner("Redacting and verifying…"):
                res = apply_redactions(st.session_state["src"], out, terms, ocr=use_ocr, lang=lang, boxes=boxes,
                                       use_ner=use_ner)
            with open(out, "rb") as fh:
                st.session_state["result"] = (fh.read(), res)
            st.rerun()  # so the risk panel above shows the measured residual risk straight away
        if "result" in st.session_state:
            data, res = st.session_state["result"]
            if res.verification.ok:
                st.success(f"Redacted {res.hits} areas and verified the saved file: every box is black and "
                           "none of the redacted text can be read back.")
            else:
                st.error("Verification FAILED - do not share this file. Fix the items below "
                         "(add the missing text under 'Extra terms' or draw a box) and redact again.")
                for problem in res.verification.problems:
                    st.write("• " + problem)
            quality = res.quality
            grade_box = {"A": st.success, "B": st.success, "C": st.warning, "F": st.error}[quality.grade]
            grade_box(f"**Quality of the saved file: {quality.grade}** ({quality.score}/100)")
            with st.expander("Why this grade?", expanded=quality.grade in ("C", "F")):
                for severity, text in quality.issues:
                    st.write(f"**{severity}** - {text}")
                for note in quality.notes:
                    st.caption(note)
                if not quality.issues and not quality.notes:
                    st.caption("No issues.")
            st.download_button("Download redacted PDF", data, "redacted.pdf", "application/pdf",
                               help=None if res.verification.ok else "Verification failed - check the problems first")

        with st.expander("Teach the benchmark with this review"):
            st.caption("Saves this document and your decisions as a labeled case, on THIS computer only "
                       "(`benchmark/local`, never uploaded, never committed): what you kept ticked is what must be found, "
                       "what you unticked is what must not be flagged, and the terms you typed in are what the detector "
                       "missed. Then the detector is scored on it. The copy of the document is the original, so it still "
                       "contains the personal data.")
            case_name = st.text_input("Case name", value=Path(upload.name).stem, key="capture_name")
            if st.button("Save as benchmark case", key="capture_save"):
                accepted = [(r["type"], r["text"]) for r in rows if r["redact"]]
                rejected = [(r["type"], r["text"]) for r in rows if not r["redact"]]
                manual = [t.strip() for t in extra.splitlines() if t.strip()]
                label = save_case(st.session_state["src"], lang, accepted, rejected, manual, boxes,
                                  name=case_name, source=upload.name)
                with st.spinner("Scoring the detector on this case…"):
                    st.session_state["capture"] = (label.name, evaluate_case(label))
            if "capture" in st.session_state:
                name, score = st.session_state["capture"]
                st.success(f"Saved `{name}`. On this document the detector scores:")
                c1, c2 = st.columns(2)
                c1.metric("Recall (items found)", f"{score.recall:.0%}", f"{score.covered}/{score.items}", delta_color="off", delta_arrow="off")
                c2.metric("Precision (findings that were right)", f"{score.precision:.0%}", f"{score.useful}/{score.findings}", delta_color="off", delta_arrow="off")
                for kind, text, share in score.missed:
                    st.write(f"**Missed** [{kind}] {text} ({share:.0%} covered)")
                for kind, text, value in score.false_positives:
                    st.write(f"**Wrongly flagged** [{kind}] {text} (score {value})")

    with right:
        st.subheader("Preview (red = detections, blue = manual boxes)")
        pages = cached_preview(st.session_state["src"], tuple(terms), use_ocr, lang, tuple(boxes))
        n = st.slider("Page", 1, len(pages), 1) if len(pages) > 1 else 1
        drag = streamlit_image_coordinates(
            Image.open(io.BytesIO(pages[n - 1])), width=640, click_and_drag=True,
            key=f"canvas_{n}_{len(boxes)}",
        )
        new_box = drag_to_box(drag, n - 1)
        if new_box and new_box != st.session_state["last_drag"]:
            st.session_state["last_drag"] = new_box
            st.session_state["boxes"] = [*box_rows, box_to_row(new_box)]
            st.rerun()
