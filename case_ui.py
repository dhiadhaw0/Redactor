"""Case mode of the review UI: several documents treated as one case.

Everything is scanned first, what was found is merged into one list of entities you decide on once, and every
file is then redacted with those decisions. Each file is still verified and graded on its own.
"""
import os
import tempfile
from pathlib import Path

import streamlit as st

from benchmark.capture import evaluate_case, save_case
from redactor.caseset import (SHARE_THRESHOLD, build_registry, build_zip, process_set, scan_files, set_audit,
                              terms_for_file)
from redactor import ocr as ocr_module
from redactor.loader import convert
from redactor.redact import preview_pages
from redactor.risk import assess_set, remaining_after
from redactor.scoring import DEFAULT_THRESHOLD

GRADE_BOX = {"A": st.success, "B": st.success, "C": st.warning, "F": st.error}


@st.cache_data(show_spinner="Rendering preview…")
def _preview(pdf, terms, ocr, lang):
    return preview_pages(pdf, list(terms), ocr=ocr, lang=lang)


def _load(uploads, use_ner, use_ocr, lang, profile=None):
    """Convert and scan every upload once per distinct set of files and options."""
    key = (tuple((u.name, u.size) for u in uploads), use_ner, use_ocr, lang, ocr_module.BACKEND,
           str(profile.as_dict()) if profile else None)
    if st.session_state.get("case_key") == key:
        return
    tmp = tempfile.mkdtemp()
    pdfs, notes = [], {}
    for i, u in enumerate(uploads):
        raw = os.path.join(tmp, f"{i}_upload{Path(u.name).suffix.lower()}")
        with open(raw, "wb") as fh:
            fh.write(u.getvalue())
        pdf = os.path.join(tmp, f"{i}.pdf")
        notes[u.name] = convert(raw, pdf)
        pdfs.append((u.name, pdf))
    bar = st.progress(0.0, text="Scanning…")
    scans = scan_files(pdfs, use_ner=use_ner, ocr=use_ocr, lang=lang, profile=profile,
                       progress=lambda i, n, name: bar.progress(i / n, text=f"Scanning {name} ({i + 1}/{n})"))
    bar.empty()
    st.session_state.update(case_key=key, case_tmp=tmp, case_scans=scans, case_registry=build_registry(scans),
                            case_notes=notes)
    for stale in ("case_result", "case_zip", "case_audit"):
        st.session_state.pop(stale, None)


def render_case(uploads, lang, use_ner, use_ocr, profile=None):
    _load(uploads, use_ner, use_ocr, lang, profile)
    scans, registry = st.session_state["case_scans"], st.session_state["case_registry"]

    st.info(f"**Case of {len(scans)} files.** What is found in any of them is merged into one list. Decide once; "
            "each file is then redacted, verified and graded on its own.", icon="🗂️")
    for name, dropped in st.session_state.get("case_notes", {}).items():
        if dropped:
            st.info(f"{name}: converted to PDF. Not carried over from the original: " + "; ".join(dropped) + ".", icon="📄")
    hidden = sum(len(scan.suppressed) for scan in scans)
    if hidden:
        st.info(f"{hidden} finding(s) hidden by the allow list of profile '{profile.name}'.", icon="🙈")
    for scan in scans:
        for w in scan.warnings:
            st.warning(f"{scan.name}: {w}", icon="⚠️")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Privacy risk of the set")
        risk_box = st.container()

        st.subheader("1. Decide once for all files")
        c1, c2, c3 = st.columns([2, 1, 2])
        threshold = c1.slider("Pre-select entities scoring at least", 0, 100,
                              profile.min_score if profile and profile.min_score is not None else DEFAULT_THRESHOLD, key="case_threshold",
                              help="The same confidence score as for one file. Moving it resets your ticks.")
        share = c2.toggle("Share across files", value=True, key="case_share",
                          help="Search every file for the confident findings of the others (a surname confirmed in "
                               "one file is removed from all). Turn off for unrelated documents.")
        share_at = c3.slider("Share findings scoring at least", 0, 100,
                             profile.share_threshold if profile and profile.share_threshold is not None else SHARE_THRESHOLD,
                             key="case_share_at", disabled=not share,
                             help="Only confident findings are spread, so a common word flagged in one file cannot "
                                  "be blacked out of every other file.")
        rows = st.data_editor(
            [{"redact": e.score >= threshold, "score": e.score, "type": e.kind, "text": e.text,
              "found in": ", ".join(e.files)} for e in registry],
            column_config={
                "redact": st.column_config.CheckboxColumn("Redact?"),
                "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                "type": st.column_config.TextColumn("Type", disabled=True),
                "text": st.column_config.TextColumn("Text", disabled=True),
                "found in": st.column_config.TextColumn("Found in", disabled=True),
            },
            hide_index=True, width="stretch", key=f"case_editor_{threshold}_{st.session_state['case_key']}",
        )
        selected = {e.key for e, r in zip(registry, rows) if r["redact"]}
        per_file_terms = {s.name: terms_for_file(s, registry, selected, share, share_at) for s in scans}

        # ---- risk panel: before vs after ----
        before = assess_set({s.name: s.findings for s in scans})
        measured = st.session_state.get("case_result")
        if measured is not None:
            after, label = measured.risk_after, "After redaction (measured on the saved files)"
        else:
            after = assess_set({s.name: remaining_after(s.findings, per_file_terms[s.name]) for s in scans})
            label = "After redaction (estimate for your current ticks)"
        with risk_box:
            a, b = st.columns(2)
            for col, name, risk in ((a, "Original documents, together", before), (b, label, after)):
                col.metric(name, f"{risk.score}/100", risk.level.upper(), delta_color="off", delta_arrow="off")
                col.progress(risk.score / 100)
            with st.expander("Why this score?"):
                st.caption("Each distinct identity counts once, but findings from different files are pooled: a name in "
                           "one file and an IBAN in another still trigger the 'financial data tied to a named person' bonus.")
                st.table([{"type": k, "findings": str(n), "points": round(p, 1)} for k, n, p in before.by_kind]
                         + [{"type": "bonus: " + r, "findings": "", "points": pts} for r, pts in before.bonuses])

        st.subheader("2. Redact all files")
        st.caption(f"{len(selected)} of {len(registry)} entities selected · "
                   f"{sum(len(t) for t in per_file_terms.values())} searches across {len(scans)} files")
        if st.button("Apply to all files", type="primary", key="case_apply"):
            out_dir = os.path.join(st.session_state["case_tmp"], "out")
            bar = st.progress(0.0, text="Redacting…")
            result = process_set(scans, out_dir, selected, registry, share, share_at, ocr=use_ocr, lang=lang, use_ner=use_ner,
                                 progress=lambda i, n, name: bar.progress(i / n, text=f"Redacting {name} ({i + 1}/{n})"))
            bar.empty()
            audit = set_audit(result, registry, share, share_at)
            st.session_state.update(case_result=result, case_audit=audit, case_zip=build_zip(result, audit))
            st.rerun()  # so the risk panel above shows the measured numbers straight away

        result = st.session_state.get("case_result")
        if result is not None:
            GRADE_BOX[result.worst_grade](
                f"**Set grade: {result.worst_grade}** (the worst file decides) · "
                f"{'every file verified' if result.ok else 'at least one file FAILED verification or is graded F: do not share it'}"
                + (f" · {result.shared_terms} searches came from other files" if result.shared_terms else ""))
            st.dataframe(
                [{"file": o.name, "areas": o.result.hits,
                  "verified": "yes" if o.result.verification.ok else "NO",
                  "grade": o.result.quality.grade, "score": o.result.quality.score,
                  "residual risk": o.result.risk_after.score} for o in result.outcomes],
                hide_index=True, width="stretch")
            for o in result.outcomes:
                q = o.result.quality
                if q.issues or o.result.verification.problems:
                    with st.expander(f"{o.name}: grade {q.grade}", expanded=q.grade in ("C", "F")):
                        for problem in o.result.verification.problems:
                            st.write("• " + problem)
                        for severity, text in q.issues:
                            st.write(f"**{severity}** - {text}")
                        for note in q.notes:
                            st.caption(note)
            st.download_button("Download all (zip with audit log)", st.session_state["case_zip"], "redacted_case.zip",
                               "application/zip", help=None if result.ok else "At least one file failed: check before sharing")


        with st.expander("Teach the benchmark with this review"):
            st.caption("Saves each file and your decisions as a labeled case, on THIS computer only (`benchmark/local`, never "
                       "uploaded, never committed). The copies are the originals, so they still contain the personal data.")
            if st.button("Save every file as a benchmark case", key="case_capture"):
                rows_out = []
                for scan in scans:
                    own = {f.text.casefold(): f for f in scan.findings}
                    mine = [e for e in registry if scan.name in e.files]
                    accepted = [(e.kind, own.get(e.text.casefold(), e).text) for e in mine if e.key in selected]
                    rejected = [(e.kind, e.text) for e in mine if e.key not in selected]
                    label = save_case(scan.pdf, lang, accepted, rejected, name=scan.name, source=scan.name)
                    s = evaluate_case(label)
                    rows_out.append({"case": label.name, "items": s.items, "recall": f"{s.recall:.0%}",
                                     "findings": s.findings, "precision": f"{s.precision:.0%}",
                                     "missed": len(s.missed), "wrongly flagged": len(s.false_positives)})
                st.session_state["case_capture_rows"] = rows_out
            if st.session_state.get("case_capture_rows"):
                st.dataframe(st.session_state["case_capture_rows"], hide_index=True, width="stretch")

    with right:
        st.subheader("Preview (red = will be removed)")
        name = st.selectbox("File", [s.name for s in scans], key="case_preview_file")
        scan = next(s for s in scans if s.name == name)
        pages = _preview(scan.pdf, tuple(per_file_terms[name]), use_ocr, lang)
        n = st.slider("Page", 1, len(pages), 1, key="case_preview_page") if len(pages) > 1 else 1
        st.image(pages[n - 1], width="stretch")
        st.caption("Manual boxes are available when a file is opened on its own.")
