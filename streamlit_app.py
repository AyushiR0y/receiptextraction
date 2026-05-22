from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import pdf_clean as extractor


ACCENT_BLUE = "#005EAC"
ACCENT_ORANGE = "#F58220"
BG = "#F7FAFD"
TEXT = "#16324F"


st.set_page_config(page_title="Commission Extractor", page_icon="CA", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {BG};
        color: {TEXT};
    }}
    .hero {{
        background: linear-gradient(135deg, rgba(0,94,172,0.10), rgba(245,130,32,0.08));
        border: 1px solid rgba(0,94,172,0.12);
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
    }}
    .hero h1 {{
        margin: 0;
        font-size: 2rem;
        color: {TEXT};
    }}
    .hero p {{
        margin: 0.35rem 0 0;
        color: rgba(22,50,79,0.78);
    }}
    .section-card {{
        background: white;
        border: 1px solid rgba(0,94,172,0.10);
        border-radius: 16px;
        padding: 1rem 1rem 0.4rem;
        box-shadow: 0 10px 30px rgba(22,50,79,0.05);
    }}
    .stButton > button {{
        background: {ACCENT_BLUE};
        color: white;
        border: 0;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }}
    .stButton > button:hover {{
        background: {ACCENT_ORANGE};
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _normalize_uploaded_name(name: str) -> Path:
    parts = []
    for chunk in str(name).replace("\\", "/").split("/"):
        chunk = chunk.strip()
        if not chunk or chunk in {".", ".."}:
            continue
        parts.append(chunk)
    return Path(*parts) if parts else Path("upload.bin")


def _stage_uploaded_files(uploaded_files, staging_dir: Path) -> list[Path]:
    staged: list[Path] = []
    for index, uploaded in enumerate(uploaded_files, start=1):
        relative_name = _normalize_uploaded_name(uploaded.name)
        target = staging_dir / f"upload_{index}" / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(uploaded.getbuffer())
        staged.append(target)
    return staged


def _append_log(logs: list[str], message: str, placeholder) -> None:
    logs.append(message)
    with placeholder.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Processing log")
        st.code("\n".join(logs[-250:]) or "No run yet.", language="text")
        st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    <div class="hero">
        <h1>Commission Extractor</h1>
        <p>Upload receipt files or point the app at a local folder, then process and download the mapped Excel.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([1.2, 0.8], gap="large")

with col_left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Inputs")
    local_folder = st.text_input(
        "Local folder path",
        value="",
        placeholder=r"C:\Users\Ayushi.Roy01\Documents\commission\extracted_invoices",
        help="Use this when the files are already on your machine. Uploading files is supported below too.",
    )
    uploaded_files = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        help="Upload PDFs, images, ZIPs, or Excel workbooks.",
    )
    pdf_password = st.text_input("PDF password (optional)", value="", type="password")
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Run")
    st.warning("Do not close this tab while processing is running.")
    process_clicked = st.button("Process files", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


if "result_path" not in st.session_state:
    st.session_state.result_path = ""
if "logs" not in st.session_state:
    st.session_state.logs = []

log_placeholder = st.empty()
status_placeholder = st.empty()
download_placeholder = st.empty()


def render_logs() -> None:
    with log_placeholder.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Processing log")
        st.code("\n".join(st.session_state.logs[-250:]) or "No run yet.", language="text")
        st.markdown("</div>", unsafe_allow_html=True)


render_logs()


if process_clicked:
    st.session_state.logs = []
    render_logs()

    input_paths: list[Path] = []
    staged_root = Path(tempfile.mkdtemp(prefix="commission_streamlit_"))

    if local_folder.strip():
        folder_path = Path(local_folder.strip())
        if not folder_path.exists():
            st.error(f"Folder does not exist: {folder_path}")
            st.stop()
        if not folder_path.is_dir():
            st.error(f"Path is not a folder: {folder_path}")
            st.stop()
        folder_candidates = list(extractor.find_candidate_files(folder_path))
        input_paths.extend(folder_candidates)
        _append_log(st.session_state.logs, f"Discovered {len(folder_candidates)} files in folder: {folder_path}", log_placeholder)

    if uploaded_files:
        staged_files = _stage_uploaded_files(uploaded_files, staged_root)
        input_paths.extend(staged_files)
        _append_log(st.session_state.logs, f"Staged {len(staged_files)} uploaded file(s).", log_placeholder)

    if not input_paths:
        st.warning("Add a folder path or upload at least one file before processing.")
        st.stop()

    unique_inputs: list[Path] = []
    seen_inputs: set[str] = set()
    for candidate in input_paths:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen_inputs:
            continue
        seen_inputs.add(key)
        unique_inputs.append(candidate)

    extractor.load_agent_codes_from_xlsx()
    _append_log(st.session_state.logs, "Loaded agent codes.", log_placeholder)

    all_rows: list[extractor.ReceiptLineItem] = []
    progress = st.progress(0)
    total = len(unique_inputs)

    for index, file_path in enumerate(unique_inputs, start=1):
        _append_log(st.session_state.logs, f"Processing {index}/{total}: {file_path}", log_placeholder)
        try:
            rows = extractor.process_path(file_path, override_password=pdf_password or None)
            if rows:
                all_rows.extend(rows)
                _append_log(st.session_state.logs, f"  -> {len(rows)} row(s)", log_placeholder)
            else:
                _append_log(st.session_state.logs, "  -> no extractable receipt rows", log_placeholder)
        except Exception as exc:
            _append_log(st.session_state.logs, f"  -> failed: {exc}", log_placeholder)
            all_rows.append(extractor.build_placeholder_row(str(file_path), "", f"Extraction failed: {exc}"))
        progress.progress(index / total)

    df = extractor.rows_to_dataframe(all_rows)
    if not df.empty and extractor.AGENT_CODE_BY_NAME:
        df = extractor.apply_agent_code_mapping_to_dataframe(df)

    output_dir = Path(tempfile.mkdtemp(prefix="commission_streamlit_output_"))
    output_file = output_dir / "commission_results.xlsx"
    df.to_excel(output_file, index=False)
    st.session_state.result_path = str(output_file)

    _append_log(st.session_state.logs, f"Wrote {len(df)} row(s) to {output_file}", log_placeholder)
    status_placeholder.success(f"Done. {len(df)} row(s) written.")
    render_logs()


if st.session_state.result_path:
    result_file = Path(st.session_state.result_path)
    if result_file.exists():
        st.download_button(
            label="Download Excel",
            data=result_file.read_bytes(),
            file_name=result_file.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
