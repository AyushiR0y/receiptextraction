from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

import pdf_clean as extractor


# ── Brand ──────────────────────────────────────────────────────────────────
PRIMARY      = "#005EAC"
PRIMARY_DARK = "#004A8C"
PRIMARY_SOFT = "#DAF8FF"
ACCENT       = "#F58220"
ACCENT_SOFT  = "#FEF0E4"
BG           = "#F4F7FC"
SURFACE      = "#FFFFFF"
TEXT         = "#0D1F33"
TEXT_MUTED   = "#5A7492"
BORDER       = "rgba(0,94,172,0.12)"
SHADOW       = "rgba(0,94,172,0.08)"


st.set_page_config(
    page_title="Commission Extractor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [data-testid="stAppViewContainer"], .stApp {{
    background: {BG} !important;
    color: {TEXT} !important;
    font-family: 'DM Sans', sans-serif !important;
}}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display: none; }}

/* ── Main container ── */
[data-testid="stAppViewContainer"] > .main {{ padding: 2rem 2.5rem 4rem; }}
[data-testid="block-container"] {{ max-width: 1240px; margin: 0 auto; padding: 0; }}

/* ── Top nav bar ── */
.topbar {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0 0 1.75rem;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 2rem;
}}
.topbar-logo {{
    width: 40px; height: 40px;
    background: {PRIMARY};
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    box-shadow: 0 4px 12px rgba(0,94,172,0.28);
}}
.topbar-title {{
    font-size: 1.55rem;
    font-weight: 700;
    color: {TEXT};
    letter-spacing: -0.02em;
}}
.topbar-sub {{
    font-size: 0.95rem;
    color: {TEXT_MUTED};
    font-weight: 400;
}}
.topbar-badge {{
    margin-left: auto;
    background: {PRIMARY_SOFT};
    color: {PRIMARY};
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    border: 1px solid rgba(0,94,172,0.18);
}}

/* ── Section headers ── */
.sec-header {{
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: {PRIMARY};
    margin-bottom: 0.55rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

/* ── Card ── */
.card {{
    background: transparent;
    border: 0;
    border-radius: 0;
    padding: 1rem 0 1rem;
    box-shadow: none;
    margin-bottom: 0.9rem;
}}

.section-divider {{
    height: 1px;
    background: rgba(0,94,172,0.14);
    margin: 0.75rem 0 1.25rem;
}}

/* ── Stat chips in hero ── */
.stats-row {{
    display: flex;
    gap: 0.75rem;
    margin-top: 1.1rem;
    flex-wrap: wrap;
}}
.stat-chip {{
    background: {PRIMARY_SOFT};
    border: 1px solid rgba(0,94,172,0.15);
    border-radius: 12px;
    padding: 0.5rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
}}
.stat-chip .val {{
    font-size: 1.15rem;
    font-weight: 700;
    color: {PRIMARY};
    line-height: 1;
}}
.stat-chip .lbl {{
    font-size: 0.64rem;
    color: {TEXT_MUTED};
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

/* ── Tip banner ── */
.tip-banner {{
    background: {ACCENT_SOFT};
    border: 1px solid rgba(245,130,32,0.2);
    border-radius: 12px;
    padding: 0.65rem 1rem;
    font-size: 0.84rem;
    color: #8B4A0A;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1.25rem;
}}

/* ── Step indicators ── */
.step-list {{
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0.25rem 0;
}}
.step-item {{
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
}}
.step-num {{
    width: 26px; height: 26px;
    background: {PRIMARY};
    color: white;
    border-radius: 50%;
    font-size: 0.72rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
    box-shadow: 0 2px 8px rgba(0,94,172,0.3);
}}
.step-text {{
    font-size: 0.8rem;
    color: {TEXT};
    line-height: 1.5;
}}
.step-text strong {{ color: {PRIMARY}; font-weight: 600; }}

/* ── Streamlit overrides ── */
.stTextInput > div > div > input,
.stTextInput > div > div > input:focus {{
    border: 1.5px solid {BORDER} !important;
    border-radius: 10px !important;
    background: {BG} !important;
    color: {TEXT} !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    box-shadow: none !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: {PRIMARY} !important;
    box-shadow: 0 0 0 3px rgba(0,94,172,0.12) !important;
}}

.stFileUploader > div {{
    border: 2px dashed rgba(0,94,172,0.25) !important;
    border-radius: 14px !important;
    background: {PRIMARY_SOFT} !important;
    transition: border-color 0.2s, background 0.2s !important;
}}
.stFileUploader > div:hover {{
    border-color: {PRIMARY} !important;
    background: rgba(0,94,172,0.08) !important;
}}

/* Submit / primary button */
.stFormSubmitButton > button, .stButton > button {{
    background: linear-gradient(135deg, {PRIMARY}, {PRIMARY_DARK}) !important;
    color: white !important;
    border: 0 !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.4rem !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 4px 14px rgba(0,94,172,0.28) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}}
.stFormSubmitButton > button:hover, .stButton > button:hover {{
    background: linear-gradient(135deg, {ACCENT}, #D4690E) !important;
    box-shadow: 0 6px 20px rgba(245,130,32,0.35) !important;
    transform: translateY(-1px) !important;
}}
.stFormSubmitButton > button:active, .stButton > button:active {{
    transform: translateY(0px) !important;
}}

.submit-row {{
    display: flex;
    justify-content: center;
    margin-top: 0.75rem;
}}

.submit-row .stFormSubmitButton {{
    width: 100%;
    max-width: 340px;
}}

/* Download button */
[data-testid="stDownloadButton"] button {{
    background: linear-gradient(135deg, #1DA462, #158C52) !important;
    box-shadow: 0 4px 14px rgba(29,164,98,0.3) !important;
}}
[data-testid="stDownloadButton"] button:hover {{
    background: linear-gradient(135deg, #18C06E, #12A348) !important;
    box-shadow: 0 6px 20px rgba(29,164,98,0.4) !important;
}}

/* Progress bar */
.stProgress > div > div > div > div {{
    background: linear-gradient(90deg, {PRIMARY}, {ACCENT}) !important;
    border-radius: 999px !important;
}}
.stProgress > div > div > div {{
    background: {PRIMARY_SOFT} !important;
    border-radius: 999px !important;
    height: 6px !important;
}}

/* Data editor */
[data-testid="stDataEditor"] {{
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1.5px solid {BORDER} !important;
}}

/* Code block */
.stCode {{
    border-radius: 12px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    background: #F0F4FA !important;
}}

/* Alerts */
.stAlert {{
    border-radius: 12px !important;
    border-left-width: 4px !important;
}}

/* Column gaps */
[data-testid="column"] {{ padding: 0 0.5rem !important; }}
[data-testid="column"]:first-child {{ padding-left: 0 !important; }}
[data-testid="column"]:last-child {{ padding-right: 0 !important; }}

/* Label styling */
.stTextInput label, .stFileUploader label {{
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: {TEXT_MUTED} !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}}

/* Subheader */
h3 {{
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: {TEXT} !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0.75rem !important;
}}

/* Caption */
.stCaption {{ color: {TEXT_MUTED} !important; font-size: 0.73rem !important; }}

/* Smaller body text inside cards */
.card p, .card li {{
    font-size: 0.8rem;
    line-height: 1.45;
}}

/* Expander tweaks */
details summary {{
    font-weight: 700;
    font-size: 1.02rem;
    color: {PRIMARY};
}}

details > div {{
    padding-top: 0.35rem;
}}

/* Warning */
.stWarning {{ background: {ACCENT_SOFT} !important; color: #7A3B0A !important; }}
</style>
""", unsafe_allow_html=True)

SUPPORTED_TYPES = ["pdf", "jpg", "jpeg", "png", "zip", "xlsx", "xlsm", "xlsb", "xls"]


def _default_password_table() -> pd.DataFrame:
    rows = [{"Bank Name": name, "Password": password} for name, password in extractor.BANK_PASSWORDS.items()]
    return pd.DataFrame(rows, columns=["Bank Name", "Password"])


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
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-header">📋 Processing log</div>', unsafe_allow_html=True)
        st.code("\n".join(logs[-250:]) or "No run yet.", language="text")
        st.markdown("</div>", unsafe_allow_html=True)


def _password_map_from_table(table: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    if table is None or table.empty:
        return result
    for _, row in table.iterrows():
        bank_name = str(row.get("Bank Name", "") or "").strip().lower()
        password = str(row.get("Password", "") or "").strip()
        if bank_name and password:
            result[bank_name] = password
    return result


def _match_password(source_text: str, custom_passwords: dict[str, str]) -> str:
    probe = (source_text or "").lower()
    for bank_name in sorted(custom_passwords.keys(), key=len, reverse=True):
        if bank_name and bank_name in probe:
            return custom_passwords[bank_name]
    for bank_name, password in extractor.BANK_PASSWORDS.items():
        if bank_name and bank_name in probe:
            return password
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">💼</div>
    <div>
        <div class="topbar-title">Commission Extractor</div>
        <div class="topbar-sub">Automated receipt processing &amp; Excel export</div>
    </div>
    <div class="topbar-badge">v2.0</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TIP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tip-banner">
    💡 <strong>Tip:</strong>&nbsp;Do not close the tab when running. Folder upload works directly in the browser. The local path field only works when running the app on your own machine.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN FORM
# ─────────────────────────────────────────────────────────────────────────────
with st.form("processing_form"):
    col_left, col_right = st.columns([1.3, 0.7], gap="large")

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-header">📂 File inputs</div>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload files or folder",
            key="uploaded_files",
            accept_multiple_files="directory",
            type=SUPPORTED_TYPES,
            help="Pick multiple files or a whole folder.",
        )

        st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)

        local_folder = st.text_input(
            "Local folder path (local app only)",
            key="local_folder",
            value="",
            placeholder=r"C:\Users\Ayushi.Roy01\Documents\commission\extracted_invoices",
            help="Only works when Streamlit runs on your computer.",
        )

        st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)

        pdf_password = st.text_input(
            "PDF password override (optional)",
            key="pdf_password",
            value="",
            type="password",
            help="Applies to all files. Leave blank to use bank-specific passwords below.",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card" style="height:100%;display:flex;flex-direction:column;">', unsafe_allow_html=True)
        st.markdown('<div class="sec-header">⚡ How it works</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="step-list">
            <div class="step-item">
                <div class="step-num">1</div>
                <div class="step-text">Upload your <strong>receipt files</strong>, folder, or ZIP archive</div>
            </div>
            <div class="step-item">
                <div class="step-num">2</div>
                <div class="step-text">Files are <strong>staged, deduplicated</strong> and normalised automatically</div>
            </div>
            <div class="step-item">
                <div class="step-num">3</div>
                <div class="step-text">Agent codes are <strong>matched & mapped</strong> from the master sheet</div>
            </div>
            <div class="step-item">
                <div class="step-num">4</div>
                <div class="step-text">Download a clean, ready-to-use <strong>Excel workbook</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

        process_clicked = st.form_submit_button(
            "🚀  Process files",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# BANK PASSWORDS + STATS ROW
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
pwd_col, stat_col = st.columns([1.2, 0.8], gap="large")

with pwd_col:
    with st.expander("🔐 Bank passwords", expanded=False):
        st.caption("Edit or add bank/password pairs. Matching runs against the file path or filename.")

        password_table = st.data_editor(
            _default_password_table(),
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "Bank Name": st.column_config.TextColumn("Bank Name", width="large"),
                "Password":  st.column_config.TextColumn("Password",  width="medium"),
            },
            key="password_table_editor",
        )

with stat_col:
    with st.expander("📊 Supported formats", expanded=False):
        st.markdown("""
        <div class="stats-row">
            <div class="stat-chip"><span class="val">PDF</span><span class="lbl">Encrypted &amp; plain</span></div>
            <div class="stat-chip"><span class="val">XLS·X</span><span class="lbl">Excel sheets</span></div>
            <div class="stat-chip"><span class="val">IMG</span><span class="lbl">JPG / PNG</span></div>
            <div class="stat-chip"><span class="val">ZIP</span><span class="lbl">Bulk archives</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-header">✅ Output features</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="step-list">
            <div class="step-item">
                <div class="step-num" style="background:#1DA462">✓</div>
                <div class="step-text">Duplicate receipts removed automatically</div>
            </div>
            <div class="step-item">
                <div class="step-num" style="background:#1DA462">✓</div>
                <div class="step-text">Agent codes mapped from master list</div>
            </div>
            <div class="step-item">
                <div class="step-num" style="background:#1DA462">✓</div>
                <div class="step-text">Single-click Excel download</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "result_path" not in st.session_state:
    st.session_state.result_path = ""
if "logs" not in st.session_state:
    st.session_state.logs = []

log_placeholder    = st.empty()
status_placeholder = st.empty()
download_placeholder = st.empty()


def render_logs() -> None:
    with log_placeholder.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-header">📋 Processing log</div>', unsafe_allow_html=True)
        st.code("\n".join(st.session_state.logs[-250:]) or "No run yet.", language="text")
        st.markdown("</div>", unsafe_allow_html=True)


render_logs()

# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING LOGIC
# ─────────────────────────────────────────────────────────────────────────────
if process_clicked:
    st.session_state.logs = []
    render_logs()
    custom_passwords = _password_map_from_table(password_table)

    uploaded_files = st.session_state.get("uploaded_files") or []
    local_folder   = st.session_state.get("local_folder", "") or ""
    pdf_password   = st.session_state.get("pdf_password", "") or ""

    input_paths: list[Path] = []
    staged_root = Path(tempfile.mkdtemp(prefix="commission_streamlit_"))

    if local_folder.strip():
        folder_path = Path(local_folder.strip())
        if not folder_path.exists():
            st.error(f"Folder does not exist: {folder_path}. If this is Streamlit Cloud, use folder upload instead.")
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
            password_override = pdf_password.strip() if pdf_password.strip() else _match_password(str(file_path), custom_passwords)
            rows = extractor.process_path(file_path, override_password=password_override or None)
            if rows:
                all_rows.extend(rows)
                _append_log(st.session_state.logs, f"  → {len(rows)} row(s) extracted", log_placeholder)
            else:
                _append_log(st.session_state.logs, "  → no extractable receipt rows", log_placeholder)
        except Exception as exc:
            _append_log(st.session_state.logs, f"  → failed: {exc}", log_placeholder)
            all_rows.append(extractor.build_placeholder_row(str(file_path), "", f"Extraction failed: {exc}"))
        progress.progress(index / total)

    df = extractor.rows_to_dataframe(all_rows)
    if not df.empty and extractor.AGENT_CODE_BY_NAME:
        df = extractor.apply_agent_code_mapping_to_dataframe(df)

    output_dir  = Path(tempfile.mkdtemp(prefix="commission_streamlit_output_"))
    output_file = output_dir / "commission_results.xlsx"
    df.to_excel(output_file, index=False)
    st.session_state.result_path = str(output_file)

    _append_log(st.session_state.logs, f"✅ Wrote {len(df)} row(s) to {output_file}", log_placeholder)
    status_placeholder.success(f"✅  Done — {len(df)} rows written successfully.")
    render_logs()

# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.result_path:
    result_file = Path(st.session_state.result_path)
    if result_file.exists():
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="card" style="border-color:rgba(29,164,98,0.25);background:rgba(29,164,98,0.03)">', unsafe_allow_html=True)
        st.markdown('<div class="sec-header" style="color:#1DA462">📥 Download result</div>', unsafe_allow_html=True)
        st.download_button(
            label="⬇️  Download Excel workbook",
            data=result_file.read_bytes(),
            file_name=result_file.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)