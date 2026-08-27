from __future__ import annotations

import os
import datetime
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

import commission_extractor.api as extractor

# ── Propagate Streamlit secrets into os.environ so extractor can find API keys ──
# On Streamlit Cloud, secrets live in st.secrets, not os.environ.
# Keys recognised by the extractor (Google Vision, Azure OpenAI, etc.) are forwarded here.
_SECRET_KEYS = [
    "GOOGLE_VISION_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
]
try:
    for _k in _SECRET_KEYS:
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
    # Reset the extractor's cached API-key values so they are re-read from the
    # now-populated environment on the first call of this session.
    extractor._extractor.GOOGLE_VISION_API_KEY = None
    extractor._extractor.AZURE_OPENAI_CONFIG_CACHE = None
except Exception:
    pass  # st.secrets not available (local run without secrets.toml) — no-op

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

SUPPORTED_TYPES = ["pdf", "jpg", "jpeg", "png", "zip", "doc", "docx"]


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
    """Write uploaded files to a temp directory, automatically expanding ZIPs."""
    import zipfile
    staged: list[Path] = []
    for index, uploaded in enumerate(uploaded_files, start=1):
        relative_name = _normalize_uploaded_name(uploaded.name)
        target = staging_dir / f"upload_{index}" / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = uploaded.getbuffer()
        target.write_bytes(raw)

        # Expand ZIP archives — add the contents, not the ZIP itself.
        if target.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(target, "r") as zf:
                    extract_dir = target.parent / target.stem
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    zf.extractall(extract_dir)
                    for member in zf.infolist():
                        if member.is_dir():
                            continue
                        extracted = extract_dir / member.filename
                        if extracted.suffix.lower() in {
                            ".pdf", ".jpg", ".jpeg", ".png", ".doc", ".docx"
                        }:
                            staged.append(extracted)
            except Exception:
                # If unzipping fails, fall back to treating it as a regular file.
                staged.append(target)
        else:
            staged.append(target)
    return staged


def _append_log(logs: list[str], message: str, placeholder) -> None:
    logs.append(message)
    with placeholder.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-header">📋 Processing log</div>', unsafe_allow_html=True)
        st.code("\n".join(logs[-250:]) or "No run yet.", language="text")
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MIS TRACKING — Google Sheets via Apps Script
# After every run, one row is silently appended to a Google Sheet you own.
# The visitor sees nothing. No SMTP, no webhook, no corporate firewall issues.
# This works because the POST goes from Streamlit Cloud → script.google.com
# (plain HTTPS — not blocked by any known corporate policy).
#
# Setup (one-time, ~5 minutes):
#   1. Go to https://sheets.google.com and create a new sheet called
#      "Commission MIS Log" (or any name you like).
#   2. Go to https://script.google.com → New Project → paste the Apps Script
#      code from mis_appscript.gs in this repo → Save → Deploy → New deployment
#      → Type: Web app → Execute as: Me → Who has access: Anyone → Deploy.
#   3. Copy the generated Web app URL and add it as a Streamlit secret:
#        MIS_GOOGLE_SHEET_URL = "https://script.google.com/macros/s/.../exec"
# ─────────────────────────────────────────────────────────────────────────────

import json
import smtplib
import logging as _logging
import urllib.request
import urllib.error
from email.message import EmailMessage


def _get_secret(key: str, default: str = "") -> str:
    """Read a value from st.secrets first, then os.environ, then default."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


def _count_pdf_pages(path: Path) -> int:
    """Return the page count of a PDF, or 0 for non-PDFs / unreadable files."""
    if path.suffix.lower() != ".pdf":
        return 0
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(path)).pages)
    except Exception:
        return 0


def _log_mis_to_google_sheet(row: dict) -> tuple[bool, str]:
    """POST one MIS row to the Google Apps Script web-app endpoint.
    Google Apps Script redirects POST → GET internally (302), so we must
    follow the Location header with a second GET request.
    Returns (success, detail).
    """
    url = _get_secret("MIS_GOOGLE_SHEET_URL")
    if not url:
        return False, "MIS_GOOGLE_SHEET_URL not set"
    try:
        data = json.dumps(row).encode("utf-8")
        # Step 1: initial POST (do NOT auto-follow redirects — handle manually).
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Disable auto-redirect so we can re-issue as GET.
        opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None  # suppress redirect

        no_redirect_opener = urllib.request.build_opener(_NoRedirect())
        try:
            with no_redirect_opener.open(req, timeout=30) as resp:
                resp.read()
                return True, "logged (no-redirect)"
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308):
                # Follow the redirect as a GET — this is normal for Apps Script.
                location = exc.headers.get("Location", "")
                if location:
                    with urllib.request.urlopen(location, timeout=30) as r2:
                        r2.read()
                    return True, "logged (redirect followed)"
                return False, f"Redirect with no Location header (HTTP {exc.code})"
            return False, f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:
        return False, str(exc)


def _send_mis_gmail(subject: str, body: str) -> tuple[bool, str]:
    """Send MIS report via Gmail SMTP (App Password).
    Secrets needed: GMAIL_USERNAME, GMAIL_APP_PASSWORD, MIS_EMAIL_TO.
    MIS_EMAIL_TO may be a comma-separated list of addresses.
    """
    user = _get_secret("GMAIL_USERNAME")
    password = _get_secret("GMAIL_APP_PASSWORD")
    recipient_raw = _get_secret("MIS_EMAIL_TO")
    if not (user and password and recipient_raw):
        return False, "GMAIL_USERNAME / GMAIL_APP_PASSWORD / MIS_EMAIL_TO not set"
    # Support multiple recipients separated by commas.
    recipients = [r.strip() for r in recipient_raw.split(",") if r.strip()]
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = ", ".join(recipients)
        msg.set_content(body)
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        return True, f"sent via Gmail to {len(recipients)} recipient(s)"
    except Exception as exc:
        return False, str(exc)


def _deliver_mis(subject: str, body: str) -> tuple[bool, str]:
    """Try Google Sheet first (primary), then Gmail (fallback).
    Returns (success, channel_used_or_error).
    """
    errors: list[str] = []

    gsheet_url = _get_secret("MIS_GOOGLE_SHEET_URL")
    if gsheet_url:
        ok, detail = _log_mis_to_google_sheet({"_subject": subject, "_body": body})
        # _log_mis_to_google_sheet is for structured rows; for email-style body
        # use it only when called from the sheet path.  The row dict is passed
        # directly from the call site; this wrapper is for the email fallback.
        if ok:
            return True, f"Google Sheet: {detail}"
        errors.append(f"Sheet: {detail}")

    if _get_secret("GMAIL_USERNAME"):
        ok, detail = _send_mis_gmail(subject, body)
        if ok:
            return True, f"Gmail: {detail}"
        errors.append(f"Gmail: {detail}")

    if not errors:
        return False, "no MIS channel configured"
    return False, " | ".join(errors)


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


# ── Per-page progress: pipe the extractor's INFO logs into the Streamlit log ──
class _StreamlitLogHandler(_logging.Handler):
    """Routes extractor log messages that mention 'page' into the live Streamlit log."""

    def __init__(self, logs: list[str], placeholder):
        super().__init__(_logging.INFO)
        self._logs = logs
        self._placeholder = placeholder
        # Only forward lines that describe per-page activity.
        self._keywords = ("page", "ocr", "low text", "forcing ocr", "rendered", "extracted")

    def emit(self, record: _logging.LogRecord) -> None:
        msg = record.getMessage().lower()
        if any(k in msg for k in self._keywords):
            _append_log(self._logs, f"  [extractor] {record.getMessage()}", self._placeholder)



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
    💡 <strong>Tip:</strong>&nbsp;Upload individual files, a whole folder, or a ZIP archive — all at once. ZIPs are unpacked automatically.
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
            "Upload files, folder, or ZIP",
            key="uploaded_files",
            accept_multiple_files=True,
            type=SUPPORTED_TYPES,
            help="Select individual PDFs/images, a whole folder, or ZIP archives. ZIPs are unpacked automatically.",
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
# SUPPORTED FORMATS STRIP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
with st.expander("� Supported formats & output features", expanded=False):
    st.markdown("""
    <div class="stats-row">
        <div class="stat-chip"><span class="val">PDF</span><span class="lbl">Encrypted &amp; plain</span></div>
        <div class="stat-chip"><span class="val">IMG</span><span class="lbl">JPG / PNG</span></div>
        <div class="stat-chip"><span class="val">ZIP</span><span class="lbl">Auto-unpacked</span></div>
        <div class="stat-chip"><span class="val">DOC</span><span class="lbl">Word documents</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
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

    uploaded_files = st.session_state.get("uploaded_files") or []

    input_paths: list[Path] = []
    staged_root = Path(tempfile.mkdtemp(prefix="commission_streamlit_"))

    if uploaded_files:
        staged_files = _stage_uploaded_files(uploaded_files, staged_root)
        input_paths.extend(staged_files)
        _append_log(st.session_state.logs, f"Staged {len(staged_files)} file(s) (ZIPs unpacked).", log_placeholder)

    if not input_paths:
        st.warning("Upload at least one file before processing.")
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

    # ── Attach live log handler so per-page extractor messages appear in the UI ──
    _st_log_handler = _StreamlitLogHandler(st.session_state.logs, log_placeholder)
    _extractor_logger = _logging.getLogger("receipt_extractor")
    _extractor_logger.addHandler(_st_log_handler)

    # ── MIS instrumentation: reset per-run counters and capture start time ──
    _IST_TZ = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    mis_started = datetime.datetime.now(_IST_TZ)
    extractor._extractor.GOOGLE_VISION_CALL_COUNT = 0
    extractor._extractor.AZURE_AI_CALL_COUNT = 0
    extractor._extractor.AZURE_AI_INPUT_CHARS = 0
    extractor._extractor.AZURE_AI_OUTPUT_CHARS = 0
    mis_file_details: list[dict] = []

    all_rows: list[extractor.ReceiptLineItem] = []
    progress = st.progress(0)
    total = len(unique_inputs)

    for index, file_path in enumerate(unique_inputs, start=1):
        _append_log(st.session_state.logs, f"Processing {index}/{total}: {file_path}", log_placeholder)
        file_pages = _count_pdf_pages(file_path)
        file_rows = 0
        file_status = ""
        try:
            password_override = _match_password(str(file_path), extractor.BANK_PASSWORDS)
            rows = extractor.process_path(file_path, override_password=password_override or None)
            if rows:
                all_rows.extend(rows)
                file_rows = len(rows)
                file_status = "ok"
                _append_log(st.session_state.logs, f"  → {len(rows)} row(s) extracted", log_placeholder)
            else:
                file_status = "no rows"
                _append_log(st.session_state.logs, "  → no extractable receipt rows", log_placeholder)
                # Hint the user when this looks like a scanned PDF and Vision is not configured.
                if str(file_path).lower().endswith(".pdf") and not extractor._extractor.get_google_vision_api_key():
                    _append_log(
                        st.session_state.logs,
                        "  ⚠ This may be a scanned PDF.  GOOGLE_VISION_API_KEY is not set — "
                        "add it to Streamlit secrets so cloud OCR is available.",
                        log_placeholder,
                    )
        except Exception as exc:
            file_status = f"failed: {exc}"
            _append_log(st.session_state.logs, f"  → failed: {exc}", log_placeholder)
            all_rows.append(extractor.build_placeholder_row(str(file_path), "", f"Extraction failed: {exc}"))
        mis_file_details.append(
            {"name": Path(file_path).name, "pages": file_pages, "rows": file_rows, "status": file_status}
        )
        progress.progress(index / total)

    # Post-extraction AI + arithmetic re-validation of the commission math so that
    # any row marked "Math Valid = YES" is genuinely valid (no false positives).
    try:
        _append_log(st.session_state.logs, "Running post-extraction math validation…", log_placeholder)
        extractor.revalidate_rows_math(all_rows)
    except Exception as exc:
        _append_log(st.session_state.logs, f"  → math validation skipped: {exc}", log_placeholder)

    df = extractor.rows_to_dataframe(all_rows)
    if not df.empty and extractor.AGENT_CODE_BY_NAME:
        df = extractor.apply_agent_code_mapping_to_dataframe(df)

    output_dir  = Path(tempfile.mkdtemp(prefix="commission_streamlit_output_"))
    output_file = output_dir / "commission_results.xlsx"
    df.to_excel(output_file, index=False)

    # Final safety net: re-validate the written workbook so no "Math Valid = YES"
    # row is a false positive (corrects the file in place if needed).
    # use_ai=False because revalidate_rows_math already ran the AI check above
    # on the same rows — calling it again on the Excel would be a duplicate AI call.
    try:
        summary = extractor.validate_excel_math(output_file, use_ai=False)
        if summary.get("downgraded"):
            _append_log(
                st.session_state.logs,
                f"  → math validation corrected {summary['downgraded']} false positive(s)",
                log_placeholder,
            )
    except Exception as exc:
        _append_log(st.session_state.logs, f"  → final math validation skipped: {exc}", log_placeholder)

    # ── Remove the live log handler now that processing is complete ──
    _extractor_logger.removeHandler(_st_log_handler)

    st.session_state.result_path = str(output_file)

    _append_log(st.session_state.logs, f"✅ Wrote {len(df)} row(s) to {output_file}", log_placeholder)
    status_placeholder.success(f"✅  Done — {len(df)} rows written successfully.")
    render_logs()

    # ── Silent MIS logging — Google Sheet primary, Gmail fallback ──
    try:
        _IST = _IST_TZ
        mis_finished = datetime.datetime.now(_IST)
        duration_s = (mis_finished - mis_started).total_seconds()
        total_pages = sum(d.get("pages", 0) for d in mis_file_details)
        _DT_FMT = "%d %m %Y %I:%M:%S %p IST"
        mis_row = {
            "Timestamp"       : mis_finished.strftime(_DT_FMT),
            "Run Started"     : mis_started.strftime(_DT_FMT),
            "Duration (s)"    : round(duration_s, 1),
            "Files Processed" : len(mis_file_details),
            "Total Pages"     : total_pages,
            "Rows Extracted"  : sum(d.get("rows", 0) for d in mis_file_details),
            "Rows Written"    : len(df),
            "GV OCR Calls"    : int(extractor._extractor.GOOGLE_VISION_CALL_COUNT),
            "AI (LLM) Calls"  : int(extractor._extractor.AZURE_AI_CALL_COUNT),
            "AI Input Chars"  : int(extractor._extractor.AZURE_AI_INPUT_CHARS),
            "AI Output Chars" : int(extractor._extractor.AZURE_AI_OUTPUT_CHARS),
            "File Names"      : ", ".join(d["name"] for d in mis_file_details),
            "Per-file Detail" : " | ".join(
                f"{d['name']} (pg:{d['pages']} rows:{d['rows']} {d.get('status','')})"
                for d in mis_file_details
            ),
        }

        # ── Estimated cost (GPT-4o-mini pricing: $0.15/1M input, $0.60/1M output tokens)
        # 1 token ≈ 4 characters.  1 USD ≈ 84 INR
        _USD_TO_INR    = 84
        _ai_in_tokens  = mis_row["AI Input Chars"]  / 4
        _ai_out_tokens = mis_row["AI Output Chars"] / 4
        _est_cost_usd  = (_ai_in_tokens * 0.15 + _ai_out_tokens * 0.60) / 1_000_000
        _est_cost_inr  = _est_cost_usd * _USD_TO_INR
        mis_row["Est. AI Cost (INR)"] = f"₹{_est_cost_inr:.4f}"

        mis_errors: list[str] = []

        # Primary: Google Sheet
        sheet_url = _get_secret("MIS_GOOGLE_SHEET_URL")
        if sheet_url:
            ok, detail = _log_mis_to_google_sheet(mis_row)
            if not ok:
                mis_errors.append(f"Sheet failed: {detail}")

        # Fallback / parallel: Gmail (MIS_EMAIL_TO supports comma-separated list)
        if _get_secret("GMAIL_USERNAME"):
            subject = (
                f"[Commission Extractor] MIS — {len(mis_file_details)} file(s), "
                f"{total_pages} page(s), {mis_finished:%Y-%m-%d %H:%M}"
            )
            body_lines = [
                "Commission Extractor — MIS Usage Report",
                "=" * 44,
                f"Timestamp    : {mis_row['Timestamp']}",
                f"Duration     : {mis_row['Duration (s)']}s",
                f"Files        : {mis_row['Files Processed']}",
                f"Pages        : {mis_row['Total Pages']}",
                f"Rows written : {mis_row['Rows Written']}",
                f"GV OCR calls : {mis_row['GV OCR Calls']}",
                f"AI calls     : {mis_row['AI (LLM) Calls']}",
                f"AI in tokens : ~{int(_ai_in_tokens)}",
                f"AI out tokens: ~{int(_ai_out_tokens)}",
                f"Est. AI cost : {mis_row['Est. AI Cost (INR)']}  (GPT-4o-mini rates, 1 USD ≈ 84 INR)",
                "",
                "Per-file detail:",
                mis_row["Per-file Detail"].replace(" | ", "\n"),
            ]
            ok, detail = _send_mis_gmail(subject, "\n".join(body_lines))
            if not ok:
                mis_errors.append(f"Gmail failed: {detail}")

        if not sheet_url and not _get_secret("GMAIL_USERNAME"):
            mis_errors.append("no MIS channel configured (set MIS_GOOGLE_SHEET_URL or GMAIL_USERNAME)")

        if mis_errors:
            _append_log(st.session_state.logs, f"(MIS: {'; '.join(mis_errors)})", log_placeholder)

    except Exception as exc:
        _append_log(st.session_state.logs, f"(MIS error: {exc})", log_placeholder)


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