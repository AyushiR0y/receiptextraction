from __future__ import annotations

import os
import html
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
PRIMARY_DARK = "#00447D"
PRIMARY_SOFT = "#E8F2FC"
ACCENT       = "#F58220"
ACCENT_SOFT  = "#FEF0E4"
SUCCESS      = "#12A150"
BG           = "#F4F7FC"
SURFACE      = "#FFFFFF"
TEXT         = "#0D1F33"
TEXT_MUTED   = "#5A7492"
BORDER       = "rgba(13,31,51,0.09)"


_LOGO_PATH = Path(__file__).parent / "logo.png"

st.set_page_config(
    page_title="Commission Extractor",
    page_icon=str(_LOGO_PATH) if _LOGO_PATH.exists() else ":material/receipt_long:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# Plain (non-f) string so CSS braces need no escaping.  Colours are inlined to
# keep the stylesheet readable; they mirror the brand constants above.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: #F4F7FC !important;
    color: #0D1F33 !important;
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Hide Streamlit chrome ─────────────────────────────────────────────────
   Keep the header and toolbar in the DOM: the "expand sidebar" control lives
   inside them, so hiding either makes a collapsed sidebar impossible to
   reopen.  Hide only the menu / deploy / status items instead.            */
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbarActions"],
[data-testid="stMainMenu"], #MainMenu,
[data-testid="stAppDeployButton"],
[data-testid="stStatusWidget"] { display: none !important; }

/* Sidebar expand control (appears in the toolbar once the sidebar is closed) */
[data-testid="stExpandSidebarButton"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="stExpandSidebarButton"] button {
    background: #FFFFFF !important;
    border: 1px solid rgba(0,94,172,0.20) !important;
    border-radius: 10px !important;
    color: #005EAC !important;
    box-shadow: 0 2px 8px rgba(0,94,172,0.14) !important;
}
[data-testid="stExpandSidebarButton"] button:hover {
    background: #E8F2FC !important;
    border-color: #005EAC !important;
}

/* ── Page canvas ───────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] > .main { padding: 0 !important; }
[data-testid="stMainBlockContainer"], [data-testid="block-container"] {
    max-width: 1120px !important;
    padding: 1.1rem 2.4rem 5rem !important;
    margin: 0 auto !important;
}
[data-testid="stVerticalBlock"] { gap: 0.75rem; }

/* Inline SVG icons (no emoji anywhere in the UI) */
.ic { flex-shrink: 0; vertical-align: -0.18em; }

/* ═══════════════════════════════════════════════════════════════════════
   SIDEBAR
   ═══════════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #E9F2FC 0%, #DCE9F8 100%) !important;
    border-right: 1px solid rgba(0,94,172,0.14) !important;
    width: 268px !important;
    min-width: 268px !important;
}
/* Pull the sidebar content up: Streamlit reserves a tall header strip for the
   collapse chevron, which pushes the brand block far down the panel. */
[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
    padding: 0.4rem 0.75rem 0 !important;
    min-height: 0 !important;
    height: auto !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding: 0.15rem 1.05rem 2rem !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.35rem; }

/* Keep the collapse chevron permanently visible, not hover-only */
[data-testid="stSidebarCollapseButton"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="stSidebarCollapseButton"] button { color: #005EAC !important; }

.sb-brand {
    display: flex; align-items: center; gap: 0.65rem;
    padding: 0 0 1.1rem;
}
.sb-mark {
    width: 42px; height: 42px; border-radius: 13px;
    background: linear-gradient(140deg, #0071CE, #00447D);
    color: #fff; font-weight: 700; font-size: 1.15rem;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 6px 16px rgba(0,68,125,0.30);
    letter-spacing: -0.02em;
}
.sb-brand-name { font-size: 0.95rem; font-weight: 700; letter-spacing: -0.02em; color: #0D1F33; line-height: 1.15; }
.sb-brand-sub  { font-size: 0.68rem; color: #5A7492; font-weight: 500; letter-spacing: 0.02em; }

.sb-label {
    font-size: 0.63rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.11em; color: #6E8CAB;
    margin: 1.05rem 0 0.5rem;
}

.sb-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
.sb-metric {
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(0,94,172,0.13);
    border-radius: 12px;
    padding: 0.55rem 0.65rem;
}
.sb-metric .v { font-size: 1.22rem; font-weight: 700; color: #00447D; line-height: 1.05; letter-spacing: -0.03em; }
.sb-metric .k { font-size: 0.6rem; font-weight: 600; color: #6E8CAB; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.12rem; }

.sb-empty {
    border: 1px dashed rgba(0,94,172,0.28);
    border-radius: 14px;
    background: rgba(255,255,255,0.45);
    padding: 1.7rem 0.9rem;
    text-align: center;
}
.sb-empty .ico { color: #8CA9C4; }
.sb-empty .t   { font-size: 0.8rem; font-weight: 600; color: #4A688B; margin-top: 0.5rem; }
.sb-empty .s   { font-size: 0.68rem; color: #7C97B3; margin-top: 0.2rem; }

.sb-run {
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(0,94,172,0.13);
    border-left: 3px solid #12A150;
    border-radius: 10px;
    padding: 0.5rem 0.65rem;
    margin-bottom: 0.4rem;
}
.sb-run .rt { font-size: 0.74rem; font-weight: 700; color: #0D1F33; }
.sb-run .rs { font-size: 0.66rem; color: #5A7492; margin-top: 0.1rem; }

.sb-foot {
    display: flex; align-items: flex-start; gap: 0.45rem;
    margin-top: 1.5rem; padding-top: 1rem;
    border-top: 1px solid rgba(0,94,172,0.15);
    font-size: 0.68rem; color: #7C97B3; line-height: 1.55;
}
.sb-foot .ic { color: #93AFC7; margin-top: 0.12rem; }

/* ═══════════════════════════════════════════════════════════════════════
   PAGE TITLE
   ═══════════════════════════════════════════════════════════════════════ */
.page-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
    margin-bottom: 2rem;
}
.page-head-text { min-width: 0; }
.page-logo {
    height: 46px;
    width: auto;
    max-width: 210px;
    object-fit: contain;
    flex-shrink: 0;
}
.page-title {
    display: flex; align-items: center; gap: 0.7rem;
    font-size: 1.85rem; font-weight: 700; letter-spacing: -0.035em; color: #0D1F33;
    line-height: 1.15;
}
.page-title .glyph {
    width: 38px; height: 38px; border-radius: 11px;
    background: #E8F2FC; color: #005EAC;
    display: inline-flex; align-items: center; justify-content: center;
}
.page-sub { font-size: 0.95rem; color: #5A7492; margin-top: 0.45rem; font-weight: 400; }

/* ═══════════════════════════════════════════════════════════════════════
   STEPPER
   ═══════════════════════════════════════════════════════════════════════ */
.stepper {
    display: flex; align-items: center;
    background: #FFFFFF;
    border: 1px solid rgba(13,31,51,0.07);
    border-radius: 18px;
    padding: 1.25rem 1.75rem;
    box-shadow: 0 1px 2px rgba(13,31,51,0.04), 0 10px 30px -20px rgba(0,94,172,0.25);
    margin-bottom: 1.9rem;
}
.step { display: flex; align-items: center; gap: 0.7rem; }
.step-dot {
    width: 30px; height: 30px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700; flex-shrink: 0;
    background: #EDF2F8; color: #8CA5BE; border: 1px solid rgba(13,31,51,0.07);
    transition: all .25s ease;
}
.step.active .step-dot {
    background: linear-gradient(140deg, #0071CE, #00447D); color: #fff; border-color: transparent;
    box-shadow: 0 4px 12px rgba(0,94,172,0.32);
}
.step.done .step-dot { background: #E7F7EE; color: #12A150; border-color: rgba(18,161,80,0.30); }
.step-t { font-size: 0.86rem; font-weight: 700; color: #8CA5BE; letter-spacing: -0.01em; line-height: 1.2; }
.step-s { font-size: 0.71rem; color: #A3B6C9; margin-top: 0.08rem; }
.step.active .step-t, .step.done .step-t { color: #0D1F33; }
.step.active .step-s, .step.done .step-s { color: #5A7492; }
.step-line { flex: 1; height: 2px; background: #E4EBF3; margin: 0 1.1rem; border-radius: 2px; }
.step-line.filled { background: linear-gradient(90deg, #0071CE, #7FC4F5); }

/* ═══════════════════════════════════════════════════════════════════════
   CARDS
   ═══════════════════════════════════════════════════════════════════════ */
.card {
    background: #FFFFFF;
    border: 1px solid rgba(13,31,51,0.07);
    border-radius: 18px;
    padding: 1.3rem 1.4rem 1.4rem;
    box-shadow: 0 1px 2px rgba(13,31,51,0.04), 0 10px 30px -18px rgba(0,94,172,0.22);
    margin-bottom: 1rem;
}
.card-head {
    display: flex; align-items: center; gap: 0.55rem;
    font-size: 0.92rem; font-weight: 700; color: #0D1F33; letter-spacing: -0.015em;
    margin-bottom: 0.2rem;
}
.card-head .ic { color: #005EAC; }
.card-head .count {
    margin-left: auto; font-size: 0.72rem; font-weight: 600; color: #5A7492;
    background: #F1F5FA; border-radius: 999px; padding: 0.22rem 0.65rem;
}
.card-sub { font-size: 0.79rem; color: #5A7492; margin-bottom: 1rem; line-height: 1.5; }

/* Streamlit-rendered blocks that need to sit inside a visual card */
.stack-card {
    background: #FFFFFF;
    border: 1px solid rgba(13,31,51,0.07);
    border-radius: 18px;
    box-shadow: 0 1px 2px rgba(13,31,51,0.04), 0 10px 30px -18px rgba(0,94,172,0.22);
    padding: 1.3rem 1.4rem 0.4rem;
    margin-bottom: 0.15rem;
}
.stack-card + div [data-testid="stElementContainer"] { margin-top: 0; }

/* ═══════════════════════════════════════════════════════════════════════
   FILE UPLOADER → drop zone
   ═══════════════════════════════════════════════════════════════════════ */
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploaderDropzone"], section[data-testid="stFileUploadDropzone"] {
    border: 2px dashed rgba(0,94,172,0.30) !important;
    border-radius: 16px !important;
    background: linear-gradient(180deg, #FAFCFF 0%, #F1F7FE 100%) !important;
    padding: 2.6rem 1.5rem !important;
    min-height: 210px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    transition: border-color .2s ease, background .2s ease, transform .2s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover, section[data-testid="stFileUploadDropzone"]:hover {
    border-color: #005EAC !important;
    background: linear-gradient(180deg, #F4F9FF 0%, #E7F1FD 100%) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 0.15rem !important;
    color: #0D1F33 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] svg { fill: #005EAC !important; color: #005EAC !important; width: 2.1rem !important; height: 2.1rem !important; opacity: .9; }
[data-testid="stFileUploaderDropzoneInstructions"] span {
    font-size: 1.12rem !important; font-weight: 700 !important;
    color: #0D1F33 !important; letter-spacing: -0.02em !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] small {
    font-size: 0.78rem !important; color: #5A7492 !important; font-weight: 400 !important;
}
/* "Browse files" button inside the dropzone */
[data-testid="stFileUploaderDropzone"] button, section[data-testid="stFileUploadDropzone"] button {
    background: linear-gradient(135deg, #005EAC, #00447D) !important;
    color: #FFFFFF !important;
    border: 0 !important;
    border-radius: 11px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    margin-top: 1rem !important;
    box-shadow: 0 4px 14px rgba(0,94,172,0.28) !important;
    transition: transform .18s ease, box-shadow .18s ease !important;
}
[data-testid="stFileUploaderDropzone"] button:hover, section[data-testid="stFileUploadDropzone"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(0,94,172,0.34) !important;
}
/* Staged-file rows rendered by Streamlit */
[data-testid="stFileUploaderFile"] {
    background: #FFFFFF !important;
    border: 1px solid rgba(13,31,51,0.09) !important;
    border-radius: 12px !important;
    padding: 0.55rem 0.7rem !important;
    margin-top: 0.5rem !important;
    box-shadow: 0 1px 2px rgba(13,31,51,0.04) !important;
}
[data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileName"] {
    font-size: 0.82rem !important; font-weight: 600 !important; color: #0D1F33 !important;
}
[data-testid="stFileUploaderFile"] small { font-size: 0.7rem !important; color: #5A7492 !important; }
[data-testid="stFileUploaderFile"] svg { color: #005EAC !important; fill: #005EAC !important; }
[data-testid="stFileUploaderDeleteBtn"] button { color: #8CA5BE !important; }
[data-testid="stFileUploaderDeleteBtn"] button:hover { color: #D6453F !important; background: rgba(214,69,63,0.08) !important; }

/* ═══════════════════════════════════════════════════════════════════════
   STAGING SUMMARY (right rail)
   ═══════════════════════════════════════════════════════════════════════ */
.summary-hero {
    background: linear-gradient(150deg, #F3F9FF, #E8F2FC);
    border: 1px solid rgba(0,94,172,0.14);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.7rem;
}
.summary-hero .n { font-size: 2.15rem; font-weight: 700; color: #00447D; line-height: 1; letter-spacing: -0.04em; }
.summary-hero .l { font-size: 0.68rem; font-weight: 600; color: #5A7492; text-transform: uppercase; letter-spacing: 0.09em; margin-top: 0.25rem; }
.summary-hero .b { font-size: 0.75rem; color: #5A7492; margin-top: 0.45rem; }

.kind-row { display: flex; flex-direction: column; gap: 0.4rem; }
.kind {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.42rem 0.55rem;
    border-radius: 10px;
    background: #F8FAFD;
    border: 1px solid rgba(13,31,51,0.06);
}
.kind .tag {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.06em;
    padding: 0.2rem 0.45rem; border-radius: 6px; min-width: 42px; text-align: center;
}
.tag.pdf { background: #FDECEC; color: #C0392B; }
.tag.img { background: #E8F2FC; color: #005EAC; }
.tag.zip { background: #FDF3E4; color: #B4690E; }
.tag.doc { background: #FDF3E4; color: #B4690E; }
.tag.oth { background: #EEF2F7; color: #5A7492; }
.kind .nm { font-size: 0.78rem; color: #0D1F33; font-weight: 500; }
.kind .ct { margin-left: auto; font-size: 0.76rem; font-weight: 700; color: #5A7492; }

.stage-empty {
    border: 1px dashed rgba(13,31,51,0.16);
    border-radius: 14px;
    padding: 2.4rem 1rem;
    text-align: center;
    background: #FAFCFE;
}
.stage-empty .i { color: #A9BED2; }
.stage-empty .t { font-size: 0.83rem; font-weight: 600; color: #5A7492; margin-top: 0.55rem; }
.stage-empty .s { font-size: 0.72rem; color: #93A9BF; margin-top: 0.2rem; }

/* ═══════════════════════════════════════════════════════════════════════
   FORMAT STRIP
   ═══════════════════════════════════════════════════════════════════════ */
.fmt-label {
    font-size: 0.63rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.11em; color: #8CA5BE; margin: 0 0 0.7rem;
}
.fmt-strip { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; }
.fmt {
    display: flex; align-items: center; gap: 0.5rem;
    background: #FFFFFF;
    border: 1px solid rgba(13,31,51,0.08);
    border-radius: 12px;
    padding: 0.45rem 0.8rem 0.45rem 0.5rem;
    box-shadow: 0 1px 2px rgba(13,31,51,0.04);
}
.fmt .tag { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.06em; padding: 0.22rem 0.45rem; border-radius: 6px; }
.fmt .d { font-size: 0.76rem; color: #5A7492; }

/* ═══════════════════════════════════════════════════════════════════════
   BUTTONS
   ═══════════════════════════════════════════════════════════════════════ */
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(135deg, #005EAC, #00447D) !important;
    color: #FFFFFF !important;
    border: 0 !important;
    border-radius: 13px !important;
    padding: 0.78rem 1.4rem !important;
    font-weight: 700 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.94rem !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 6px 18px -4px rgba(0,94,172,0.45) !important;
    transition: transform .18s ease, box-shadow .18s ease, background .2s ease !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #0071CE, #00538F) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 10px 26px -6px rgba(0,94,172,0.52) !important;
}
.stButton > button:active, .stFormSubmitButton > button:active { transform: translateY(0) !important; }
.stButton > button:focus:not(:active) { color: #FFFFFF !important; }
.stButton > button p, .stFormSubmitButton > button p { font-weight: 700 !important; }

/* Download button — success tone */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #12A150, #0C8340) !important;
    box-shadow: 0 6px 18px -4px rgba(18,161,80,0.45) !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: linear-gradient(135deg, #16B85C, #0E9349) !important;
    box-shadow: 0 10px 26px -6px rgba(18,161,80,0.52) !important;
}

/* ═══════════════════════════════════════════════════════════════════════
   PROGRESS + LOG CONSOLE
   ═══════════════════════════════════════════════════════════════════════ */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #005EAC, #F58220) !important;
    border-radius: 999px !important;
}
.stProgress > div > div > div {
    background: #E4EDF7 !important;
    border-radius: 999px !important;
    height: 7px !important;
}

[data-testid="stCode"] { border-radius: 0 0 14px 14px !important; overflow: hidden !important; }
[data-testid="stCode"] pre {
    background: #0B1B2B !important;
    border: 1px solid rgba(0,94,172,0.28) !important;
    border-radius: 14px !important;
    padding: 0.9rem 1rem !important;
    max-height: 330px !important;
    overflow-y: auto !important;
}
[data-testid="stCode"] code, [data-testid="stCode"] pre span {
    color: #C6DDF2 !important;
    font-family: 'DM Mono', ui-monospace, monospace !important;
    font-size: 0.755rem !important;
    line-height: 1.62 !important;
}
[data-testid="stCode"] pre::-webkit-scrollbar { width: 8px; }
[data-testid="stCode"] pre::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.16); border-radius: 8px; }
[data-testid="stCodeBlock"] button { color: #7FA6C8 !important; }

.console-head {
    display: flex; align-items: center; gap: 0.5rem;
    font-size: 0.85rem; font-weight: 700; color: #0D1F33;
    margin: 0.6rem 0 0.65rem;
}
.console-head .ic { color: #005EAC; }
.console-head .live {
    margin-left: auto; font-size: 0.63rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #5A7492;
    background: #F1F5FA; border-radius: 999px; padding: 0.18rem 0.55rem;
}

/* ═══════════════════════════════════════════════════════════════════════
   RESULT PANEL
   ═══════════════════════════════════════════════════════════════════════ */
.result-card {
    background: linear-gradient(150deg, #F2FBF6, #E7F7EE);
    border: 1px solid rgba(18,161,80,0.24);
    border-radius: 18px;
    padding: 1.3rem 1.4rem 0.7rem;
    box-shadow: 0 10px 30px -18px rgba(18,161,80,0.4);
}
.result-card .rh {
    display: flex; align-items: center; gap: 0.55rem;
    font-size: 0.95rem; font-weight: 700; color: #0C8340;
}
.result-card .rs { font-size: 0.8rem; color: #40765A; margin: 0.3rem 0 1rem; line-height: 1.5; }

/* ═══════════════════════════════════════════════════════════════════════
   MISC STREAMLIT OVERRIDES
   ═══════════════════════════════════════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: 13px !important;
    border: 1px solid rgba(13,31,51,0.08) !important;
    font-size: 0.85rem !important;
}
[data-testid="stExpander"] {
    border: 1px solid rgba(13,31,51,0.07) !important;
    border-radius: 14px !important;
    background: #FFFFFF !important;
    box-shadow: 0 1px 2px rgba(13,31,51,0.04) !important;
}
[data-testid="stExpander"] summary { font-weight: 700 !important; font-size: 0.86rem !important; color: #0D1F33 !important; }
[data-testid="stExpander"] summary:hover { color: #005EAC !important; }

[data-testid="stColumn"] { padding: 0 0.45rem !important; }
[data-testid="stColumn"]:first-child { padding-left: 0 !important; }
[data-testid="stColumn"]:last-child  { padding-right: 0 !important; }

.spacer-xs { height: 0.35rem; }
.spacer-sm { height: 0.75rem; }
.spacer-md { height: 1.25rem; }

@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] { padding: 1rem 1rem 3rem !important; }
    .page-head { flex-direction: column; align-items: flex-start; gap: 0.9rem; }
    .page-logo { height: 34px; order: -1; }
    .page-title { font-size: 1.4rem; }
    .stepper { flex-direction: column; align-items: flex-start; gap: 0.75rem; }
    .step-line { display: none; }
}
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


# ─────────────────────────────────────────────────────────────────────────────
# PRESENTATION HELPERS  (display only — no effect on extraction)
# ─────────────────────────────────────────────────────────────────────────────

# Line icons drawn inline as SVG so the UI carries no emoji.  Each entry is the
# body of a 24×24 stroke icon that inherits the surrounding text colour.
_ICON_PATHS = {
    "extract":  '<path d="M12 3v11"/><path d="m8 11 4 4 4-4"/>'
                '<path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>',
    "upload":   '<path d="M12 16V4"/><path d="m8 8 4-4 4 4"/>'
                '<path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>',
    "stack":    '<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 16 9 5 9-5"/><path d="m3 12 9 5 9-5"/>',
    "inbox":    '<path d="M22 12h-6l-2 3h-4l-2-3H2"/>'
                '<path d="M5.4 5.1 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.4-6.9A2 2 0 0 0 16.8 4H7.2a2 2 0 0 0-1.8 1.1Z"/>',
    "activity": '<path d="m4 17 6-6-6-6"/><path d="M12 19h8"/>',
    "check":    '<path d="M22 11.1V12a10 10 0 1 1-5.9-9.1"/><path d="m22 4-10 10-3-3"/>',
    "clock":    '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "shield":   '<path d="M12 3 5 6v5c0 4.4 3 8.4 7 9.6 4-1.2 7-5.2 7-9.6V6l-7-3Z"/>',
    "receipt":  '<path d="M5 3v18l2-1.4 2 1.4 2-1.4 2 1.4 2-1.4 2 1.4V3l-2 1.4L13 3l-2 1.4L9 3 7 4.4 5 3Z"/>'
                '<path d="M9 8h6"/><path d="M9 12h6"/>',
}


@st.cache_data(show_spinner=False)
def _logo_data_uri(max_height: int = 120) -> str:
    """Return logo.png as a downscaled base64 data URI (empty string if absent).

    Raw HTML cannot reference local files, so the image is inlined.  It is
    resized first to keep the inlined payload small, and cached so the encode
    happens once per session rather than on every rerun.
    """
    if not _LOGO_PATH.exists():
        return ""
    try:
        import base64
        import io
        from PIL import Image

        with Image.open(_LOGO_PATH) as img:
            img = img.convert("RGBA")
            if img.height > max_height:
                scale = max_height / img.height
                img = img.resize((max(1, round(img.width * scale)), max_height), Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return ""


def _icon(name: str, size: int = 16, stroke: float = 1.8) -> str:
    """Return an inline SVG icon that inherits the current text colour."""
    body = _ICON_PATHS.get(name, "")
    return (
        f'<svg class="ic" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{body}</svg>'
    )


_KIND_BY_SUFFIX = {
    ".pdf":  ("PDF", "pdf", "PDF documents"),
    ".jpg":  ("IMG", "img", "Images"),
    ".jpeg": ("IMG", "img", "Images"),
    ".png":  ("IMG", "img", "Images"),
    ".zip":  ("ZIP", "zip", "Archives"),
    ".doc":  ("DOC", "doc", "Word documents"),
    ".docx": ("DOC", "doc", "Word documents"),
}


def _human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.0f} {unit}" if unit in ("B", "KB") else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def _kind_of(file_name: str) -> tuple[str, str, str]:
    return _KIND_BY_SUFFIX.get(Path(str(file_name)).suffix.lower(), ("FILE", "oth", "Other"))


def _console_block(placeholder, lines: list[str]) -> None:
    """Render the activity log into `placeholder`.  Stays hidden until a run starts."""
    if not lines:
        placeholder.empty()
        return
    with placeholder.container():
        st.markdown(
            f'<div class="console-head">{_icon("activity")}Activity log'
            f'<span class="live">{len(lines)} entries</span></div>',
            unsafe_allow_html=True,
        )
        st.code("\n".join(lines[-250:]), language="text")


def _append_log(logs: list[str], message: str, placeholder) -> None:
    logs.append(message)
    _console_block(placeholder, logs)


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
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "result_path" not in st.session_state:
    st.session_state.result_path = ""
if "logs" not in st.session_state:
    st.session_state.logs = []
if "runs" not in st.session_state:
    st.session_state.runs = []          # display-only history of this session

# Files already picked in a previous rerun — used to drive the stepper and the
# staging summary before the uploader widget is re-instantiated below.
_pending = st.session_state.get("uploaded_files") or []
_has_files  = bool(_pending)
_has_result = bool(st.session_state.result_path and Path(st.session_state.result_path).exists())


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="sb-brand">
        <div class="sb-mark">{_icon("receipt", 20)}</div>
        <div>
            <div class="sb-brand-name">Commission Extractor</div>
            <div class="sb-brand-sub">Finance automation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _staged_bytes = sum(getattr(f, "size", 0) or 0 for f in _pending)

    st.markdown(f"""
    <div class="sb-label">Ready to extract</div>
    <div class="sb-metrics">
        <div class="sb-metric"><div class="v">{len(_pending)}</div><div class="k">Files</div></div>
        <div class="sb-metric"><div class="v">{_human_size(_staged_bytes)}</div><div class="k">Size</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Recent runs</div>', unsafe_allow_html=True)
    if st.session_state.runs:
        _run_html = "".join(
            f'<div class="sb-run"><div class="rt">{html.escape(r["time"])}</div>'
            f'<div class="rs">{r["files"]} file(s) &rarr; {r["rows"]} row(s)</div></div>'
            for r in reversed(st.session_state.runs[-6:])
        )
        st.markdown(_run_html, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="sb-empty">
            <div class="ico">{_icon("clock", 22)}</div>
            <div class="t">No runs yet</div>
            <div class="s">Your results will appear here</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sb-foot">{_icon("shield", 14)}
        <span>Your files stay private and are deleted automatically when you close this page.</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEAD
# ─────────────────────────────────────────────────────────────────────────────
_logo_uri = _logo_data_uri()
_logo_html = (
    f'<img class="page-logo" src="{_logo_uri}" alt="Company logo">' if _logo_uri else ""
)

st.markdown(f"""
<div class="page-head">
    <div class="page-head-text">
        <div class="page-title"><span class="glyph">{_icon("extract", 20)}</span>Commission Extraction</div>
        <div class="page-sub">Upload your receipts and download a ready-to-use Excel workbook.</div>
    </div>
    {_logo_html}
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEPPER  (reflects real state: files staged → result written)
# ─────────────────────────────────────────────────────────────────────────────
def _step_class(step_no: int) -> str:
    active = 3 if _has_result else (2 if _has_files else 1)
    if step_no < active:
        return "step done"
    if step_no == active:
        return "step active"
    return "step"


st.markdown(f"""
<div class="stepper">
    <div class="{_step_class(1)}">
        <div class="step-dot">1</div>
        <div><div class="step-t">Upload receipts</div><div class="step-s">Add files, folder or ZIP</div></div>
    </div>
    <div class="step-line {'filled' if _has_files else ''}"></div>
    <div class="{_step_class(2)}">
        <div class="step-dot">2</div>
        <div><div class="step-t">Extract &amp; map</div><div class="step-s">Read fields, match agents</div></div>
    </div>
    <div class="step-line {'filled' if _has_result else ''}"></div>
    <div class="{_step_class(3)}">
        <div class="step-dot">3</div>
        <div><div class="step-t">Review &amp; download</div><div class="step-s">Validate math and export</div></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD ROW — drop zone (left) + staging summary & CTA (right)
# ─────────────────────────────────────────────────────────────────────────────
col_drop, col_stage = st.columns([1.55, 0.95], gap="medium")

with col_drop:
    st.markdown(f"""
    <div class="stack-card">
        <div class="card-head">{_icon("upload")}Upload receipts</div>
        <div class="card-sub">Add as many files as you like. Password-protected PDFs and ZIP folders
        are handled for you.</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload files, folder, or ZIP",
        key="uploaded_files",
        accept_multiple_files=True,
        type=SUPPORTED_TYPES,
        label_visibility="collapsed",
        help="Select individual PDFs/images, a whole folder, or ZIP archives. ZIPs are unpacked automatically.",
    )

with col_stage:
    _files = uploaded_files or []
    if _files:
        _total_bytes = sum(getattr(f, "size", 0) or 0 for f in _files)
        _by_kind: dict[str, dict] = {}
        for f in _files:
            tag, cls, label = _kind_of(f.name)
            entry = _by_kind.setdefault(tag, {"cls": cls, "label": label, "n": 0})
            entry["n"] += 1
        _kind_html = "".join(
            f'<div class="kind"><span class="tag {v["cls"]}">{html.escape(k)}</span>'
            f'<span class="nm">{html.escape(v["label"])}</span>'
            f'<span class="ct">{v["n"]}</span></div>'
            for k, v in sorted(_by_kind.items(), key=lambda kv: -kv[1]["n"])
        )
        st.markdown(f"""
        <div class="card">
            <div class="card-head">{_icon("stack")}Selected files</div>
            <div class="summary-hero">
                <div class="n">{len(_files)}</div>
                <div class="l">Files ready</div>
                <div class="b">{_human_size(_total_bytes)} in total</div>
            </div>
            <div class="kind-row">{_kind_html}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card">
            <div class="card-head">{_icon("stack")}Selected files</div>
            <div class="stage-empty">
                <div class="i">{_icon("inbox", 26)}</div>
                <div class="t">Nothing selected yet</div>
                <div class="s">Add receipts on the left to begin</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    process_clicked = st.button(
        "Start extraction",
        use_container_width=True,
        type="primary",
    )

st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SUPPORTED FORMATS STRIP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fmt-label">Accepted formats</div>
<div class="fmt-strip">
    <div class="fmt"><span class="tag pdf">PDF</span><span class="d">Including password-protected</span></div>
    <div class="fmt"><span class="tag img">IMG</span><span class="d">JPG &amp; PNG scans</span></div>
    <div class="fmt"><span class="tag zip">ZIP</span><span class="d">Folders &amp; archives</span></div>
    <div class="fmt"><span class="tag doc">DOC</span><span class="d">Word documents</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE / STATUS / DOWNLOAD PLACEHOLDERS
# ─────────────────────────────────────────────────────────────────────────────
status_placeholder   = st.empty()
progress_placeholder = st.empty()
log_placeholder      = st.empty()
download_placeholder = st.empty()


def render_logs() -> None:
    _console_block(log_placeholder, st.session_state.logs)


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
    progress = progress_placeholder.progress(0)
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
                        "  ! This may be a scanned PDF.  GOOGLE_VISION_API_KEY is not set — "
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

    _append_log(st.session_state.logs, f"Done. Wrote {len(df)} row(s) to {output_file}", log_placeholder)
    status_placeholder.success(f"Done — {len(df)} row(s) written successfully.")
    progress_placeholder.empty()
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

        # Display-only run history for the sidebar (no effect on extraction).
        st.session_state.runs.append({
            "time"   : mis_finished.strftime("%d %b %Y, %I:%M %p"),
            "files"  : len(mis_file_details),
            "rows"   : len(df),
            "seconds": round(duration_s, 1),
        })

    except Exception as exc:
        _append_log(st.session_state.logs, f"(MIS error: {exc})", log_placeholder)


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.result_path:
    result_file = Path(st.session_state.result_path)
    if result_file.exists():
        with download_placeholder.container():
            st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-card">
                <div class="rh">{_icon("check", 18)}Extraction complete</div>
                <div class="rs">Your workbook is ready — one row per receipt, with agent codes filled in
                and the commission amounts checked.</div>
            </div>
            """, unsafe_allow_html=True)
            st.download_button(
                label="Download Excel workbook",
                data=result_file.read_bytes(),
                file_name=result_file.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
