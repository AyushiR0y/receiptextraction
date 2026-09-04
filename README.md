# Commission Extractor

Reads insurance commission / brokerage invoices (PDF, scanned images, ZIP archives,
Excel workbooks) and turns them into one normalized Excel workbook — one row per
invoice line item, with agent codes filled in from the agent master and the GST
arithmetic re-checked.

Two front ends share the same extraction engine:

| Front end | Entry point | Use for |
| --- | --- | --- |
| Streamlit web app | `streamlit_app.py` | Upload-and-download UI, deployed on Streamlit Community Cloud |
| Command line | `pdf_clean.py` / `python -m commission_extractor.cli` | Batch runs over a folder, scripted or scheduled jobs |

---

## What it does

1. **Discovers files** — walks a folder recursively for `.pdf`, `.jpg`, `.jpeg`,
   `.png`, `.zip`, `.xlsx`, `.xlsm`, `.xlsb`, `.xls`. ZIPs are unpacked and their
   contents processed as if they had been passed directly.
2. **Reads text** — embedded PDF text first (pypdf / PyMuPDF); falls back to OCR
   for scanned pages. OCR order: RapidOCR (local, bundled) → EasyOCR / Tesseract
   if installed → Google Cloud Vision when `GOOGLE_VISION_API_KEY` is set. Page
   text is cached under `.ocr_cache/`, so a re-run of the same document is
   near-instant. Encrypted PDFs are opened with the built-in `BANK_PASSWORDS`
   table in [extractor.py](commission_extractor/extractor.py#L89) or a password
   you pass in.
3. **Extracts fields** — a large regex/heuristic layer plus issuer-specific
   handlers (J&K Bank, India Post, Dhanlaxmi, City Union, NKGSB, Probitas, Ethika,
   Catalyst, InCred, J.B. Boda, Mahindra, Bajaj Housing, Coverkraft, Motilal …).
   When a page is ambiguous, an optional Azure OpenAI call re-reads it and the
   answer is reconciled against the regex result.
4. **Normalizes tax** — [tax_logic.py](commission_extractor/tax_logic.py) decides
   CGST+SGST vs IGST vs UTGST, repairs known issuer quirks, and back-fills
   brokerage from total and GST only when the 18% relationship is unambiguous.
5. **Maps agents** — matches the extracted party name / PAN against
   `agent_details_full.xlsx` (any sheet exposing `AGENT_CODE` + `Name` columns) to
   fill `AGENT_CODE` and the clean agent name, with fuzzy matching at a 0.6
   similarity threshold as fallback.
6. **Validates the math** — every row is re-checked arithmetically (and optionally
   by the LLM) so no row stays marked `Math Valid = YES` when brokerage + GST does
   not reconcile to the invoice total. The written workbook is validated once more
   in place as a final safety net.
7. **Writes output** — `<output>.xlsx` with the 27 columns of `OUTPUT_COLUMNS`,
   plus `<output>_audit.xlsx` with per-source coverage and an optional
   reconciliation against a previous run.

### Output columns

`AGENT_CODE`, `Agent Name`, `Agent PAN`, `Name of Service Receipient`,
`BALIC STATE`, `BALIC GSTN`, `BROKER GSTN STATE`, `BROKER GSTN`,
`Vendor Inv Date`, `Vendor Inv No`, `Total Inv Amt`, `BROKERAGE Amount`,
`CGST @ 9%`, `SGST @ 9%`, `UTGST`, `IGST`, `GST TOTAL AMT`, `DATE_FROM`,
`DATE_TO`, `Narration`, `Type`, `Micro/Non Micro`, `SAC Code`, `Source File`,
`Source Page`, `Math Valid`, `Missing Field and Why`

---

## Layout

```
streamlit_app.py              Streamlit UI: upload -> run -> live log -> download
pdf_clean.py                  Legacy CLI shim (re-exports the package API)
commission_extractor/
  api.py                      Public import surface
  cli.py                      main() for the command line
  workflow.py                 Sequential run() - appends rows to the workbook as it goes
  extractor.py                The engine: OCR, parsing, mapping, validation (~7.8k lines)
  tax_logic.py                GST mode selection and issuer-specific repairs
  mapping.py                  Agent master loading + name/PAN matching
  io_ops.py / dataframe_ops.py / engine.py    Focused facades over extractor.py
agent_details_full.xlsx       Agent master (AGENT_CODE / Name / PAN / GSTN)
logo.png                      Branding shown in the app header
requirements.txt              Python dependencies
packages.txt                  APT packages for Streamlit Cloud (poppler-utils)
.streamlit/config.toml        Theme
.streamlit/secrets.toml.example    Template for the Streamlit Cloud secrets block
```

`workflow.run()` processes files one at a time and appends each result to the
workbook immediately, so a crash still leaves partial output on disk.
`extractor.run()` is a parallel variant (`ThreadPoolExecutor`, 8 workers by
default) that writes once at the end — faster for large batches when OCR goes
through Google Vision, which is network-bound rather than CPU-bound. The CLI uses
the sequential path.

---

## Setup

Requires **Python 3.10+** (the checked-in virtualenv is 3.10.10).

```powershell
# PowerShell, from the project root
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# bash / Git Bash
python -m venv venv
source venv/Scripts/activate      # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

Poppler is optional — page rendering prefers PyMuPDF, which needs no external
binary. For the `pdf2image` fallback, either drop a Poppler build in
`poppler-25.12.0/Library/bin` next to the project (auto-detected) or install
`poppler-utils` on Linux.

### Configuration

Create a `.env` in the project root (it is git-ignored); the extractor loads it
automatically on first use:

```ini
GOOGLE_VISION_API_KEY=<your key>
AZURE_OPENAI_API_KEY=<your key>
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-10-21
```

| Variable | Required | Effect |
| --- | --- | --- |
| `GOOGLE_VISION_API_KEY` | No, but needed for scanned PDFs | Enables Google Vision OCR on low-confidence pages |
| `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY` | No | Enables the LLM re-read of ambiguous pages and the AI math check |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | No | Deployment to call; defaults to `gpt-4o-mini` |
| `AZURE_OPENAI_API_VERSION` | No | Defaults to `2024-10-21` |
| `OCR_CACHE` | No | Set to `0` to disable the `.ocr_cache/` page-text cache |
| `ARITHMETIC_AUTOFILL` | No | Set to `1` to let the extractor derive amounts arithmetically; off by default so only values read from the document are written |

Both integrations are optional. Without them extraction still runs on local OCR
and the regex layer alone, with lower recall on scanned documents.

---

## Running the Streamlit app

```powershell
venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

```bash
# after activating the venv
streamlit run streamlit_app.py
```

Opens on <http://localhost:8501>; choose another port with
`streamlit run streamlit_app.py --server.port 8502`.

Upload individual PDFs/images, whole folders, or ZIP archives — ZIPs are unpacked
for you and password-protected PDFs are matched against the built-in bank
password table. The activity log streams per-page extractor progress, and the
finished workbook is offered as a download (written to a temp directory, never
into the repo).

### Deploying on Streamlit Community Cloud

The app is pure Streamlit and runs on the free tier. Point the deployment at
`streamlit_app.py`; `requirements.txt` and `packages.txt` are picked up
automatically. Copy `.streamlit/secrets.toml.example` into **App → Settings →
Secrets** and replace the placeholders — the app forwards those secrets into
`os.environ` at startup so the extractor finds them.

---

## Running from the command line

```powershell
venv\Scripts\python.exe pdf_clean.py --input "C:\path\to\invoices" --output "results.xlsx"
```

```bash
python pdf_clean.py --input ./invoices --output ./results.xlsx
```

Equivalent module form:

```bash
python -m commission_extractor.cli --input ./invoices --output ./results.xlsx
```

| Flag | Required | Meaning |
| --- | --- | --- |
| `--input` | yes | File or folder to process (folders are walked recursively) |
| `--output` | yes | Destination `.xlsx` |
| `--password` | no | Override password for encrypted PDFs, tried before the built-in bank table |
| `--verbose` | no | Debug-level logging |
| `--baseline-output` | no | A previous results workbook; the audit report then lists rows missing/added against it |
| `--audit-output` | no | Audit workbook path; defaults to `<output>_audit.xlsx` |

Examples:

```bash
# single file
python pdf_clean.py --input "./test/3346_Bajaj Life Insurance Limited.pdf" --output out.xlsx

# whole folder, verbose, reconciled against last month's run
python pdf_clean.py --input ./invoices --output ./sep.xlsx \
  --baseline-output ./aug.xlsx --audit-output ./sep_audit.xlsx --verbose

# encrypted PDFs sharing one password
python pdf_clean.py --input ./invoices --output out.xlsx --password "MyPassword"

# skip the OCR cache for a clean re-read
OCR_CACHE=0 python pdf_clean.py --input ./invoices --output out.xlsx
```

On Windows PowerShell, set an environment variable for the run with
`$env:OCR_CACHE = "0"` before the command.

### Ad-hoc check of the agent mapping

```bash
python test_agent_mapping.py
```

Loads the agent master, re-applies the mapping to an existing
`commission invoices extracted.xlsx`, and prints before/after fill rates. It is a
scratch script rather than a test-suite entry point, and needs that workbook
present in the working directory.

---

## MIS logging

After every Streamlit run the app silently records one usage row (files, pages,
rows, OCR and LLM call counts, estimated AI cost in INR, per-file detail). It
posts to a Google Apps Script web app first and falls back to Gmail SMTP.
Configure it through secrets: `MIS_GOOGLE_SHEET_URL`, `GMAIL_USERNAME`,
`GMAIL_APP_PASSWORD`, `MIS_EMAIL_TO` (comma-separated addresses supported). With
none of these set the run works normally and the log simply notes that no MIS
channel is configured.

> `mis_appscript.gs` in the repo is currently empty — the Apps Script that
> receives the POST has to be pasted in before that path will work.

---

## Notes and gotchas

- **Secrets do not belong in git.** `.env` and `.streamlit/secrets.toml` are
  git-ignored; the `.example` file is the only one meant to be committed, and it
  should carry placeholders rather than live credentials.
- `.ocr_cache/` grows without bound. Delete it to reclaim space; page text is
  re-OCR'd on the next run.
- The cache is keyed on file digest + page, so re-running the same batch after a
  code change still reuses old OCR text — clear the cache when debugging OCR
  itself.
- `Math Valid = NO` is informational, not a failure: the row is still written,
  with `Missing Field and Why` explaining what did not reconcile.
- `agent_details_full.xlsx` must sit in the working directory (or beside the
  package) or every `AGENT_CODE` comes out blank; the log warns when it is not
  found.
