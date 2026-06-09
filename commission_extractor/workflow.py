from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

from . import extractor as _extractor
from .dataframe_ops import rows_to_dataframe, write_audit_report
from .io_ops import ReceiptLineItem, build_placeholder_row, find_candidate_files, process_path
from .mapping import AGENT_CODE_BY_NAME, apply_agent_code_mapping_to_dataframe, load_agent_codes_from_xlsx

LOGGER = _extractor.LOGGER
# Do NOT copy the counter integers here — integers are immutable and the copy
# would freeze at 0 forever.  Read them live from _extractor at summary time.


def run(
    input_path: Path,
    output_file: Path,
    password: Optional[str] = None,
    baseline_output: Optional[Path] = None,
    audit_output: Optional[Path] = None,
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    load_agent_codes_from_xlsx()

    # Create an initial, header-only Excel so a file exists immediately
    # and partial progress is visible if processing fails later.
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        import pandas as _pd

        header_df = _pd.DataFrame(columns=_extractor.OUTPUT_COLUMNS)
        header_df.to_excel(output_file, index=False)
        LOGGER.info("Created initial output workbook: %s", output_file)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to create initial output workbook %s: %s", output_file, exc)

    # helper to append rows incrementally so partial results persist
    def _append_rows(output_file: Path, row_dicts: List[dict]):
        if not row_dicts:
            return 0
        cols = list(_extractor.OUTPUT_COLUMNS)
        # Try openpyxl for efficient append
        try:
            from openpyxl import load_workbook
            import tempfile
            import shutil

            wb = load_workbook(filename=str(output_file))
            ws = wb.active
            for rd in row_dicts:
                row = [rd.get(c, "") for c in cols]
                ws.append(row)
            # Save to a temp file then atomically replace to avoid permission issues
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xlsx")
            try:
                os.close(tmp_fd)
                wb.save(tmp_path)
                shutil.move(tmp_path, str(output_file))
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
            return len(row_dicts)
        except Exception:
            LOGGER.debug("openpyxl append failed; falling back to CSV or full rewrite", exc_info=True)

        # Fallback: append to a CSV so we always persist results even if Excel engine missing
        try:
            import csv

            csv_path = output_file.with_suffix(".csv")
            write_header = not csv_path.exists()
            with csv_path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=cols)
                if write_header:
                    writer.writeheader()
                for rd in row_dicts:
                    writer.writerow({k: rd.get(k, "") for k in cols})
            LOGGER.info("Appended %s rows to CSV fallback %s", len(row_dicts), csv_path)
            return len(row_dicts)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to persist rows for %s: %s", output_file, exc)
            return 0

    all_rows: List[ReceiptLineItem] = []
    processed_sources: List[str] = []
    for file_path in find_candidate_files(input_path):
        processed_sources.append(str(file_path))
        LOGGER.info("Processing: %s", file_path)
        try:
            rows = process_path(file_path, password)
            if rows:
                # immediately append these rows to the output so partial results persist
                row_dicts = [r.values for r in rows]
                appended = _append_rows(output_file, row_dicts)
                LOGGER.info("Appended %s rows from %s", appended, file_path)
                all_rows.extend(rows)
            else:
                all_rows.append(build_placeholder_row(str(file_path), "", "No identifiable receipt data extracted from document"))
                # persist placeholder row as well
                _append_rows(output_file, [all_rows[-1].values])
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Failed to process %s: %s", file_path, exc)
            all_rows.append(build_placeholder_row(str(file_path), "", f"Extraction failed: {exc}"))
            # persist the failure placeholder
            _append_rows(output_file, [all_rows[-1].values])

    # After iterating all files, build final dataframe (for audit/reconciliation)
    df = rows_to_dataframe(all_rows)

    if not df.empty and AGENT_CODE_BY_NAME:
        df = apply_agent_code_mapping_to_dataframe(df)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_file, index=False)
    write_audit_report(
        output_file=output_file,
        df=df,
        processed_sources=processed_sources,
        baseline_output=baseline_output,
        audit_output=audit_output,
    )
    LOGGER.info("Wrote %s rows to %s", len(df), output_file)
    LOGGER.info(
        "Usage summary | google_vision_calls=%s | azure_ai_calls=%s | azure_ai_input_chars=%s | azure_ai_output_chars=%s",
        _extractor.GOOGLE_VISION_CALL_COUNT,
        _extractor.AZURE_AI_CALL_COUNT,
        _extractor.AZURE_AI_INPUT_CHARS,
        _extractor.AZURE_AI_OUTPUT_CHARS,
    )
    return output_file


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract receipt data from PDF/JPG/ZIP/Excel and write normalized Excel output.",
    )
    parser.add_argument("--input", required=True, help="Input file or folder path")
    parser.add_argument("--output", required=True, help="Output Excel file path (.xlsx)")
    parser.add_argument(
        "--password",
        required=False,
        default=None,
        help="Optional override password for encrypted PDFs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logs",
    )
    parser.add_argument(
        "--baseline-output",
        required=False,
        default=None,
        help="Optional previous output Excel file for non-blocking reconciliation",
    )
    parser.add_argument(
        "--audit-output",
        required=False,
        default=None,
        help="Optional audit report Excel file path; defaults to <output>_audit.xlsx",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _extractor.configure_logging(args.verbose)
    run(
        Path(args.input),
        Path(args.output),
        args.password,
        Path(args.baseline_output) if args.baseline_output else None,
        Path(args.audit_output) if args.audit_output else None,
    )
