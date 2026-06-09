from __future__ import annotations

"""Input traversal and document processing operations."""

from . import extractor as _extractor

ReceiptLineItem = _extractor.ReceiptLineItem
BANK_PASSWORDS = _extractor.BANK_PASSWORDS

find_candidate_files = _extractor.find_candidate_files
process_spreadsheet = _extractor.process_spreadsheet
process_pdf = _extractor.process_pdf
process_image = _extractor.process_image
process_path = _extractor.process_path
build_placeholder_row = _extractor.build_placeholder_row

__all__ = [
    "ReceiptLineItem",
    "BANK_PASSWORDS",
    "find_candidate_files",
    "process_spreadsheet",
    "process_pdf",
    "process_image",
    "process_path",
    "build_placeholder_row",
]
