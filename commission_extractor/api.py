from __future__ import annotations

"""Public API surface for the commission extractor.

This module provides a readable, stable entrypoint while delegating extraction
logic to engine.py.
"""

from typing import List

from .import extractor as _extractor
from .dataframe_ops import OUTPUT_COLUMNS, rows_to_dataframe
from .io_ops import BANK_PASSWORDS, ReceiptLineItem, build_placeholder_row, find_candidate_files, process_image, process_path, process_pdf, process_spreadsheet
from .mapping import AGENT_CODE_BY_NAME, AGENT_NAME_BY_CODE, apply_agent_code_mapping_to_dataframe, load_agent_codes_from_xlsx
from .workflow import build_arg_parser, run

configure_logging = _extractor.configure_logging

# Keep broad compatibility for existing imports, including advanced utilities.
__all__ = [name for name in dir(_extractor) if not name.startswith("_")]


def __getattr__(name: str):
    return getattr(_extractor, name)


def __dir__() -> List[str]:
    return sorted(set(globals().keys()) | set(__all__))
