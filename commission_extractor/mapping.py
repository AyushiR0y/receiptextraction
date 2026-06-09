from __future__ import annotations

"""Agent and identity mapping helpers.

This module provides a focused import surface while delegating implementation
to extractor.py.
"""

from . import extractor as _extractor

AGENT_CODE_BY_NAME = _extractor.AGENT_CODE_BY_NAME
AGENT_NAME_BY_CODE = _extractor.AGENT_NAME_BY_CODE
AGENT_INFO_BY_PAN = _extractor.AGENT_INFO_BY_PAN
AGENT_INFO_BY_CODE = _extractor.AGENT_INFO_BY_CODE

load_agent_codes_from_xlsx = _extractor.load_agent_codes_from_xlsx
find_best_matching_agent_code = _extractor.find_best_matching_agent_code
apply_agent_code_mapping_to_dataframe = _extractor.apply_agent_code_mapping_to_dataframe

__all__ = [
    "AGENT_CODE_BY_NAME",
    "AGENT_NAME_BY_CODE",
    "AGENT_INFO_BY_PAN",
    "AGENT_INFO_BY_CODE",
    "load_agent_codes_from_xlsx",
    "find_best_matching_agent_code",
    "apply_agent_code_mapping_to_dataframe",
]
