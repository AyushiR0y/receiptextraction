from __future__ import annotations

"""Compatibility facade for historical engine imports.

New code should prefer focused modules such as mapping.py, io_ops.py,
dataframe_ops.py, and workflow.py.
"""

from .extractor import *
from .workflow import build_arg_parser, main, run
