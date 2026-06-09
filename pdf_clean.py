from __future__ import annotations

"""Compatibility entrypoint for legacy imports.

The extractor implementation now lives under commission_extractor.
"""

from commission_extractor.api import *
from commission_extractor.cli import main


if __name__ == "__main__":
    main()
