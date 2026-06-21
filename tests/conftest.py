"""Test path setup for the nested Aurora checkout."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AURORA_ROOT = ROOT / "aurora"

for path in (ROOT, AURORA_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
