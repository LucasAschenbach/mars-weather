"""Import path helpers for the nested Aurora checkout."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_aurora_on_path() -> None:
    """Make the vendored Aurora package importable from repo-root scripts."""

    repo_root = Path(__file__).resolve().parents[1]
    aurora_root = repo_root / "aurora"
    if (aurora_root / "aurora" / "__init__.py").exists():
        path = str(aurora_root)
        if path not in sys.path:
            sys.path.insert(0, path)
