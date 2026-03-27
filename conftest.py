from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
VENDORED_PARMED_ROOT = REPO_ROOT / "ParmEd"

for candidate in (REPO_ROOT, VENDORED_PARMED_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
