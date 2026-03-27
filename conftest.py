from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
VENDORED_PARMED_ROOT = REPO_ROOT / "ParmEd"

for candidate in (REPO_ROOT, VENDORED_PARMED_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _path_is_within(path: str | None, root: Path) -> bool:
    if not path:
        return False
    try:
        Path(path).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


existing_parmed = sys.modules.get("parmed")
if existing_parmed is not None and not _path_is_within(getattr(existing_parmed, "__file__", None), VENDORED_PARMED_ROOT):
    for module_name in list(sys.modules):
        if module_name == "parmed" or module_name.startswith("parmed."):
            del sys.modules[module_name]
