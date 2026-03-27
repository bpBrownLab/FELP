from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
VENDORED_PARMED_ROOT = REPO_ROOT / "ParmEd"

def _promote_sys_path_entry(path: Path) -> None:
    target = path.resolve()
    kept_entries: list[str] = []
    for entry in sys.path:
        try:
            if Path(entry).resolve() == target:
                continue
        except Exception:
            pass
        if entry == str(path):
            continue
        kept_entries.append(entry)
    sys.path[:] = [str(path)] + kept_entries


for candidate in (REPO_ROOT, VENDORED_PARMED_ROOT):
    _promote_sys_path_entry(candidate)


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
