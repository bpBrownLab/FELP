from __future__ import annotations

import importlib.util
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


def _purge_parmed_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "parmed" or module_name.startswith("parmed."):
            del sys.modules[module_name]


def _force_load_vendored_parmed() -> None:
    package_root = VENDORED_PARMED_ROOT / "parmed"
    init_file = package_root / "__init__.py"
    if not init_file.is_file():
        return

    existing_parmed = sys.modules.get("parmed")
    if existing_parmed is not None and _path_is_within(getattr(existing_parmed, "__file__", None), VENDORED_PARMED_ROOT):
        return

    _purge_parmed_modules()
    spec = importlib.util.spec_from_file_location(
        "parmed",
        init_file,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to create an import spec for vendored ParmEd at {init_file}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules["parmed"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        _purge_parmed_modules()
        raise


_force_load_vendored_parmed()
