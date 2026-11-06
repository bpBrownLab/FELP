"""Thin application entry point for FELP."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _bootstrap_package() -> None:
    package_root = Path(__file__).with_name("felp")
    if "felp" in sys.modules and getattr(sys.modules["felp"], "__path__", None):
        return

    spec = importlib.util.spec_from_file_location(
        "felp",
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to bootstrap the FELP package.")

    module = importlib.util.module_from_spec(spec)
    sys.modules["felp"] = module
    spec.loader.exec_module(module)


def main(argv: list[str] | None = None) -> int:
    _bootstrap_package()
    from felp.__main__ import main as app_main

    return app_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


