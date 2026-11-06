"""Shared dispatch helper for standalone FELP wrapper scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_felp_entrypoint():
    """Load the top-level ``felp.py`` entry point as a module."""

    entrypoint = Path(__file__).resolve().parents[1] / "felp.py"
    spec = importlib.util.spec_from_file_location("felp_entrypoint", entrypoint)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the FELP entry point.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dispatch(subcommand: str, argv: list[str] | None = None) -> int:
    """Forward wrapper-script arguments to a specific FELP subcommand."""

    module = _load_felp_entrypoint()
    forwarded_argv = [subcommand, *(argv if argv is not None else sys.argv[1:])]
    return module.main(forwarded_argv)
