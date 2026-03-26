#! /usr/bin/env python
"""Compatibility wrapper for the rewritten PyEMMA MSM pipeline.

This legacy entry point now forwards directly to:

    python felp.py msm-fit ...
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_felp_entrypoint():
    entrypoint = Path(__file__).with_name("felp.py")
    spec = importlib.util.spec_from_file_location("felp_entrypoint", entrypoint)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the FELP entry point.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    module = _load_felp_entrypoint()
    forwarded_argv = ["msm-fit", *(argv if argv is not None else sys.argv[1:])]
    return module.main(forwarded_argv)


if __name__ == "__main__":
    raise SystemExit(main())
