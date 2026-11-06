"""Convenience wrapper for ``python felp.py felp-plot``."""

from __future__ import annotations

import sys

from _dispatch import dispatch


if __name__ == "__main__":
    raise SystemExit(dispatch("felp-plot", sys.argv[1:]))
