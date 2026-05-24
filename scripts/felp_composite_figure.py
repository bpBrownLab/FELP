"""Convenience wrapper for ``python felp.py composite-figure``."""

from __future__ import annotations

import sys

from _dispatch import dispatch


if __name__ == "__main__":
    raise SystemExit(dispatch("composite-figure", sys.argv[1:]))
