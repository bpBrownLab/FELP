"""Convenience wrapper for ``python felp.py run``."""

from __future__ import annotations

import sys

from _dispatch import dispatch


if __name__ == "__main__":
    raise SystemExit(dispatch("run", sys.argv[1:]))
