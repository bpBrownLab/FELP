"""Convenience wrapper for ``python felp.py msm-fit``."""

from __future__ import annotations

import sys

from _dispatch import dispatch


if __name__ == "__main__":
    raise SystemExit(dispatch("msm-fit", sys.argv[1:]))
