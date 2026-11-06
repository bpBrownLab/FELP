"""Convenience wrapper for ``python felp.py msm-plot``."""

from __future__ import annotations

import sys

from _dispatch import dispatch


if __name__ == "__main__":
    raise SystemExit(dispatch("msm-plot", sys.argv[1:]))
