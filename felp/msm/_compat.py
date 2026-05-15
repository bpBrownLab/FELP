"""Compatibility patches for legacy PyEMMA dependencies."""

from __future__ import annotations


def patch_threadpoolctl_missing_library_paths() -> None:
    """Silence threadpoolctl callback tracebacks for stale shared-library paths.

    Some PyEMMA/deeptime validation paths call ``threadpoolctl`` repeatedly while
    estimating MSMs. On this NFS/HPC stack, ``dl_iterate_phdr`` can report a
    library path that disappears before ``threadpoolctl`` resolves it, producing
    noisy "Exception ignored on calling ctypes callback" tracebacks. Returning an
    empty path makes threadpoolctl skip that stale entry.
    """

    try:
        import threadpoolctl
    except ImportError:
        return

    original = getattr(threadpoolctl, "_realpath", None)
    if original is None or getattr(original, "_felp_missing_path_safe", False):
        return

    def safe_realpath(filepath):
        try:
            return original(filepath)
        except (FileNotFoundError, OSError, TypeError, ValueError):
            return ""

    safe_realpath._felp_missing_path_safe = True  # type: ignore[attr-defined]
    threadpoolctl._realpath = safe_realpath
