"""Custom exceptions for FELP."""


class FelpError(Exception):
    """Base exception for FELP."""


class ConfigurationError(FelpError):
    """Raised when CLI or config inputs are invalid."""


class DependencyError(FelpError):
    """Raised when an optional runtime dependency is unavailable."""


class ConversionError(FelpError):
    """Raised when structure conversion fails."""


class MicrostateAssignmentError(FelpError):
    """Raised when a frame cannot be assigned to a microstate."""
