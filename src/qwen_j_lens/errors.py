"""Project-specific exceptions."""


class QwenJLenseError(RuntimeError):
    """Base package error."""


class MissingModelError(QwenJLenseError):
    """Raised when requested model files are unavailable locally."""


class ArtifactError(QwenJLenseError):
    """Raised when an artifact is missing or invalid."""

