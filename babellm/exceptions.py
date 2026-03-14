"""Exception hierarchy for babellm."""


class BabelLMError(Exception):
    """Root exception for all babellm errors."""

    pass


class ProviderError(BabelLMError):
    """Exception raised when a provider returns an error."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        raw: dict | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.raw = raw
        super().__init__(message)


class AuthenticationError(ProviderError):
    """Raised when authentication fails (401/403)."""

    pass


class RateLimitError(ProviderError):
    """Raised when rate limit is exceeded (429)."""

    pass


class ModelNotFoundError(ProviderError):
    """Raised when a model is not found (404)."""

    pass


class ConnectionError(BabelLMError):
    """Raised when a network connection error occurs."""

    pass


class StreamingError(BabelLMError):
    """Raised when an error occurs during streaming."""

    pass
