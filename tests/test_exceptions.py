"""Tests for babellm.exceptions module."""

from babellm.exceptions import (
    AuthenticationError,
    BabelLMError,
    ConnectionError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    StreamingError,
)


class TestBabelLMError:
    """Test base BabelLMError."""

    def test_error_creation(self):
        """Test creating base error."""
        error = BabelLMError("Test error")
        assert str(error) == "Test error"

    def test_error_inheritance(self):
        """Test that all exceptions inherit from BabelLMError."""
        assert issubclass(ProviderError, BabelLMError)
        assert issubclass(AuthenticationError, BabelLMError)
        assert issubclass(RateLimitError, BabelLMError)
        assert issubclass(ModelNotFoundError, BabelLMError)
        assert issubclass(ConnectionError, BabelLMError)
        assert issubclass(StreamingError, BabelLMError)


class TestProviderError:
    """Test ProviderError exception."""

    def test_provider_error_creation(self):
        """Test creating a provider error."""
        raw = {"error": "Something went wrong"}
        error = ProviderError("Provider failed", status_code=500, raw=raw)
        assert str(error) == "Provider failed"
        assert error.status_code == 500
        assert error.raw == raw

    def test_provider_error_without_raw(self):
        """Test provider error without raw data."""
        error = ProviderError("Error", status_code=400)
        assert error.status_code == 400
        assert error.raw is None


class TestAuthenticationError:
    """Test AuthenticationError exception."""

    def test_auth_error_401(self):
        """Test authentication error with 401 status."""
        error = AuthenticationError("Auth failed", status_code=401)
        assert error.status_code == 401
        assert issubclass(AuthenticationError, ProviderError)

    def test_auth_error_403(self):
        """Test authentication error with 403 status."""
        error = AuthenticationError("Forbidden", status_code=403)
        assert error.status_code == 403


class TestRateLimitError:
    """Test RateLimitError exception."""

    def test_rate_limit_error(self):
        """Test rate limit error."""
        error = RateLimitError("Rate limited", status_code=429)
        assert error.status_code == 429
        assert issubclass(RateLimitError, ProviderError)


class TestModelNotFoundError:
    """Test ModelNotFoundError exception."""

    def test_model_not_found_error(self):
        """Test model not found error."""
        error = ModelNotFoundError("Model not found", status_code=404)
        assert error.status_code == 404
        assert issubclass(ModelNotFoundError, ProviderError)


class TestConnectionError:
    """Test ConnectionError exception."""

    def test_connection_error(self):
        """Test connection error."""
        error = ConnectionError("Network failed")
        assert str(error) == "Network failed"
        assert issubclass(ConnectionError, BabelLMError)


class TestStreamingError:
    """Test StreamingError exception."""

    def test_streaming_error(self):
        """Test streaming error."""
        error = StreamingError("Stream interrupted")
        assert str(error) == "Stream interrupted"
        assert issubclass(StreamingError, BabelLMError)
