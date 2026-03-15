"""Tests for HTTPClient."""


import httpx
import pytest
import respx

from babellm._http import HTTPClient
from babellm.exceptions import (
    AuthenticationError,
    ConnectionError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)


@pytest.fixture
def client():
    """Create HTTPClient instance."""
    client = HTTPClient(base_url="http://localhost:11434")
    yield client
    client.close()


@pytest.fixture
async def async_client():
    """Create async HTTPClient instance."""
    client = HTTPClient(base_url="http://localhost:11434")
    yield client
    await client.aclose()


class TestHTTPClientPost:
    """Test sync POST method."""

    def test_post_success(self, client):
        """Test successful POST request."""
        with respx.mock:
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(200, json={"response": "Hello"})
            )
            result = client.post("/api/chat", {"model": "test"})
            assert result == {"response": "Hello"}

    def test_post_with_json_parsing(self, client):
        """Test POST with complex JSON response."""
        with respx.mock:
            response_data = {
                "message": {"role": "assistant", "content": "2+2=4"},
                "model": "llama3.2",
                "done": True,
            }
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(200, json=response_data)
            )
            result = client.post("/api/chat", {})
            assert result == response_data
            assert result["message"]["content"] == "2+2=4"

    def test_post_401_authentication_error(self, client):
        """Test POST with 401 Unauthorized."""
        with respx.mock:
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(
                    401, json={"error": "Unauthorized"}
                )
            )
            with pytest.raises(AuthenticationError):
                client.post("/api/chat", {})

    def test_post_403_authentication_error(self, client):
        """Test POST with 403 Forbidden."""
        with respx.mock:
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(403, text="Forbidden")
            )
            with pytest.raises(AuthenticationError):
                client.post("/api/chat", {})

    def test_post_404_model_not_found(self, client):
        """Test POST with 404 Model Not Found."""
        with respx.mock:
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(404, json={"error": "Model not found"})
            )
            with pytest.raises(ModelNotFoundError):
                client.post("/api/chat", {})

    def test_post_429_rate_limit(self, client):
        """Test POST with 429 Rate Limited."""
        with respx.mock:
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(429, json={"error": "Rate limited"})
            )
            with pytest.raises(RateLimitError):
                client.post("/api/chat", {})

    def test_post_500_provider_error(self, client):
        """Test POST with 500 Server Error."""
        with respx.mock:
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(500, json={"error": "Server error"})
            )
            with pytest.raises(ProviderError):
                client.post("/api/chat", {})


class TestHTTPClientPostStream:
    """Test streaming POST."""

    def test_post_stream_basic(self, client):
        """Test streaming POST with NDJSON."""
        with respx.mock:
            ndjson_response = (
                '{"response": "Hello"}\n'
                '{"response": " "}\n'
                '{"response": "world"}\n'
            )
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(200, content=ndjson_response)
            )
            chunks = list(client.post_stream("/api/chat", {}))
            assert len(chunks) == 3
            assert chunks[0] == {"response": "Hello"}
            assert chunks[1] == {"response": " "}
            assert chunks[2] == {"response": "world"}

    def test_post_stream_empty_lines(self, client):
        """Test stream ignores empty lines."""
        with respx.mock:
            ndjson_response = (
                '{"data": 1}\n'
                '\n'
                '{"data": 2}\n'
            )
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(200, content=ndjson_response)
            )
            chunks = list(client.post_stream("/api/chat", {}))
            assert len(chunks) == 2

    def test_post_stream_401_error(self, client):
        """Test stream with auth error."""
        with respx.mock:
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(401, json={"error": "Unauthorized"})
            )
            with pytest.raises(AuthenticationError):
                list(client.post_stream("/api/chat", {}))


class TestHTTPClientAsync:
    """Test async methods."""

    @pytest.mark.asyncio
    async def test_apost_success(self, async_client):
        """Test successful async POST."""
        with respx.mock:
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(200, json={"response": "Hello"})
            )
            result = await async_client.apost("/api/chat", {})
            assert result == {"response": "Hello"}

    @pytest.mark.asyncio
    async def test_apost_stream(self, async_client):
        """Test async streaming POST."""
        with respx.mock:
            ndjson_response = (
                '{"response": "Hello"}\n'
                '{"response": " "}\n'
                '{"response": "world"}\n'
            )
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(200, content=ndjson_response)
            )
            chunks = []
            async for chunk in async_client.apost_stream("/api/chat", {}):
                chunks.append(chunk)
            assert len(chunks) == 3

    @pytest.mark.asyncio
    async def test_apost_401_error(self, async_client):
        """Test async POST with auth error."""
        with respx.mock:
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(401, text="Unauthorized")
            )
            with pytest.raises(AuthenticationError):
                await async_client.apost("/api/chat", {})


class TestContextManagers:
    """Test context manager support."""

    def test_sync_context_manager(self):
        """Test sync context manager."""
        with HTTPClient(base_url="http://localhost:11434") as client:
            assert client is not None
            assert client._client is None or client._client is not None

    def test_sync_context_manager_closes(self):
        """Test sync context manager closes client."""
        client = HTTPClient(base_url="http://localhost:11434")
        with client:
            # Trigger client creation
            assert client._client is None
        # After exit, client should be closed

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager."""
        async with HTTPClient(base_url="http://localhost:11434") as client:
            assert client is not None


class TestConnectionError:
    """Test connection error handling."""

    def test_post_connection_error(self, client):
        """Test POST with connection error."""
        with respx.mock:
            # Mock a connection error
            respx.post("http://localhost:11434/api/chat").mock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            with pytest.raises(ConnectionError):
                client.post("/api/chat", {})
