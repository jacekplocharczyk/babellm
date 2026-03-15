"""Tests for OllamaProvider."""

import pytest
import respx
import httpx

from babellm.providers.ollama.client import OllamaProvider
from babellm.types import Message, Role


@pytest.fixture
def provider():
    """Create OllamaProvider instance."""
    provider = OllamaProvider()
    yield provider
    provider.close()


@pytest.fixture
async def async_provider():
    """Create async OllamaProvider instance."""
    provider = OllamaProvider()
    yield provider
    await provider.aclose()


class TestOllamaProviderChat:
    """Test chat endpoint."""

    def test_chat_basic(self, provider):
        """Test basic chat request."""
        with respx.mock:
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "message": {"role": "assistant", "content": "Hello!"},
                        "model": "llama3.2",
                        "done": True,
                    },
                )
            )
            response = provider.chat(
                messages=[Message(role=Role.USER, content="Hi")],
                model="llama3.2",
            )
            assert response.message.content == "Hello!"
            assert response.model == "llama3.2"
            assert response.done is True

    def test_chat_with_temperature(self, provider):
        """Test chat with temperature parameter."""
        with respx.mock:
            request = respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "message": {"role": "assistant", "content": "Response"},
                        "done": True,
                    },
                )
            )
            provider.chat(
                messages=[Message(role=Role.USER, content="Hi")],
                model="llama3.2",
                temperature=0.5,
            )
            # Verify temperature was included in request
            assert request.called

    def test_chat_with_max_tokens(self, provider):
        """Test chat with max_tokens parameter."""
        with respx.mock:
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "message": {"role": "assistant", "content": "Short"},
                        "done": True,
                    },
                )
            )
            response = provider.chat(
                messages=[Message(role=Role.USER, content="Hi")],
                model="llama3.2",
                max_tokens=50,
            )
            assert response.message.content == "Short"

    def test_chat_multi_turn(self, provider):
        """Test multi-turn conversation."""
        with respx.mock:
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "message": {"role": "assistant", "content": "It's 4"},
                        "done": True,
                    },
                )
            )
            messages = [
                Message(role=Role.SYSTEM, content="You are helpful."),
                Message(role=Role.USER, content="What is 2+2?"),
            ]
            response = provider.chat(messages=messages, model="llama3.2")
            assert response.message.content == "It's 4"


class TestOllamaProviderChatStream:
    """Test streaming chat."""

    def test_chat_stream_basic(self, provider):
        """Test basic streaming chat."""
        with respx.mock:
            ndjson = (
                '{"message": {"content": "Hello"}, "done": false}\n'
                '{"message": {"content": " "}, "done": false}\n'
                '{"message": {"content": "world"}, "done": true}\n'
            )
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(200, content=ndjson)
            )
            chunks = list(
                provider.chat_stream(
                    messages=[Message(role=Role.USER, content="Hi")],
                    model="llama3.2",
                )
            )
            assert len(chunks) == 3
            assert chunks[0].delta == "Hello"
            assert chunks[1].delta == " "
            assert chunks[2].delta == "world"
            assert chunks[2].done is True


class TestOllamaProviderGenerate:
    """Test generate endpoint."""

    def test_generate_basic(self, provider):
        """Test basic generate request."""
        with respx.mock:
            respx.post("http://localhost:11434/api/generate").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "response": "def hello():\n    return 'world'",
                        "model": "codellama",
                        "done": True,
                    },
                )
            )
            response = provider.generate(
                prompt="def hello():",
                model="codellama",
            )
            assert "def hello()" in response.text
            assert response.model == "codellama"

    def test_generate_with_system(self, provider):
        """Test generate with system prompt."""
        with respx.mock:
            respx.post("http://localhost:11434/api/generate").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "response": "x = 5\ny = 3",
                        "model": "codellama",
                        "done": True,
                    },
                )
            )
            response = provider.generate(
                prompt="Create variables",
                system="Output code only.",
                model="codellama",
            )
            assert response.text == "x = 5\ny = 3"


class TestOllamaProviderGenerateStream:
    """Test streaming generate."""

    def test_generate_stream_basic(self, provider):
        """Test basic streaming generate."""
        with respx.mock:
            ndjson = (
                '{"response": "def ", "done": false}\n'
                '{"response": "hello():", "done": true}\n'
            )
            respx.post("http://localhost:11434/api/generate").mock(
                return_value=httpx.Response(200, content=ndjson)
            )
            chunks = list(
                provider.generate_stream(
                    prompt="def hello():",
                    model="codellama",
                )
            )
            assert len(chunks) == 2
            assert chunks[0].delta == "def "


class TestOllamaProviderEmbed:
    """Test embedding endpoint."""

    def test_embed_single_string(self, provider):
        """Test embedding a single string."""
        with respx.mock:
            embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
            respx.post("http://localhost:11434/api/embed").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "embedding": embedding,
                        "model": "nomic-embed-text",
                    },
                )
            )
            response = provider.embed(
                input="Hello world",
                model="nomic-embed-text",
            )
            assert response.embedding == embedding

    def test_embed_list_of_strings(self, provider):
        """Test embedding a list of strings."""
        with respx.mock:
            embedding = [0.1, 0.2]
            respx.post("http://localhost:11434/api/embed").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "embedding": embedding,
                        "model": "nomic-embed-text",
                    },
                )
            )
            response = provider.embed(
                input=["Hello", "world"],
                model="nomic-embed-text",
            )
            assert response.embedding == embedding


class TestOllamaProviderAsync:
    """Test async methods."""

    @pytest.mark.asyncio
    async def test_achat(self, async_provider):
        """Test async chat."""
        with respx.mock:
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "message": {"role": "assistant", "content": "Hello!"},
                        "done": True,
                    },
                )
            )
            response = await async_provider.achat(
                messages=[Message(role=Role.USER, content="Hi")],
                model="llama3.2",
            )
            assert response.message.content == "Hello!"

    @pytest.mark.asyncio
    async def test_achat_stream(self, async_provider):
        """Test async streaming chat."""
        with respx.mock:
            ndjson = (
                '{"message": {"content": "Hi"}, "done": true}\n'
            )
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(200, content=ndjson)
            )
            chunks = []
            async for chunk in async_provider.achat_stream(
                messages=[Message(role=Role.USER, content="Hi")],
                model="llama3.2",
            ):
                chunks.append(chunk)
            assert len(chunks) == 1
            assert chunks[0].delta == "Hi"

    @pytest.mark.asyncio
    async def test_agenerate(self, async_provider):
        """Test async generate."""
        with respx.mock:
            respx.post("http://localhost:11434/api/generate").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "response": "generated text",
                        "done": True,
                    },
                )
            )
            response = await async_provider.agenerate(
                prompt="Test",
                model="llama3.2",
            )
            assert response.text == "generated text"

    @pytest.mark.asyncio
    async def test_agenerate_stream(self, async_provider):
        """Test async streaming generate."""
        with respx.mock:
            ndjson = '{"response": "text", "done": true}\n'
            respx.post("http://localhost:11434/api/generate").mock(
                return_value=httpx.Response(200, content=ndjson)
            )
            chunks = []
            async for chunk in async_provider.agenerate_stream(
                prompt="Test",
                model="llama3.2",
            ):
                chunks.append(chunk)
            assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_aembed(self, async_provider):
        """Test async embed."""
        with respx.mock:
            respx.post("http://localhost:11434/api/embed").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "embedding": [0.1, 0.2],
                        "model": "test",
                    },
                )
            )
            response = await async_provider.aembed(
                input="test",
                model="test",
            )
            assert response.embedding == [0.1, 0.2]


class TestOllamaProviderCustomization:
    """Test customization options."""

    def test_custom_base_url(self):
        """Test provider with custom base URL."""
        provider = OllamaProvider(base_url="http://192.168.1.100:11434")
        assert provider._http.base_url == "http://192.168.1.100:11434"

    def test_custom_timeout(self):
        """Test provider with custom timeout."""
        provider = OllamaProvider(timeout=300.0)
        assert provider._http.timeout == 300.0

    def test_custom_headers(self):
        """Test provider with custom headers."""
        headers = {"X-Custom": "header"}
        provider = OllamaProvider(headers=headers)
        assert provider._http._headers == headers


class TestOllamaProviderContextManager:
    """Test context manager support."""

    def test_sync_context_manager(self):
        """Test sync context manager."""
        with OllamaProvider() as provider:
            assert isinstance(provider, OllamaProvider)

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager."""
        async with OllamaProvider() as provider:
            assert isinstance(provider, OllamaProvider)
