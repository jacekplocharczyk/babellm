"""Tests for babellm.types module."""

from babellm.types import (
    ChatChunk,
    ChatResponse,
    EmbeddingResponse,
    GenerateChunk,
    GenerateResponse,
    Message,
    Role,
    Usage,
)


class TestRole:
    """Test Role enum."""

    def test_role_values(self):
        """Test role enum string values."""
        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.TOOL == "tool"

    def test_role_from_string(self):
        """Test creating role from string."""
        assert Role("system") == Role.SYSTEM
        assert Role("user") == Role.USER


class TestMessage:
    """Test Message dataclass."""

    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"

    def test_message_with_system_role(self):
        """Test message with system role."""
        msg = Message(role=Role.SYSTEM, content="You are helpful.")
        assert msg.role == Role.SYSTEM
        assert msg.content == "You are helpful."


class TestUsage:
    """Test Usage dataclass."""

    def test_usage_defaults(self):
        """Test usage defaults to zero."""
        usage = Usage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_usage_with_values(self):
        """Test usage with token counts."""
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30


class TestChatResponse:
    """Test ChatResponse dataclass."""

    def test_chat_response_creation(self):
        """Test creating a chat response."""
        msg = Message(role=Role.ASSISTANT, content="2+2 = 4")
        response = ChatResponse(message=msg, model="llama3.2", done=True)
        assert response.message == msg
        assert response.model == "llama3.2"
        assert response.done is True
        assert response.usage is None
        assert response.raw == {}

    def test_chat_response_with_usage(self):
        """Test chat response with usage data."""
        msg = Message(role=Role.ASSISTANT, content="Response")
        usage = Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        response = ChatResponse(message=msg, model="test", done=True, usage=usage)
        assert response.usage == usage

    def test_chat_response_with_raw(self):
        """Test chat response with raw data."""
        msg = Message(role=Role.ASSISTANT, content="Response")
        raw_data = {"key": "value", "nested": {"data": 123}}
        response = ChatResponse(
            message=msg, model="test", done=True, raw=raw_data
        )
        assert response.raw == raw_data


class TestGenerateResponse:
    """Test GenerateResponse dataclass."""

    def test_generate_response_creation(self):
        """Test creating a generate response."""
        response = GenerateResponse(
            text="def hello():\n    return 'world'", model="codellama", done=True
        )
        assert response.text == "def hello():\n    return 'world'"
        assert response.model == "codellama"
        assert response.done is True

    def test_generate_response_with_usage(self):
        """Test generate response with usage."""
        usage = Usage(prompt_tokens=5, completion_tokens=20, total_tokens=25)
        response = GenerateResponse(
            text="text", model="test", done=True, usage=usage
        )
        assert response.usage == usage


class TestEmbeddingResponse:
    """Test EmbeddingResponse dataclass."""

    def test_embedding_response_creation(self):
        """Test creating an embedding response."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        response = EmbeddingResponse(embedding=embedding, model="nomic-embed-text")
        assert response.embedding == embedding
        assert response.model == "nomic-embed-text"
        assert response.raw == {}

    def test_embedding_response_with_raw(self):
        """Test embedding response with raw data."""
        embedding = [0.1, 0.2]
        raw = {"something": "data"}
        response = EmbeddingResponse(
            embedding=embedding, model="test", raw=raw
        )
        assert response.raw == raw


class TestChatChunk:
    """Test ChatChunk dataclass."""

    def test_chat_chunk_creation(self):
        """Test creating a chat chunk."""
        chunk = ChatChunk(delta="Hello ", model="llama3.2", done=False)
        assert chunk.delta == "Hello "
        assert chunk.model == "llama3.2"
        assert chunk.done is False

    def test_chat_chunk_done(self):
        """Test chat chunk with done flag."""
        chunk = ChatChunk(delta="", model="test", done=True)
        assert chunk.done is True


class TestGenerateChunk:
    """Test GenerateChunk dataclass."""

    def test_generate_chunk_creation(self):
        """Test creating a generate chunk."""
        chunk = GenerateChunk(delta="def ", model="codellama", done=False)
        assert chunk.delta == "def "
        assert chunk.model == "codellama"
        assert chunk.done is False
